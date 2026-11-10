"""
OKF-based retrieval: answer questions by navigating the Open Knowledge Format
bundle in ./okf/ (built by okf_build.py) instead of vector search.

Retrieval steps:
  1. Exact lookup: any NAICS code mentioned in the question (e.g. "524126")
     resolves directly to its OKF page.
  2. Keyword search: field-weighted BM25 over frontmatter (title, description,
     tags) and body. Frontmatter is boosted because OKF pages are curated
     around one entity each.
  3. Link expansion: pages linked from the top hits (e.g. a NAICS code's
     parent) are added as supporting context — the "graph" part of OKF.
  4. Grounded generation with the same provider/prompt as the RAG path
     (see providers.py — local Ollama Qwen by default).

Usage (CLI):
    python okf_retrieve.py "What NAICS code covers direct property and casualty insurers?"
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from rag_query import Citation, RAGResult, format_response

OKF_DIR = Path(os.environ.get("OKF_DIR", "okf"))
TOP_K = 5
MAX_EXPANSION = 3
SNIPPET_CHARS = 6000

FIELD_WEIGHTS = {"title": 3, "description": 2, "tags": 1, "body": 1}
BM25_K1, BM25_B = 1.2, 0.75

_STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it its of on or "
    "our that the their there this to under what when where which who why will with "
    "my me we you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_LINK = re.compile(r"\]\((/[^)\s]+\.md)\)")
_NAICS_CODE = re.compile(r"(?<!\d)(\d{2,6}|\d{2}-\d{2})(?!\d)")


@dataclass
class OkfDoc:
    link: str            # bundle-absolute path, e.g. "/naics/us/524126.md"
    meta: dict
    body: str

    @property
    def title(self) -> str:
        return str(self.meta.get("title", self.link))

    @property
    def links(self) -> list[str]:
        return _LINK.findall(self.body)


def tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) > 4 and tok.endswith("ies"):
            tok = tok[:-3] + "y"
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        out.append(tok)
    return out


def parse_doc(path: Path, root: Path) -> OkfDoc:
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    body = text
    if text.startswith("---\n"):
        end = text.index("\n---", 4)
        meta = yaml.safe_load(text[4:end]) or {}
        body = text[end + 4 :].lstrip("\n")
    return OkfDoc(link="/" + path.relative_to(root).as_posix(), meta=meta, body=body)


class OkfIndex:
    """In-memory BM25F-style index over an OKF bundle."""

    def __init__(self, docs: list[OkfDoc]):
        self.docs = docs
        self.by_link = {d.link: d for d in docs}
        self.by_code: dict[str, list[OkfDoc]] = {}
        for d in docs:
            if d.meta.get("code"):
                self.by_code.setdefault(str(d.meta["code"]), []).append(d)

        self.tfs: list[Counter] = []
        self.lengths: list[int] = []
        df: Counter = Counter()
        for d in docs:
            tf: Counter = Counter()
            fields = {
                "title": d.title,
                "description": str(d.meta.get("description", "")),
                "tags": " ".join(map(str, d.meta.get("tags") or [])),
                "body": d.body,
            }
            for name, text in fields.items():
                for tok in tokenize(text):
                    tf[tok] += FIELD_WEIGHTS[name]
            self.tfs.append(tf)
            self.lengths.append(sum(tf.values()))
            df.update(tf.keys())
        n = max(1, len(docs))
        self.avg_len = sum(self.lengths) / n if docs else 1.0
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @classmethod
    def load(cls, root: Path = OKF_DIR) -> "OkfIndex":
        if not root.exists():
            raise FileNotFoundError(
                f"OKF bundle not found at ./{root}. Run `python okf_build.py` first."
            )
        return cls([parse_doc(p, root) for p in sorted(root.rglob("*.md"))])

    def bm25(self, query: str, lob_filter: str | None = None) -> list[tuple[float, OkfDoc]]:
        q = tokenize(query)
        scored = []
        for i, d in enumerate(self.docs):
            if d.meta.get("type") == "Index":
                continue
            if lob_filter and d.meta.get("lob") != lob_filter:
                continue
            tf, dl = self.tfs[i], self.lengths[i]
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (BM25_K1 + 1) / (
                        f + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avg_len)
                    )
            if s > 0:
                scored.append((s, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def code_lookup(self, query: str, lob_filter: str | None = None) -> list[OkfDoc]:
        if lob_filter and lob_filter != "NAICS Classification":
            return []
        hits: list[OkfDoc] = []
        for code in _NAICS_CODE.findall(query):
            hits.extend(self.by_code.get(code, []))
        return hits

    def search(
        self, query: str, lob_filter: str | None = None, top_k: int = TOP_K
    ) -> list[tuple[float, OkfDoc]]:
        """Exact code hits (score 1.0) if any, else BM25 hits normalised to (0, 1];
        then up to MAX_EXPANSION linked pages."""
        results: list[tuple[float, OkfDoc]] = []
        seen: set[str] = set()
        for d in self.code_lookup(query, lob_filter):
            if d.link not in seen:
                results.append((1.0, d))
                seen.add(d.link)
        # An exact code match is authoritative: keyword-search filler around it
        # only adds near-miss codes the answer model may wrongly attribute.
        ranked = [] if results else self.bm25(query, lob_filter)
        top = ranked[0][0] if ranked else 1.0
        for s, d in ranked:
            if len(results) >= top_k:
                break
            if d.link not in seen:
                results.append((round(s / top, 4), d))
                seen.add(d.link)
        results = results[:top_k]

        # Link expansion: pull in pages the best hits point at (parent codes,
        # cross-references), scored just below the page that linked them.
        added = 0
        for s, d in list(results[:2]):
            for link in d.links:
                if added >= MAX_EXPANSION:
                    break
                nb = self.by_link.get(link)
                if nb and nb.link not in seen and nb.meta.get("type") != "Index":
                    results.append((round(s * 0.5, 4), nb))
                    seen.add(nb.link)
                    added += 1
        return results


@lru_cache(maxsize=1)
def get_index() -> OkfIndex:
    return OkfIndex.load(OKF_DIR)


def query_okf(
    question: str,
    lob_filter: str | None = None,
    top_k: int = TOP_K,
) -> RAGResult:
    """OKF retrieval + grounded generation. Same return type as rag_query.query."""
    from providers import generate_answer

    hits = get_index().search(question, lob_filter=lob_filter, top_k=top_k)
    if not hits and lob_filter:  # a wrong LOB guess shouldn't yield nothing
        hits = get_index().search(question, top_k=top_k)

    documents = [
        {
            "id": str(i),
            "title": f"{d.title}  |  {d.meta.get('resource', d.link)}  |  {d.meta.get('lob', '')}",
            "snippet": d.body[:SNIPPET_CHARS],
        }
        for i, (_, d) in enumerate(hits)
    ]
    answer, grounded = (
        generate_answer(question, documents)
        if documents
        else ("No pages in the OKF knowledge bundle matched this question.", [])
    )
    return RAGResult(
        question=question,
        answer=answer,
        citations=[
            Citation(
                source=str(d.meta.get("source", d.link)),
                page=int(d.meta.get("page") or 0),
                lob=str(d.meta.get("lob", "")),
                relevance_score=s,
            )
            for s, d in hits
        ],
        grounded_spans=grounded,
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip() or "What NAICS code covers insurance carriers?"
    print(format_response(query_okf(q)))
