"""
RAG query engine for P&C insurance underwriting guidelines.

Usage (CLI):
    python rag_query.py "What are eligibility criteria for frame construction?"

Usage (module):
    from rag_query import query, format_response
    result = query("Does a CGL policy cover completed operations claims?")
    print(format_response(result))
"""

import os
import sys
from dataclasses import dataclass, field

import cohere
import chromadb

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "underwriting_guidelines"
EMBED_MODEL = "embed-english-v3.0"
GENERATE_MODEL = "command-r-plus-08-2024"
TOP_K = 5

SYSTEM_PROMPT = """\
You are a senior P&C insurance underwriting assistant with deep expertise across \
Commercial Property, Commercial General Liability (CGL), and Personal Auto lines of business.

When answering:
- Ground every statement in the provided reference documents.
- Be precise: cite specific policy provisions, eligibility thresholds, exclusions, or \
  limit structures when relevant.
- Use professional underwriting terminology (occurrence vs. claims-made, TIV, BI/PD, \
  named-storm deductible, MVR, etc.).
- If the documents do not fully address the question, clearly state what is missing \
  rather than speculating.
- Keep your answer well-structured: a direct answer first, then supporting detail."""


@dataclass
class Citation:
    source: str
    page: int
    lob: str
    relevance_score: float  # 1 - cosine distance


@dataclass
class RAGResult:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded_spans: list[dict] = field(default_factory=list)  # raw Cohere citations


def query(
    question: str,
    lob_filter: str | None = None,
    top_k: int = TOP_K,
) -> RAGResult:
    """
    Retrieve top_k chunks from ChromaDB, then generate a grounded answer
    via Cohere Command R+ with inline citations.

    Args:
        question:   Natural-language underwriting question.
        lob_filter: Optionally restrict retrieval to one LOB
                    ("Commercial Property", "Commercial General Liability", "Personal Auto").
        top_k:      Number of chunks to retrieve.

    Returns:
        RAGResult with the generated answer and source citations.
    """
    if not COHERE_API_KEY:
        raise EnvironmentError("COHERE_API_KEY environment variable not set.")

    co = cohere.Client(api_key=COHERE_API_KEY)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    # ── 1. Embed query ────────────────────────────────────────────────────────
    embed_resp = co.embed(
        texts=[question],
        model=EMBED_MODEL,
        input_type="search_query",
    )
    query_vec = embed_resp.embeddings[0]

    # ── 2. Retrieve from Chroma ───────────────────────────────────────────────
    where = {"lob": {"$eq": lob_filter}} if lob_filter else None
    n = min(top_k, collection.count())

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[str] = results["documents"][0]
    metas: list[dict] = results["metadatas"][0]
    distances: list[float] = results["distances"][0]

    # ── 3. Build Cohere documents payload (grounded generation) ───────────────
    cohere_docs = [
        {
            "id": str(i),
            "title": f"{m['source']}  |  Page {m['page']}  |  {m['lob']}",
            "snippet": chunk,
        }
        for i, (chunk, m) in enumerate(zip(chunks, metas))
    ]

    # ── 4. Generate answer with Command R+ ───────────────────────────────────
    chat_resp = co.chat(
        model=GENERATE_MODEL,
        message=question,
        preamble=SYSTEM_PROMPT,
        documents=cohere_docs,
        temperature=0.1,
    )

    # ── 5. Build result ───────────────────────────────────────────────────────
    source_citations = [
        Citation(
            source=m["source"],
            page=m["page"],
            lob=m["lob"],
            relevance_score=round(1.0 - d, 4),
        )
        for m, d in zip(metas, distances)
    ]

    grounded = []
    if chat_resp.citations:
        for c in chat_resp.citations:
            grounded.append(
                {
                    "text": c.text,
                    "document_ids": c.document_ids,
                    "start": c.start,
                    "end": c.end,
                }
            )

    return RAGResult(
        question=question,
        answer=chat_resp.text,
        citations=source_citations,
        grounded_spans=grounded,
    )


def format_response(result: RAGResult) -> str:
    """Human-readable rendering of a RAGResult."""
    lines = [
        f"{'─'*70}",
        f"QUESTION:  {result.question}",
        f"{'─'*70}",
        "",
        result.answer,
        "",
        f"{'─'*70}",
        "SOURCES (ranked by relevance):",
    ]
    seen: set[tuple[str, int]] = set()
    for c in sorted(result.citations, key=lambda x: x.relevance_score, reverse=True):
        key = (c.source, c.page)
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"  [{c.relevance_score:.4f}]  {c.source}  "
            f"(page {c.page})  —  {c.lob}"
        )
    lines.append(f"{'─'*70}")
    return "\n".join(lines)


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        q = "What are eligibility criteria for frame construction commercial property?"
    result = query(q)
    print(format_response(result))
