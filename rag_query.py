"""
RAG query engine for P&C insurance underwriting guidelines.

Usage (CLI):
    python rag_query.py "What are eligibility criteria for frame construction?"

Usage (module):
    from rag_query import query, format_response
    result = query("Does a CGL policy cover completed operations claims?")
    print(format_response(result))
"""

import sys
from dataclasses import dataclass, field

import providers
from providers import SYSTEM_PROMPT, generate_answer  # noqa: F401  (re-exported)

CHROMA_DIR = "chroma_db"
TOP_K = 5


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
    grounded_spans: list[dict] = field(default_factory=list)  # citation spans


def query(
    question: str,
    lob_filter: str | None = None,
    top_k: int = TOP_K,
) -> RAGResult:
    """
    Retrieve top_k chunks from ChromaDB, then generate a grounded answer with
    inline citations. Embedding/generation backends come from `providers`
    (local Ollama Qwen by default, Cohere optional).

    Args:
        question:   Natural-language underwriting question.
        lob_filter: Optionally restrict retrieval to one LOB
                    ("Commercial Property", "Commercial General Liability", "Personal Auto").
        top_k:      Number of chunks to retrieve.

    Returns:
        RAGResult with the generated answer and source citations.
    """
    import chromadb  # imported lazily so the OKF path doesn't need it

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    name = providers.collection_name()
    try:
        collection = chroma_client.get_collection(name)
    except Exception as exc:
        raise RuntimeError(
            f"Chroma collection {name!r} not found — run `python ingest.py` with "
            f"EMBED_PROVIDER={providers.EMBED_PROVIDER} first."
        ) from exc

    # ── 1. Embed query ────────────────────────────────────────────────────────
    query_vec = providers.embed([question], input_type="search_query")[0]

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

    # ── 3–4. Grounded generation ─────────────────────────────────────────────
    answer, grounded = generate_answer(
        question,
        [
            {
                "id": str(i),
                "title": f"{m['source']}  |  Page {m['page']}  |  {m['lob']}",
                "snippet": chunk,
            }
            for i, (chunk, m) in enumerate(zip(chunks, metas))
        ],
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

    return RAGResult(
        question=question,
        answer=answer,
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
