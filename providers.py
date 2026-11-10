"""
Pluggable embedding and generation providers: Cohere (hosted) or Ollama (local).

Selected with environment variables (defaults keep everything local):

    EMBED_PROVIDER=ollama | cohere          (default: ollama)
    GENERATE_PROVIDER=ollama | cohere       (default: ollama)
    OLLAMA_URL=http://localhost:11434
    OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
    OLLAMA_GENERATE_MODEL=qwen3:8b
    COHERE_API_KEY=...                      (only needed for a cohere provider)

Each embedding provider writes to its own ChromaDB collection (see
`collection_name()`): vectors from different models live in different spaces
and must never be mixed in one index.
"""

from __future__ import annotations

import os
import re
import time

import httpx

EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "ollama").lower()
GENERATE_PROVIDER = os.environ.get("GENERATE_PROVIDER", "ollama").lower()

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_GENERATE_MODEL = os.environ.get("OLLAMA_GENERATE_MODEL", "qwen3:8b")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))

COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
COHERE_EMBED_MODEL = "embed-english-v3.0"
COHERE_GENERATE_MODEL = "command-r-plus-08-2024"

BASE_COLLECTION = "underwriting_guidelines"

# Qwen3-Embedding is instruction-aware: queries get a task prefix, documents don't.
QWEN_QUERY_INSTRUCT = (
    "Instruct: Given a P&C insurance underwriting or industry classification "
    "question, retrieve passages from guideline documents that answer it\nQuery: "
)

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


def _require(name: str, value: str) -> str:
    if not value:
        raise EnvironmentError(f"{name} environment variable not set.")
    return value


def collection_name(provider: str | None = None) -> str:
    """Chroma collection for an embedding provider. Cohere keeps the original name."""
    provider = (provider or EMBED_PROVIDER).lower()
    if provider == "cohere":
        return BASE_COLLECTION
    model = re.sub(r"[^a-zA-Z0-9_-]+", "-", OLLAMA_EMBED_MODEL).strip("-")
    return f"{BASE_COLLECTION}__{model}"


# ── Embedding ────────────────────────────────────────────────────────────────

def embed(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed texts. input_type: "search_document" or "search_query"."""
    if input_type not in ("search_document", "search_query"):
        raise ValueError(f"bad input_type {input_type!r}")
    if EMBED_PROVIDER == "cohere":
        return _cohere_embed(texts, input_type)
    if EMBED_PROVIDER == "ollama":
        return _ollama_embed(texts, input_type)
    raise ValueError(f"Unknown EMBED_PROVIDER {EMBED_PROVIDER!r}")


def _cohere_embed(texts: list[str], input_type: str) -> list[list[float]]:
    import cohere

    co = cohere.Client(api_key=_require("COHERE_API_KEY", COHERE_API_KEY))
    for attempt in range(5):
        try:
            return co.embed(texts=texts, model=COHERE_EMBED_MODEL, input_type=input_type).embeddings
        except cohere.errors.too_many_requests_error.TooManyRequestsError:
            wait = 60 * (attempt + 1)
            print(f"  rate-limited — waiting {wait}s …")
            time.sleep(wait)
    raise RuntimeError("Cohere embed rate-limit retries exhausted.")


def _ollama_embed(texts: list[str], input_type: str) -> list[list[float]]:
    if input_type == "search_query":
        texts = [QWEN_QUERY_INSTRUCT + t for t in texts]
    resp = httpx.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": OLLAMA_EMBED_MODEL, "input": texts},
        timeout=600,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama embed failed ({resp.status_code}): {resp.text[:200]}. "
            f"Is Ollama running and `ollama pull {OLLAMA_EMBED_MODEL}` done?"
        )
    return resp.json()["embeddings"]


# ── Grounded generation ──────────────────────────────────────────────────────

def generate_answer(question: str, documents: list[dict]) -> tuple[str, list[dict]]:
    """
    Answer `question` grounded in `documents` ({"id", "title", "snippet"}).
    Returns (answer_text, grounded_spans). Cohere returns character-level
    citation spans; Ollama answers cite documents inline as [id] and spans are
    derived from those markers.
    """
    if GENERATE_PROVIDER == "cohere":
        return _cohere_generate(question, documents)
    if GENERATE_PROVIDER == "ollama":
        return _ollama_generate(question, documents)
    raise ValueError(f"Unknown GENERATE_PROVIDER {GENERATE_PROVIDER!r}")


def _cohere_generate(question: str, documents: list[dict]) -> tuple[str, list[dict]]:
    import cohere

    co = cohere.Client(api_key=_require("COHERE_API_KEY", COHERE_API_KEY))
    resp = co.chat(
        model=COHERE_GENERATE_MODEL,
        message=question,
        preamble=SYSTEM_PROMPT,
        documents=documents,
        temperature=0.1,
    )
    grounded = [
        {"text": c.text, "document_ids": c.document_ids, "start": c.start, "end": c.end}
        for c in (resp.citations or [])
    ]
    return resp.text, grounded


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _ollama_generate(question: str, documents: list[dict]) -> tuple[str, list[dict]]:
    context = "\n\n".join(
        f"[{d['id']}] {d['title']}\n{d['snippet']}" for d in documents
    )
    prompt = (
        f"Reference documents:\n\n{context}\n\n---\n\nQuestion: {question}\n\n"
        "Answer using only the reference documents. After each claim, cite the "
        "supporting document number(s) in square brackets, e.g. [0] or [1, 3]."
    )
    resp = httpx.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_GENERATE_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_ctx": OLLAMA_NUM_CTX},
        },
        timeout=600,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama chat failed ({resp.status_code}): {resp.text[:200]}. "
            f"Is `ollama pull {OLLAMA_GENERATE_MODEL}` done?"
        )
    text = resp.json()["message"]["content"].strip()
    valid = {str(d["id"]) for d in documents}
    grounded = []
    for m in _CITE.finditer(text):
        ids = [i.strip() for i in m.group(1).split(",") if i.strip() in valid]
        if ids:
            grounded.append({"text": m.group(0), "document_ids": ids, "start": m.start(), "end": m.end()})
    return text, grounded
