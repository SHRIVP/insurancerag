"""
Ingest P&C insurance guideline PDFs into ChromaDB.

Pipeline:
  1. Load each PDF from ./data/ with PyMuPDF
  2. Chunk by paragraph, targeting ~300 tokens per chunk
  3. Embed chunks (search_document) with the configured provider —
     local Ollama qwen3-embedding by default, or Cohere embed-english-v3.0
  4. Store in local ChromaDB with metadata: source, page, lob.
     Each embedding provider has its own collection (providers.collection_name()).

Usage:
    python ingest.py                         # Ollama / Qwen3-Embedding (default)
    EMBED_PROVIDER=cohere python ingest.py   # Cohere
"""

import os
import re
import sys
import time
from pathlib import Path

import chromadb
import fitz  # PyMuPDF

import providers

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = providers.collection_name()
TARGET_TOKENS = 300
EMBED_BATCH = 90 if providers.EMBED_PROVIDER == "cohere" else 32  # Cohere max is 96

# Filename-prefix → LOB label
LOB_MAP = {
    "commercial_property": "Commercial Property",
    "cgl": "Commercial General Liability",
    "personal_auto": "Personal Auto",
    "naics": "NAICS Classification",
}


# ── Token estimation ─────────────────────────────────────────────────────────

def _token_estimate(text: str) -> int:
    """Rough token count: ~0.75 words per token (BPE approximation)."""
    return max(1, len(text.split()) * 4 // 3)


# ── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_page(text: str, target: int = TARGET_TOKENS) -> list[str]:
    """
    Split page text into chunks near `target` tokens.
    Strategy: split on blank lines (paragraphs), then greedily merge
    until the next paragraph would exceed the target.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for para in paragraphs:
        para_tokens = _token_estimate(para)
        if buf and buf_tokens + para_tokens > target:
            chunks.append("\n\n".join(buf))
            buf, buf_tokens = [], 0
        # If a single paragraph is longer than the target, hard-split it
        if para_tokens > target:
            words = para.split()
            step = max(1, target * 3 // 4)  # words ≈ tokens * 0.75
            for i in range(0, len(words), step):
                segment = " ".join(words[i : i + step])
                if segment:
                    chunks.append(segment)
        else:
            buf.append(para)
            buf_tokens += para_tokens

    if buf:
        chunks.append("\n\n".join(buf))

    return [c for c in chunks if len(c.strip()) >= 60]


# ── LOB detection ────────────────────────────────────────────────────────────

def _detect_lob(filename: str) -> str:
    stem = Path(filename).stem.lower()
    for prefix, label in LOB_MAP.items():
        if stem.startswith(prefix):
            return label
    return "Unknown"


# ── Per-PDF ingestion ────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: Path,
    collection: chromadb.Collection,
) -> int:
    lob = _detect_lob(pdf_path.name)
    doc = fitz.open(str(pdf_path))

    all_texts: list[str] = []
    all_meta: list[dict] = []
    all_ids: list[str] = []
    chunk_counter = 0

    for page_idx in range(len(doc)):
        page_text = doc[page_idx].get_text()
        if not page_text.strip():
            continue
        for chunk in _chunk_page(page_text):
            all_texts.append(chunk)
            all_meta.append(
                {
                    "source": pdf_path.name,
                    "page": page_idx + 1,
                    "lob": lob,
                }
            )
            all_ids.append(f"{pdf_path.stem}_p{page_idx+1}_c{chunk_counter}")
            chunk_counter += 1

    doc.close()

    if not all_texts:
        print(f"  WARNING: no text extracted from {pdf_path.name}")
        return 0

    # Embed in batches and upsert into Chroma
    # Rate-limit: trial keys allow 100k tokens/min; each batch ≈ 90 chunks × 300 tokens
    # = ~27k tokens. Sleep 20s between batches to stay well under the limit.
    for i in range(0, len(all_texts), EMBED_BATCH):
        batch_texts = all_texts[i : i + EMBED_BATCH]
        embeddings = providers.embed(batch_texts, input_type="search_document")

        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=all_meta[i : i + EMBED_BATCH],
            ids=all_ids[i : i + EMBED_BATCH],
        )

        # Cohere trial keys: pace batches to stay under 100k tokens/min
        if providers.EMBED_PROVIDER == "cohere" and i + EMBED_BATCH < len(all_texts):
            time.sleep(20)
        print(f"  {min(i + EMBED_BATCH, len(all_texts))}/{len(all_texts)} chunks", end="\r")

    return len(all_texts)


# ── Main ─────────────────────────────────────────────────────────────────────

def _already_ingested(collection: chromadb.Collection, filename: str) -> bool:
    """Return True if this PDF already has chunks in the collection."""
    result = collection.get(where={"source": {"$eq": filename}}, limit=1)
    return len(result["ids"]) > 0


def main() -> None:
    if providers.EMBED_PROVIDER == "cohere" and not providers.COHERE_API_KEY:
        sys.exit("ERROR: COHERE_API_KEY environment variable not set.")

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"ERROR: No PDFs found in ./{DATA_DIR}. Run download_data.py first.")

    print(f"Embedding provider: {providers.EMBED_PROVIDER}  →  collection {COLLECTION_NAME!r}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Open existing collection or create it fresh
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for pdf in pdfs:
        if _already_ingested(collection, pdf.name):
            print(f"[skip] {pdf.name}  (already in collection)")
            continue
        print(f"\nIngesting: {pdf.name}  (LOB: {_detect_lob(pdf.name)})")
        n = ingest_pdf(pdf, collection)
        print(f"  {n} chunks stored")
        total_chunks += n

    print(f"\n{'='*50}")
    print(f"Ingestion complete.  Total chunks: {total_chunks}")
    print(f"Collection count:    {collection.count()}")
    print(f"Chroma DB path:      ./{CHROMA_DIR}/")


if __name__ == "__main__":
    main()
