# P&C Insurance Underwriting Guidelines RAG

Retrieval-Augmented Generation over publicly available P&C insurance documents.

**Lines of business:** Commercial Property · Commercial General Liability · Personal Auto · NAICS Classification

**Stack:** PyMuPDF · Qwen3-Embedding + Qwen3 8B via Ollama (local, default) or Cohere ·
ChromaDB (local) · Open Knowledge Format (OKF) bundle · TypeSafe Jev router

Each question is routed by **Jev** to one of two retrieval paths:

- **OKF** — structured lookup in an [Open Knowledge Format](https://agentic-ai.readthedocs.io/en/latest/Standards/open-knowledge-format/)
  bundle (markdown + YAML frontmatter, cross-linked). Best for specific entities:
  NAICS codes and industries, a code's parent/sub-industries, a named form section.
- **RAG** — semantic vector search over chunks. Best for open-ended, conceptual, or
  cross-document underwriting questions.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

### 2. Pull the local models (Ollama)

```bash
ollama pull qwen3-embedding:0.6b   # embeddings (1024-dim)
ollama pull qwen3:8b               # answer generation
```

Everything runs locally by default. Providers are picked by environment variables
(see `providers.py`):

| Variable | Default | Options |
|---|---|---|
| `EMBED_PROVIDER` | `ollama` | `ollama`, `cohere` |
| `GENERATE_PROVIDER` | `ollama` | `ollama`, `cohere` |
| `OLLAMA_EMBED_MODEL` | `qwen3-embedding:0.6b` | e.g. `qwen3-embedding:4b` |
| `OLLAMA_GENERATE_MODEL` | `qwen3:8b` | any Ollama chat model |
| `OLLAMA_URL` | `http://localhost:11434` | |
| `COHERE_API_KEY` | — | only for a `cohere` provider |

Each embedding provider/model gets its **own** Chroma collection
(`underwriting_guidelines` for Cohere, `underwriting_guidelines__qwen3-embedding-0-6b`
for the default), since vectors from different models can't be mixed. Switching
`EMBED_PROVIDER` or `OLLAMA_EMBED_MODEL` means running `ingest.py` again for that collection.

### 3. Download source PDFs

```bash
python download_data.py
```

Downloads 10 publicly available documents into `./data/`:

| File | Source | LOB |
|------|--------|-----|
| `commercial_property_il_doi_guide.pdf` | Illinois DOI — Commercial Property Insurance Consumer Guide | Commercial Property |
| `commercial_property_ca_doi_guide.pdf` | California DOI — Commercial Insurance Consumer Guide | Commercial Property |
| `commercial_property_oregon_pc_guide.pdf` | Oregon DFR — Guide to Commercial P&C Insurance | Commercial Property |
| `cgl_ny_policy_form.pdf` | New York OGS — ISO CGL Policy Form CG 00 01 | CGL |
| `cgl_wi_consumer_guide.pdf` | Wisconsin OCI — Consumer Guide to Commercial Liability (PI-045) | CGL |
| `personal_auto_naic_guide.pdf` | NAIC — Consumer's Guide to Auto Insurance | Personal Auto |
| `personal_auto_tx_statistical_plan.pdf` | Texas DOI — Private Passenger Auto Statistical Plan | Personal Auto |
| `personal_auto_fl_toolkit.pdf` | Florida DFS — Automobile Insurance Toolkit | Personal Auto |
| `naics_usa_2022_manual.pdf` | US Census Bureau — NAICS 2022 Manual | NAICS Classification |
| `naics_canada_2017_manual.pdf` | Statistics Canada — NAICS Canada 2017 Manual | NAICS Classification |

### 4. Ingest into ChromaDB

```bash
python ingest.py
```

Chunks PDFs by paragraph (~300 tokens), embeds with the configured provider, and stores
in a local ChromaDB collection at `./chroma_db/`. Ingestion is **incremental** — already
ingested PDFs are skipped, so re-running only processes new files.

> **Note — Cohere trial keys** (`EMBED_PROVIDER=cohere`): the free tier allows 100k
> tokens/minute; `ingest.py` paces batches and retries on rate limits.

### 5. Build the OKF bundle

```bash
python okf_build.py
```

Writes `./okf/` (~4,200 pages, a few seconds, no API calls):

- `okf/naics/us/<code>.md`, `okf/naics/ca/<code>.md` — one page per NAICS code
  (sector → national industry), linked to its parent, children, and cross-referenced codes
- `okf/<lob>/<source>/p<N>.md` — one page per guideline PDF page
- `index.md` entry points at the root, per LOB, per source, per NAICS edition

### 6. Set your TypeSafe API key (for the router)

```bash
export TYPESAFE_API_KEY="your-key-here"   # https://console.typesafe.ai/
```

Without it the router falls back to RAG (or OKF when the question contains a NAICS code).

### 7. Query

**Claude Desktop (recommended)** — see [Claude Desktop Setup](#claude-desktop-setup) below.

**Web UI (local):**
```bash
python app.py        # → http://localhost:7860
```
Ask a question, pick *Auto (Jev)* / *Force OKF* / *Force RAG*, and see the routing
decision, the answer, and its sources side by side.

**CLI — routed:**
```bash
python router.py "What does NAICS 524126 cover?"                      # → OKF
python router.py "How does NAICS Canada classify real estate businesses?"  # → OKF
python router.py "Does a CGL policy cover completed operations claims?"   # → RAG
python router.py --route-only "How do auto and property deductibles compare?"
```

**CLI — force one path:**
```bash
python okf_retrieve.py "What sub-industries are under insurance carriers?"
python rag_query.py "What are eligibility criteria for frame construction commercial property?"
python rag_query.py "Does a CGL policy cover completed operations claims?"
python rag_query.py "What driving record violations disqualify a personal auto applicant?"
python rag_query.py "What NAICS code covers insurance carriers?"
python rag_query.py "How does NAICS Canada classify real estate businesses?"
```

**Jupyter notebook:**
```bash
jupyter notebook demo.ipynb
```

---

## Claude Desktop Setup

The MCP server exposes the RAG pipeline as a tool directly inside Claude Desktop.

### One-time setup

Add the following to `~/Library/Application Support/Claude/claude_desktop_config.json`
(merge into the existing `mcpServers` block if one already exists):

```json
{
  "mcpServers": {
    "insurance-rag": {
      "command": "/path/to/your/python",
      "args": ["/path/to/insurancerag/mcp_server.py"],
      "env": {
        "TYPESAFE_API_KEY": "your-key-here"
      }
    }
  }
}
```

Replace `/path/to/your/python` with the output of `which python` and
`/path/to/insurancerag` with the absolute path to this directory.

Restart Claude Desktop. A hammer icon (🔨) in the chat bar confirms the tool loaded.

### Usage in Claude Desktop

Just ask underwriting or classification questions naturally:

- *"What are eligibility criteria for frame construction commercial property?"*
- *"Does a CGL policy cover completed operations claims?"*
- *"What NAICS code applies to residential building contractors in Canada?"*
- *"A business with NAICS code 236115 wants CGL coverage — what underwriting considerations apply?"*

Claude will call `ask_underwriting`, which routes the question with Jev to OKF or
RAG, generates a grounded answer with the local Qwen model, and returns it with
source citations.

**Filtering by LOB:** Claude will pass a `lob` filter when the question is clearly
scoped to one line. You can also be explicit:
> *"Using only Personal Auto sources, what violations trigger a surcharge?"*

---

## Architecture

```
download_data.py          fetch PDFs from public URLs → ./data/
        ↓
ingest.py
  _already_ingested()     skip PDFs already in collection (incremental)
  fitz.open(pdf)          PyMuPDF page-by-page text extraction
  _chunk_page()           paragraph-split, ~300 token target
  providers.embed(..., "search_document")   Qwen3-Embedding (Ollama) or Cohere
  collection.add()        ChromaDB PersistentClient (cosine space)
        ↓
chroma_db/                local vector store (persisted on disk)
        ↓
rag_query.py
  providers.embed(query, "search_query")   (Qwen adds a task instruction)
  collection.query(top_k=5, where={lob_filter})
  providers.generate_answer(...)           qwen3:8b with [n] citations, or Command R+
  → RAGResult(answer, citations, grounded_spans)
        ↓
okf_build.py              PDFs → ./okf/ OKF bundle (NAICS graph + guideline pages)
okf_retrieve.py           exact code lookup (or field-weighted BM25) → link expansion
                          → same providers.generate_answer() as rag_query
        ↓
router.py                 one Jev request, three parallel questions:
                            path    Choice  okf | rag
                            lob     Choice  4 LOBs (speculative)
                            one_lob Noul    companion: does exactly one LOB apply?
                          policy in code: NAICS code → okf; path confidence ≥ 0.6
                          else rag; LOB filter only if one_lob ≥ 0.7 and lob p ≥ 0.7
        ↓
mcp_server.py             MCP tools: ask_underwriting (routed),
                          query_underwriting_guidelines (RAG with LOB filter)
demo.ipynb                interactive notebook with 5 sample queries
```

### Metadata stored per chunk

| Field | Example |
|-------|---------|
| `source` | `cgl_ny_policy_form.pdf` |
| `page` | `3` |
| `lob` | `Commercial General Liability` |

LOB is inferred from the filename prefix:
`commercial_property_*`, `cgl_*`, `personal_auto_*`, `naics_*`.

---

## Sample Queries (Python)

```python
from rag_query import query, format_response

# LOB-scoped query
r = query(
    "What are eligibility criteria for frame construction commercial property?",
    lob_filter="Commercial Property",
)
print(format_response(r))

# NAICS query
r = query(
    "What NAICS code covers insurance carriers in Canada?",
    lob_filter="NAICS Classification",
)
print(format_response(r))

# Cross-LOB query (no filter)
r = query("How is flood exposure handled across different lines of business?")
print(format_response(r))
```

---

## File Structure

```
insurancerag/
├── data/                         # PDF source documents (git-ignored)
├── chroma_db/                    # ChromaDB vector store (git-ignored)
├── download_data.py              # fetch public PDFs
├── ingest.py                     # chunk → embed → store (incremental)
├── okf/                          # OKF knowledge bundle (git-ignored)
├── rag_query.py                  # retrieve → generate → cite
├── okf_build.py                  # build the OKF bundle from PDFs
├── okf_retrieve.py               # OKF lookup/BM25 retrieval → generate → cite
├── providers.py                  # embedding/generation backends: Ollama (Qwen) or Cohere
├── app.py                        # local Gradio web UI (python app.py)
├── router.py                     # Jev router: OKF vs RAG + LOB filter
├── tests/                        # pytest (no API keys needed)
├── mcp_server.py                 # MCP tool server for Claude Desktop
├── demo.ipynb                    # interactive demo notebook
├── requirements.txt
└── README.md
```

---

## Notes

- **Adding new documents:** drop a PDF into `./data/` with the correct filename prefix
  (`commercial_property_`, `cgl_`, `personal_auto_`, or `naics_`) and re-run `ingest.py`.
  Only the new file will be embedded.
- **LOB filtering:** pass `lob_filter=` in `query()` to restrict retrieval to one LOB.
- **Router thresholds** (`PATH_MIN_CONFIDENCE`, `LOB_MIN_PROBABILITY`, `ONE_LOB_MIN` in
  `router.py`) are starting points — tune them on your own labeled questions. Jev's
  *confidence* measures how concentrated its answer is, so a 0.52/0.48 split has very low
  confidence and correctly falls back to RAG.
- **Tests:** `pytest tests` (uses fake Jev responses and a tiny temp OKF bundle).
- **Adding documents to OKF:** re-run `okf_build.py` (it rebuilds `./okf/` from scratch).
  Omit for cross-LOB questions.
- **No restart needed** when adding new data — `rag_query.py` opens a fresh ChromaDB
  connection on every call, so new chunks are immediately available.
