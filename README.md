# P&C Insurance Underwriting Guidelines RAG

Retrieval-Augmented Generation over publicly available P&C insurance documents.

**Lines of business:** Commercial Property · Commercial General Liability · Personal Auto · NAICS Classification

**Stack:** PyMuPDF · Cohere Embed v3 · ChromaDB (local) · Cohere Command R+ (08-2024)

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

### 2. Set your Cohere API key

```bash
export COHERE_API_KEY="your-key-here"
```

Get a free key at <https://dashboard.cohere.com/>.

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

Chunks PDFs by paragraph (~300 tokens), embeds with `embed-english-v3.0`, and stores
in a local ChromaDB collection at `./chroma_db/`. Ingestion is **incremental** — already
ingested PDFs are skipped, so re-running only processes new files.

> **Note — Cohere trial keys:** The free tier allows 100k tokens/minute. `ingest.py`
> automatically paces embed batches and retries on rate-limit errors. The full corpus
> (~2,700 chunks) takes a few minutes on a trial key.

### 5. Query

**Claude Desktop (recommended)** — see [Claude Desktop Setup](#claude-desktop-setup) below.

**CLI:**
```bash
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
        "COHERE_API_KEY": "your-key-here"
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

Claude will automatically call `query_underwriting_guidelines`, retrieve from ChromaDB,
generate a grounded answer via Cohere, and return it with source citations.

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
  co.embed(..., input_type="search_document")   Cohere Embed v3
  collection.add()        ChromaDB PersistentClient (cosine space)
        ↓
chroma_db/                local vector store (persisted on disk)
        ↓
rag_query.py
  co.embed(query, input_type="search_query")
  collection.query(top_k=5, where={lob_filter})
  co.chat(model="command-r-plus-08-2024", documents=[...])   grounded gen
  → RAGResult(answer, citations, grounded_spans)
        ↓
mcp_server.py             MCP tool wrapper for Claude Desktop
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
├── rag_query.py                  # retrieve → generate → cite
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
  Omit for cross-LOB questions.
- **No restart needed** when adding new data — `rag_query.py` opens a fresh ChromaDB
  connection on every call, so new chunks are immediately available.
