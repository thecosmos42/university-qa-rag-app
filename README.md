# university-qa-rag

A Retrieval-Augmented Generation (RAG) backend for answering questions about VU Amsterdam master's programmes. Built with FastAPI, Supabase (pgvector), and Gemini 2.5 Flash.

## Demo

![Landing Page](demo-images/landing-page.png)
![Example 1](demo-images/example-1.png)
![Example 2](demo-images/example-2.png)
![Example 3](demo-images/example-3.png)

---

## How It Works

1. **Ingestion** — PDFs in `data/pdfs/` are chunked, embedded, and stored in Supabase alongside their metadata (programme name, type, URL).
2. **Query** — The `/query` endpoint embeds the user's question, runs a hybrid search (semantic + BM25 keyword) via a Supabase RPC, then feeds the top-k chunks to Gemini to generate a grounded answer.
3. **Sources** — Each answer includes cited source chunks with clickable `href` links back to the VU programme page.

```
PDF files
    └─► ingestion_pipeline.py
            ├─ chunk text (RecursiveCharacterTextSplitter)
            ├─ embed chunks (bge-base-en-v1.5)
            └─ upsert to Supabase (documents + chunks tables)

User question
    └─► POST /query
            ├─ embed question
            ├─ hybrid_search RPC (pgvector cosine + BM25, RRF fusion)
            ├─ Gemini 2.5 Flash generates answer
            └─ return { answer, sources[] }
```

---

## Project Structure

```
rag-app/
├── src/
│   ├── main.py                  # FastAPI app + CORS + lifespan hook
│   ├── api/
│   │   └── query.py             # POST /query endpoint
│   ├── config/
│   │   ├── config.py            # All tunable parameters
│   │   ├── db_config.py         # Supabase connection config
│   │   ├── logger.py            # Structured logging
│   │   └── queries_supabase.sql # Schema + hybrid_search SQL function
│   ├── db/
│   │   └── supabase_db.py       # CRUD helpers (upsert documents/chunks)
│   ├── ingestion/
│   │   ├── ingestion_pipeline.py  # Orchestrates ingest → chunk → embed → store
│   │   ├── embedder_factory.py    # Backend-aware embedder factory
│   │   ├── embedder_local.py      # sentence-transformers (GPU-aware)
│   │   ├── embedder_gemini.py     # Gemini Embedding API
│   │   └── embedder_hf.py         # HuggingFace Serverless Inference API
│   └── llm/
│       ├── gemini_llm.py          # Gemini 2.5 Flash wrapper (multi-key rotation)
│       ├── llm_wrapper.py         # LLM interface
│       ├── program_extractor.py   # Extract programme name from query
│       └── query_rewriter.py      # Optional query rewriting
├── data/
│   ├── pdfs/                    # Source PDFs + sidecar .json metadata
│   └── embeddings_cache/        # Per-chunk embedding cache (speeds up re-ingest)
├── university_docs/             # Scraped programme metadata (JSON)
├── web_scrape.py                # Playwright scraper for VU study guides
├── requirements.txt             # Production dependencies
├── requirements-dev.txt         # Dev/local dependencies (torch, sentence-transformers)
├── .env.example                 # Environment variable template
└── logs/                        # App logs (when LOG_TO_FILE=true)
```

---

## Prerequisites

- Python 3.11+
- A [Supabase](https://supabase.com) project with the `pgvector` extension enabled
- A [Google AI Studio](https://aistudio.google.com/app/apikey) API key (for LLM + optionally embeddings)
- A [HuggingFace](https://huggingface.co/settings/tokens) read token (if using `EMBEDDING_BACKEND=huggingface`)

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/your-org/rag-app.git
cd rag-app
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Production deps (no GPU required)
pip install -r requirements.txt

# Only needed for EMBEDDING_BACKEND=local (downloads torch + sentence-transformers)
pip install -r requirements-dev.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Primary Gemini API key |
| `GOOGLE_API_KEY_2` … `_9` | No | Extra keys for rate-limit rotation |
| `SUPABASE_URL` | Yes | Your Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Yes | Supabase service role key |
| `HF_TOKEN` | If HF backend | HuggingFace read token |
| `ALLOWED_ORIGINS` | Production | Comma-separated CORS origins |
| `LOG_TO_FILE` | No | `true` to write `logs/app.log` |
| `DEBUG_MODE` | No | `true` to include debug info in responses |

### 3. Set up the Supabase database

1. Enable the `pgvector` extension in Supabase: **Database → Extensions → vector**.
2. Open **SQL Editor** in your Supabase dashboard and run the full contents of [`src/config/queries_supabase.sql`](src/config/queries_supabase.sql). This creates the `documents` and `chunks` tables and the `hybrid_search` RPC function.

---

## Embedding Backends

Set `EMBEDDING_BACKEND` in `src/config/config.py` (or override via env):

| Value | Model | Requires | Notes |
|---|---|---|---|
| `"huggingface"` | `BAAI/bge-base-en-v1.5` | `HF_TOKEN` | **Recommended for production** — no GPU needed |
| `"local"` | `BAAI/bge-base-en-v1.5` | `requirements-dev.txt` | Fastest; GPU auto-detected |
| `"gemini"` | `gemini-embedding-001` | `GOOGLE_API_KEY` | Same key as LLM |

> **Important:** Ingestion and query **must use the same backend**. Changing `EMBEDDING_BACKEND` after ingestion requires clearing `data/embeddings_cache/` and re-running ingestion with `--force`.

---

## Ingesting Documents

Place PDFs in `data/pdfs/`. Each PDF should follow the naming convention:

```
[DocType]_[ProgramType]_[ProgramName].pdf
```

Optionally place a sidecar `<filename>.pdf.json` alongside each PDF with metadata:

```json
{
  "program_url": "https://vu.nl/en/education/master/...",
  "faculty": "Faculty of Science",
  "academic_year": "2025-2026"
}
```

Then run the ingestion pipeline:

```bash
# First-time ingest
python -m src.ingestion.ingestion_pipeline

# Force re-ingest (clears cache, re-embeds everything)
python -m src.ingestion.ingestion_pipeline --force
```

---

## Running the Server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## API

### `POST /query`

**Request:**
```json
{
  "question": "What are the admission requirements for the AI master?",
  "semantic_weight": 0.6,
  "keyword_weight": 0.4,
  "limit": 10
}
```

**Response:**
```json
{
  "answer": "The AI master requires...",
  "sources": [
    {
      "title": "Master Artificial Intelligence",
      "program_type": "Masters",
      "doc_type": "Studiegids",
      "chunk": "Admission requires a bachelor's degree in...",
      "href": "https://vu.nl/en/education/master/artificial-intelligence"
    }
  ]
}
```

---

## Key Configuration (`src/config/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_BACKEND` | `"huggingface"` | Embedding backend (`local` / `gemini` / `huggingface`) |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `100` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `ENABLE_HISTORY` | `False` | Include conversation history in prompt |

---

## Deployment

The server is designed for zero-downtime cold starts on platforms like Render or Railway. All heavy initialisation (model loading, DB connections) happens inside the FastAPI lifespan hook **after** the port is bound, so the health check passes immediately.

**Recommended production config:**
- `EMBEDDING_BACKEND=huggingface` (no GPU required, same vectors as local)
- `LOG_TO_FILE=false` (ephemeral filesystems have no persistent `/logs`)
- `ALLOWED_ORIGINS=https://your-frontend.vercel.app`
- `DEBUG_MODE=false`

---

## Frontend

The companion Next.js frontend lives in [`rag-app-ui/`](../rag-app-ui). Point it at this backend by setting `NEXT_PUBLIC_API_URL=http://localhost:8000` in its `.env.local`.
