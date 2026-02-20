# Engineering Report: University RAG Application
**Date:** 2026-02-20  
**Scope:** Full forensic reconstruction of the agent session  
**Classification:** Internal Engineering Documentation  

---

# 1. Project Overview

This project is a **Retrieval-Augmented Generation (RAG) application** built to answer student questions about a **VU Amsterdam AI Master's programme study guide** (PDF: `Studiegids.pdf`, 84 pages).

**Architecture:**
- **Backend:** Python + FastAPI (`src/main.py`)
- **Frontend:** Next.js React app (`rag-app-ui/`, runs on `localhost:3000`)
- **Database:** Supabase PostgreSQL with `pgvector` extension
- **Embedding model:** Google `gemini-embedding-001` (768 dimensions, current)
- **LLM:** Google `gemini-2.5-flash` (for answer generation)
- **Search:** Hybrid search (semantic + BM25 keyword) with Reciprocal Rank Fusion (RRF)

**Pipeline flow:**
```
PDF → PDFIngester → AcademicChunker → Embedder (Gemini) → Supabase pgvector
User query → Embedder → hybrid_search SQL → RAGGenerator → Gemini LLM → Response
```

**Evidence:** Reconstructed from files: `src/main.py`, `src/api/query.py`, `src/ingestion/ingestion_pipeline.py`, `src/config/queries_supabase.sql`, `src/api/RAGGenerator.py`, `src/llm/gemini_llm.py`.

---

# 2. Implemented Features

## 2.1 Document-Aware Academic Chunker
- **File:** `src/ingestion/academic_chunker.py`
- **Description:** Custom chunker that detects course boundaries via regex (`Course Code` pattern), extracts structured metadata (course_name, course_code, credits, period, faculty, language, degree_name, specialisation), and keeps each course as a self-contained chunk. Implements Databricks "Context-Enriched Chunking" (Strategy #5) with neighbour context.
- **Status:** Implemented. Context block prepending was later **removed from embedded text** (kept in metadata only) to reduce token consumption.
- **Evidence (conversation summary):** "document-aware chunking" listed as implemented fix. Agent created `academic_chunker.py` with context enrichment.
- **Evidence (current code):** `_sections_to_documents()` method now uses only `_build_metadata_header()`, not `_build_context_block()`, for the chunk text body.

## 2.2 Hybrid Search with RRF
- **File:** `src/config/queries_supabase.sql`
- **Description:** SQL function `hybrid_search` combining:
  - Semantic similarity search (cosine distance via pgvector `<=>`)
  - BM25 keyword search (`ts_rank_cd` + `to_tsvector`)
  - Reciprocal Rank Fusion for score combination
  - Optional metadata filters (course_name, course_code, degree_name)
- **Status:** Implemented. RRF constant `k` was changed from 60 → 1 for better score separation on small datasets.
- **Evidence (chat):** Agent identified "semantic search bug" and "BM25 synonym blindness" as root causes. Implemented RRF fusion. Later changed k=60 to k=1 based on log analysis showing clustered scores.

## 2.3 Query Rewriter
- **File:** `src/rag/query_rewriter.py`
- **Description:** Pre-query processing with synonym expansion, abbreviation expansion (NLP → Natural Language Processing), and metadata extraction from natural language queries.
- **Status:** Implemented but **currently disabled** (commented out in `query.py` per user request).
- **Evidence (chat):** User explicitly asked to "comment out the query rewrite functionality from query.py, i don't want to use it now."

## 2.4 Embeddings Caching
- **File:** `src/ingestion/ingestion_pipeline.py`
- **Description:** After generating embeddings, they are saved to `data/embeddings_cache/{filename}.json`. On subsequent runs (without `--force`), cached embeddings are loaded if valid (correct count, valid JSON).
- **Status:** Implemented.
- **Evidence (chat):** User requested: "save the embeddings in a local file so that it can read it from there in case some error like this happens next time." Agent implemented `json.dump`/`json.load` with validation in the pipeline.

## 2.5 Gemini Embeddings with Rate Limiting
- **File:** `src/ingestion/embedder.py`
- **Description:** Gemini `gemini-embedding-001` with `output_dimensionality=768`. Batch size reduced to 20, cooldown of 8s between batches, 60s wait on 429 errors, max 5 retries.
- **Status:** Current production configuration.
- **Evidence (chat):** User reported hitting Gemini free-tier limits (RPM 200/100, TPM 31.4k/30k). Agent reduced batch size from 100→20, increased cooldown from 2s→8s, retry wait from 30s→60s, retries from 3→5.

## 2.6 HuggingFace Embedder (Alternative)
- **File:** `src/ingestion/embedder_hf.py`
- **Description:** Alternative embedder using HuggingFace `InferenceClient`. Includes `save_embeddings()` and `load_embeddings()` methods. Default model: `intfloat/multilingual-e5-large-instruct`.
- **Status:** **Not in active use.** Created during the Qwen3 migration attempt, then reverted. File still exists but is not imported anywhere in the active code path.
- **Evidence (chat):** User asked to switch to Qwen3-Embedding-0.6B, encountered multiple API errors, then asked to "revert back to the gemini text embeddings model everywhere."

## 2.7 Structured Metadata Columns
- **File:** `src/config/queries_supabase.sql`, `src/db/supabase_db.py`
- **Description:** `chunks` table has dedicated columns: `course_name`, `course_code`, `degree_name`, `section_type` for SQL-level filtering in hybrid_search.
- **Status:** Implemented.
- **Evidence (conversation summary):** "metadata filtering" listed as implemented fix. SQL schema includes these columns with B-tree indexes.

## 2.8 HNSW Vector Index
- **File:** `src/config/queries_supabase.sql`
- **Description:** HNSW index on `embedding` column using `vector_cosine_ops` with `m=16, ef_construction=64` for approximate nearest-neighbour search.
- **Status:** Defined in SQL file.
- **Outcome:** Not explicitly confirmed whether user applied this in Supabase SQL Editor.

## 2.9 Timing Instrumentation
- **File:** `src/api/query.py`
- **Description:** Per-request timing breakdown logged: `embed`, `search`, `llm`, `total`.
- **Status:** Implemented.
- **Evidence (chat):** Agent added timing in the query optimization round.

## 2.10 Prompt Engineering
- **File:** `src/rag/prompt_builder.py`
- **Description:** Structured prompt with numbered SOURCEs, citation rules requiring `[SOURCE N, Page M]` format, and optional conversation history (disabled via `ENABLE_HISTORY=False`).
- **Status:** Implemented. Was present before this session began.

---

# 3. Problems Encountered

## P1: Retrieval Returning Incorrect Results
- **Description:** The RAG system was returning wrong or irrelevant answers to user queries.
- **Technical context:** Initial forensic debugging request from user.
- **Root causes identified (per conversation summary):**
  1. Semantic search SQL bug
  2. BM25 synonym blindness (different stems not matched)
  3. Chunking contamination (cross-course content in chunks)
  4. Lack of structured metadata for filtering
- **Severity:** Critical — core functionality broken.

## P2: Qwen3-Embedding-0.6B 404 Error
- **Description:** `HfHubHTTPError: Client error '404 Not Found'` when calling `https://router.huggingface.co/hf-inference/models/Qwen/Qwen3-Embedding-0.6B/pipeline/feature-extraction`
- **Technical context:** User attempted to switch from Gemini to Qwen3-Embedding-0.6B via HuggingFace Inference API.
- **Root cause:** Model not available for Inference API at that endpoint.
- **Severity:** Blocking — embedding pipeline could not run.

## P3: intfloat/multilingual-e5-large-instruct 500 Error
- **Description:** `Server error '500 Internal Server Error'` from HuggingFace Inference API for `intfloat/multilingual-e5-large-instruct`.
- **Technical context:** Fallback model after Qwen3 failed.
- **Root cause:** Server-side issue on HuggingFace.
- **Severity:** Blocking.

## P4: TypeError: ndarray Not JSON Serializable
- **Description:** `TypeError: Object of type ndarray is not JSON serializable` when inserting embeddings into Supabase via `httpx`.
- **Technical context:** HuggingFace `feature_extraction` returned numpy arrays instead of Python lists.
- **File:** `src/db/supabase_db.py` (line 93, `insert(batch).execute()`)
- **Root cause:** HuggingFace InferenceClient returns `np.ndarray`; Supabase client uses `json.dumps` which cannot serialize numpy types.
- **Severity:** Blocking.

## P5: Code Error in embedder_hf.py
- **Description:** Embedding logic was accidentally placed inside `__init__` method instead of the `encode` method, creating orphaned code.
- **Technical context:** Caused by a bad edit during the HF embedder refactor.
- **Root cause:** Agent error during file editing — code was misplaced.
- **Severity:** Blocking — class could not be instantiated.
- **Evidence (chat):** User said "there is an error in embedder_hf.py file." Agent confirmed: "It seems I introduced an error in the last turn by misplacing some code."

## P6: Vector Dimension Mismatch (384 vs 768)
- **Description:** `postgrest.exceptions.APIError: {'message': 'different vector dimensions 384 and 768'}`
- **Technical context:** After switching ingestion to `sentence-transformers/all-MiniLM-L6-v2` (384-dim), the query endpoint was still using Gemini embedder (768-dim). Database had mixed-dimension vectors.
- **File:** `src/api/query.py` (still imported old `src.ingestion.embedder`)
- **Root cause:** Query-side embedder not updated to match ingestion-side embedder after model switch.
- **Severity:** Critical — all queries failed.
- **Evidence (chat):** Error appeared twice. Agent traced it to `query.py` importing wrong embedder.

## P7: Gemini Free-Tier Rate Limiting
- **Description:** User hitting RPM 200/100 and TPM 31.4k/30k limits on Gemini embedding API.
- **Technical context:** 247 chunks at ~1200 chars each, plus verbose context blocks prepended to each chunk, sent in batches of 100 with only 2s delay.
- **Root cause:** Too many chunks, too large chunks (with context overhead), batch size too high, cooldown too short.
- **Severity:** Blocking — ingestion could not complete.

## P8: Query Response Taking ~10 Seconds
- **Description:** Average query response time of 10 seconds.
- **Technical context:** User explicitly reported this performance issue.
- **Root causes identified by agent:**
  1. SQL CTEs had no inner LIMIT — ranked ALL chunks before final LIMIT
  2. No HNSW index — brute-force vector distance scan
  3. 20 chunks sent to LLM — too much context for generation
  4. `RAGGenerator()` instantiated per request
  5. Excessive logging (6 log lines per result)
- **Severity:** High — poor user experience.

## P9: RRF Scores Clustered / Near-Identical
- **Description:** Combined RRF scores showed almost no differentiation. Example: Ranks 2-8 all scored between 0.448 and 0.440 (spread < 0.01). One result scored `8.44e-05`.
- **Technical context:** RRF constant `k=60` designed for millions of documents, not ~120 chunks.
- **Root cause:** With k=60, `1/(60+rank)` produces near-identical values across small rank ranges. Rank 1: `1/61=0.01639`, Rank 30: `1/90=0.01111`.
- **Evidence (logs):** Combined scores: `0.448, 0.447, 0.447, 0.442, 0.441, 0.441, 0.440`.
- **Severity:** Medium — retrieval worked but ranking lacked meaningful differentiation.

## P10: Vector Similarity Returning 0.0
- **Description:** Early log entries showed `Vector Similarity: 0` for all results.
- **Technical context:** Queries during the dimension-mismatch era when semantic search was broken.
- **Evidence (logs):** Multiple entries showing `Vector Similarity: 0` (2026-02-19 01:07 to 01:10). Later entries (01:43) show normal values like `0.706, 0.747, 0.746`.
- **Severity:** Was critical, resolved by fixing dimension mismatch.

---

# 4. Fixes Attempted

| # | Problem | Fix Description | File(s) Modified | Worked? | Evidence |
|---|---------|----------------|-------------------|---------|----------|
| F1 | P1: Bad retrieval | Implemented RRF hybrid search, metadata extraction, document-aware chunking | `queries_supabase.sql`, `academic_chunker.py`, `query_rewriter.py` | Yes (per conversation summary) | Conversation summary: "All forensic debugging steps completed, RRF and metadata implemented" |
| F2 | P2: Qwen3 404 | Switched model to `intfloat/multilingual-e5-large-instruct` | `embedder_hf.py` | No — led to P3 | P3 error message in chat |
| F3 | P3: e5-large 500 | Switched model to `sentence-transformers/all-MiniLM-L6-v2` | `ingestion_pipeline.py` | Partial — model worked but dimension mismatch | Led to P6 |
| F4 | P4: ndarray TypeError | Added `isinstance(result, np.ndarray)` check with `.tolist()` conversion | `embedder_hf.py` | Yes | Fix applied, next error was different (not ndarray-related) |
| F5 | P5: Code in `__init__` | Removed orphaned code block from `__init__`, kept clean `encode()` method | `embedder_hf.py` | Yes | User confirmed fix was correct by proceeding |
| F6 | P6: Dimension mismatch | Updated `query.py` to import `embedder_hf` with matching model | `query.py` | Yes (temporarily) | Error changed from dimension mismatch to other issues |
| F7 | P6: Dimension mismatch (recurrence) | Reverted everything back to Gemini embedder everywhere | `query.py`, `ingestion_pipeline.py`, `queries_supabase.sql` | Yes | User request: "revert back to the gemini text embeddings model everywhere" |
| F8 | P7: Rate limiting | Increased chunk size 1200→2500, removed context block from embedded text, reduced batch size 100→20, increased cooldown 2s→8s | `ingestion_pipeline.py`, `academic_chunker.py`, `embedder.py` | Outcome not explicitly confirmed | Changes applied but user did not report success/failure of ingestion after this |
| F9 | P8: Slow queries | Inner LIMITs on CTEs, HNSW index, limit 20→10, RAGGenerator reuse, compact logging, timing instrumentation | `queries_supabase.sql`, `query.py` | Outcome not explicitly confirmed | SQL changes require manual application in Supabase Editor |
| F10 | P9: Clustered RRF scores | Changed RRF constant k=60→k=1 | `queries_supabase.sql` | Outcome not explicitly confirmed | Theoretical analysis provided showing improved separation |
| F11 | P4: Embeddings backup | Added `save_embeddings()` / `load_embeddings()` + caching in pipeline | `embedder_hf.py`, `ingestion_pipeline.py` | Yes (mechanism works) | Code verified, caching logic with validation implemented |

---

# 5. Confirmed Working Fixes

Based on explicit evidence in chat:

1. **F1: Hybrid search + RRF + metadata + academic chunker** — Conversation summary states "All forensic debugging steps completed."
2. **F4: ndarray → list conversion** — Subsequent errors were unrelated to serialization.
3. **F5: Code placement fix in embedder_hf.py** — User proceeded without further complaint about this issue.
4. **F7: Revert to Gemini embeddings** — User successfully used the system after this revert (subsequent queries worked, logs show semantic scores > 0).

---

# 6. Failed or Partial Fixes

| Fix | Status | Reason |
|-----|--------|--------|
| F2: Switch to `intfloat/multilingual-e5-large-instruct` | Failed | HuggingFace returned 500 Internal Server Error |
| F3: Switch to `sentence-transformers/all-MiniLM-L6-v2` | Partial | Model worked but created dimension mismatch (384 vs 768/1024); ultimately reverted |
| F6: Update `query.py` to use HF embedder | Superseded | Entire HF embedder approach was reverted in F7 |
| F8: Rate limiting optimizations | Unconfirmed | User has not yet reported whether ingestion completes within Gemini free-tier limits after these changes |
| F9: Query performance optimizations | Unconfirmed | SQL changes require manual execution in Supabase; user has not reported timing results |
| F10: RRF k=1 tuning | Unconfirmed | SQL change requires manual execution in Supabase |

---

# 7. Architectural Evolution Timeline

## Phase 1: Initial State (Pre-Session)
- **Chunker:** Basic character-based splitting (config: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`)
- **Embedder:** Gemini `gemini-embedding-001` with `output_dimensionality=768`
- **Database:** Supabase pgvector
- **Search:** Unknown initial state (conversation summary mentions "semantic search bug")
- **LLM:** Gemini `gemini-2.5-flash`
- **Other existing files:** `local_llm.py`, `local_embedder.py`, `chroma_db_local.py` — suggesting earlier ChromaDB + local model architecture that was migrated to Supabase + Gemini

## Phase 2: Forensic Debugging & Hybrid Search (Early Session, from Conversation Summary)
- **Root causes identified:** Semantic search SQL bug, BM25 synonym blindness, chunking contamination, missing metadata
- **Changes:**
  - Created `academic_chunker.py` — document-aware chunking with course boundary detection
  - Created `query_rewriter.py` — synonym/abbreviation expansion
  - Implemented `hybrid_search` SQL function with RRF (k=60)
  - Added structured metadata columns to `chunks` table
  - Added metadata indexes (B-tree on `course_name`, `course_code`, `degree_name`)
  - Added full-text search indexes (GIN on `to_tsvector`, trigram)

## Phase 3: Embedding Model Migration Attempt (Mid Session)
- **Goal:** Switch from Gemini to open-source Qwen3-Embedding-0.6B
- **Sequence:**
  1. Created `embedder_hf.py` with HuggingFace InferenceClient
  2. Changed `ingestion_pipeline.py` to import `embedder_hf`
  3. Changed DB schema to `vector(1024)` for Qwen3
  4. Qwen3 returned 404 → switched to `intfloat/multilingual-e5-large-instruct`
  5. e5-large returned 500 → switched to `sentence-transformers/all-MiniLM-L6-v2`
  6. Changed DB schema to `vector(384)` for MiniLM
  7. ndarray serialization error → added `.tolist()` conversion
  8. Code misplaced in `__init__` → fixed
  9. Dimension mismatch (384 vs 768) → updated `query.py`
  10. Dimension mismatch recurred → **full revert to Gemini**
- **Outcome:** Net zero. All HF changes reverted. `embedder_hf.py` left as unused file.

## Phase 4: Token Budget Optimization (Late Session)
- **Problem:** Gemini free-tier limits hit during ingestion
- **Changes:**
  - `max_chunk_size`: 1200 → 2500 (fewer chunks)
  - `chunk_overlap`: 100 → 150
  - Removed context block from embedded text (kept in metadata)
  - `batch_size`: 100 → 20
  - Cooldown: 2s → 8s
  - Retry wait: 30s → 60s, retries: 3 → 5
  - Added embeddings caching to `data/embeddings_cache/`

## Phase 5: Query Performance Optimization (Late Session)
- **Problem:** ~10s per query
- **Changes:**
  - Inner LIMITs (3x final, max 30) on both CTEs
  - Removed `> 0.1` semantic threshold filter (forced full scan)
  - Added HNSW index definition
  - Default limit: 20 → 10
  - `RAGGenerator` instantiated once at module level
  - Logging reduced from 6 lines/result to 1
  - Added per-request timing instrumentation
  - Switched BM25 from `ts_rank * 2.0` to `ts_rank_cd`

## Phase 6: RRF Tuning (Latest)
- **Problem:** Clustered scores with k=60
- **Change:** k=60 → k=1 for better rank separation on ~120 chunks

## Phase 7: Query Rewriter Disabled (Latest)
- **Change:** Query rewriting commented out in `query.py` per user request. All metadata filters passed as `None`.

---

# 8. Current System State

## 8.1 Retrieval Pipeline (Query Time)
```
User question
  → (query rewriting DISABLED)
  → Gemini embedding (768-dim, RETRIEVAL_QUERY)
  → hybrid_search SQL (RRF k=1, semantic_weight=0.6, keyword_weight=0.4)
  → Top 10 results
  → PromptBuilder formats as numbered SOURCEs
  → Gemini 2.5-flash generates answer
  → JSON response with answer, sources, debug info
```

## 8.2 Ingestion Pipeline
```
PDF files in data/pdfs/
  → PDFIngester (LangChain PyPDF)
  → AcademicChunker (max_chunk_size=2500, overlap=150, min=100)
     - Course boundary detection via "Course Code" regex
     - Metadata extraction (course_name, code, credits, period, etc.)
     - Context enrichment metadata (prev/next course) stored in metadata dict
     - Only compact metadata header prepended to chunk text
  → Gemini embedding (768-dim, batch_size=20, cooldown=8s)
  → Cache embeddings to data/embeddings_cache/
  → Supabase insert (batch_size=50)
```

## 8.3 Database Schema
- **`documents` table:** id, filename (unique), title, checksum, created_at
- **`chunks` table:** id, document_id (FK), content, embedding `vector(768)`, page_number, chunk_index, metadata (jsonb), course_name, course_code, degree_name, section_type, created_at
- **Indexes:** GIN (FTS), GIN (trigram), B-tree (course_name, course_code, degree_name), HNSW (embedding, cosine, m=16, ef=64) — HNSW may not be applied yet
- **Extensions:** `vector`, `pg_trgm`

## 8.4 Files in Active Use
| File | Role |
|------|------|
| `src/main.py` | FastAPI app entry point |
| `src/api/query.py` | `/query` POST endpoint |
| `src/api/RAGGenerator.py` | Bridges search results → prompt → LLM |
| `src/rag/prompt_builder.py` | Constructs grounded prompt with SOURCEs |
| `src/llm/gemini_llm.py` | Gemini 2.5-flash client |
| `src/ingestion/embedder.py` | Gemini embedding client (active) |
| `src/ingestion/academic_chunker.py` | Document-aware chunker |
| `src/ingestion/ingestion_pipeline.py` | Orchestrates ingestion |
| `src/ingestion/pdf_ingester.py` | PDF loading |
| `src/db/supabase_db.py` | Database operations |
| `src/config/queries_supabase.sql` | Schema + SQL functions |
| `src/config/config.py` | Configuration constants |
| `src/config/db_config.py` | Supabase URL/key |
| `src/config/logger.py` | Logging setup |

## 8.5 Files NOT in Active Use (Dead Code)
| File | Reason |
|------|--------|
| `src/ingestion/embedder_hf.py` | Remnant of HF migration; not imported |
| `src/ingestion/local_embedder.py` | Pre-session local model approach |
| `src/llm/local_llm.py` | Pre-session local LLM approach |
| `src/db/chroma_db_local.py` | Pre-session ChromaDB approach |
| `src/rag/query_rewriter.py` | Implemented but disabled in query.py |
| `src/rag/retriever.py` | Not imported anywhere in active path |

---

# 9. Remaining Risks & Weaknesses

## R1: SQL Changes Not Confirmed Applied
The following SQL changes were written to `queries_supabase.sql` but require **manual execution** in Supabase SQL Editor:
- HNSW index creation
- Updated `hybrid_search` function (inner LIMITs, `ts_rank_cd`, k=1)
- Schema change to `vector(768)` (if it was ever changed to 384/1024)

> **Risk:** If not applied, the database is still running the old `hybrid_search` function with k=60, no inner LIMITs, and possibly the wrong vector dimension.

## R2: Gemini Free-Tier Rate Limits
Rate limiting fixes (batch_size=20, cooldown=8s) were implemented but:
> "Outcome not explicitly confirmed" — user has not reported whether full ingestion now completes within limits.

**Calculation:** ~120 chunks × ~500 tokens avg ≈ 60k tokens total, sent in 6 batches over ~48s. This should stay under 30k TPM but is tight.

## R3: Query Rewriter Disabled
The query rewriter provided synonym expansion and metadata filtering. With it disabled:
- BM25 cannot match synonyms (e.g., "mandatory" won't find "compulsory")
- No metadata filters are applied (all searches scan all chunks)
- Abbreviations are not expanded (e.g., "NLP" won't match "Natural Language Processing")

> This was a **user choice**, not a technical failure.

## R4: Hardcoded API Keys
- Gemini API key is hardcoded in `src/config/config.py`
- HuggingFace token is hardcoded in `embedder_hf.py` and previously in `ingestion_pipeline.py`

> **Risk:** Security concern. These should be in environment variables / `.env` file.

## R5: `query.py` Has Syntax Error
The current `query.py` has an indentation issue in the fallback block:
```python
    if not db_results.data:
        ...
        db_results = sb.rpc(...).execute()
            
            for i, res in enumerate(db_results.data):  # ← wrong indentation
```
Lines 92-97 have fallback logging code indented under the `.execute()` call instead of under the `if` block.
> **Risk:** This will cause an `IndentationError` when the fallback path is triggered (empty result set).

## R6: Dead Code Accumulation
Multiple unused files and commented-out code blocks exist:
- `embedder_hf.py` — entire file unused
- `local_llm.py`, `local_embedder.py`, `chroma_db_local.py` — pre-migration remnants
- `retriever.py` — not imported
- Old chunker code commented out at top of `academic_chunker.py` (~280 lines of comments)
- `RAGGenerator.format_context()` method commented out

## R7: Embeddings Cache Consistency
The embeddings cache (`data/embeddings_cache/`) validates only on count:
```python
if not isinstance(embeddings, list) or len(embeddings) != len(chunk_dicts):
```
It does **not** validate:
- Dimension of cached vectors
- Whether the embedding model changed since caching
- Whether chunk content changed (only file-level checksum is checked, but chunks depend on chunker settings)

> **Risk:** If chunker parameters change (e.g., `max_chunk_size`), cached embeddings may not align with new chunks even if the PDF hasn't changed.

## R8: No Score Threshold Filtering
All retrieved chunks are passed to the LLM regardless of relevance score. Low-scoring results (e.g., `combined_score < 0.05`) add noise to the prompt without contributing useful context.

## R9: Single-Document System
The system currently processes only one PDF (`Studiegids.pdf`). The architecture supports multiple documents, but:
- No multi-document testing evidence exists
- The `document_ids` filter in the query endpoint defaults to empty (searches all)
- The academic chunker's regex patterns are tuned specifically for VU Amsterdam course catalogs

---

# 10. Chronological Timeline of Changes

| # | Phase | Action | Files | Outcome |
|---|-------|--------|-------|---------|
| 1 | Forensic Debug | Identified 4 root causes of retrieval failure | Analysis only | — |
| 2 | Forensic Debug | Created `academic_chunker.py` with course-boundary chunking | `academic_chunker.py` | Implemented |
| 3 | Forensic Debug | Created `query_rewriter.py` with synonym/abbreviation expansion | `query_rewriter.py` | Implemented, later disabled |
| 4 | Forensic Debug | Implemented `hybrid_search` SQL with RRF (k=60) | `queries_supabase.sql` | Implemented |
| 5 | Forensic Debug | Added structured metadata columns + indexes | `queries_supabase.sql`, `supabase_db.py` | Implemented |
| 6 | HF Migration | Created `embedder_hf.py` for Qwen3-Embedding-0.6B | `embedder_hf.py` | Created |
| 7 | HF Migration | Changed pipeline to use HF embedder | `ingestion_pipeline.py` | Applied then reverted |
| 8 | HF Migration | Changed DB schema to `vector(1024)` | `queries_supabase.sql` | Applied then reverted |
| 9 | HF Migration | Qwen3 404 error → switched to e5-large | `embedder_hf.py` | Failed (500 error) |
| 10 | HF Migration | Switched to `all-MiniLM-L6-v2` (384-dim) | `ingestion_pipeline.py` | Partial (dimension mismatch) |
| 11 | HF Migration | Fixed ndarray serialization | `embedder_hf.py` | Fixed |
| 12 | HF Migration | Fixed code placement in `__init__` | `embedder_hf.py` | Fixed |
| 13 | HF Migration | Integrated caching (save/load embeddings) | `embedder_hf.py`, `ingestion_pipeline.py` | Implemented |
| 14 | HF Migration | Updated `query.py` to use HF embedder + MiniLM | `query.py` | Applied then reverted |
| 15 | HF Migration | **Full revert to Gemini embeddings everywhere** | `query.py`, `ingestion_pipeline.py`, `queries_supabase.sql` | Reverted to Gemini |
| 16 | HF Migration | Moved caching from HF embedder methods to standalone `json.dump`/`json.load` | `ingestion_pipeline.py` | Implemented |
| 17 | Token Budget | Increased `max_chunk_size` 1200→2500 | `ingestion_pipeline.py` | Applied |
| 18 | Token Budget | Removed context block from chunk text (kept in metadata) | `academic_chunker.py` | Applied |
| 19 | Token Budget | Reduced batch_size 100→20, cooldown 2s→8s, retries 3→5, wait 30s→60s | `embedder.py` | Applied |
| 20 | Query Perf | Added inner LIMITs to CTEs (3x final, max 30) | `queries_supabase.sql` | Applied (needs SQL execution) |
| 21 | Query Perf | Removed `> 0.1` semantic threshold filter | `queries_supabase.sql` | Applied (needs SQL execution) |
| 22 | Query Perf | Added HNSW index definition | `queries_supabase.sql` | Applied (needs SQL execution) |
| 23 | Query Perf | Default limit 20→10 | `query.py` | Applied |
| 24 | Query Perf | `RAGGenerator` instantiated once at module level | `query.py` | Applied |
| 25 | Query Perf | Compact logging (6→1 line per result) | `query.py` | Applied |
| 26 | Query Perf | Added timing instrumentation | `query.py` | Applied |
| 27 | Query Perf | Switched BM25 to `ts_rank_cd` | `queries_supabase.sql` | Applied (needs SQL execution) |
| 28 | Rewriter | Disabled query rewriter per user request | `query.py` | Applied |
| 29 | RRF Tuning | Changed RRF constant k=60→k=1 | `queries_supabase.sql` | Applied (needs SQL execution) |

---

# Appendix: Contradiction / Revert Log

| Decision | Later Reversed? | Reason |
|----------|----------------|--------|
| Switch to Qwen3-Embedding-0.6B | Yes | 404 Not Found on HF Inference API |
| Switch to `intfloat/multilingual-e5-large-instruct` | Yes | 500 Internal Server Error |
| Switch to `sentence-transformers/all-MiniLM-L6-v2` | Yes | Dimension mismatch (384 vs 768) cascaded through system |
| DB schema `vector(1024)` for Qwen3 | Yes | Reverted to `vector(768)` for Gemini |
| DB schema `vector(384)` for MiniLM | Yes | Reverted to `vector(768)` for Gemini |
| `query.py` imports `embedder_hf` | Yes | Reverted to `embedder` (Gemini) |
| `ingestion_pipeline.py` imports `embedder_hf` | Yes | Reverted to `embedder` (Gemini) |
| Query rewriter enabled | Yes (disabled) | User preference, not technical failure |
| RRF k=60 | Yes (→ k=1) | Poor score separation for small dataset |

---

*End of Report*
