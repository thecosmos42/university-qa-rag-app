"""
Central configuration module.

All tunable parameters for the RAG pipeline live here.
Every other module imports what it needs from this file
so that changing behaviour never requires hunting through code.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — anchored to this file so paths resolve correctly regardless
# of the working directory when uvicorn / the process manager starts.
# config.py lives at <root>/src/config/config.py → three .parent() calls.
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Directory where user PDF files are placed for ingestion.
PDF_INPUT_DIR: Path = _PROJECT_ROOT / "data" / "pdfs"

# Root directory where ChromaDB will persist its vector store on disk.
# Kept outside src/ so it is writable even when source tree is read-only.
CHROMA_PERSIST_DIR: Path = _PROJECT_ROOT / "data" / "chromadb"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Number of characters per chunk (before overlap).
CHUNK_SIZE: int = 800

# Number of overlapping characters between consecutive chunks.
# Overlap ensures that context at chunk boundaries is not lost during retrieval.
CHUNK_OVERLAP: int = 100

# Ordered list of separator strings that RecursiveCharacterTextSplitter tries
# before falling back to the next one.  It attempts to split on double-newlines
# (paragraphs) first, then single newlines, then spaces, then individual
# characters.  Keeping natural boundaries intact improves retrieval quality.
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Which embedding backend to use for BOTH ingestion and query time.
# Changing this value REQUIRES clearing data/embeddings_cache/ and re-ingesting
# with --force, otherwise stored vectors and query vectors will be mismatched.
#
#   "local"        — BAAI/bge-base-en-v1.5 via sentence-transformers (no API key, GPU-aware)
#   "gemini"       — gemini-embedding-001 via Google Generative AI API (requires GOOGLE_API_KEY)
#   "huggingface"  — BAAI/bge-base-en-v1.5 via HF Serverless Inference API (requires HF_TOKEN)
EMBEDDING_BACKEND: str = "huggingface"

# Sentence-Transformers model name used when EMBEDDING_BACKEND = "local".
# BAAI/bge-base-en-v1.5 outputs 768-dim vectors (matches Supabase vector(768))
# and runs on GPU automatically when available.
EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"

# HuggingFace Inference API settings (used when EMBEDDING_BACKEND = "huggingface").
# BAAI/bge-base-en-v1.5 is used here too — ensures the SAME model and vector
# space is used regardless of whether you run locally or via the HF API.
HF_EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"


# ---------------------------------------------------------------------------
# Vector Store (ChromaDB)
# ---------------------------------------------------------------------------
# Name of the ChromaDB collection that stores embedded chunks.
CHROMA_COLLECTION_NAME: str = "ms_ai_program"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# How many chunks to retrieve for a single query.
TOP_K_RESULTS: int = 5


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Hugging Face model identifier.  Must fit in local VRAM / RAM.
# "google/gemma-2b-it" is a good default; swap for any instruction-tuned model.
LLM_MODEL_NAME: str = "google/gemma-2b-it"

# Maximum number of new tokens the LLM is allowed to generate per answer.
LLM_MAX_NEW_TOKENS: int = 512

# Sampling temperature.  0 = greedy (deterministic); higher = more creative.
LLM_TEMPERATURE: float = 0.1

# Flag for enabling history context for prompts
ENABLE_HISTORY: bool = False

# Logger path — absolute so it works regardless of cwd.
LOGGER_PATH: str = str(_PROJECT_ROOT / "logs")

import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set. Add it to your .env file.")

# Support multiple Gemini keys for rate-limit rotation.
# Add GOOGLE_API_KEY_2, GOOGLE_API_KEY_3, … to .env as needed.
GOOGLE_API_KEYS: list[str] = [GOOGLE_API_KEY] if GOOGLE_API_KEY else []
for _i in range(2, 10):
    _extra = os.environ.get(f"GOOGLE_API_KEY_{_i}")
    if _extra:
        GOOGLE_API_KEYS.append(_extra)
    else:
        break

# HuggingFace token — only required when EMBEDDING_BACKEND = "huggingface".
# Get a free read token at: https://huggingface.co/settings/tokens
HF_TOKEN: str | None = os.environ.get("HF_TOKEN")