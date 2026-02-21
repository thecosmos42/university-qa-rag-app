"""
Embedder Factory
================
Single source of truth for which embedding backend is active.

Both the ingestion pipeline and the query pipeline call `get_embedder()`.
This guarantees that ingestion-time and query-time embeddings always come
from the same model — mismatching the two is the #1 cause of broken retrieval.

To switch backends:
    1. Change EMBEDDING_BACKEND in src/config/config.py
    2. Delete data/embeddings_cache/ (or run ingestion with --force)
    3. Re-ingest all documents

Supported backends
------------------
  "gemini"       — gemini-embedding-001 via Google GenAI API (requires GOOGLE_API_KEY)
  "local"        — BAAI/bge-base-en-v1.5 via sentence-transformers (no quota, GPU-aware)
  "huggingface"  — BAAI/bge-base-en-v1.5 via HF Serverless Inference API (requires HF_TOKEN)
"""

import json
import os
from pathlib import Path

from src.config.config import EMBEDDING_BACKEND

# File that records which backend was used to build the current embeddings cache.
# Lives next to the cache directory so it travels with it.
_CACHE_META_PATH = Path("data/embeddings_cache/.backend")


def get_embedder():
    """
    Return an initialised Embedder for the configured backend.

    All three embedder classes expose the same interface:
        embedder.encode(texts: list[str], task_type: str) -> list[list[float]]
    so callers need no backend-specific logic.
    """
    backend = EMBEDDING_BACKEND.lower().strip()

    if backend == "local":
        from src.ingestion.local_embedder import Embedder
        return Embedder()

    if backend == "gemini":
        from src.ingestion.embedder import Embedder
        return Embedder()

    if backend == "huggingface":
        from src.ingestion.embedder_hf import Embedder
        return Embedder()

    raise ValueError(
        f"Unknown EMBEDDING_BACKEND '{EMBEDDING_BACKEND}' in config.py. "
        "Valid values: 'local', 'gemini', 'huggingface'."
    )


# ---------------------------------------------------------------------------
# Cache-backend consistency helpers
# Used by the ingestion pipeline to detect and warn about backend switches.
# ---------------------------------------------------------------------------

def record_cache_backend() -> None:
    """
    Write the current EMBEDDING_BACKEND to the cache metadata file.
    Call this after a successful ingestion run.
    """
    _CACHE_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_META_PATH.write_text(json.dumps({"backend": EMBEDDING_BACKEND}))


def check_cache_backend() -> None:
    """
    Warn (and prompt) if the active EMBEDDING_BACKEND differs from the one
    that was used to build the existing embeddings cache.

    Raises RuntimeError if the user does not confirm they want to proceed,
    because continuing would silently mix embeddings from different models
    in the database.
    """
    if not _CACHE_META_PATH.exists():
        # No metadata yet — first run, or cache was manually cleared. Fine.
        return

    try:
        meta = json.loads(_CACHE_META_PATH.read_text())
        cached_backend = meta.get("backend", "unknown")
    except Exception:
        # Unreadable metadata — treat as unknown, warn but don't block.
        print(
            "[EmbedderFactory] WARNING: Could not read cache metadata. "
            "Ensure EMBEDDING_BACKEND is consistent with your stored vectors."
        )
        return

    if cached_backend != EMBEDDING_BACKEND:
        msg = (
            f"\n{'='*70}\n"
            f"[EmbedderFactory] BACKEND MISMATCH DETECTED\n"
            f"  Cache was built with : '{cached_backend}'\n"
            f"  Current backend is   : '{EMBEDDING_BACKEND}'\n\n"
            f"  Stored document embeddings and query embeddings will be in\n"
            f"  DIFFERENT vector spaces — retrieval will return random results.\n\n"
            f"  Fix: delete data/embeddings_cache/ and re-run ingestion with --force\n"
            f"  to rebuild all embeddings using the new backend.\n"
            f"{'='*70}\n"
        )
        print(msg)
        raise RuntimeError(
            f"EMBEDDING_BACKEND changed from '{cached_backend}' to '{EMBEDDING_BACKEND}'. "
            "Clear data/embeddings_cache/ and re-ingest with --force before querying."
        )
