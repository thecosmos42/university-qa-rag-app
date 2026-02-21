"""
HuggingFace Inference API Embedder
====================================
Uses the HF Serverless Inference API to embed text with BAAI/bge-base-en-v1.5.

  - 768-dim output — matches Supabase vector(768) exactly
  - Free-tier compatible (requires HF_TOKEN from huggingface.co/settings/tokens)
  - Batched requests to minimise API round-trips
  - Automatic retry with exponential backoff for:
      503 — model cold start (free tier spins down idle models, ~20s warm-up)
      429 — rate limit
  - L2-normalised output (required for cosine similarity in pgvector)

Notes on free-tier cold starts
-------------------------------
HF Serverless Inference spins down models after ~15 minutes of inactivity.
The first request after an idle period returns 503 with "Model is currently
loading". The retry logic here waits and re-tries automatically, so the caller
just experiences a slightly slower first response (~20s) rather than an error.
"""

import time
import numpy as np
from typing import List

from huggingface_hub import InferenceClient
from src.config.config import HF_TOKEN, HF_EMBEDDING_MODEL_NAME


# Max texts per API call.
# BAAI/bge-base chunks are ~800 chars; 32 texts ≈ ~25KB payload — well within
# the HF Inference API's 2MB limit.
_BATCH_SIZE = 32

# Retry configuration
_MAX_RETRIES = 6
_BASE_BACKOFF_S = 5     # initial wait (seconds)
_MAX_BACKOFF_S = 60     # cap to avoid excessively long waits


class Embedder:
    """HuggingFace Inference API embedder using BAAI/bge-base-en-v1.5 (768-dim)."""

    def __init__(
        self,
        model_id: str = HF_EMBEDDING_MODEL_NAME,
        hf_token: str | None = None,
    ):
        token = hf_token or HF_TOKEN
        if not token:
            raise ValueError(
                "HF_TOKEN is not set. Add HF_TOKEN=your_token to your .env file.\n"
                "Get a free read token at: https://huggingface.co/settings/tokens"
            )
        self.model_id = model_id
        self.client = InferenceClient(api_key=token)
        print(f"[Embedder-HF] Model: '{self.model_id}' — HF Inference API")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_retriable(exc: Exception) -> bool:
        """Return True for errors that warrant a wait-and-retry."""
        msg = str(exc).lower()
        return (
            "503" in msg
            or "loading" in msg
            or "429" in msg
            or "rate limit" in msg
            or "too many requests" in msg
        )

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Call the HF Inference API for one batch of texts.
        Returns float32 numpy array of shape (len(texts), 768), L2-normalised.
        Retries on cold-start (503) and rate-limit (429) errors.
        """
        backoff = _BASE_BACKOFF_S

        for attempt in range(_MAX_RETRIES):
            try:
                raw = self.client.feature_extraction(texts, model=self.model_id)
                arr = np.array(raw, dtype=np.float32)

                # Defensive shape handling:
                #   2D (n, 768)        — already pooled, use directly
                #   3D (n, seq, 768)   — token-level output, mean-pool over seq axis
                if arr.ndim == 3:
                    arr = arr.mean(axis=1)

                if arr.ndim != 2 or arr.shape[1] != 768:
                    raise ValueError(
                        f"Unexpected embedding shape {arr.shape} from model '{self.model_id}'. "
                        "Expected (n, 768)."
                    )

                # L2 normalise so dot product == cosine similarity in pgvector
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)  # guard against zero vectors
                return arr / norms

            except Exception as e:
                if not self._is_retriable(e):
                    raise  # non-retriable (e.g. auth error, bad payload) — fail fast

                if attempt < _MAX_RETRIES - 1:
                    if "503" in str(e) or "loading" in str(e).lower():
                        print(
                            f"[Embedder-HF] Model cold start — "
                            f"waiting {backoff}s before retry "
                            f"({attempt + 1}/{_MAX_RETRIES})…"
                        )
                    else:
                        print(
                            f"[Embedder-HF] Rate limited — "
                            f"waiting {backoff}s before retry "
                            f"({attempt + 1}/{_MAX_RETRIES})…"
                        )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, _MAX_BACKOFF_S)
                else:
                    raise RuntimeError(
                        f"[Embedder-HF] All {_MAX_RETRIES} retries failed. Last error: {e}"
                    ) from e

    # ------------------------------------------------------------------
    # Public API  (mirrors local_embedder.Embedder and embedder.Embedder)
    # ------------------------------------------------------------------

    def encode(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Encode a list of texts into 768-dim L2-normalised vectors.

        task_type is accepted for interface compatibility with the Gemini embedder
        but is not used — BAAI/bge-base-en-v1.5 is trained for retrieval by default.

        Returns a list of float lists with shape (len(texts), 768).
        """
        if not texts:
            return []

        chunks: list[np.ndarray] = []
        total_batches = (len(texts) + _BATCH_SIZE - 1) // _BATCH_SIZE

        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i: i + _BATCH_SIZE]
            batch_num = i // _BATCH_SIZE + 1
            print(f"  [Embedder-HF] Batch {batch_num}/{total_batches} ({len(batch)} texts)…")
            chunks.append(self._embed_batch(batch))

        return np.vstack(chunks).tolist()  # (n_total, 768) → list[list[float]]
