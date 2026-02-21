"""
Local GPU Embedding Module

Uses sentence-transformers with automatic GPU acceleration.
BAAI/bge-base-en-v1.5 outputs 768-dim vectors — matches Supabase vector(768).
No API keys, no quotas, no rate limits.
"""

from typing import List
import torch
from sentence_transformers import SentenceTransformer
from src.config.config import EMBEDDING_MODEL_NAME


class Embedder:
    """Local sentence-transformers embedder with automatic GPU support."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Embedder] Loading '{model_name}' on {self.device.upper()}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print(f"[Embedder] Ready — output dim: {self.model.get_sentence_embedding_dimension()}")

    def encode(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        """
        Encode texts into vectors.
        task_type is accepted for interface compatibility with the Gemini embedder
        but is not used — BGE models are trained for retrieval by default.

        Returns list of float lists with shape (len(texts), 768).
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=64,          # RTX 5060 can handle large batches
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # BGE models benefit from L2 normalisation
        )
        return embeddings.tolist()