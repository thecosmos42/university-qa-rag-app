# rag_system/embedder.py
"""
Embedding Module

Convert text strings into dense numerical vectors
"""

from typing import List
from sentence_transformers import SentenceTransformer
from src.config.config import EMBEDDING_MODEL_NAME


class Embedder:
    """Wraps a sentence-transformers model to produce text embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Load the embedding model into memory.

        Args:
            model_name: A sentence-transformers model identifier.
                        Defaults to the value in config.py.
        """
        print(f"[Embedder] Loading model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        print("[Embedder] Model loaded.")

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode a list of text strings into vectors.

        Args:
            texts: One or more strings to embed.

        Returns:
            A list of embedding vectors (list of floats), one per input string.
            Each vector has the same dimensionality as the model's output
            (384 for all-MiniLM-L6-v2).
        """
        if not texts:
            return []

        # sentence-transformers handles batching internally.
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # .encode() returns a NumPy array; convert to plain Python lists
        # so the result is JSON-serialisable and ChromaDB-compatible.
        return embeddings.tolist()