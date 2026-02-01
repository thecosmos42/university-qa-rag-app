# rag_system/retriever.py
"""
Retriever Module

Responsibility: Given a user question, embed it and fetch the most relevant
text chunks from the vector store.

This is the bridge between the user's query and the stored knowledge.
It combines the Embedder and VectorStore into a single, easy-to-call interface.
"""

from typing import List

from src.ingestion.embedder import Embedder
from src.vector_store.persist_data import VectorStore
from src.config.config import TOP_K_RESULTS


class Retriever:
    """Embeds a query and retrieves the most relevant chunks."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        """
        Args:
            embedder:     An initialised Embedder instance (shared with the
                          ingestion pipeline so the same model is used).
            vector_store: An initialised VectorStore instance.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: int = TOP_K_RESULTS) -> List[dict]:
        """
        Find the chunks most relevant to a natural-language question.

        Steps:
            1. Embed the question using the same model that was used during
               ingestion (critical — mismatched models break retrieval).
            2. Run a nearest-neighbour search in the vector store.
            3. Return the matching chunks with their metadata.

        Args:
            question: The user's question as a plain string.
            top_k:    How many chunks to retrieve.

        Returns:
            A list of chunk dicts (see VectorStore.query for the exact shape).
        """
        # encode() expects a list; we pass a single-element list and take [0]
        query_embedding = self.embedder.encode([question])[0]

        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        print(f"[Retriever] Retrieved {len(results)} chunks for the query.")
        return results