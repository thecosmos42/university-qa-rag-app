# rag_system/vector_store.py
"""
Persist and retrieve text chunks using ChromaDB as a local,
file-backed vector database.
"""

from typing import List

import chromadb
from chromadb.config import Settings

from src.config.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, TOP_K_RESULTS


class VectorStore:
    """Local, persistent vector store backed by ChromaDB."""

    def __init__(
        self,
        persist_dir: str = str(CHROMA_PERSIST_DIR),
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        """
        Initialise (or reconnect to) a persistent ChromaDB instance.

        Args:
            persist_dir:     Directory on disk where ChromaDB stores its data.
            collection_name: Name of the collection to use or create.
        """
        self.collection_name = collection_name

        # PersistentClient keeps data on disk across runs automatically.
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # get_or_create is idempotent: safe to call every time.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        print(
            f"[VectorStore] Connected to collection '{self.collection_name}' "
            f"— {self.collection.count()} existing vectors"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[dict],
        embeddings: List[List[float]],
    ) -> None:
        """
        Store text chunks and their pre-computed embeddings.

        Args:
            chunks:     List of chunk dicts (must contain "text", "source",
                        "chunk_index" keys — as produced by TextChunker).
            embeddings: Parallel list of embedding vectors, one per chunk.

        Raises:
            ValueError: If chunks and embeddings have different lengths.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                "chunks and embeddings must have the same length "
                f"(got {len(chunks)} vs {len(embeddings)})"
            )

        # ChromaDB requires unique string IDs.  We generate a deterministic
        # ID from source + chunk_index so that re-ingesting the same PDFs
        # simply overwrites duplicates instead of creating them.
        ids = [
            f"{chunk['source']}__chunk_{chunk['chunk_index']}"
            for chunk in chunks
        ]

        texts = [chunk["text"] for chunk in chunks]

        metadata = [
            {
                "source": chunk["source"],
                "page": chunk.get("page", 0),
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadata,
        )
        print(f"[VectorStore] Upserted {len(chunks)} chunks.")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = TOP_K_RESULTS,
    ) -> List[dict]:
        """
        Retrieve the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: The embedding vector of the user's question.
            top_k:           Number of results to return.

        Returns:
            A list of result dicts, each containing:
                {
                    "text":     str,   # the chunk text
                    "source":   str,   # original PDF filename
                    "distance": float  # cosine distance (lower = more similar)
                }
            Results are ordered from most similar to least similar.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns nested lists (one list per query); we only
        # ever pass a single query, so unwrap the outer layer.
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text": doc,
                "source": meta["source"],
                "distance": dist,
            }
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]