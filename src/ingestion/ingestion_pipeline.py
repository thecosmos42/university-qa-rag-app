# rag_system/ingest.py
"""
Ingestion Pipeline

Responsibility: Orchestrate the full ingestion flow:
    PDF files → LangChain Documents → chunks → embeddings → vector store

Run this script once (or re-run it whenever new PDFs are added) to populate
the vector database.  The query pipeline (main.py) reads from that database
at runtime.

Usage:
    python ingest.py
"""

from pathlib import Path
from typing import List

from src.ingestion.pdf_ingester import PDFIngester
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import Embedder
from src.vector_store.persist_data import VectorStore


def _documents_to_dicts(chunks) -> List[dict]:
    """
    Convert LangChain Document objects into the plain dicts that the
    Embedder and VectorStore expect.

    This is the single translation point between LangChain's Document type
    and the rest of the pipeline.  If the LangChain interface ever changes,
    only this function needs updating.

    Args:
        chunks: List of LangChain Document objects as returned by
                TextChunker.chunk_documents().

    Returns:
        A list of dicts, each with:
            {
                "text":        str,   # chunk text
                "source":      str,   # original PDF filename (basename only)
                "chunk_index": int    # 0-based position within that file
            }
    """
    result: List[dict] = []
    for doc in chunks:
        # .metadata["source"] is a full path from the loader; we keep only
        # the filename so ChromaDB IDs stay short and portable.
        source_path = doc.metadata.get("source", "unknown")
        source_name = Path(source_path).name

        result.append({
            "text": doc.page_content,
            "source": source_name,
            "chunk_index": doc.metadata.get("chunk_index", 0),
        })
    return result


def run_ingestion() -> None:
    """Execute the full ingestion pipeline end-to-end."""

    # 1. Load PDFs → list of LangChain Documents (one per page)
    ingester = PDFIngester()
    documents = ingester.ingest_all()

    # 2. Split pages into overlapping chunks (still LangChain Documents)
    chunker = TextChunker()
    chunks = chunker.chunk_documents(documents)

    # 3. Translate LangChain Documents → plain dicts for the rest of the pipeline
    chunk_dicts = _documents_to_dicts(chunks)

    # 4. Embed every chunk
    embedder = Embedder()
    texts = [chunk["text"] for chunk in chunk_dicts]
    embeddings = embedder.encode(texts)

    # 5. Store chunks + embeddings in ChromaDB
    store = VectorStore()
    store.add_chunks(chunk_dicts, embeddings)

    print("\n✔ Ingestion complete")


if __name__ == "__main__":
    run_ingestion()