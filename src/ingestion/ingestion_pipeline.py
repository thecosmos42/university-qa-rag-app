# rag_system/ingestion_pipeline.py
"""
Ingestion Pipeline
Orchestrate the full ingestion flow:
PDF files -> LangChain Documents -> chunks -> embeddings -> vector store

Run this script once (or re-run it whenever new PDFs are added) to populate
the vector database.  The query pipeline (main.py) reads from that database
at runtime.

Usage:
    python -m src.ingestion.ingestion_pipeline
"""

from pathlib import Path
from typing import List
from src.ingestion.pdf_ingester import PDFIngester
from src.ingestion.chunker import TextChunker
from src.ingestion.embedder import Embedder
from src.db.supabase_db import Supabase
from src.config.db_config import SUPABASE_URL, SUPABASE_SERVICE_KEY


def _documents_to_dicts(chunks) -> List[dict]:
    """
    Convert LangChain Document objects into the plain dicts that the
    Embedder and Supabase expect.

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
                "page":        int    # 1-based position within that file
            }
    """
    result: List[dict] = []
    for doc in chunks:
        source_path = doc.metadata.get("source", "unknown")
        source_name = Path(source_path).name

        result.append({
            "text": doc.page_content,
            "source": source_name,
            "page": doc.metadata.get("page", 0),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        })
    # Print a sample of the first 2 chunks to verify
    for i, chunk in enumerate(result[:2]):
        print(f"--- After translating Langchain to dict {i} ---")
        print(f"Content Preview: {chunk}")
    return result

from src.config.config import PDF_INPUT_DIR
import os
import hashlib

def calculate_checksum(file_path: str) -> str:
    """Helper to hash the file content."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

import numpy as np

def pad_embedding(embedding: list, target_dim: int = 768) -> list:
    # Convert to numpy array
    emb_array = np.array(embedding)
    
    # Calculate how many zeros to add
    padding_size = target_dim - len(emb_array)
    
    if padding_size > 0:
        # Add zeros to the end
        padded = np.pad(emb_array, (0, padding_size), mode='constant')
        return padded.tolist()
    
    return embedding # Already correct or larger

def run_ingestion() -> None:
    store = Supabase(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    ingester = PDFIngester()
    chunker = TextChunker()
    embedder = Embedder()

    pdf_files = [f for f in os.listdir(PDF_INPUT_DIR) if f.endswith(".pdf")]

    for filename in pdf_files:
        file_path = os.path.join(PDF_INPUT_DIR, filename)

        checksum = calculate_checksum(file_path)

        if not store.needs_update(filename, checksum):
            print(f"[Skip] {filename} is already up to date.")
            continue

        print(f"[Process] {filename} changed. Starting pipeline...")

        # Load → chunk
        documents = ingester.load_single_pdf(file_path)
        chunks = chunker.chunk_documents(documents)
        chunk_dicts = _documents_to_dicts(chunks)

        # Embed
        texts = [c["text"] for c in chunk_dicts]
        embeddings = embedder.encode(texts)

        #COMMENT this line after integrating 768 dim embeddings model
        # embeddings=pad_embedding(embeddings)

        # Single atomic replace
        print("Adding chunks")
        store.replace_document(
            filename=filename,
            title=filename,
            checksum=checksum,
            chunks=chunk_dicts,
            embeddings=embeddings,
        )

    print("--- Ingestion Complete ---")



# def run_ingestion() -> None:
#     """Execute the full ingestion pipeline end-to-end."""

#     # Load PDFs as list of LangChain Documents (one per page)
#     ingester = PDFIngester()
#     documents = ingester.ingest_all()

#     # Split pages into overlapping chunks
#     chunker = TextChunker()
#     chunks = chunker.chunk_documents(documents)

#     # Translate LangChain Documents to plain dicts for the rest of the pipeline
#     chunk_dicts = _documents_to_dicts(chunks)

#     # Embed every chunk
#     embedder = Embedder()
#     texts = [chunk["text"] for chunk in chunk_dicts]
#     embeddings = embedder.encode(texts)

#     # store chunks + embeddings in ChromaDB
#     if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
#         raise RuntimeError("Supabase config variables not set")
#     store = Supabase(SUPABASE_URL, SUPABASE_SERVICE_KEY)
#     store.add_chunks(chunk_dicts, embeddings)

#     print("\n✔ Ingestion complete")


if __name__ == "__main__":
    run_ingestion()