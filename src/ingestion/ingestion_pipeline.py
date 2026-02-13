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
from src.config.config import PDF_INPUT_DIR
import os
import hashlib


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

def calculate_checksum(file_path: str) -> str:
    """Helper to hash the file content."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

import numpy as np

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

        # Load chunk
        documents = ingester.load_single_pdf(file_path)
        chunks = chunker.chunk_documents(documents)
        chunk_dicts = _documents_to_dicts(chunks)

        # Embed
        texts = [c["text"] for c in chunk_dicts]
        embeddings = embedder.encode(texts)

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


if __name__ == "__main__":
    run_ingestion()