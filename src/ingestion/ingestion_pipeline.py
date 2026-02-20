# rag_system/ingestion_pipeline.py
"""
Ingestion Pipeline
Orchestrate the full ingestion flow:
PDF files -> LangChain Documents -> course-aware chunks -> embeddings -> vector store

Run this script once (or re-run it whenever new PDFs are added) to populate
the vector database.  The query pipeline (main.py) reads from that database
at runtime.

Usage:
    python -m src.ingestion.ingestion_pipeline
    python -m src.ingestion.ingestion_pipeline --force   # Force re-ingestion
"""

from pathlib import Path
from typing import List
from src.ingestion.pdf_ingester import PDFIngester
from src.ingestion.academic_chunker import AcademicChunker
from src.ingestion.embedder import Embedder
from src.db.supabase_db import Supabase
from src.config.db_config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.config.config import PDF_INPUT_DIR
import os
import sys
import json
import hashlib


def _documents_to_dicts(chunks) -> List[dict]:
    """
    Convert LangChain Document objects into the plain dicts that the
    Embedder and Supabase expect.

    Now includes structured metadata extracted by AcademicChunker:
    course_name, course_code, credits, degree_name, section_type.

    Args:
        chunks: List of LangChain Document objects as returned by
                AcademicChunker.chunk_documents().

    Returns:
        A list of dicts with text, source, page, chunk_index, and metadata.
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
            # Structured metadata from AcademicChunker
            "course_name": doc.metadata.get("course_name", ""),
            "course_code": doc.metadata.get("course_code", ""),
            "credits": doc.metadata.get("credits", ""),
            "degree_name": doc.metadata.get("degree_name", ""),
            "section_type": doc.metadata.get("section_type", ""),
        })
    
    # Print a sample of first 3 chunks to verify metadata extraction
    for i, chunk in enumerate(result[:3]):
        print(f"--- Chunk {i} ---")
        print(f"  Course: {chunk['course_name']} | Code: {chunk['course_code']}")
        print(f"  Section: {chunk['section_type']} | Page: {chunk['page']}")
        print(f"  Content: {chunk['text'][:120]}...")
    
    return result

def calculate_checksum(file_path: str) -> str:
    """Helper to hash the file content."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_ingestion(force: bool = False) -> None:
    store = Supabase(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    ingester = PDFIngester()
    chunker = AcademicChunker(
        max_chunk_size=2500,
        chunk_overlap=150,
        min_chunk_size=100,
    )
    embedder = Embedder()

    # Define a cache directory for embeddings
    EMBEDDINGS_CACHE_DIR = "data/embeddings_cache"
    os.makedirs(EMBEDDINGS_CACHE_DIR, exist_ok=True)

    pdf_files = [f for f in os.listdir(PDF_INPUT_DIR) if f.endswith(".pdf")]

    for filename in pdf_files:
        file_path = os.path.join(PDF_INPUT_DIR, filename)
        embeddings_path = os.path.join(EMBEDDINGS_CACHE_DIR, f"{filename}.json")

        checksum = calculate_checksum(file_path)

        if force:
            store.force_update(filename)
            print(f"[Force] Re-ingesting {filename}...")
        elif not store.needs_update(filename, checksum):
            print(f"[Skip] {filename} is already up to date.")
            continue

        print(f"[Process] {filename} — Starting pipeline...")

        # Load pages from PDF
        documents = ingester.load_single_pdf(file_path)
        
        # Course-aware chunking (detects course boundaries, extracts metadata)
        chunks = chunker.chunk_documents(documents)
        chunk_dicts = _documents_to_dicts(chunks)

        embeddings = None
        # Try to load embeddings from cache if not forcing re-ingestion
        if not force and os.path.exists(embeddings_path):
            print(f"[Cache] Loading embeddings from {embeddings_path}")
            try:
                with open(embeddings_path, "r", encoding="utf-8") as f:
                    embeddings = json.load(f)
                # Basic validation
                if not isinstance(embeddings, list) or len(embeddings) != len(chunk_dicts):
                    print("[Warning] Cached embeddings are invalid. Regenerating.")
                    embeddings = None
            except Exception as e:
                print(f"[Error] Failed to load cached embeddings: {e}. Regenerating.")
                embeddings = None

        # If embeddings weren't loaded from cache, generate them
        if embeddings is None:
            print("[Embed] Generating new embeddings...")
            texts = [c["text"] for c in chunk_dicts]
            embeddings = embedder.encode(texts)
            print(f"[Cache] Saving embeddings to {embeddings_path}")
            with open(embeddings_path, "w", encoding="utf-8") as f:
                json.dump(embeddings, f)


        # Store in Supabase with structured metadata
        print(f"[Store] Inserting {len(chunk_dicts)} chunks into DB...")
        store.replace_document(
            filename=filename,
            title=filename,
            checksum=checksum,
            chunks=chunk_dicts,
            embeddings=embeddings,
        )

    print("--- Ingestion Complete ---")


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    run_ingestion(force=force_flag)