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

from src.ingestion.pdf_ingester import PDFIngester
from src.ingestion.curriculum_chunker import CurriculumChunker
from src.ingestion.policy_chunker import PolicyChunker
from src.ingestion.embedder import Embedder
from src.db.supabase_db import Supabase
from src.config.db_config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.config.config import PDF_INPUT_DIR
import os
import sys
import json
import hashlib


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
    curriculum_chunker = CurriculumChunker(
        code_field_synonyms=["Module Nr."],
        credits_field_synonyms=["ECTS", "Credits"],
        period_field_synonyms=["Period", "Semester", "Trimester"],
        track_keywords=["specialisatie", "track"],
    )
    policy_chunker = PolicyChunker()
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

        if filename.lower().startswith("ter"):
            print("[Chunker] Using PolicyChunker for TER document")
            chunk_dicts = policy_chunker.chunk_pdf(file_path)
        else:
            print("[Chunker] Using CurriculumChunker for curriculum document")
            # Load pages from PDF
            documents = ingester.load_single_pdf(file_path)
            # Course-aware chunking (detects course boundaries, extracts metadata)
            chunk_dicts = curriculum_chunker.chunk_documents(documents)

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