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
from src.ingestion.embedder_factory import get_embedder, check_cache_backend, record_cache_backend
from src.db.supabase_db import Supabase
from src.config.db_config import SUPABASE_URL, SUPABASE_SERVICE_KEY
from src.config.config import PDF_INPUT_DIR, EMBEDDING_BACKEND
import os
import re
import sys
import json
import hashlib


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case. e.g. ArtificialIntelligence → artificial_intelligence"""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def parse_filename_metadata(filename: str) -> dict:
    """
    Parse doc_type, program_type, and program_name from the filename convention:
        [DocType]_[ProgramType]_[CourseName].pdf

    Examples:
        Studiegids_Masters_ArtificialIntelligence.pdf
            → { doc_type: "curriculum", program_type: "masters", program_name: "artificial_intelligence" }
        TER_Masters_ArtificialIntelligence.pdf
            → { doc_type: "ter", program_type: "masters", program_name: "artificial_intelligence" }
        TER_Bachelors_ClinicalAndDevelopmentalPsychopathology.pdf
            → { doc_type: "ter", program_type: "bachelors", program_name: "clinical_and_developmental_psychopathology" }

    Falls back to None values if the filename does not match the convention.
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_", 2)

    if len(parts) < 3:
        print(f"[Warning] '{filename}' does not match naming convention. "
              "Expected [DocType]_[ProgramType]_[CourseName].pdf. "
              "Metadata fields will be None.")
        return {"doc_type": None, "program_type": None, "program_name": None}

    doc_type_raw, program_type_raw, program_name_raw = parts[0], parts[1], parts[2]

    # Normalise doc_type: "Studiegids" → "curriculum", everything else lowercased as-is
    doc_type = "curriculum" if doc_type_raw.lower() == "studiegids" else doc_type_raw.lower()

    # Normalise program_type to lowercase; convert program_name from CamelCase to snake_case
    program_type = program_type_raw.lower()
    program_name = camel_to_snake(program_name_raw)

    # Strip common filename suffixes that are not part of the canonical program name
    _STRIP_SUFFIXES = [
        "_joint_degree",
        "_research_master",
        "_2_years",
        "_ft",
    ]
    for suffix in _STRIP_SUFFIXES:
        if program_name.endswith(suffix):
            program_name = program_name[: -len(suffix)]
            break

    return {
        "doc_type": doc_type,
        "program_type": program_type,
        "program_name": program_name,
    }


def calculate_checksum(file_path: str) -> str:
    """Helper to hash the file content."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_ingestion(force: bool = False) -> None:
    # Guard against accidental backend switches that would corrupt the DB.
    # Skipped on --force because the user is explicitly rebuilding everything.
    if not force:
        check_cache_backend()

    print(f"[Embedder] Backend: '{EMBEDDING_BACKEND}'")

    store = Supabase(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    ingester = PDFIngester()
    curriculum_chunker = CurriculumChunker(
        code_field_synonyms=["Module Nr."],
        credits_field_synonyms=["ECTS", "Credits"],
        period_field_synonyms=["Period", "Semester", "Trimester"],
        track_keywords=["specialisatie", "track"],
    )
    policy_chunker = PolicyChunker()
    embedder = get_embedder()

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

        # Parse program-level metadata from the filename convention
        file_meta = parse_filename_metadata(filename)
        print(
            f"[Meta] doc_type={file_meta['doc_type']} | "
            f"program_type={file_meta['program_type']} | "
            f"program_name={file_meta['program_name']}"
        )

        # Load program_url from the sidecar JSON file if it exists
        # Sidecar convention: data/pdfs/<filename>.json
        sidecar_path = os.path.join(PDF_INPUT_DIR, f"{filename}.json")
        program_url: str | None = None
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as f:
                    sidecar = json.load(f)
                program_url = sidecar.get("program_url") or None
                if program_url:
                    print(f"[Meta] program_url={program_url}")
                else:
                    print(f"[Meta] program_url not found in sidecar — will be NULL in DB")
            except Exception as e:
                print(f"[Warning] Could not read sidecar {sidecar_path}: {e}")
        else:
            print(f"[Meta] No sidecar JSON found for {filename} — program_url will be NULL")

        if file_meta["doc_type"] == "ter":
            print("[Chunker] Using PolicyChunker for TER document")
            chunk_dicts = policy_chunker.chunk_pdf(file_path)
        else:
            print("[Chunker] Using CurriculumChunker for curriculum document")
            # Load pages from PDF
            documents = ingester.load_single_pdf(file_path)
            # Course-aware chunking (detects course boundaries, extracts metadata)
            chunk_dicts = curriculum_chunker.chunk_documents(documents)

        # Inject program-level metadata into every chunk so it is stored in the DB
        for chunk in chunk_dicts:
            chunk.setdefault("doc_type", file_meta["doc_type"])
            chunk.setdefault("program_type", file_meta["program_type"])
            chunk.setdefault("program_name", file_meta["program_name"])

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
            program_url=program_url,
        )

    # Record which backend was used so future runs can detect mismatches.
    record_cache_backend()
    print("--- Ingestion Complete ---")


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    run_ingestion(force=force_flag)