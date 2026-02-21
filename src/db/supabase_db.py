import hashlib
from typing import List, Optional
from supabase import create_client, Client
from pathlib import Path


class Supabase:
    """
    A controller class for Supabase operations, specifically designed 
    for managing RAG (Retrieval-Augmented Generation) document storage.
    """
    def __init__(self, url: str, key: str):
        self.supabase: Client = create_client(url, key)
    
    def clean_text(self, text: str) -> str:
        """Removes illegal characters that cause PostgreSQL insert errors."""
        if not text:
            return text
        return text.replace("\x00", "")


    def replace_document(
        self,
        *,
        filename: str,
        title: str,
        checksum: str,
        chunks: list[dict],
        embeddings: list[list[float]],
        program_url: str | None = None,
    ) -> None:
        """
        Performs an atomic-like refresh of a document: 
        Updates metadata, purges old vectors, and inserts new ones.
        Now includes structured metadata (course_name, course_code, etc.)
        """
        # Upsert document row
        doc_res = (
            self.supabase
            .table("documents")
            .upsert(
                {
                    "filename": filename,
                    "title": title,
                    "checksum": checksum,
                    "program_url": program_url,
                },
                on_conflict="filename",
            )
            .execute()
        )

        doc_id = doc_res.data[0]["id"]

        # Remove old chunks
        (
            self.supabase
            .table("chunks")
            .delete()
            .eq("document_id", doc_id)
            .execute()
        )

        # Insert new chunks with structured metadata
        payload = [
            {
                "document_id": doc_id,
                "content": self.clean_text(c["text"]),
                "embedding": e,
                "page_number": c.get("page", 0),
                "chunk_index": c["chunk_index"],
                "metadata": {
                    "source": filename,
                    "course_name": c.get("course_name", ""),
                    "course_code": c.get("course_code", ""),
                    "credits": c.get("credits", ""),
                    "section_type": c.get("section_type", ""),
                },
                # Program-level columns (parsed from filename convention)
                "doc_type": c.get("doc_type") or None,
                "program_type": c.get("program_type") or None,
                "program_name": c.get("program_name") or None,
            }
            for c, e in zip(chunks, embeddings)
        ]

        print(f"[Supabase] Inserting {len(payload)} chunks for '{filename}'")

        if payload:
            # Insert in batches of 50 to avoid payload size limits
            batch_size = 50
            for i in range(0, len(payload), batch_size):
                batch = payload[i:i + batch_size]
                self.supabase.table("chunks").insert(batch).execute()
                print(f"  Batch {i // batch_size + 1} inserted ({len(batch)} chunks)")

        print(f"[OK] {filename} ingested ({len(payload)} chunks)")


    def needs_update(self, filename: str, checksum: str) -> bool:
        """
        Compares local file hash against DB hash to avoid redundant embedding API calls.
        """
        res = (
            self.supabase
            .table("documents")
            .select("checksum")
            .eq("filename", filename)
            .execute()
        )

        if not res.data:
            return True

        return res.data[0]["checksum"] != checksum
    
    def force_update(self, filename: str) -> None:
        """
        Force re-ingestion by deleting existing document entry.
        Useful when schema changes require re-processing.
        """
        (
            self.supabase
            .table("documents")
            .delete()
            .eq("filename", filename)
            .execute()
        )
        print(f"[Force] Cleared '{filename}' from DB for re-ingestion.")