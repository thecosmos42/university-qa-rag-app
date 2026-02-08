import hashlib
from typing import List, Optional
from supabase import create_client, Client
from pathlib import Path


class Supabase:
    def __init__(self, url: str, key: str):
        self.supabase: Client = create_client(url, key)
    
    def clean_text(self, text: str) -> str:
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
    ) -> None:
        # Upsert document row
        doc_res = (
            self.supabase
            .table("documents")
            .upsert(
                {
                    "filename": filename,
                    "title": title,
                    "checksum": checksum,
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

        # Insert new chunks
        payload = [
            {
                "document_id": doc_id,
                "content": self.clean_text(c["text"]),
                "embedding": e,
                "page_number": c.get("page", 0),
                "chunk_index": c["chunk_index"],
                "metadata": {"source": filename},
            }
            for c, e in zip(chunks, embeddings)
        ]

        print(f"The PAyload is: {payload}")

        if payload:
            self.supabase.table("chunks").insert(payload).execute()

        print(f"[OK] {filename} ingested ({len(payload)} chunks)")


    def needs_update(self, filename: str, checksum: str) -> bool:
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