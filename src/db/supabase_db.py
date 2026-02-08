import hashlib
from typing import List, Optional
from supabase import create_client, Client


class Supabase:
    def __init__(self, url: str, key: str):
        self.supabase: Client = create_client(url, key)

    def _calculate_checksum(self, file_path: str) -> str:
        """Helper to hash the file content."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    from pathlib import Path

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


    # def add_chunks(self, chunks: List[dict], embeddings: List[List[float]], file_path: str, checksum: str):
    #     filename = file_path.split("/")[-1] # Extract name from path
    #     current_checksum = self._calculate_checksum(file_path)

    #     # CHECK if this version already exists
    #     existing_doc = self.supabase.table("documents") \
    #         .select("id", "checksum") \
    #         .eq("filename", filename) \
    #         .execute()

    #     if existing_doc.data:
    #         if existing_doc.data[0]["checksum"] == current_checksum:
    #             print(f"Skipping {filename}: Already up to date.")
    #             return False # No changes made
            
    #         # If checksum is different, we'll update the document 
    #         # and let 'on delete cascade' handle the old chunks (if set up)
    #         # or manually delete them here.
    #         doc_id = existing_doc.data[0]["id"]
    #         self.supabase.table("chunks").delete().eq("document_id", doc_id).execute()

    #     # UPSERT the document record
    #     doc_payload = {
    #         "filename": filename,
    #         "checksum": current_checksum,
    #         "title": filename
    #     }
    #     doc_res = self.supabase.table("documents").upsert(doc_payload, on_conflict="filename").execute()
    #     doc_id = doc_res.data[0]["id"]

    #     # INSERT the new chunks
    #     chunk_payload = [
    #         {
    #             "document_id": doc_id,
    #             "content": c["text"],
    #             "embedding": e,
    #             "page_number": c.get("page", 0),
    #             "chunk_index": c["chunk_index"],
    #             "metadata": {"source": filename}
    #         }
    #         for c, e in zip(chunks, embeddings)
    #     ]

    #     self.supabase.table("chunks").insert(chunk_payload).execute()
    #     print(f"Successfully updated {filename} ({len(chunk_payload)} chunks).")
    #     return True
    
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