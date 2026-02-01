# rag_system/chunker.py

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


class TextChunker:
    """Splits LangChain Documents into overlapping chunks using
    RecursiveCharacterTextSplitter."""

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: list[str] = CHUNK_SEPARATORS,
    ):
        """
        Args:
            chunk_size:    Target character length for each chunk.
            chunk_overlap: Number of characters shared between consecutive
                           chunks.
            separators:    Ordered list of strings the splitter tries as
                           break points.  See config.py for the default list
                           and the reasoning behind the order.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )


    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split every Document produced by PDFIngester into smaller chunks.

        The splitter preserves and propagates each input Document's metadata
        (source path, page number, etc.) into every chunk it produces.  A
        "chunk_index" key is added so downstream code can reconstruct ordering
        if needed.

        Args:
            documents: List of LangChain Document objects as returned by
                       PDFIngester.ingest_all().

        Returns:
            A flat list of Document objects, one per chunk.  Each chunk's
            metadata contains everything the original had, plus:
                "chunk_index" : int  — 0-based position within its source file.
        """
        chunks: List[Document] = self.splitter.split_documents(documents)

        # Assign a chunk_index per source file for deterministic IDs
        source_counters: dict[str, int] = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            index = source_counters.get(source, 0)
            chunk.metadata["chunk_index"] = index
            source_counters[source] = index + 1

        print(
            f"[TextChunker] Produced {len(chunks)} chunks "
            f"from {len(documents)} page(s)"
        )
        return chunks