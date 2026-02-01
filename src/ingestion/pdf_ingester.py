# src/ingestion/pdf_ingester.py
"""
PDF Ingestion Module

Responsibility: Read PDF files from disk and return them as LangChain
Document objects.  Everything downstream consumes that standard type, so
swapping loaders here never touches the rest of the pipeline.

Table handling:
    PyMuPDFLoader extracts text in reading order, which flattens tables
    into unsstructured word runs.  This module uses pymupdf's native table
    detection API to find tables, converts them to a structured text
    format (one row per line, headers repeated on each row), and splices
    that back into the page text.  The chunker and embedder then see
    meaningful sentence-like strings instead of word salad.

Loader choice:
    PyMuPDFLoader  (primary)  — backed by pymupdf (fitz).  Produces the
                                cleanest text extraction for most PDFs, with
                                good handling of multi-column layouts.
    PyPDFLoader    (fallback) — backed by PyPDF2.  Useful if a specific PDF
                                fails with pymupdf.  To switch, change the
                                _LOADER_CLASS attribute on the class.
                                Note: PyPDFLoader does NOT support table
                                extraction, so tables will remain flattened
                                if you switch to it.
"""

from pathlib import Path
from typing import List

import fitz  # pymupdf — used directly for table extraction
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_core.documents import Document

from src.config.config import PDF_INPUT_DIR


class PDFIngester:
    """Scans a directory for PDF files and loads them via LangChain loaders,
    with an additional table-extraction pass using pymupdf directly.

    Attributes:
        _LOADER_CLASS: The LangChain loader class used for base text
                       extraction.  Swap to PyPDFLoader if pymupdf causes
                       issues with a particular file (tables will lose
                       structure in that case).
    """

    # Change this single line to switch between loaders project-wide.
    _LOADER_CLASS = PyMuPDFLoader

    def __init__(self, input_dir: Path = PDF_INPUT_DIR):
        """
        Args:
            input_dir: Path to the directory containing PDF files.
                       Defaults to the path defined in config.py.
        """
        self.input_dir = Path(input_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_all(self) -> List[Document]:
        """
        Load every PDF in input_dir and return a flat list of LangChain
        Document objects with tables properly structured.

        Each Document has:
            .page_content  — extracted text for that page, with any tables
                             converted to structured row-per-line format.
            .metadata      — a dict that always contains at least:
                                 "source"      : full file path (str)
                                 "page"        : 0-based page number (int)

        Returns:
            List[Document] — one Document per page across all PDFs.

        Raises:
            FileNotFoundError: If input_dir does not exist.
            ValueError:        If no PDF files are found in input_dir.
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(
                f"PDF input directory not found: {self.input_dir}"
            )

        pdf_paths = sorted(self.input_dir.glob("*.pdf"))

        if not pdf_paths:
            raise ValueError(
                f"No PDF files found in: {self.input_dir}"
            )

        all_documents: List[Document] = []

        for pdf_path in pdf_paths:
            docs = self._load_single_pdf(pdf_path)
            all_documents.extend(docs)
            print(
                f"[PDFIngester] Loaded '{pdf_path.name}' "
                f"— {len(docs)} page(s)"
            )

        print(
            f"[PDFIngester] Total: {len(all_documents)} pages "
            f"from {len(pdf_paths)} file(s)"
        )
        return all_documents

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_single_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load a PDF via LangChain, then enrich each page's text with
        structured table representations extracted by pymupdf directly.

        Args:
            pdf_path: Path to a single .pdf file.

        Returns:
            A list of Document objects, one per page, with tables structured.
        """
        # Step 1 — base text extraction via LangChain loader
        loader = self._LOADER_CLASS(str(pdf_path))
        documents: List[Document] = loader.load()

        # Step 2 — explicitly stamp metadata["page"] on every Document.
        #
        # WHY THIS IS NECESSARY:  PyMuPDFLoader does not reliably set
        # metadata["page"].  Depending on the langchain-community version
        # it may use "page_label" (a string), or omit the key entirely.
        # Every .get("page", 0) downstream was hitting the default of 0
        # on every chunk, making all sources display as "page 1".
        #
        # The loader always returns Documents in page order (0, 1, 2, …),
        # so enumerate gives us the correct 0-based index.  We stamp it
        # here — the single earliest point — so everything downstream
        # (chunker, ingestion_pipeline, ChromaDB, display) just works.
        for i, doc in enumerate(documents):
            doc.metadata["page"] = i

        # Step 3 — table extraction pass (only if using pymupdf-backed loader)
        if self._LOADER_CLASS is PyMuPDFLoader:
            documents = self._enrich_with_tables(pdf_path, documents)

        return documents

    @staticmethod
    def _enrich_with_tables(
        pdf_path: Path,
        documents: List[Document],
    ) -> List[Document]:
        """
        Use pymupdf's table finder to detect tables on each page, convert
        them to structured text, and append that to the page's content.

        The structured text is APPENDED (not spliced in place) because
        reliably locating and removing the flattened table text from the
        base extraction is fragile.  Appending is safe: the chunker will
        produce separate chunks for the structured version, and the
        embedding of the structured rows will be far more semantically
        useful than the flattened original.

        Table format (one row per line, header repeated):
            | Header1 | Header2 | Header3 |
            | val1    | val2    | val3    |
            ...

        Args:
            pdf_path:  Path to the PDF (opened again with fitz directly).
            documents: The LangChain Documents from the base loader.

        Returns:
            The same list of Documents, with structured table text appended
            to any page that contained a detected table.
        """
        doc = fitz.open(str(pdf_path))
        table_count = 0

        for page_doc in documents:
            page_num = page_doc.metadata.get("page", 0)

            # fitz page indices match LangChain's 0-based page numbers
            if page_num >= len(doc):
                continue

            page = doc[page_num]
            tables = page.find_tables()

            if not tables:
                continue

            structured_parts: List[str] = []
            for table in tables:
                rows = table.to_pandas().values.tolist()
                headers = table.to_pandas().columns.tolist()

                if not rows:
                    continue

                # Build a pipe-delimited markdown-style table
                header_line = "| " + " | ".join(str(h) for h in headers) + " |"
                separator   = "| " + " | ".join("-" * max(len(str(h)), 3) for h in headers) + " |"
                row_lines   = [
                    "| " + " | ".join(str(cell) for cell in row) + " |"
                    for row in rows
                ]

                table_text = "\n".join([header_line, separator] + row_lines)
                structured_parts.append(table_text)
                table_count += 1

            if structured_parts:
                # Append with a clear separator so the chunker can split here
                page_doc.page_content += (
                    "\n\n[Structured Tables]\n\n"
                    + "\n\n".join(structured_parts)
                )

        doc.close()

        if table_count:
            print(f"[PDFIngester] Extracted {table_count} table(s) from '{pdf_path.name}'")

        return documents