# rag_system/pdf_ingester.py
"""
PDF Ingestion Module

Responsibility: Read PDF files from disk and return them as LangChain
Document objects.  Everything downstream consumes that standard type, so
swapping loaders here never touches the rest of the pipeline.

Loader choice:
    PyMuPDFLoader  (primary)  — backed by pymupdf (fitz).  Produces the
                                cleanest text extraction for most PDFs, with
                                good handling of multi-column layouts.
    PyPDFLoader    (fallback) — backed by PyPDF2.  Useful if a specific PDF
                                fails with pymupdf.  To switch, change the
                                _LOADER_CLASS attribute on the class.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_core.documents import Document

from src.config.config import PDF_INPUT_DIR


class PDFIngester:
    """Scans a directory for PDF files and loads them via LangChain loaders.

    Attributes:
        _LOADER_CLASS: The LangChain loader class used for extraction.
                       Swap to PyPDFLoader if pymupdf causes issues with
                       a particular file.
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



    def ingest_all(self) -> List[Document]:
        """
        Load every PDF in input_dir and return a flat list of LangChain
        Document objects.

        Each Document has:
            .page_content  — the extracted text for that page.
            .metadata      — a dict that always contains at least:
                                 "source"      : full file path (str)
                                 "page"        : 0-based page number (int)

        The list is ordered: all pages of the first PDF in filename order,
        then all pages of the second PDF, and so on.

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


    def _load_single_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Use the configured LangChain loader to extract all pages from one PDF.

        Args:
            pdf_path: Absolute or relative path to a single .pdf file.

        Returns:
            A list of Document objects, one per page.
        """
        loader = self._LOADER_CLASS(str(pdf_path))
        return loader.load()