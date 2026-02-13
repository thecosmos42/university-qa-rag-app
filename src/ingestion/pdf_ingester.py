# src/ingestion/pdf_ingester.py
"""
Read PDF files from disk using PyMuPDFLoader and return them as LangChain
Document objects. 
"""

from pathlib import Path
from typing import List
# import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from src.config.config import PDF_INPUT_DIR


class PDFIngester:
    """
    Scans a directory for PDF files and loads them via LangChain loaders
    """

    _LOADER_CLASS = PyMuPDFLoader

    def __init__(self, input_dir: Path = PDF_INPUT_DIR):
        """
        Args:
            input_dir: Path(defined in config.py) to the directory containing PDF files.
        """
        self.input_dir = Path(input_dir)

    def ingest_all(self) -> List[Document]:
        """
        Load every PDF in input_dir and return a flat list of LangChain
        Document objects

        Each Document has:
            .page_content: extracted text for that page
            .metadata    : a dict that always contains at least:
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
                f"PDF Ingester could not find input directory: {self.input_dir}"
            )

        pdf_paths = sorted(self.input_dir.glob("*.pdf"))

        if not pdf_paths:
            raise ValueError(
                f"PDF Ingester could not find PDF files in: {self.input_dir}"
            )

        all_documents: List[Document] = []

        for pdf_path in pdf_paths:
            docs = self.load_single_pdf(pdf_path)
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


    def load_single_pdf(self, pdf_path: Path) -> List[Document]:
        """
        Load a PDF via LangChain

        Args:
            pdf_path: Path to a single .pdf file.

        Returns:
            A list of Document objects, one per page, with tables structured.
        """
        loader = self._LOADER_CLASS(str(pdf_path))
        documents: List[Document] = loader.load()

        # explicitly add metadata["page"] on every Document.
        for i, doc in enumerate(documents):
            doc.metadata["page"] = i+1
            doc.metadata["source"] = Path(pdf_path).name

        # table extraction pass (only if using pymupdf-backed loader)
        # if self._LOADER_CLASS is PyMuPDFLoader:
        #     documents = self._enrich_with_tables(pdf_path, documents)

        return documents

    # @staticmethod
    # def _enrich_with_tables(
    #     pdf_path: Path,
    #     documents: List[Document],
    # ) -> List[Document]:
    #     """
    #     Use pymupdf's table finder to detect tables on each page, convert
    #     them to structured text, and append that to the page's content.

    #     The structured text is APPENDED (not spliced in place) because
    #     reliably locating and removing the flattened table text from the
    #     base extraction is fragile.  Appending is safe: the chunker will
    #     produce separate chunks for the structured version, and the
    #     embedding of the structured rows will be far more semantically
    #     useful than the flattened original.

    #     Table format (one row per line, header repeated):
    #         | Header1 | Header2 | Header3 |
    #         | val1    | val2    | val3    |
    #         ...

    #     Args:
    #         pdf_path:  Path to the PDF (opened again with fitz directly).
    #         documents: The LangChain Documents from the base loader.

    #     Returns:
    #         The same list of Documents, with structured table text appended
    #         to any page that contained a detected table.
    #     """
    #     doc = fitz.open(str(pdf_path))
    #     table_count = 0

    #     for page_doc in documents:
    #         page_num = page_doc.metadata.get("page", 0)

    #         # fitz page indices match LangChain's 0-based page numbers
    #         if page_num >= len(doc):
    #             continue

    #         page = doc[page_num]
    #         tables = page.find_tables()

    #         if not tables:
    #             continue

    #         structured_parts: List[str] = []
    #         for table in tables:
    #             rows = table.to_pandas().values.tolist()
    #             headers = table.to_pandas().columns.tolist()

    #             if not rows:
    #                 continue

    #             # Build a pipe-delimited markdown-style table
    #             header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    #             separator   = "| " + " | ".join("-" * max(len(str(h)), 3) for h in headers) + " |"
    #             row_lines   = [
    #                 "| " + " | ".join(str(cell) for cell in row) + " |"
    #                 for row in rows
    #             ]

    #             table_text = "\n".join([header_line, separator] + row_lines)
    #             structured_parts.append(table_text)
    #             table_count += 1

    #         if structured_parts:
    #             # Append with a clear separator so the chunker can split here
    #             page_doc.page_content += (
    #                 "\n\n[Structured Tables]\n\n"
    #                 + "\n\n".join(structured_parts)
    #             )

    #     doc.close()

    #     if table_count:
    #         print(f"[PDFIngester] Extracted {table_count} table(s) from '{pdf_path.name}'")

    #     return documents