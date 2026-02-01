# src/debug_tables.py
"""
Table Extraction Debug Viewer

Runs the PDF ingester and dumps every page's full content to the console.
This is the exact state of each Document AFTER table enrichment has run
and BEFORE the chunker splits anything — so you see precisely what the
chunker and embedder will receive.

Each page is boxed so you can instantly spot:
    • Where the original extracted text ends
    • Whether a [Structured Tables] block was appended
    • The exact markdown format of each detected table

Usage:
    python -m debug_tables

    # Or, to inspect only a single PDF file, pass its path:
    python -m debug_tables path/to/specific.pdf
"""

import sys
from pathlib import Path

from src.ingestion.pdf_ingester import PDFIngester


def dump_documents(documents) -> None:
    """Print every Document with clear visual boundaries."""

    for doc in documents:
        source = Path(doc.metadata.get("source", "unknown")).name
        page   = doc.metadata.get("page", "?")

        # ── top border ─────────────────────────────────────────────
        print("┌" + "─" * 70 + "┐")
        print(f"│  {source}  —  page {page + 1} (0-based index: {page})".ljust(71) + "│")
        print("├" + "─" * 70 + "┤")

        # ── content, with each line indented and capped at terminal width ──
        for line in doc.page_content.split("\n"):
            # Wrap long lines so nothing scrolls off-screen
            while len(line) > 66:
                print(f"│  {line[:66]}  │")
                line = line[66:]
            print(f"│  {line:<66}  │")

        # ── bottom border ──────────────────────────────────────────
        print("└" + "─" * 70 + "┘")
        print()


def main() -> None:
    if len(sys.argv) > 1:
        # Single-file mode: wrap the path in a temporary one-file directory
        # by passing it directly to _load_single_pdf via a thin wrapper.
        target = Path(sys.argv[1])
        if not target.is_file():
            print(f"Error: '{target}' is not a file.")
            sys.exit(1)

        ingester = PDFIngester()
        documents = ingester._load_single_pdf(target)
        print(f"\n[debug_tables] Inspecting single file: {target.name}\n")
    else:
        # Default: run the full ingester on the configured PDF directory
        ingester = PDFIngester()
        documents = ingester.ingest_all()
        print()  # blank line after ingester's own prints

    dump_documents(documents)


if __name__ == "__main__":
    main()