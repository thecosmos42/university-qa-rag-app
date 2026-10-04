"""Check the benchmark labels against the source PDFs.

    python -m eval.validate_dataset

For every answerable item, each gold snippet must occur on one of its gold pages in a
gold PDF. For unanswerable items, none of `absent_terms` may occur anywhere in the
programme's PDFs (so abstention really is the correct behaviour).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

from eval.metrics import norm
from eval.run_benchmark import DEFAULT_DATASET, load_dataset

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdfs"
_cache: dict[str, list[str]] = {}


def pages(filename: str) -> list[str]:
    if filename not in _cache:
        with pymupdf.open(PDF_DIR / filename) as doc:
            _cache[filename] = [norm(p.get_text()) for p in doc]
    return _cache[filename]


def validate(items: list[dict]) -> list[str]:
    errors: list[str] = []
    for it in items:
        iid = it["id"]
        if it.get("answerable", True):
            if not (it.get("gold_files") and it.get("gold_pages") and it.get("gold_snippets")):
                errors.append(f"{iid}: answerable item needs gold_files, gold_pages and gold_snippets")
                continue
            for snippet in it["gold_snippets"]:
                found = False
                for f in it["gold_files"]:
                    pg = pages(f)
                    for p in it["gold_pages"]:
                        if 1 <= p <= len(pg) and norm(snippet) in pg[p - 1]:
                            found = True
                if not found:
                    errors.append(f"{iid}: snippet not found on gold pages: {snippet!r}")
        else:
            prog = it["question"]
            for f in sorted(PDF_DIR.glob("*.pdf")):
                # Check every PDF of the programmes the question is about (by name in filename).
                stem = f.stem.split("_", 2)[-1].lower()
                if stem in it.get("programmes", [stem]):
                    for term in it.get("absent_terms", []):
                        if any(norm(term) in pg for pg in pages(f.name)):
                            errors.append(f"{iid}: absent term {term!r} actually appears in {f.name}")
    return errors


def main() -> int:
    items = load_dataset(DEFAULT_DATASET)
    errors = validate(items)
    for e in errors:
        print("ERROR", e)
    n_ans = sum(1 for i in items if i.get("answerable", True))
    print(f"{len(items)} items ({n_ans} answerable, {len(items) - n_ans} unanswerable): "
          f"{'OK' if not errors else f'{len(errors)} error(s)'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
