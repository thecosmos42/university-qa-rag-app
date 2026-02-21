#!/usr/bin/env python3
"""
VU Academic TER Chunker
=======================
A class-based, context-rich, hierarchical chunker for VU Teaching and
Examination Regulation (TER) PDFs.

Quick start
-----------
    from ter_chunker import PolicyChunker

    chunker = PolicyChunker(
        code_field_synonyms    = ["course code", "Module Nr.", "Code"],
        credits_field_synonyms = ["ECTS", "EC", "Credits", "nr of EC"],
        period_field_synonyms  = ["Period", "Semester", "Trimester", "Block"],
        track_keywords         = ["specialisatie", "track", "specialisation"],
    )

    # accepts: file paths (str/Path), dicts, LangChain/LlamaIndex Documents
    chunk_dicts = chunker.chunk_documents(documents)

    # or a single PDF:
    chunk_dicts = chunker.chunk_pdf("TER_ComputerScience.pdf")

    # serialise:
    import json
    json.dump(chunk_dicts, open("chunks.json", "w"), indent=2, ensure_ascii=False)

Synonym parameters
------------------
code_field_synonyms
    Column headers that identify the course-code column in curriculum tables.
    Different TERs use "course code", "Module Nr.", "Code", "Vakcode", …
    The chunker uses these to parse structured course rows and expose them
    as structured metadata on each course-table chunk.

credits_field_synonyms
    Column headers for ECTS credits. Common variants: "ECTS", "EC",
    "Credits", "nr of EC", "studiepunten".

period_field_synonyms
    Column headers for when a course is offered. Variants: "Period",
    "Semester", "Trimester", "Block", "Kwartaal".

track_keywords
    Words that signal a track/specialisation header *inside* a course table.
    Used to tag each parsed course row with its parent track. Variants:
    "specialisatie", "track", "specialisation", "profiel", "afstudeerrichting".

VU structural invariants (intentionally not configurable)
---------------------------------------------------------
  Sections 1-5  → label "A"   (Faculty-wide rules)
  Sections 6-8  → label "B1"  (Programme-specific general provisions)
  Sections 9-11 → label "B2"  (Programme-specific curriculum)

  These are mandated by the VU Executive Board OER template and are
  identical across every VU programme. Article 1.2 is always "Definitions".

Input formats accepted by chunk_documents()
-------------------------------------------
  • str / Path            - treated as a PDF file path
  • dict                  - must have a "source" or "path" key pointing to PDF,
                            or "page_content"/"text"/"content" with inline text
                            and optionally a "metadata" sub-dict
  • LangChain Document    - object with .page_content and .metadata
  • LlamaIndex Document   - object with .text and .metadata

Output format
-------------
  list[dict] — one dict per chunk, JSON-serialisable, with fields:

    chunk_id        str   deterministic MD5-based ID
    document_name   str   e.g. "TER MSc Artificial Intelligence 2025-2026"
    institution     str   "Vrije Universiteit Amsterdam"
    faculty         str   e.g. "Science"
    programme       str   e.g. "MSc Artificial Intelligence"
    degree_level    str   "MSc" | "BSc"
    croho_number    str   e.g. "66981"
    academic_year   str   e.g. "2025-2026"
    source_path     str   absolute path of the source PDF
    section_label   str   "A" | "B1" | "B2" | "Appendix I" | …
    section_number  str   "3"
    section_title   str   "Assessment and examination"
    article_number  str   "3.5"
    article_title   str   "Examination opportunities"
    chunk_type      str   "policy_text" | "course_table" | "appendix" | "definitions"
    track           str   track/specialisation name (course_table chunks only)
    courses         list  parsed course rows (course_table chunks only)
                          each row: {code, name, credits, level, period, track}
    page_start      int   first PDF page (1-indexed)
    page_end        int   last PDF page (1-indexed)
    char_count      int
    token_estimate  int   ≈ char_count // 4
    context_prefix  str   auto-generated context sentence (prepended before embed)
    text            str   ← EMBED THIS: context_prefix + "\\n\\n" + text_body
    text_body       str   ← raw article text; use for display / citation
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Union

try:
    import pdfplumber
except ImportError:
    raise ImportError(
        "pdfplumber is required: pip install pdfplumber"
    )


# ─────────────────────────────────────────────────────────────────────────────
# VU OER template structural constants  (invariant across all programmes)
# ─────────────────────────────────────────────────────────────────────────────

_VU_SECTION_LABEL: dict[str, str] = {
    "1": "A",  "2": "A",  "3": "A",  "4": "A",  "5": "A",
    "6": "B1", "7": "B1", "8": "B1",
    "9": "B2", "10": "B2","11": "B2",
}

_SECTION_LABEL_HUMAN: dict[str, str] = {
    "A":  "Section A — Faculty-wide rules",
    "B1": "Section B1 — Programme-specific general provisions",
    "B2": "Section B2 — Programme-specific curriculum content",
}

_CHUNK_TYPE_HUMAN: dict[str, str] = {
    "policy_text":  "policy article",
    "course_table": "curriculum / course table",
    "appendix":     "appendix",
    "definitions":  "definitions article",
}

_DEFINITIONS_ARTICLE      = "1.2"
_DEFAULT_TABLE_MAX_CHARS  = 3_000
_DEFAULT_COURSE_THRESHOLD = 4        # min course-code hits to classify as table

# Regex: VU course codes like XM_0052, X_400111, XMU_0037, P_MBRIMAG_AI
_RE_VU_COURSE_CODE = re.compile(r"\b[A-Z]+_[A-Z0-9_]{3,}\b")
_RE_ARTICLE        = re.compile(r"^Article\s+(\d{1,2}\.\d{1,2})\s+(.{3,100})$", re.M)
_RE_APPENDIX       = re.compile(r"^(Appendix\s+(?:I{1,4}V?|IV|VI?))\b", re.M | re.I)
_RE_PAGE_MARKER    = re.compile(r"<<<PAGE (\d+)>>>")

# Table-cell noise that must not be mistaken for section headings
_RE_SECTION_NOISE = re.compile(
    r"Advice OLC|Ordinance CvB|legal provision|approval FGV|"
    r"enter into force|Not applicable|is excluded|are eligible|"
    r"programme starts|may only take|CROHO|see Appendix",
    re.I,
)


# ─────────────────────────────────────────────────────────────────────────────
# Internal chunk dataclass  (converted to dict before returning to caller)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Chunk:
    chunk_id: str       = ""
    document_name: str  = ""
    institution: str    = "Vrije Universiteit Amsterdam"
    faculty: str        = ""
    programme: str      = ""
    degree_level: str   = ""
    croho_number: str   = ""
    academic_year: str  = ""
    source_path: str    = ""
    section_label: str  = ""
    section_number: str = ""
    section_title: str  = ""
    article_number: str = ""
    article_title: str  = ""
    chunk_type: str     = "policy_text"
    track: str          = ""          # populated for course_table chunks
    courses: list       = field(default_factory=list)  # parsed course rows
    page_start: int     = 0
    page_end: int       = 0
    chunk_index: int    = 0
    char_count: int     = 0
    token_estimate: int = 0
    context_prefix: str = ""
    text: str           = ""          # ← embed this
    text_body: str      = ""          # ← display / citation

    def finalise(self, prefix_fn) -> None:
        self.context_prefix = prefix_fn(self)
        self.text           = f"{self.context_prefix}\n\n{self.text_body}".strip()
        self.char_count     = len(self.text)
        self.token_estimate = max(1, self.char_count // 4)
        slug = re.sub(
            r"[^a-z0-9]", "_",
            self.article_number.lower() if self.article_number
            else self.section_label.lower(),
        )
        self.chunk_id = f"ter_{slug}_{hashlib.md5(self.text.encode()).hexdigest()[:10]}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["page"] = self.page_start   # alias expected by the DB layer
        return d


# ─────────────────────────────────────────────────────────────────────────────
# PolicyChunker
# ─────────────────────────────────────────────────────────────────────────────

class PolicyChunker:
    """
    Configurable, generalizable chunker for VU Teaching and Examination
    Regulation (TER) PDFs.

    Parameters
    ----------
    code_field_synonyms : list[str]
        Column headers that mean "course code" in curriculum tables.
        e.g. ["course code", "Module Nr.", "Code", "Vakcode"]

    credits_field_synonyms : list[str]
        Column headers that mean "ECTS credits".
        e.g. ["ECTS", "EC", "Credits", "nr of EC", "studiepunten"]

    period_field_synonyms : list[str]
        Column headers that mean "teaching period".
        e.g. ["Period", "Semester", "Trimester", "Block", "Kwartaal"]

    track_keywords : list[str]
        Words (case-insensitive) that signal a track/specialisation sub-header
        inside a course table.
        e.g. ["specialisatie", "track", "specialisation", "profiel"]

    table_max_chars : int
        Maximum characters before a course-table chunk is split. Default 3000.

    course_code_threshold : int
        Minimum number of VU course-code pattern matches for an article to be
        classified as a course_table rather than policy_text. Default 4.

    verbose : bool
        Print detection and landmark debug output. Default False.
    """

    def __init__(
        self,
        code_field_synonyms:    list[str] | None = None,
        credits_field_synonyms: list[str] | None = None,
        period_field_synonyms:  list[str] | None = None,
        track_keywords:         list[str] | None = None,
        table_max_chars:        int  = _DEFAULT_TABLE_MAX_CHARS,
        course_code_threshold:  int  = _DEFAULT_COURSE_THRESHOLD,
        verbose:                bool = False,
    ) -> None:
        # Synonym lists — merge user-supplied with sensible defaults
        self.code_field_synonyms    = self._merge(
            code_field_synonyms,
            ["course code", "Code", "Vakcode", "Module Nr.", "Course code"],
        )
        self.credits_field_synonyms = self._merge(
            credits_field_synonyms,
            ["ECTS", "EC", "Credits", "nr of EC", "studiepunten", "credits"],
        )
        self.period_field_synonyms  = self._merge(
            period_field_synonyms,
            ["Period", "Semester", "Trimester", "Block", "Kwartaal", "period"],
        )
        self.track_keywords         = self._merge(
            track_keywords,
            ["specialisatie", "track", "specialisation", "specialization",
             "profiel", "afstudeerrichting", "Main", "Specialisation"],
        )
        self.table_max_chars       = table_max_chars
        self.course_code_threshold = course_code_threshold
        self.verbose               = verbose

        # Build compiled patterns from synonyms
        self._re_track_header = self._build_track_re()

    # ── public API ────────────────────────────────────────────────────────────

    def chunk_documents(
        self,
        documents: list[Any],
        *,
        # per-document metadata overrides (applied to ALL documents in the batch)
        programme:    str = "",
        degree_level: str = "",
        faculty:      str = "",
        croho_number: str = "",
        academic_year: str = "",
    ) -> list[dict]:
        """
        Chunk a list of documents. Returns a flat list of chunk dicts.

        Each element of `documents` may be:
          • str or Path           — path to a PDF file
          • dict                  — {"source": "/path/to.pdf", ...}
                                    or {"path": "/path/to.pdf", ...}
          • LangChain Document    — .page_content + .metadata
          • LlamaIndex Document   — .text + .metadata

        The optional keyword arguments override auto-detected metadata for
        every document in the batch. Leave empty to rely on auto-detection
        (recommended for mixed batches of different programmes).
        """
        overrides = dict(
            programme=programme,
            degree_level=degree_level,
            faculty=faculty,
            croho_number=croho_number,
            academic_year=academic_year,
        )
        all_chunks: list[dict] = []
        for doc in documents:
            pdf_path, inline_text, extra_meta = self._resolve_document(doc)
            merged_overrides = {**overrides, **extra_meta}

            if pdf_path:
                chunks = self._chunk_pdf_path(pdf_path, **merged_overrides)
            elif inline_text:
                # Inline text path: wrap as synthetic full_text (no page markers)
                chunks = self._chunk_inline_text(
                    inline_text, source_path="", **merged_overrides
                )
            else:
                raise ValueError(
                    f"Cannot resolve document to a PDF path or text content: {doc!r}"
                )
            all_chunks.extend(chunks)
        return all_chunks

    def chunk_pdf(
        self,
        pdf_path: str | Path,
        *,
        programme:    str = "",
        degree_level: str = "",
        faculty:      str = "",
        croho_number: str = "",
        academic_year: str = "",
    ) -> list[dict]:
        """
        Chunk a single PDF file. Convenience wrapper around chunk_documents().

        Returns list[dict] with one dict per chunk.
        """
        return self._chunk_pdf_path(
            str(pdf_path),
            programme=programme,
            degree_level=degree_level,
            faculty=faculty,
            croho_number=croho_number,
            academic_year=academic_year,
        )

    def to_json(self, chunks: list[dict], indent: int = 2) -> str:
        """Serialise chunk dicts to a JSON string."""
        return json.dumps(chunks, indent=indent, ensure_ascii=False)

    def save_json(self, chunks: list[dict], path: str | Path, indent: int = 2) -> None:
        """Write chunk dicts to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(chunks, indent), encoding="utf-8")
        if self.verbose:
            print(f"[PolicyChunker] Saved {len(chunks)} chunks → {out}")

    def summary(self, chunks: list[dict]) -> dict:
        """
        Return a summary dict describing the chunk set.
        Useful for logging or sanity-checking after chunking.
        """
        by_type: dict[str, int] = {}
        by_sec:  dict[str, int] = {}
        total_tok = 0
        programmes: set[str] = set()
        for c in chunks:
            by_type[c["chunk_type"]]   = by_type.get(c["chunk_type"], 0) + 1
            by_sec[c["section_label"]] = by_sec.get(c["section_label"], 0) + 1
            total_tok += c["token_estimate"]
            if c["programme"]:
                programmes.add(c["programme"])
        return {
            "total_chunks":    len(chunks),
            "total_tokens_est":total_tok,
            "avg_tokens":      total_tok // len(chunks) if chunks else 0,
            "programmes":      sorted(programmes),
            "by_chunk_type":   by_type,
            "by_section":      dict(sorted(by_sec.items())),
        }

    # ── document resolution ───────────────────────────────────────────────────

    def _resolve_document(
        self, doc: Any
    ) -> tuple[str, str, dict]:
        """
        Normalise any supported document format into
        (pdf_path, inline_text, extra_metadata_overrides).

        Exactly one of pdf_path / inline_text will be non-empty.
        extra_metadata_overrides contains any metadata fields found in the
        document object that should override auto-detection.
        """
        extra: dict[str, str] = {}

        # ── str / Path  →  treat as PDF path ─────────────────────────────────
        if isinstance(doc, (str, Path)):
            return str(doc), "", extra

        # ── dict ──────────────────────────────────────────────────────────────
        if isinstance(doc, dict):
            # Source path
            pdf_path = doc.get("source") or doc.get("path") or doc.get("file_path", "")
            if pdf_path and Path(pdf_path).suffix.lower() == ".pdf":
                # Pull known metadata keys from the dict
                meta = doc.get("metadata", {})
                for key in ("programme", "degree_level", "faculty",
                            "croho_number", "academic_year"):
                    if meta.get(key):
                        extra[key] = meta[key]
                return str(pdf_path), "", extra

            # Inline text (e.g. pre-extracted)
            text = (doc.get("page_content")
                    or doc.get("text")
                    or doc.get("content", ""))
            meta = doc.get("metadata", {})
            for key in ("programme", "degree_level", "faculty",
                        "croho_number", "academic_year"):
                if meta.get(key):
                    extra[key] = meta[key]
            return "", str(text), extra

        # ── LangChain / LlamaIndex Document objects ───────────────────────────
        # LangChain: .page_content, .metadata
        if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            meta = doc.metadata or {}
            src  = meta.get("source", "") or meta.get("file_path", "")
            for key in ("programme", "degree_level", "faculty",
                        "croho_number", "academic_year"):
                if meta.get(key):
                    extra[key] = meta[key]
            if src and Path(src).suffix.lower() == ".pdf":
                return str(src), "", extra
            return "", str(doc.page_content), extra

        # LlamaIndex: .text, .metadata, optionally .doc_id
        if hasattr(doc, "text") and hasattr(doc, "metadata"):
            meta = doc.metadata or {}
            src  = meta.get("file_path", "") or meta.get("source", "")
            for key in ("programme", "degree_level", "faculty",
                        "croho_number", "academic_year"):
                if meta.get(key):
                    extra[key] = meta[key]
            if src and Path(src).suffix.lower() == ".pdf":
                return str(src), "", extra
            return "", str(doc.text), extra

        raise TypeError(
            f"Unsupported document type: {type(doc)}. "
            "Expected str, Path, dict, LangChain Document, or LlamaIndex Document."
        )

    # ── PDF chunking pipeline ─────────────────────────────────────────────────

    def _chunk_pdf_path(
        self,
        pdf_path: str,
        *,
        programme:    str = "",
        degree_level: str = "",
        faculty:      str = "",
        croho_number: str = "",
        academic_year: str = "",
    ) -> list[dict]:
        if self.verbose:
            print(f"[PolicyChunker] Reading: {pdf_path}")
        first_text, full_text = self._extract_full_text(pdf_path)
        return self._run_pipeline(
            full_text, first_text,
            source_path   = str(Path(pdf_path).resolve()),
            programme     = programme,
            degree_level  = degree_level,
            faculty       = faculty,
            croho_number  = croho_number,
            academic_year = academic_year,
        )

    def _chunk_inline_text(
        self,
        text: str,
        *,
        source_path:  str = "",
        programme:    str = "",
        degree_level: str = "",
        faculty:      str = "",
        croho_number: str = "",
        academic_year: str = "",
    ) -> list[dict]:
        return self._run_pipeline(
            text, text[:4000],  # use first 4000 chars as "first pages"
            source_path   = source_path,
            programme     = programme,
            degree_level  = degree_level,
            faculty       = faculty,
            croho_number  = croho_number,
            academic_year = academic_year,
        )

    def _run_pipeline(
        self,
        full_text:    str,
        first_text:   str,
        *,
        source_path:  str,
        programme:    str,
        degree_level: str,
        faculty:      str,
        croho_number: str,
        academic_year: str,
    ) -> list[dict]:
        # 1. Detect metadata
        detected       = self._detect_metadata(first_text, full_text)
        section_titles = self._detect_section_titles(full_text)

        meta = {
            "programme":    programme    or detected.get("programme",    ""),
            "degree_level": degree_level or detected.get("degree_level", ""),
            "faculty":      faculty      or detected.get("faculty",      ""),
            "croho_number": croho_number or detected.get("croho_number", ""),
            "academic_year":academic_year or detected.get("academic_year",""),
            "source_path":  source_path,
        }
        meta["document_name"] = (
            f"TER {meta['programme']} {meta['academic_year']}".strip()
        )

        if self.verbose:
            print("[auto-detect] Metadata:")
            for k, v in meta.items():
                print(f"  {k:<20}: {v!r}")
            print("[auto-detect] Section titles:")
            for k, v in sorted(section_titles.items(), key=lambda x: int(x[0])):
                print(f"  {k:>2}. {v}")

        # 2. Build landmarks
        landmarks = self._build_landmarks(full_text)

        if self.verbose:
            print(f"[landmarks] {len(landmarks)} total")
            for o, t, d in landmarks:
                print(f"  @{o:6d}  {t:10s}  {d}")

        # 3. Slice into chunks
        chunks: list[_Chunk] = []
        for idx, (offset, ltype, data) in enumerate(landmarks):
            next_off = (
                landmarks[idx + 1][0] if idx + 1 < len(landmarks) else len(full_text)
            )
            ps, pe = self._page_range(full_text, offset, next_off)
            body   = re.sub(r"<<<PAGE \d+>>>", "", full_text[offset:next_off]).strip()

            c = self._make_base_chunk(meta)
            c.page_start, c.page_end = ps, pe

            if ltype == "article":
                c.article_number = data["number"]
                c.article_title  = data["title"]
                lbl, snum, stitle = self._resolve_section(
                    data["number"], section_titles
                )
                c.section_label  = lbl
                c.section_number = snum
                c.section_title  = stitle
                hits = len(_RE_VU_COURSE_CODE.findall(body))
                c.chunk_type = (
                    "course_table" if hits >= self.course_code_threshold
                    else "definitions" if data["number"] == _DEFINITIONS_ARTICLE
                    else "policy_text"
                )
                if c.chunk_type == "course_table":
                    c.courses = self._parse_course_rows(body)
                    c.track   = self._detect_primary_track(body)

            else:  # appendix
                c.section_label  = data["label"]
                c.article_number = ""
                c.article_title  = ""
                c.chunk_type     = "appendix"

            c.text_body = body
            c.finalise(self._build_prefix)
            chunks.append(c)

        # 4. Split large course-table chunks
        chunks = self._split_large_tables(chunks)

        for i, c in enumerate(chunks):
            c.chunk_index = i
        return [c.to_dict() for c in chunks]

    # ── PDF text extraction ───────────────────────────────────────────────────

    @staticmethod
    def _extract_full_text(pdf_path: str) -> tuple[str, str]:
        """Returns (first_pages_text, full_text_with_page_markers)."""
        pages: list[dict] = []
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                pages.append({"page": i + 1, "text": page.extract_text() or ""})
        first_text = "\n".join(p["text"] for p in pages[:5])
        full_text  = "\n".join(
            f"\n<<<PAGE {p['page']}>>>\n{p['text']}" for p in pages
        )
        return first_text, full_text

    @staticmethod
    def _page_range(full_text: str, start: int, end: int) -> tuple[int, int]:
        markers = [
            (m.start(), int(m.group(1)))
            for m in _RE_PAGE_MARKER.finditer(full_text)
        ]
        ps = pe = 1
        for pos, pg in markers:
            if pos <= start: ps = pg
            if pos <= end:   pe = pg
        return ps, pe

    # ── Metadata auto-detection ───────────────────────────────────────────────

    @staticmethod
    def _detect_metadata(first_text: str, full_text: str) -> dict[str, str]:
        meta: dict[str, str] = {}

        # Programme name + degree level
        m = re.search(
            r"(Master['\u2019]s|Bachelor['\u2019]s)\s+programme\s+in\s+"
            r"([A-Z][A-Za-z &\-]{2,60})",
            first_text, re.I,
        )
        if m:
            meta["degree_level"] = "MSc" if "master" in m.group(1).lower() else "BSc"
            meta["programme"]    = f"{meta['degree_level']} {m.group(2).strip()}"
        else:
            meta["degree_level"] = ""
            meta["programme"]    = ""

        # Faculty
        m = re.search(r"Faculty\s+of\s+([A-Z][A-Za-z &]{2,40})", first_text, re.I)
        meta["faculty"] = m.group(1).strip() if m else ""

        # CROHO number
        m = re.search(r"CROHO[- ]?(?:number\s*)?(\d{5,6})", full_text, re.I)
        meta["croho_number"] = m.group(1) if m else ""

        # Academic year
        m = re.search(r"(20\d{2}[-\u2013]20\d{2})", first_text)
        meta["academic_year"] = m.group(1).replace("\u2013", "-") if m else ""

        return meta

    @staticmethod
    def _detect_section_titles(full_text: str) -> dict[str, str]:
        """Extract section titles 1-11 from TOC (first clean occurrence each)."""
        seen: dict[str, str] = {}
        for m in re.finditer(r"^(\d{1,2})\.\s+([A-Z][^\n]{3,100})$", full_text, re.M):
            num, title = m.group(1), m.group(2).strip()
            if num not in _VU_SECTION_LABEL:        continue
            if num in seen:                         continue
            if _RE_SECTION_NOISE.search(title):     continue
            if len(title) > 100:                    continue
            seen[num] = title
        return seen

    # ── Landmark index ────────────────────────────────────────────────────────

    @staticmethod
    def _build_landmarks(full_text: str) -> list[tuple[int, str, dict]]:
        landmarks: list[tuple[int, str, dict]] = []
        for m in _RE_ARTICLE.finditer(full_text):
            landmarks.append(
                (m.start(), "article",
                 {"number": m.group(1), "title": m.group(2).strip()})
            )
        seen_app: set[str] = set()
        for m in _RE_APPENDIX.finditer(full_text):
            key = m.group(1).strip().lower()
            if key not in seen_app:
                seen_app.add(key)
                landmarks.append((m.start(), "appendix", {"label": m.group(1).strip()}))
        landmarks.sort(key=lambda x: x[0])
        return landmarks

    # ── Section resolution ────────────────────────────────────────────────────

    @staticmethod
    def _resolve_section(
        art_num: str, section_titles: dict[str, str]
    ) -> tuple[str, str, str]:
        sec_num = art_num.split(".")[0]
        lbl     = _VU_SECTION_LABEL.get(sec_num, "")
        title   = section_titles.get(sec_num, "")
        return lbl, sec_num, title

    # ── Course row parser  (synonym-aware) ────────────────────────────────────

    def _parse_course_rows(self, body: str) -> list[dict]:
        """
        Parse structured course rows from a course-table article body.

        Uses self.code_field_synonyms / credits_field_synonyms /
        period_field_synonyms to locate column positions, then extracts one
        dict per course row:
            {code, name, credits, level, period, track}

        Falls back gracefully if table structure is non-standard.
        """
        rows: list[dict] = []
        current_track = ""

        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue

            # Track/specialisation header detection
            track_hit = self._re_track_header.search(line)
            if track_hit and not _RE_VU_COURSE_CODE.search(line):
                current_track = line.rstrip(":")
                continue

            # Course row detection: line must contain a VU course code
            code_match = _RE_VU_COURSE_CODE.search(line)
            if not code_match:
                continue

            code = code_match.group(0)

            # Credits: first integer ≤ 60 that appears after the code
            after_code = line[code_match.end():]
            credits = ""
            level   = ""
            period  = ""

            # Find all standalone integers in the tail of the line
            nums = re.findall(r"\b(\d{1,3})\b", after_code)
            if len(nums) >= 1:
                # First number that is plausible ECTS (1-60)
                for n in nums:
                    if 1 <= int(n) <= 60:
                        credits = n
                        break
            if len(nums) >= 2:
                # Second number after credits is typically level (100-900)
                for n in nums:
                    if 100 <= int(n) <= 900:
                        level = n
                        break

            # Course name: everything before the code
            name = line[: code_match.start()].strip().rstrip("1234567890").strip()

            rows.append({
                "code":    code,
                "name":    name,
                "credits": credits,
                "level":   level,
                "period":  period,      # period rarely appears in VU TER tables
                "track":   current_track,
            })

        return rows

    def _detect_primary_track(self, body: str) -> str:
        """Return the first track/specialisation header found in the body."""
        for line in body.splitlines():
            line = line.strip()
            if self._re_track_header.search(line) and not _RE_VU_COURSE_CODE.search(line):
                return line.rstrip(":")
        return ""

    # ── Context prefix builder ────────────────────────────────────────────────

    @staticmethod
    def _build_prefix(c: "_Chunk") -> str:
        """
        Generates a short, information-dense context sentence prepended to
        each chunk before embedding.  This is the contextual retrieval
        mechanism: even an isolated chunk knows its position in the document.

        Example (Article 3.5):
            [TER VU Amsterdam — MSc AI 2025-2026 | A, Article 3.5
             "Examination opportunities" | policy article]
            This policy article is part of the TER for the MSc AI (2025-2026)
            at Vrije Universiteit Amsterdam. It belongs to Section 3
            "Assessment and examination", Section A — Faculty-wide rules.
        """
        sec_human  = _SECTION_LABEL_HUMAN.get(c.section_label, c.section_label)
        type_human = _CHUNK_TYPE_HUMAN.get(c.chunk_type, c.chunk_type)
        art_part   = (
            f', Article {c.article_number} "{c.article_title}"'
            if c.article_number else ""
        )
        header = (
            f"[TER VU Amsterdam — {c.programme} {c.academic_year}"
            f" | {c.section_label}{art_part} | {type_human}]"
        )
        if c.chunk_type == "appendix":
            body = (
                f"This is {c.section_label} of the Teaching and Examination "
                f"Regulations (TER) for the {c.programme} ({c.academic_year}) "
                f"at {c.institution}."
            )
        else:
            parent = (
                f' It belongs to Section {c.section_number} "{c.section_title}"'
                if c.section_number else ""
            )
            body = (
                f"This {type_human} is part of the TER for the {c.programme} "
                f"({c.academic_year}) at {c.institution}.{parent}, {sec_human}."
            )
        return f"{header}\n{body}"

    # ── Large table splitter ──────────────────────────────────────────────────

    def _split_large_tables(self, chunks: list[_Chunk]) -> list[_Chunk]:
        result: list[_Chunk] = []
        for ch in chunks:
            if ch.chunk_type != "course_table" or len(ch.text_body) <= self.table_max_chars:
                result.append(ch)
                continue

            paras   = re.split(r"\n{2,}", ch.text_body)
            parts:  list[str] = []
            cur_len = 0
            sub_idx = 1

            def flush(ps: list[str], si: int, ch: _Chunk = ch) -> _Chunk:
                base = {k: v for k, v in ch.to_dict().items()
                        if k not in ("text", "context_prefix", "char_count",
                                     "token_estimate", "chunk_id", "text_body",
                                     "page")}
                s = _Chunk(**base)
                s.article_title = f"{ch.article_title} (part {si})"
                s.text_body     = "\n\n".join(ps)
                s.courses       = self._parse_course_rows(s.text_body)
                s.finalise(self._build_prefix)
                return s

            for para in paras:
                if cur_len + len(para) > self.table_max_chars and parts:
                    result.append(flush(parts, sub_idx))
                    sub_idx += 1
                    parts, cur_len = [para], len(para)
                else:
                    parts.append(para)
                    cur_len += len(para)
            if parts:
                result.append(flush(parts, sub_idx))
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_base_chunk(meta: dict) -> _Chunk:
        c = _Chunk()
        c.document_name  = meta["document_name"]
        c.programme      = meta["programme"]
        c.degree_level   = meta["degree_level"]
        c.faculty        = meta["faculty"]
        c.croho_number   = meta["croho_number"]
        c.academic_year  = meta["academic_year"]
        c.source_path    = meta["source_path"]
        return c

    def _build_track_re(self) -> re.Pattern:
        """Compile a pattern that matches any track/specialisation header line."""
        escaped = [re.escape(kw) for kw in self.track_keywords]
        return re.compile(r"(?:" + "|".join(escaped) + r")", re.I)

    @staticmethod
    def _merge(user: list[str] | None, defaults: list[str]) -> list[str]:
        """Merge user-supplied synonyms with defaults, preserving order, no dupes."""
        combined = list(user) if user else []
        seen = set(s.lower() for s in combined)
        for d in defaults:
            if d.lower() not in seen:
                combined.append(d)
                seen.add(d.lower())
        return combined


# ─────────────────────────────────────────────────────────────────────────────
# CLI  (python ter_chunker.py --pdf TER.pdf --output chunks.json)
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse, sys

    parser = argparse.ArgumentParser(
        description=(
            "VU Academic TER Chunker\n"
            "Chunks any VU TER PDF into context-rich, metadata-rich chunks.\n"
            "Metadata is auto-detected; use override flags only when needed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pdf",     required=True,
                        help="Path to a VU TER PDF (any programme, any year)")
    parser.add_argument("--output",  default="ter_chunks.json",
                        help="Output JSON path  (default: ter_chunks.json)")
    g = parser.add_argument_group("Synonym overrides")
    g.add_argument("--code-synonyms",    nargs="+", metavar="STR", default=[],
                   help='Extra synonyms for course-code column header')
    g.add_argument("--credits-synonyms", nargs="+", metavar="STR", default=[],
                   help='Extra synonyms for credits column header')
    g.add_argument("--period-synonyms",  nargs="+", metavar="STR", default=[],
                   help='Extra synonyms for period column header')
    g.add_argument("--track-keywords",   nargs="+", metavar="STR", default=[],
                   help='Extra keywords that signal a track header row')
    g2 = parser.add_argument_group("Metadata overrides (if auto-detection fails)")
    g2.add_argument("--programme",    default="")
    g2.add_argument("--croho",        default="")
    g2.add_argument("--year",         default="")
    g2.add_argument("--faculty",      default="")
    g2.add_argument("--degree-level", default="")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    chunker = PolicyChunker(
        code_field_synonyms    = args.code_synonyms    or None,
        credits_field_synonyms = args.credits_synonyms or None,
        period_field_synonyms  = args.period_synonyms  or None,
        track_keywords         = args.track_keywords   or None,
        verbose                = args.verbose,
    )

    chunks = chunker.chunk_pdf(
        args.pdf,
        programme    = args.programme,
        degree_level = getattr(args, "degree_level", ""),
        faculty      = args.faculty,
        croho_number = args.croho,
        academic_year= args.year,
    )

    s = chunker.summary(chunks)
    print(f"\n{'='*65}")
    print(f"  VU TER Chunker — {s['total_chunks']} chunks  |  "
          f"~{s['total_tokens_est']:,} tokens  (avg {s['avg_tokens']})")
    print(f"{'='*65}")
    for prog in s["programmes"]:
        print(f"  Programme    : {prog}")
    print(f"  Chunk types  : {s['by_chunk_type']}")
    print(f"  By section   : {s['by_section']}")

    # Show a sample course_table chunk with parsed courses
    sample_table = next((c for c in chunks if c["chunk_type"] == "course_table"), None)
    if sample_table:
        print(f"\n  ── Sample course_table chunk: {sample_table['article_number']} ──")
        print(f"     Article: {sample_table['article_title']}")
        print(f"     Track  : {sample_table['track'] or '(none)'}")
        print(f"     Parsed courses ({len(sample_table['courses'])}):")
        for row in sample_table["courses"][:5]:
            print(f"       {row['code']:<20} {row['credits']:>4} EC  lvl:{row['level']}"
                  f"  track:{row['track'][:30] if row['track'] else ''}")

    chunker.save_json(chunks, args.output)
    print(f"\n  Written → {args.output}  ({Path(args.output).stat().st_size:,} bytes)")


if __name__ == "__main__":
    _cli()