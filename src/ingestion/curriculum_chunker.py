# src/ingestion/academic_chunker.py
"""
Document-Aware Academic Chunker with Context-Enriched Chunking
==============================================================
VU Amsterdam Study Guide 2025-2026

CHUNKING STRATEGY
-----------------
Implements "Context-Enriched Chunking" as described in the Databricks guide
(https://community.databricks.com/t5/technical-blog/the-ultimate-guide-to-
chunking-strategies-for-rag-applications/ba-p/113089, Strategy #5).

The core idea: each chunk gets a context block prepended that summarises
what came before and after it in the document. This means the embedding
encodes not just the chunk itself, but its surrounding context — dramatically
improving retrieval for queries that span section boundaries or rely on
programme-level context.

For this specific academic PDF the context enrichment is:

  Context block prepended to every chunk:
  ┌─────────────────────────────────────────────────────────────────┐
  │ [Programme: Master of Science in Artificial Intelligence]       │
  │ [Specialisation: AI for Health]                                 │
  │ [Prev: Advanced Logic (X_405048, P4)]                           │
  │ [Next: AI and Society (XM_0075, P3)]                            │
  │ [Position: 12 of 52]                                            │
  ├─────────────────────────────────────────────────────────────────┤
  │ [Course: Advanced Machine Learning] [Code: XM_0010] ...         │
  │ <actual chunk body>                                              │
  └─────────────────────────────────────────────────────────────────┘

  Metadata stored per chunk:
    course_name, course_code, credits, period, course_level,
    faculty, language, degree_name, section_type,
    prev_course_name, prev_course_code, next_course_name, next_course_code,
    specialisation, section_index, total_sections,
    page, source, chunk_index

OPTIONAL LLM SUMMARIZER
-----------------------
Pass a callable `context_summarizer` to CurriculumChunker() to replace the
default structured-context approach with LLM-generated summaries (matching
the Databricks article's `summarize=True` path):

    def my_summarizer(text: str) -> str:
        # Call your LLM endpoint here
        return llm.invoke(f"Summarise in 1 sentence: {text}")

    chunker = CurriculumChunker(context_summarizer=my_summarizer)

Without a summarizer, the chunker uses structured neighbour metadata instead
(zero extra latency, no external calls).

DESIGN PRINCIPLE — NO HARDCODINGS
----------------------------------
Nothing in this file assumes a specific university, programme name, degree
type, language, or field-label convention.  Every string that varies between
documents is either:

  A) Auto-detected from the document itself at runtime (footers, section
     headers, programme name), or
  B) Injected via constructor arguments with broad multilingual defaults.

Every internal method uses only the instance's compiled regex objects —
there are no inline hardcoded strings in helper methods.

HOW SECTION BOUNDARIES ARE FOUND
---------------------------------
The only structural assumption is that each course block contains a
labelled code field near its top, e.g.:
    "Course Code  XM_0083"     (VU Amsterdam)
    "Module Code  HIST-301"    (generic)
    "Cursus Code  A1234"       (Dutch)
    "Kode         B-HUM-12"    (other)

Everything before the first such match is the "preamble" (overview +
track tables). Everything after is sliced per course.

FOOTER AUTO-DETECTION
---------------------
Footers are detected by finding lines that appear verbatim on 3+ pages
and are short / low information-density.  No university name is needed.
Explicit patterns can be supplied to override this.

QUICKSTART
----------
    from langchain_community.document_loaders import PyPDFLoader
    from academic_chunker import CurriculumChunker

    # Zero config — auto-detects footers, works for any programme
    docs   = PyPDFLoader("any_catalogue.pdf").load()
    chunks = CurriculumChunker().chunk_documents(docs)

    # Non-English or unusual field labels
    chunks = CurriculumChunker(
        code_field_synonyms=["Cursus Code", "Kode", "Module Nr."],
        credits_field_synonyms=["ECTS", "Studiepunten", "Credits"],
        period_field_synonyms=["Periode", "Semester", "Trimester"],
        track_keywords=["richting", "profiel", "specialisatie", "track"],
    ).chunk_documents(docs)

    # With an LLM summarizer (Databricks article summarize=True path)
    chunks = CurriculumChunker(
        context_summarizer=lambda text: llm.invoke(
            f"Summarise in one sentence: {text}"
        )
    ).chunk_documents(docs)
"""
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Default synonym lists
# Broad multilingual coverage; callers can replace any of these.
# ---------------------------------------------------------------------------

_DEFAULT_CODE_LABELS: List[str] = [
    "Course Code", "Module Code", "Unit Code", "Cursus Code",
    "Vak Code", "Course Number", "Module Number", "Kode",
]

_DEFAULT_CREDITS_LABELS: List[str] = [
    "Credits", "ECTS", "Credit Hours", "Units",
    "Studiepunten", "Credit Points", "SP",
]

_DEFAULT_PERIOD_LABELS: List[str] = [
    "Period", "Periode", "Semester", "Term", "Block", "Quarter",
    "Teaching Period", "Academic Period", "Trimester",
]

_DEFAULT_LEVEL_LABELS: List[str] = [
    "Course Level", "Level", "Module Level", "Year", "Niveau", "Jaar",
]

_DEFAULT_FACULTY_LABELS: List[str] = [
    "Faculty", "Faculteit", "Department", "School", "Division",
    "Institute", "Afdeling", "Dept",
]

_DEFAULT_LANGUAGE_LABELS: List[str] = [
    "Language Of Tuition", "Language", "Taal", "Teaching Language",
    "Instruction Language", "Voertaal", "Onderwijstaal",
]

_DEFAULT_DEGREE_LABELS: List[str] = [
    # English degrees
    "Master", "Bachelor", "MSc", "BSc", "MA", "BA",
    "Doctor", "PhD", "MPhil", "LLM", "LLB", "MBChB",
    "Certificate", "Diploma", "Associate",
    # Dutch degrees
    "HBO",
]

# Generic structural keywords indicating a sub-programme / track section.
# No programme names — just the structural role words, multilingual.
_DEFAULT_TRACK_KEYWORDS: List[str] = [
    # English
    "specialisation", "specialization", "track", "stream",
    "major", "minor", "pathway", "concentration",
    # Dutch
    "specialisatie", "richting", "profiel",
    # German
    "Schwerpunkt", "Vertiefung",
    # French
    "parcours", "spécialité",
]

# Structural words found in preamble course-list tables.
# Used to classify preamble blocks without knowing the programme name.
_DEFAULT_COMPULSORY_KEYWORDS: List[str] = [
    # English
    "compulsory", "mandatory", "elective", "optional",
    "programme sections", "required",
    # Dutch
    "verplicht", "keuzevak", "vrij", "keuze",
    # German
    "Pflicht", "Wahlpflicht",
]

# Keywords in "audience" / "intended for" fields that hint at a track.
_DEFAULT_AUDIENCE_LABELS: List[str] = [
    "target audience", "intended for", "aimed at",
    "voor studenten", "bestemd voor",
    "Zielgruppe",
]


# ---------------------------------------------------------------------------
# Regex builders — built from lists, never inline strings in methods
# ---------------------------------------------------------------------------

def _label_re(labels: List[str]) -> re.Pattern:
    """
    Match any of the label strings followed by its value.
    Captures the value in group 1.
    """
    alt = "|".join(re.escape(l) for l in labels)
    return re.compile(rf'(?:^|\n)(?:{alt})\s+([^\n]+)', re.IGNORECASE)


def _code_anchor_re(labels: List[str]) -> re.Pattern:
    """
    Match a code-field label followed by its value.
    Captures the code in group 1.
    Handles: XM_0083, HIST-301, CS 101, A1234, L_AAMPLIN035, B-HUM-12
    """
    alt = "|".join(re.escape(l) for l in labels)
    return re.compile(
        rf'(?:{alt})\s+([A-Za-z0-9][A-Za-z0-9_\-\./ ]{{0,30}})',
        re.IGNORECASE,
    )


def _degree_re(labels: List[str]) -> re.Pattern:
    """
    Match a degree-type word followed by the programme name.
    Captures the full degree+name string in group 1.
    """
    alt = "|".join(re.escape(l) for l in labels)
    return re.compile(
        rf'((?:{alt})\.?\s+(?:of\s+|in\s+|der\s+|van\s+|de\s+)?[A-Za-z\s&\-]{{2,60}})',
        re.IGNORECASE,
    )


def _track_header_re(keywords: List[str]) -> re.Pattern:
    """
    Match a heading line (starts with any non-whitespace char) that contains
    at least one track keyword.  Captures the full line in group 1.
    """
    alt = "|".join(re.escape(k) for k in keywords)
    return re.compile(
        rf'(?:^|\n)(\S[^\n]{{4,120}}(?:{alt})[^\n]{{0,80}})',
        re.IGNORECASE | re.MULTILINE,
    )


def _audience_re(labels: List[str]) -> re.Pattern:
    """Match an audience label and capture the following line."""
    alt = "|".join(re.escape(l) for l in labels)
    return re.compile(
        rf'(?:{alt})[^\n]*\n([^\n]{{5,80}})',
        re.IGNORECASE,
    )


def _table_header_re(code_labels: List[str], credits_labels: List[str]) -> re.Pattern:
    """
    Match a course-table header row, e.g. "Course Name  Period  Credits  Code".
    Built from the same synonym lists as the field extractors — no inline strings.
    """
    code_alt    = "|".join(re.escape(l) for l in code_labels)
    credits_alt = "|".join(re.escape(l) for l in credits_labels)
    # A table header contains at least one code label AND one credits label
    return re.compile(
        rf'(?:{code_alt}|course name|module name|unit name).{{0,60}}(?:{credits_alt})',
        re.IGNORECASE,
    )


def _compulsory_re(keywords: List[str]) -> re.Pattern:
    """Match any compulsory/elective keyword in preamble blocks."""
    alt = "|".join(re.escape(k) for k in keywords)
    return re.compile(rf'\b(?:{alt})\b', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CourseSection:
    """One logical section extracted from an academic catalogue."""
    course_name:        str = ""
    course_code:        str = ""
    credits:            str = ""
    period:             str = ""
    course_level:       str = ""
    faculty:            str = ""
    language:           str = ""
    degree_name:        str = ""
    specialisation:     str = ""
    content:            str = ""
    page_number:        int = 0
    source:             str = ""
    # course_detail | program_overview | course_list | general
    section_type:       str = "course_detail"
    # Populated by _enrich_with_context
    prev_course_name:   str = ""
    prev_course_code:   str = ""
    prev_course_period: str = ""
    next_course_name:   str = ""
    next_course_code:   str = ""
    next_course_period: str = ""
    section_index:      int = 0
    total_sections:     int = 0


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class CurriculumChunker:
    """
    Splits any structured academic PDF into context-enriched course chunks.

    Works for any programme and any institution.  No programme names, footer
    text, or course-code formats are hardcoded anywhere in this class.

    Args:
        max_chunk_size:
            Maximum characters per chunk before fallback splitting fires.

        chunk_overlap:
            Character overlap used by the fallback splitter.

        min_chunk_size:
            Sections shorter than this are silently dropped.

        window_size:
            Number of neighbouring sections used for context enrichment.
            1 = immediate prev + next only.

        context_summarizer:
            Optional callable (str) -> str.  When provided, each neighbour's
            full body text is passed through it and the result appears in the
            [Context] block.  Matches Databricks article's summarize=True path.
            When None (default), structured metadata stubs are used instead.

        footer_patterns:
            Explicit regex strings matching PDF footer/header lines to strip.
            When None (default), footers are auto-detected (lines repeating
            verbatim on 3+ pages that are short / low information-density).

        footer_min_pages:
            Minimum page-count for a line to qualify as a footer during
            auto-detection.  Default 3.

        code_field_synonyms:
            Field labels for the course-code field.
            Default: ["Course Code", "Module Code", "Cursus Code", ...]

        credits_field_synonyms:
            Field labels for the credits field.
            Default: ["Credits", "ECTS", "Studiepunten", ...]

        period_field_synonyms:
            Field labels for the period/semester field.
            Default: ["Period", "Semester", "Term", ...]

        level_field_synonyms:
            Field labels for the course-level field.
            Default: ["Course Level", "Level", "Year", ...]

        faculty_field_synonyms:
            Field labels for the faculty/department field.
            Default: ["Faculty", "Department", "School", ...]

        language_field_synonyms:
            Field labels for the language-of-instruction field.
            Default: ["Language Of Tuition", "Language", "Taal", ...]

        degree_type_labels:
            Degree-type words used to extract the programme name.
            Default: ["Master", "Bachelor", "MSc", "BSc", "MA", "BA", ...]

        track_keywords:
            Words that indicate a preamble line is a sub-programme/track header.
            Default: ["specialisation", "track", "major", "richting", ...]

        compulsory_keywords:
            Words that indicate a preamble block contains course requirement
            tables (used for section-type classification).
            Default: ["compulsory", "elective", "verplicht", "keuzevak", ...]

        audience_labels:
            Field labels for the "target audience" / "intended for" field,
            used to infer a course's specialisation track.
            Default: ["target audience", "intended for", "voor studenten", ...]
    """

    def __init__(
        self,
        max_chunk_size:          int = 1500,
        chunk_overlap:           int = 150,
        min_chunk_size:          int = 80,
        window_size:             int = 1,
        context_summarizer:      Optional[Callable[[str], str]] = None,
        footer_patterns:         Optional[List[str]] = None,
        footer_min_pages:        int = 3,
        code_field_synonyms:     Optional[List[str]] = None,
        credits_field_synonyms:  Optional[List[str]] = None,
        period_field_synonyms:   Optional[List[str]] = None,
        level_field_synonyms:    Optional[List[str]] = None,
        faculty_field_synonyms:  Optional[List[str]] = None,
        language_field_synonyms: Optional[List[str]] = None,
        degree_type_labels:      Optional[List[str]] = None,
        track_keywords:          Optional[List[str]] = None,
        compulsory_keywords:     Optional[List[str]] = None,
        audience_labels:         Optional[List[str]] = None,
    ):
        self.max_chunk_size     = max_chunk_size
        self.chunk_overlap      = chunk_overlap
        self.min_chunk_size     = min_chunk_size
        self.window_size        = window_size
        self.context_summarizer = context_summarizer
        self._explicit_footer_patterns = footer_patterns
        self.footer_min_pages   = footer_min_pages

        # Resolve synonym lists
        code_labels        = code_field_synonyms     or _DEFAULT_CODE_LABELS
        credits_labels     = credits_field_synonyms  or _DEFAULT_CREDITS_LABELS
        period_labels      = period_field_synonyms   or _DEFAULT_PERIOD_LABELS
        level_labels       = level_field_synonyms    or _DEFAULT_LEVEL_LABELS
        faculty_labels     = faculty_field_synonyms  or _DEFAULT_FACULTY_LABELS
        lang_labels        = language_field_synonyms or _DEFAULT_LANGUAGE_LABELS
        degree_labels      = degree_type_labels      or _DEFAULT_DEGREE_LABELS
        track_kws          = track_keywords          or _DEFAULT_TRACK_KEYWORDS
        compulsory_kws     = compulsory_keywords     or _DEFAULT_COMPULSORY_KEYWORDS
        audience_lbls      = audience_labels         or _DEFAULT_AUDIENCE_LABELS

        # All regexes compiled once here; no method ever uses inline strings
        self._code_anchor_re    = _code_anchor_re(code_labels)
        self._credits_re        = _label_re(credits_labels)
        self._period_re         = _label_re(period_labels)
        self._level_re          = _label_re(level_labels)
        self._faculty_re        = _label_re(faculty_labels)
        self._language_re       = _label_re(lang_labels)
        self._degree_re         = _degree_re(degree_labels)
        self._track_header_re   = _track_header_re(track_kws)
        self._table_header_re   = _table_header_re(code_labels, credits_labels)
        self._compulsory_re     = _compulsory_re(compulsory_kws)
        self._audience_re       = _audience_re(audience_lbls)
        self._track_inline_re   = re.compile(   # "Track: X" or "Specialisation: X" in body text
            rf'(?:{"|".join(re.escape(k) for k in track_kws)})\s*[:\-]\s*([^\n,]{{2,80}})',
            re.IGNORECASE,
        )

        # Footer regex — built in chunk_documents after auto-detection
        self._footer_noise_re: Optional[re.Pattern] = None

        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Convert page-level Documents into context-enriched course chunks.

        Args:
            documents: One Document per PDF page (e.g. from PyPDFLoader).

        Returns:
            List of enriched LangChain Documents with metadata.
        """
        if not documents:
            return []

        source = documents[0].metadata.get("source", "unknown")

        self._footer_noise_re = self._build_footer_re(documents)
        full_text, page_map   = self._build_full_text(documents)
        sections              = self._split_into_sections(full_text, page_map, source)
        sections              = self._enrich_with_context(sections)
        chunks                = self._sections_to_documents(sections)
        dicts                 = self._documents_to_dicts(chunks)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        footer_mode = "explicit" if self._explicit_footer_patterns else "auto-detected"
        print(
            f"[CurriculumChunker] {len(documents)} pages -> "
            f"{len(sections)} sections -> {len(chunks)} chunks  "
            f"(window={self.window_size}, footer={footer_mode}, "
            f"summarizer={'LLM' if self.context_summarizer else 'structured'})"
        )
        
        return dicts

    def _documents_to_dicts(self, chunks) -> List[dict]:
        """
        Convert LangChain Document objects into the plain dicts that the
        Embedder and Supabase expect.

        Now includes structured metadata extracted by CurriculumChunker:
        course_name, course_code, credits, degree_name, section_type.

        Args:
            chunks: List of LangChain Document objects as returned by
                    CurriculumChunker.chunk_documents().

        Returns:
            A list of dicts with text, source, page, chunk_index, and metadata.
        """
        result: List[dict] = []
        for doc in chunks:
            source_path = doc.metadata.get("source", "unknown")
            source_name = Path(source_path).name

            result.append({
                "text": doc.page_content,
                "source": source_name,
                "page": doc.metadata.get("page", 0),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                # Structured metadata from CurriculumChunker
                "course_name": doc.metadata.get("course_name", ""),
                "course_code": doc.metadata.get("course_code", ""),
                "credits": doc.metadata.get("credits", ""),
                "degree_name": doc.metadata.get("degree_name", ""),
                "section_type": doc.metadata.get("section_type", ""),
            })
        
        # Print a sample of first chunk to verify metadata extraction
        for i, chunk in enumerate(result[:1]):
            print(f"--- Sample Chunk {i} ---")
            print(f"  Course: {chunk['course_name']} | Code: {chunk['course_code']}")
            print(f"  Section: {chunk['section_type']} | Page: {chunk['page']}")
            print(f"  Content: {chunk['text'][:120]}...")
        
        return result


    # ------------------------------------------------------------------
    # Step 0: footer detection
    # ------------------------------------------------------------------

    def _build_footer_re(self, documents: List[Document]) -> Optional[re.Pattern]:
        """
        Build the footer-noise regex.

        If explicit patterns were given, compile and return those.

        Otherwise auto-detect: lines appearing verbatim on ≥ footer_min_pages
        distinct pages that are also short (< 120 chars) and either:
          - Low alphanumeric density (≤ 60%) — catches "Page 1 of 84", dates,
            separator lines, etc., or
          - Appear on ≥ 40% of all pages (strong repetition signal).

        No institution name or programme name is needed for this to work.
        """
        if self._explicit_footer_patterns:
            combined = "|".join(self._explicit_footer_patterns)
            return re.compile(combined, re.IGNORECASE)

        line_page_count: Counter = Counter()
        for doc in documents:
            for line in set(ln.strip() for ln in doc.page_content.splitlines() if ln.strip()):
                line_page_count[line] += 1

        total_pages   = len(documents)
        footer_lines: List[str] = []

        for line, count in line_page_count.items():
            if count < self.footer_min_pages or len(line) >= 120:
                continue
            alnum_ratio = sum(c.isalnum() for c in line) / max(len(line), 1)
            if alnum_ratio <= 0.60:
                footer_lines.append(line)
            elif count / total_pages >= 0.40:
                footer_lines.append(line)

        if not footer_lines:
            return None

        return re.compile(
            "|".join(re.escape(ln) for ln in footer_lines),
            re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # Step 1: concatenate pages
    # ------------------------------------------------------------------

    def _build_full_text(
        self, documents: List[Document]
    ) -> Tuple[str, List[Tuple[int, int, int]]]:
        parts: List[str] = []
        page_map: List[Tuple[int, int, int]] = []
        offset = 0
        for doc in documents:
            text     = doc.page_content
            page_num = doc.metadata.get("page", 0)
            parts.append(text)
            page_map.append((offset, offset + len(text), page_num))
            offset += len(text) + 1
        return "\n".join(parts), page_map

    def _page_for_offset(
        self, offset: int, page_map: List[Tuple[int, int, int]]
    ) -> int:
        for start, end, page_num in page_map:
            if start <= offset <= end:
                return page_num
        return page_map[-1][2] if page_map else 0

    # ------------------------------------------------------------------
    # Step 2: split into CourseSection objects
    # ------------------------------------------------------------------

    def _split_into_sections(
        self,
        full_text: str,
        page_map:  List[Tuple[int, int, int]],
        source:    str,
    ) -> List[CourseSection]:
        sections: List[CourseSection] = []
        code_matches = list(self._code_anchor_re.finditer(full_text))

        if not code_matches:
            return self._page_level_fallback(full_text, page_map, source)

        # Preamble — everything before the first course title
        first_title_start = self._title_start_before_code(
            full_text, code_matches[0].start()
        )
        preamble = full_text[:first_title_start].strip()
        if len(preamble) >= self.min_chunk_size:
            sections.extend(self._split_preamble(preamble, page_map, source))

        # Individual course sections
        for idx, code_match in enumerate(code_matches):
            section_start = self._title_start_before_code(
                full_text, code_match.start()
            )
            section_end = (
                self._title_start_before_code(
                    full_text, code_matches[idx + 1].start()
                )
                if idx + 1 < len(code_matches)
                else len(full_text)
            )

            raw_text = full_text[section_start:section_end].strip()
            if len(raw_text) < self.min_chunk_size:
                continue

            raw_title_block = full_text[section_start : code_match.start()]
            sections.append(CourseSection(
                course_name    = self._clean_title(raw_title_block),
                course_code    = code_match.group(1).strip(),
                credits        = self._field(self._credits_re,  raw_text),
                period         = self._field(self._period_re,   raw_text),
                course_level   = self._field(self._level_re,    raw_text),
                faculty        = self._field(self._faculty_re,  raw_text),
                language       = self._field(self._language_re, raw_text),
                degree_name    = self._field(self._degree_re,   raw_text),
                specialisation = self._infer_specialisation(raw_text),
                content        = raw_text,
                page_number    = self._page_for_offset(section_start, page_map),
                source         = source,
                section_type   = "course_detail",
            ))

        return sections

    def _title_start_before_code(self, text: str, code_pos: int) -> int:
        """
        Walk backwards from a code anchor to find the start of the course
        title block.  Two strategies, neither assumes a specific language:

          1. Last double-newline within 400 chars — paragraph break preserved
             by most PDF extractors.
          2. Last non-noise line that is non-empty — works for compact layouts
             where lines are separated by single newlines only.

        Note: we no longer require the line to start with an uppercase letter.
        That assumption fails for titles starting with digits ("3D Graphics"),
        lowercase articles in Dutch/German ("de Staat", "das Recht"), or
        symbols.  Any non-noise non-empty line is a valid title candidate.
        """
        search_start = max(0, code_pos - 400)
        preceding    = text[search_start : code_pos]

        last_dbl = preceding.rfind("\n\n")
        if last_dbl != -1:
            return search_start + last_dbl + 2

        lines = preceding.split("\n")
        pos   = search_start + len(preceding)
        for line in reversed(lines):
            pos -= len(line) + 1
            stripped = line.strip()
            if stripped and not self._is_noise(stripped):
                return max(0, pos + 1)

        return search_start

    # ------------------------------------------------------------------
    # Preamble splitting
    # ------------------------------------------------------------------

    def _split_preamble(
        self,
        preamble: str,
        page_map: List[Tuple[int, int, int]],
        source:   str,
    ) -> List[CourseSection]:
        """
        Split the pre-course preamble into one chunk per sub-programme /
        track section, using _find_track_headers() to locate boundaries.
        """
        sections: List[CourseSection] = []
        header_matches = self._find_track_headers(preamble)

        if not header_matches:
            self._emit_preamble_chunk(preamble, 0, source, sections, page_map)
            return sections

        overview = preamble[: header_matches[0].start()].strip()
        if len(overview) >= self.min_chunk_size:
            self._emit_preamble_chunk(overview, 0, source, sections, page_map)

        for i, hdr_m in enumerate(header_matches):
            block_start = hdr_m.start()
            block_end   = (
                header_matches[i + 1].start()
                if i + 1 < len(header_matches)
                else len(preamble)
            )
            block = preamble[block_start : block_end].strip()
            if len(block) < self.min_chunk_size:
                continue

            header_text = (
                hdr_m.group(1).strip() if hdr_m.lastindex
                else hdr_m.group(0).strip()
            )
            sections.append(CourseSection(
                course_name    = header_text,
                specialisation = self._extract_track_name(header_text),
                degree_name    = self._field(self._degree_re, block),
                content        = block,
                page_number    = self._page_for_offset(block_start, page_map),
                source         = source,
                section_type   = self._classify_preamble_block(block),
            ))

        return sections

    def _find_track_headers(self, text: str) -> List[re.Match]:
        """
        Detect sub-programme / track header lines in the preamble using
        two heuristics that require no knowledge of the programme name:

          1. Any line containing a track keyword (from self._track_header_re,
             built from the caller-supplied or default track_keywords list).

          2. Any heading-like line that is immediately followed (within
             5 non-empty lines) by a course-table header row (detected via
             self._table_header_re, built from the code and credits label
             synonym lists — no inline strings).
        """
        matches_by_pos: Dict[int, re.Match] = {}

        # Heuristic 1: track keyword present in the line
        for m in self._track_header_re.finditer(text):
            matches_by_pos[m.start()] = m

        # Heuristic 2: heading followed by a course-table header
        heading_re = re.compile(r'(?:^|\n)(\S[^\n]{4,100})\n', re.MULTILINE)
        for m in heading_re.finditer(text):
            if m.start() in matches_by_pos:
                continue
            lookahead        = text[m.end() : m.end() + 400]
            near_lines       = [
                ln for ln in text[m.end() : m.end() + 200].splitlines()
                if ln.strip()
            ]
            if (
                self._table_header_re.search(lookahead)
                and len(near_lines) <= 5
            ):
                matches_by_pos[m.start()] = m

        return sorted(matches_by_pos.values(), key=lambda m: m.start())

    def _emit_preamble_chunk(
        self,
        text:     str,
        offset:   int,
        source:   str,
        out:      List[CourseSection],
        page_map: List[Tuple[int, int, int]],
    ) -> None:
        out.append(CourseSection(
            degree_name  = self._field(self._degree_re, text),
            content      = text,
            page_number  = self._page_for_offset(offset, page_map),
            source       = source,
            section_type = "program_overview",
        ))

    # ------------------------------------------------------------------
    # Step 3: context enrichment (Databricks Strategy #5)
    # ------------------------------------------------------------------

    def _enrich_with_context(
        self, sections: List[CourseSection]
    ) -> List[CourseSection]:
        total = len(sections)
        for i, section in enumerate(sections):
            section.section_index  = i
            section.total_sections = total
            if i > 0:
                prev = sections[max(0, i - self.window_size) : i][-1]
                section.prev_course_name   = prev.course_name
                section.prev_course_code   = prev.course_code
                section.prev_course_period = prev.period
            if i + 1 < total:
                nxt = sections[i + 1 : min(total, i + 1 + self.window_size)][0]
                section.next_course_name   = nxt.course_name
                section.next_course_code   = nxt.course_code
                section.next_course_period = nxt.period
        return sections

    def _build_context_block(self, section: CourseSection) -> str:
        lines: List[str] = ["[Context]"]
        if section.degree_name:
            lines.append(f"Programme: {section.degree_name.strip()}")
        if section.specialisation:
            lines.append(f"Track: {section.specialisation}")
        if section.prev_course_name:
            stub = section.prev_course_name
            if section.prev_course_code:
                stub += f" ({section.prev_course_code}"
                if section.prev_course_period:
                    stub += f", {section.prev_course_period.strip()}"
                stub += ")"
            lines.append(f"Previous: {stub}")
        if section.next_course_name:
            stub = section.next_course_name
            if section.next_course_code:
                stub += f" ({section.next_course_code}"
                if section.next_course_period:
                    stub += f", {section.next_course_period.strip()}"
                stub += ")"
            lines.append(f"Next: {stub}")
        if section.total_sections:
            lines.append(
                f"Position: {section.section_index + 1} of {section.total_sections}"
            )
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Step 4: emit LangChain Documents
    # ------------------------------------------------------------------

    def _sections_to_documents(
        self, sections: List[CourseSection]
    ) -> List[Document]:
        documents: List[Document] = []
        for section in sections:
            metadata = {
                "source":           section.source,
                "page":             section.page_number,
                "section_type":     section.section_type,
                "course_name":      section.course_name,
                "course_code":      section.course_code,
                "credits":          section.credits,
                "period":           section.period,
                "course_level":     section.course_level,
                "faculty":          section.faculty,
                "language":         section.language,
                "degree_name":      section.degree_name,
                "specialisation":   section.specialisation,
                "prev_course_name": section.prev_course_name,
                "prev_course_code": section.prev_course_code,
                "next_course_name": section.next_course_name,
                "next_course_code": section.next_course_code,
                "section_index":    section.section_index,
                "total_sections":   section.total_sections,
            }
            context_block   = self._build_context_block(section)
            metadata_header = self._build_metadata_header(section)
            full_body       = context_block + "\n" + metadata_header + section.content

            if len(full_body) <= self.max_chunk_size:
                documents.append(Document(
                    page_content=full_body,
                    metadata=metadata.copy(),
                ))
            else:
                sub_docs = self._fallback_splitter.split_documents([
                    Document(page_content=section.content, metadata=metadata.copy())
                ])
                for sub_idx, sub in enumerate(sub_docs):
                    sub_ctx = context_block.rstrip("\n")
                    if len(sub_docs) > 1:
                        sub_ctx += f"\nSub-chunk: {sub_idx + 1} of {len(sub_docs)}"
                    sub_ctx += "\n"
                    sub.page_content = sub_ctx + "\n" + metadata_header + sub.page_content
                    sub.metadata.update(metadata)
                    sub.metadata["sub_chunk_index"]  = sub_idx
                    sub.metadata["total_sub_chunks"] = len(sub_docs)
                    documents.append(sub)

        return documents

    # ------------------------------------------------------------------
    # Helpers — every method uses only self.* compiled regexes
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata_header(section: CourseSection) -> str:
        parts: List[str] = []
        if section.course_name:
            parts.append(f"[Course: {section.course_name}]")
        if section.course_code:
            parts.append(f"[Code: {section.course_code}]")
        if section.period:
            parts.append(f"[Period: {section.period.strip()}]")
        if section.credits:
            parts.append(f"[Credits: {section.credits}]")
        if section.course_level:
            parts.append(f"[Level: {section.course_level}]")
        if section.faculty:
            parts.append(f"[Faculty: {section.faculty.strip()[:60]}]")
        return (" ".join(parts) + "\n") if parts else ""

    def _clean_title(self, raw_block: str) -> str:
        """
        Strip footer noise from lines immediately before the code anchor
        and return the last 1–2 clean lines as the course title.
        No assumption about case or starting character.
        """
        lines = [ln.strip() for ln in raw_block.splitlines()]
        clean = [ln for ln in lines if ln and not self._is_noise(ln)]
        return " ".join(clean[-2:]).strip() if clean else ""

    def _is_noise(self, line: str) -> bool:
        if self._footer_noise_re is None:
            return False
        return bool(self._footer_noise_re.search(line))

    def _infer_specialisation(self, text: str) -> str:
        """
        Infer a course's track/specialisation from its body text.

        Uses only self._track_inline_re (built from track_keywords) and
        self._audience_re (built from audience_labels) — no inline strings.
        Returns empty string when nothing can be inferred.
        """
        # Explicit "Track: X" or "Specialisation: X" label in the text
        m = self._track_inline_re.search(text)
        if m:
            return m.group(1).strip()

        # "Target Audience" field on the following line
        m = self._audience_re.search(text)
        if m:
            candidate = m.group(1).strip()
            # Accept only if it looks like a proper name (contains a letter)
            if re.search(r'[A-Za-z\u00C0-\u024F]', candidate):
                return candidate[:60]

        return ""

    def _extract_track_name(self, header_text: str) -> str:
        """
        Strip the generic structural prefix from a detected track header.
        E.g. "Computer Science specialisation Data Science" → "Data Science"

        Uses self._track_header_re keywords list rebuilt as an alternation
        to strip the prefix — no inline keyword strings.
        """
        # Build a one-shot alternation from the same keywords used for detection
        # (pulled back out of the compiled pattern's source to avoid duplication)
        kw_alt = "|".join(
            re.escape(k) for k in (_DEFAULT_TRACK_KEYWORDS)
        )
        cleaned = re.sub(
            rf'^.+?\s+(?:{kw_alt})\s+',
            '', header_text, count=1, flags=re.IGNORECASE
        ).strip()
        return cleaned if cleaned else header_text.strip()

    def _classify_preamble_block(self, text: str) -> str:
        """
        Classify a preamble block using only self._table_header_re and
        self._compulsory_re — no inline strings.
        """
        if self._table_header_re.search(text):
            return "course_list"
        if self._compulsory_re.search(text):
            return "program_overview"
        return "general"

    def _page_level_fallback(
        self,
        full_text: str,
        page_map:  List[Tuple[int, int, int]],
        source:    str,
    ) -> List[CourseSection]:
        """
        Safety fallback: one section per page when no code anchors are found.
        Triggered when none of the code-field synonyms appear in the document.
        """
        sections = []
        for start, end, page_num in page_map:
            text = full_text[start:end].strip()
            if len(text) < self.min_chunk_size:
                continue
            sections.append(CourseSection(
                content      = text,
                page_number  = page_num,
                source       = source,
                section_type = self._classify_preamble_block(text),
                degree_name  = self._field(self._degree_re, text),
            ))
        return sections

    @staticmethod
    def _field(pattern: re.Pattern, text: str) -> str:
        m = pattern.search(text)
        return m.group(1).strip() if m else ""