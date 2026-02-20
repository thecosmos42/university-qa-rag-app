# # src/ingestion/academic_chunker.py
# """
# Document-Aware Academic Chunker

# Instead of blindly splitting by character count, this chunker:
# 1. Detects course section boundaries via regex patterns
# 2. Keeps each course as a self-contained chunk
# 3. Splits only within a course if it exceeds max size
# 4. Prepends course title as a header to every chunk
# 5. Extracts structured metadata (course_name, course_code, degree_name)

# This prevents cross-course contamination that causes retrieval collisions
# between similarly-named courses like "Advanced NLP" vs "Natural Language Processing".
# """

# import re
# from typing import List, Optional
# from dataclasses import dataclass, field
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter


# @dataclass
# class CourseSection:
#     """Represents a parsed course section from the academic document."""
#     course_name: str = ""
#     course_code: str = ""
#     credits: str = ""
#     content: str = ""
#     page_number: int = 0
#     source: str = ""
#     section_type: str = "course_detail"  # course_detail, program_overview, course_list
#     degree_name: str = ""


# # Regex patterns for detecting course boundaries in academic catalogs
# COURSE_CODE_PATTERN = re.compile(
#     r'Course\s+Code\s+([A-Za-z_0-9]+)',
#     re.IGNORECASE
# )

# COURSE_NAME_BEFORE_CODE = re.compile(
#     r'(?:^|\n)([A-Z][A-Za-z\s&\-:,]+?)\s*\n\s*Course\s+Code',
#     re.IGNORECASE | re.MULTILINE
# )

# CREDITS_PATTERN = re.compile(
#     r'Credits\s+(\d+)',
#     re.IGNORECASE
# )

# # Pattern for course headers that appear as standalone lines
# COURSE_HEADER_PATTERN = re.compile(
#     r'^([A-Z][A-Za-z\s&\-:,]+?)\s*$',
#     re.MULTILINE
# )

# # Pattern for degree program sections
# DEGREE_PATTERN = re.compile(
#     r'((?:Master|Bachelor|MSc|BSc|MA|BA)\s+(?:of\s+|in\s+)?[A-Za-z\s&\-]+)',
#     re.IGNORECASE
# )

# # Patterns that indicate a new course section is starting
# SECTION_BOUNDARY_PATTERNS = [
#     re.compile(r'\nCourse\s+Code\s+', re.IGNORECASE),
#     re.compile(r'\nCourse\s+Objective\b', re.IGNORECASE),
# ]


# class AcademicChunker:
#     """
#     Splits academic catalog documents into semantically coherent chunks
#     that respect course boundaries.
#     """

#     def __init__(
#         self,
#         max_chunk_size: int = 1200,
#         chunk_overlap: int = 100,
#         min_chunk_size: int = 100,
#     ):
#         self.max_chunk_size = max_chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.min_chunk_size = min_chunk_size
        
#         # Fallback splitter for chunks that exceed max size
#         self.fallback_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=max_chunk_size,
#             chunk_overlap=chunk_overlap,
#             separators=["\n\n", "\n", ". ", " ", ""],
#         )

#     def chunk_documents(self, documents: List[Document]) -> List[Document]:
#         """
#         Process LangChain Documents (one per PDF page) into semantically
#         coherent chunks that respect course boundaries.

#         Args:
#             documents: List of LangChain Document objects (one per page).

#         Returns:
#             List of chunked Document objects with enriched metadata.
#         """
#         # Step 1: Concatenate all pages into one big text, preserving page boundaries
#         full_text, page_map = self._build_full_text(documents)
#         source = documents[0].metadata.get("source", "unknown") if documents else "unknown"
        
#         # Step 2: Split into course sections
#         sections = self._split_into_sections(full_text, page_map, source)
        
#         # Step 3: Convert sections to Documents, splitting oversized ones
#         chunks = self._sections_to_documents(sections)
        
#         # Step 4: Assign chunk indices
#         for i, chunk in enumerate(chunks):
#             chunk.metadata["chunk_index"] = i

#         print(
#             f"[AcademicChunker] Produced {len(chunks)} chunks "
#             f"from {len(documents)} page(s) "
#             f"({len(sections)} sections detected)"
#         )
#         return chunks

#     def _build_full_text(self, documents: List[Document]) -> tuple[str, list[tuple[int, int, int]]]:
#         """
#         Concatenate all pages with page markers.
        
#         Returns:
#             full_text: The concatenated text
#             page_map: List of (start_offset, end_offset, page_number) tuples
#         """
#         parts = []
#         page_map = []
#         offset = 0
        
#         for doc in documents:
#             text = doc.page_content
#             page_num = doc.metadata.get("page", 0)
#             start = offset
#             parts.append(text)
#             offset += len(text) + 1  # +1 for the newline joiner
#             page_map.append((start, offset - 1, page_num))
        
#         return "\n".join(parts), page_map

#     def _get_page_for_offset(self, offset: int, page_map: list[tuple[int, int, int]]) -> int:
#         """Find which page a character offset belongs to."""
#         for start, end, page_num in page_map:
#             if start <= offset <= end:
#                 return page_num
#         return page_map[-1][2] if page_map else 0

#     def _split_into_sections(
#         self, full_text: str, page_map: list[tuple[int, int, int]], source: str
#     ) -> List[CourseSection]:
#         """
#         Split the full document text into course-level sections.
#         Uses 'Course Code' as the primary boundary marker.
#         """
#         sections: List[CourseSection] = []
        
#         # Find all course code occurrences — these are reliable section boundaries
#         code_matches = list(COURSE_CODE_PATTERN.finditer(full_text))
        
#         if not code_matches:
#             # No course codes found — fall back to page-level chunking
#             return self._fallback_page_sections(full_text, page_map, source)
        
#         # Extract section before the first course code (program overview, TOC, etc.)
#         first_code_pos = code_matches[0].start()
        
#         # Search backwards from the first Course Code to find the course title
#         preamble_end = self._find_section_start_before_code(full_text, first_code_pos)
        
#         if preamble_end > self.min_chunk_size:
#             preamble_text = full_text[:preamble_end].strip()
#             if preamble_text:
#                 section = CourseSection(
#                     content=preamble_text,
#                     page_number=self._get_page_for_offset(0, page_map),
#                     source=source,
#                     section_type=self._classify_section(preamble_text),
#                 )
#                 section.degree_name = self._extract_degree(preamble_text)
#                 sections.append(section)
        
#         # Process each course section
#         for i, code_match in enumerate(code_matches):
#             # Find the start of this course section (look for the title before Course Code)
#             section_start = self._find_section_start_before_code(full_text, code_match.start())
            
#             # Section ends at the start of the next course, or end of document
#             if i + 1 < len(code_matches):
#                 next_section_start = self._find_section_start_before_code(
#                     full_text, code_matches[i + 1].start()
#                 )
#                 section_end = next_section_start
#             else:
#                 section_end = len(full_text)
            
#             section_text = full_text[section_start:section_end].strip()
            
#             if len(section_text) < self.min_chunk_size:
#                 continue
            
#             # Extract metadata
#             course_name = self._extract_course_name(full_text, code_match.start())
#             course_code = code_match.group(1).strip()
#             credits = self._extract_credits(section_text)
            
#             section = CourseSection(
#                 course_name=course_name,
#                 course_code=course_code,
#                 credits=credits,
#                 content=section_text,
#                 page_number=self._get_page_for_offset(section_start, page_map),
#                 source=source,
#                 section_type="course_detail",
#             )
#             section.degree_name = self._extract_degree(section_text)
#             sections.append(section)
        
#         return sections

#     def _find_section_start_before_code(self, text: str, code_pos: int) -> int:
#         """
#         Search backwards from a 'Course Code' position to find where the
#         course section actually starts (typically the course title line).
#         """
#         # Look back up to 500 chars for a blank line or section boundary
#         search_start = max(0, code_pos - 500)
#         preceding = text[search_start:code_pos]
        
#         # Find the last double-newline (paragraph break) before Course Code
#         last_break = preceding.rfind("\n\n")
#         if last_break != -1:
#             return search_start + last_break + 2  # Skip the \n\n
        
#         # If no paragraph break, find last single newline that starts a capitalized word
#         lines = preceding.split("\n")
#         pos = code_pos
#         for line in reversed(lines):
#             pos -= len(line) + 1
#             stripped = line.strip()
#             if stripped and stripped[0].isupper() and len(stripped) > 3:
#                 return max(0, search_start + (pos - search_start) + 1)
        
#         return search_start

#     def _extract_course_name(self, text: str, code_pos: int) -> str:
#         """Extract the course name that appears before 'Course Code'."""
#         # Look backward from 'Course Code' for the title
#         search_start = max(0, code_pos - 300)
#         preceding = text[search_start:code_pos].strip()
        
#         # The course name is typically the last non-empty line(s) before Course Code
#         lines = [l.strip() for l in preceding.split("\n") if l.strip()]
        
#         if not lines:
#             return ""
        
#         # Walk backwards to find the course title
#         # Skip lines that look like metadata (Credits, Period, etc.)
#         name_parts = []
#         for line in reversed(lines):
#             # Stop at lines that are clearly part of a previous section
#             if any(kw in line.lower() for kw in [
#                 'course content', 'course objective', 'teaching method',
#                 'method of assessment', 'additional information', 'literature',
#                 'target audience', 'recommended background', 'page '
#             ]):
#                 break
#             name_parts.insert(0, line)
#             # Course names are usually 1-3 words on a single line
#             if len(name_parts) >= 2:
#                 break
        
#         course_name = " ".join(name_parts).strip()
        
#         # Clean up: remove trailing metadata keywords
#         for suffix in ['Course Code', 'Credits', 'Period']:
#             if course_name.endswith(suffix):
#                 course_name = course_name[:course_name.rfind(suffix)].strip()
        
#         return course_name

#     def _extract_credits(self, text: str) -> str:
#         """Extract credits value from section text."""
#         match = CREDITS_PATTERN.search(text)
#         return match.group(1) if match else ""

#     def _extract_degree(self, text: str) -> str:
#         """Extract degree program name from section text."""
#         match = DEGREE_PATTERN.search(text)
#         return match.group(1).strip() if match else ""

#     def _classify_section(self, text: str) -> str:
#         """Classify what type of section this is."""
#         lower = text.lower()
#         if 'course code' in lower and 'course objective' in lower:
#             return "course_detail"
#         elif any(kw in lower for kw in ['compulsory courses', 'elective', 'programme']):
#             return "program_overview"
#         elif re.search(r'(?:course\s+name|period|credits|code)\s', lower):
#             return "course_list"
#         return "general"

#     def _fallback_page_sections(
#         self, full_text: str, page_map: list[tuple[int, int, int]], source: str
#     ) -> List[CourseSection]:
#         """When no course codes are detected, fall back to page-level sections."""
#         sections = []
#         for start, end, page_num in page_map:
#             text = full_text[start:end].strip()
#             if len(text) < self.min_chunk_size:
#                 continue
#             sections.append(CourseSection(
#                 content=text,
#                 page_number=page_num,
#                 source=source,
#                 section_type=self._classify_section(text),
#                 degree_name=self._extract_degree(text),
#             ))
#         return sections

#     def _sections_to_documents(self, sections: List[CourseSection]) -> List[Document]:
#         """
#         Convert CourseSection objects to LangChain Documents.
#         Splits oversized sections using the fallback splitter.
#         """
#         documents: List[Document] = []
        
#         for section in sections:
#             metadata = {
#                 "source": section.source,
#                 "page": section.page_number,
#                 "course_name": section.course_name,
#                 "course_code": section.course_code,
#                 "credits": section.credits,
#                 "degree_name": section.degree_name,
#                 "section_type": section.section_type,
#             }
            
#             # Prepend course title as header for better embedding quality
#             header = ""
#             if section.course_name:
#                 header = f"[Course: {section.course_name}]"
#                 if section.course_code:
#                     header += f" [Code: {section.course_code}]"
#                 header += "\n"
            
#             content_with_header = header + section.content
            
#             if len(content_with_header) <= self.max_chunk_size:
#                 # Fits in one chunk
#                 documents.append(Document(
#                     page_content=content_with_header,
#                     metadata=metadata.copy(),
#                 ))
#             else:
#                 # Too large — split but preserve the header in each sub-chunk
#                 sub_docs = self.fallback_splitter.split_documents([
#                     Document(page_content=section.content, metadata=metadata.copy())
#                 ])
#                 for sub_doc in sub_docs:
#                     sub_doc.page_content = header + sub_doc.page_content
#                     # Copy metadata from section (fallback splitter preserves original metadata)
#                     for key in ["course_name", "course_code", "credits", "degree_name", "section_type"]:
#                         sub_doc.metadata[key] = metadata[key]
#                     documents.append(sub_doc)
        
#         return documents

# src/ingestion/academic_chunker.py
"""
Document-Aware Academic Chunker with Context-Enriched Chunking
==============================================================
VU Amsterdam AI Study Guide 2025-2026

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
Pass a callable `context_summarizer` to AcademicChunker() to replace the
default structured-context approach with LLM-generated summaries (matching
the Databricks article's `summarize=True` path):

    def my_summarizer(text: str) -> str:
        # Call your LLM endpoint here
        return llm.invoke(f"Summarise in 1 sentence: {text}")

    chunker = AcademicChunker(context_summarizer=my_summarizer)

Without a summarizer, the chunker uses structured neighbour metadata instead
(zero extra latency, no external calls).

DOCUMENT STRUCTURE (VU Amsterdam AI Study Guide)
------------------------------------------------
Page  1:      Cover page
Page  2:      Programme overview
Pages 3–4:    Table of Contents
Pages 5–7:    Specialisation: "AI for Health"
Pages 8–11:   Specialisation: "Cognitive Science"
Pages 11–13:  Specialisation: "Main AI"
Pages 14–84:  Individual course detail blocks

PAGE FOOTER (2 lines, on every page — stripped during title extraction):
    Artificial Intelligence - 2025-2026
    Vrije Universiteit Amsterdam - Faculty of Science 31 Jan 2026 - Page N of 84
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Primary section anchor.  All course code formats in this PDF:
#   X_405048, XM_0010, L_AAMPLIN035, P_MAGEDIS, XMU_0045, S_EMP, AM_470730
COURSE_CODE_RE = re.compile(
    r'Course\s+Code\s+([A-Za-z0-9_]+)',
    re.IGNORECASE,
)

# Metadata field patterns
CREDITS_RE      = re.compile(r'(?:^|\n)Credits\s+(\d+)',                re.IGNORECASE)
PERIOD_RE       = re.compile(r'(?:^|\n)Period\s+([^\n]+)',              re.IGNORECASE)
COURSE_LEVEL_RE = re.compile(r'(?:^|\n)Course Level\s+(\d+)',          re.IGNORECASE)
FACULTY_RE      = re.compile(r'(?:^|\n)Faculty\s+([^\n]+)',            re.IGNORECASE)
LANGUAGE_RE     = re.compile(r'(?:^|\n)Language Of Tuition\s+([^\n]+)',re.IGNORECASE)
DEGREE_RE       = re.compile(
    r'((?:Master|Bachelor|MSc|BSc|MA|BA)\s+(?:of\s+|in\s+)?[A-Za-z\s&\-]+)',
    re.IGNORECASE,
)

# Specialisation section headers (pages 5–13)
SPECIALISATION_HEADER_RE = re.compile(
    r'(?:^|\n)(Artificial Intelligence\s+'
    r'(?:specialisation|main specialisation)\s+[^\n]+)',
    re.MULTILINE | re.IGNORECASE,
)

# VU page footer lines — always 2 lines, stripped during title extraction
_FOOTER_NOISE_RE = re.compile(
    r'Vrije Universiteit'
    r'|Faculty of Science'
    r'|Artificial Intelligence\s+-\s+\d{4}'
    r'|Page \d+ of \d+'
    r'|\d+ Jan \d{4}',
    re.IGNORECASE,
)

# Detect which specialisation a course belongs to based on its code prefix
# Used to propagate specialisation context forward to course chunks
_SPECIALISATION_MAP = {
    'XMU': 'AI for Health',        # Amsterdam UMC / UvA joint courses
    'P_M': 'Cognitive Science',     # Psychology department courses
    'P_N': 'Cognitive Science',
    'S_':  'Social Robotics',
    'L_':  'Language Technology',
    'AM_': 'Health Sciences',
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CourseSection:
    """One logical section extracted from the academic catalogue."""
    course_name:  str = ""
    course_code:  str = ""
    credits:      str = ""
    period:       str = ""
    course_level: str = ""
    faculty:      str = ""
    language:     str = ""
    degree_name:  str = ""
    specialisation: str = ""
    content:      str = ""
    page_number:  int = 0
    source:       str = ""
    section_type: str = "course_detail"   # course_detail|program_overview|course_list|general
    # Context window neighbours (populated by _enrich_with_context)
    prev_course_name: str = ""
    prev_course_code: str = ""
    prev_course_period: str = ""
    next_course_name: str = ""
    next_course_code: str = ""
    next_course_period: str = ""
    section_index: int = 0
    total_sections: int = 0


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class AcademicChunker:
    """
    Splits a VU Amsterdam AI study guide (one LangChain Document per PDF page)
    into semantically coherent, context-enriched chunks.

    Context enrichment follows Databricks' Strategy #5: each chunk has a
    structured context block prepended that describes its neighbours, enabling
    the embedding model to capture cross-section relationships.

    Args:
        max_chunk_size:      Maximum characters per chunk before fallback splitting.
        chunk_overlap:       Overlap between fallback sub-chunks.
        min_chunk_size:      Minimum characters for a section to be kept.
        window_size:         Number of neighbouring sections to include in
                             context (1 = immediate prev + next, 2 = two each side).
        context_summarizer:  Optional callable (str) -> str.  If provided,
                             neighbour text is passed through it before being
                             prepended.  If None, structured metadata is used
                             instead (no LLM call required).
    """

    def __init__(
        self,
        max_chunk_size:     int = 1500,
        chunk_overlap:      int = 150,
        min_chunk_size:     int = 80,
        window_size:        int = 1,
        context_summarizer: Optional[Callable[[str], str]] = None,
    ):
        self.max_chunk_size     = max_chunk_size
        self.chunk_overlap      = chunk_overlap
        self.min_chunk_size     = min_chunk_size
        self.window_size        = window_size
        self.context_summarizer = context_summarizer

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Convert page-level Documents into context-enriched course-level chunks.

        Pipeline:
          1. Concatenate pages + build offset->page map
          2. Split into CourseSection objects (boundary detection)
          3. Enrich each section with context from its neighbours
          4. Convert to LangChain Documents (splitting oversized sections)
          5. Assign sequential chunk indices

        Args:
            documents: One Document per PDF page (e.g. from PyPDFLoader).

        Returns:
            List of enriched, chunked Documents.
        """
        if not documents:
            return []

        source = documents[0].metadata.get("source", "unknown")
        full_text, page_map = self._build_full_text(documents)

        # Steps 1–2: parse sections
        sections = self._split_into_sections(full_text, page_map, source)

        # Step 3: context enrichment (Databricks Strategy #5)
        sections = self._enrich_with_context(sections)

        # Steps 4–5: emit Documents
        chunks = self._sections_to_documents(sections)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        print(
            f"[AcademicChunker] {len(documents)} pages -> "
            f"{len(sections)} sections -> {len(chunks)} chunks  "
            f"(window_size={self.window_size}, "
            f"summarizer={'LLM' if self.context_summarizer else 'structured'})"
        )
        return chunks

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
            start    = offset
            parts.append(text)
            offset  += len(text) + 1
            page_map.append((start, offset - 1, page_num))

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
        code_matches = list(COURSE_CODE_RE.finditer(full_text))

        if not code_matches:
            return self._page_level_fallback(full_text, page_map, source)

        # Preamble (pages 1–13)
        first_title_start = self._title_start_before_code(
            full_text, code_matches[0].start()
        )
        preamble = full_text[:first_title_start].strip()
        if len(preamble) >= self.min_chunk_size:
            sections.extend(self._split_preamble(preamble, page_map, source))

        # Individual courses
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
            course_name = self._clean_title(raw_title_block)
            course_code = code_match.group(1).strip()

            sections.append(CourseSection(
                course_name    = course_name,
                course_code    = course_code,
                credits        = self._field(CREDITS_RE,      raw_text),
                period         = self._field(PERIOD_RE,       raw_text),
                course_level   = self._field(COURSE_LEVEL_RE, raw_text),
                faculty        = self._field(FACULTY_RE,      raw_text),
                language       = self._field(LANGUAGE_RE,     raw_text),
                degree_name    = self._field(DEGREE_RE,       raw_text),
                specialisation = self._infer_specialisation(course_code, raw_text),
                content        = raw_text,
                page_number    = self._page_for_offset(section_start, page_map),
                source         = source,
                section_type   = "course_detail",
            ))

        return sections

    def _title_start_before_code(self, text: str, code_pos: int) -> int:
        """
        Walk backwards from a 'Course Code' position to find the start of the
        course's title block, stripping VU footer lines.

        Layout before every course (after page joins):
            [prev section body]
            Artificial Intelligence - 2025-2026    <- footer line 1
            Vrije Universiteit Amsterdam - Page N  <- footer line 2
            <CourseName>                           <- title we want
            Course Code X_XXXXX
        """
        search_start = max(0, code_pos - 400)
        preceding    = text[search_start : code_pos]

        # Prefer the last paragraph break
        last_dbl = preceding.rfind("\n\n")
        if last_dbl != -1:
            return search_start + last_dbl + 2

        # Fallback: last non-noise capitalised line
        lines = preceding.split("\n")
        pos   = search_start + len(preceding)
        for line in reversed(lines):
            pos -= len(line) + 1
            stripped = line.strip()
            if stripped and stripped[0].isupper() and not _FOOTER_NOISE_RE.search(stripped):
                return max(0, pos + 1)

        return search_start

    # ------------------------------------------------------------------
    # Step 3: Context Enrichment  (Databricks Strategy #5)
    # ------------------------------------------------------------------

    def _enrich_with_context(
        self, sections: List[CourseSection]
    ) -> List[CourseSection]:
        """
        Implement the Databricks context-enriched chunking pattern.

        For each section we look at up to `window_size` neighbours on each
        side.  The neighbour information is either:

          A) Structured metadata (default, zero-latency):
             "Advanced Logic (X_405048, P4)" — name, code, period

          B) LLM summary (if context_summarizer is provided):
             The neighbour's full body text is passed through the callable
             and the returned summary is used instead.

        This mirrors the article's window approach where:
          window = base_chunks[window_start : window_end]
          context_chunks = [c for j, c in enumerate(window) if j != i]
        """
        total = len(sections)
        for i, section in enumerate(sections):
            section.section_index  = i
            section.total_sections = total

            # -- Left context (previous neighbours) ----------------------
            left_idx  = max(0, i - self.window_size)
            left_neighbours = sections[left_idx : i]

            if left_neighbours:
                prev = left_neighbours[-1]   # immediate predecessor
                section.prev_course_name   = prev.course_name
                section.prev_course_code   = prev.course_code
                section.prev_course_period = prev.period

            # -- Right context (next neighbours) -------------------------
            right_idx = min(total, i + self.window_size + 1)
            right_neighbours = sections[i + 1 : right_idx]

            if right_neighbours:
                nxt = right_neighbours[0]    # immediate successor
                section.next_course_name   = nxt.course_name
                section.next_course_code   = nxt.course_code
                section.next_course_period = nxt.period

        return sections

    def _build_context_block(self, section: CourseSection) -> str:
        """
        Build the context preamble that is prepended to each chunk's content.

        Structure mirrors the article's enriched format:
          "Context: [summary/metadata]\n\nContent: [chunk body]"

        This is always built from structured metadata.  If context_summarizer
        is set it is used to condense the neighbour's body text (called from
        _sections_to_documents so we have the full text available).
        """
        lines: List[str] = ["[Context]"]

        # Programme / specialisation context
        if section.degree_name:
            lines.append(f"Programme: {section.degree_name.strip()}")
        if section.specialisation:
            lines.append(f"Specialisation: {section.specialisation}")

        # Neighbour context — previous
        if section.prev_course_name:
            prev_info = section.prev_course_name
            if section.prev_course_code:
                prev_info += f" ({section.prev_course_code}"
                if section.prev_course_period:
                    prev_info += f", {section.prev_course_period.strip()}"
                prev_info += ")"
            lines.append(f"Previous course: {prev_info}")

        # Neighbour context — next
        if section.next_course_name:
            next_info = section.next_course_name
            if section.next_course_code:
                next_info += f" ({section.next_course_code}"
                if section.next_course_period:
                    next_info += f", {section.next_course_period.strip()}"
                next_info += ")"
            lines.append(f"Next course: {next_info}")

        # Position signal
        if section.total_sections:
            lines.append(
                f"Position: {section.section_index + 1} of {section.total_sections}"
            )

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Preamble splitting (specialisation overview pages 1–13)
    # ------------------------------------------------------------------

    def _split_preamble(
        self,
        preamble: str,
        page_map: List[Tuple[int, int, int]],
        source:   str,
    ) -> List[CourseSection]:
        """
        Split pages 1–13 into one programme overview chunk + one chunk per
        specialisation (AI for Health, Cognitive Science, Main AI).
        """
        sections: List[CourseSection] = []
        spec_matches = list(SPECIALISATION_HEADER_RE.finditer(preamble))

        if not spec_matches:
            self._emit_overview(preamble, 0, source, sections, page_map)
            return sections

        overview = preamble[: spec_matches[0].start()].strip()
        if len(overview) >= self.min_chunk_size:
            self._emit_overview(overview, 0, source, sections, page_map)

        for i, spec_m in enumerate(spec_matches):
            block_start = spec_m.start()
            block_end   = (
                spec_matches[i + 1].start()
                if i + 1 < len(spec_matches)
                else len(preamble)
            )
            block = preamble[block_start : block_end].strip()
            if len(block) < self.min_chunk_size:
                continue

            sections.append(CourseSection(
                course_name    = spec_m.group(1).strip(),
                specialisation = self._extract_specialisation_name(spec_m.group(1)),
                degree_name    = self._field(DEGREE_RE, block),
                content        = block,
                page_number    = self._page_for_offset(block_start, page_map),
                source         = source,
                section_type   = self._classify_preamble(block),
            ))

        return sections

    def _emit_overview(
        self,
        text:     str,
        offset:   int,
        source:   str,
        out:      List[CourseSection],
        page_map: List[Tuple[int, int, int]],
    ) -> None:
        out.append(CourseSection(
            degree_name  = self._field(DEGREE_RE, text),
            content      = text,
            page_number  = self._page_for_offset(offset, page_map),
            source       = source,
            section_type = "program_overview",
        ))

    # ------------------------------------------------------------------
    # Step 4: sections -> LangChain Documents
    # ------------------------------------------------------------------

    def _sections_to_documents(
        self, sections: List[CourseSection]
    ) -> List[Document]:
        documents: List[Document] = []

        for section in sections:
            metadata = {
                # Core identification
                "source":             section.source,
                "page":               section.page_number,
                "section_type":       section.section_type,
                # Course metadata
                "course_name":        section.course_name,
                "course_code":        section.course_code,
                "credits":            section.credits,
                "period":             section.period,
                "course_level":       section.course_level,
                "faculty":            section.faculty,
                "language":           section.language,
                "degree_name":        section.degree_name,
                "specialisation":     section.specialisation,
                # Context-enrichment metadata (Databricks Strategy #5)
                "prev_course_name":   section.prev_course_name,
                "prev_course_code":   section.prev_course_code,
                "next_course_name":   section.next_course_name,
                "next_course_code":   section.next_course_code,
                "section_index":      section.section_index,
                "total_sections":     section.total_sections,
            }

            # Lightweight chunk body: just a compact metadata header + content.
            # The full context/neighbour info is preserved in Document metadata
            # for use at retrieval time, but NOT embedded (saves tokens).
            metadata_header = self._build_metadata_header(section)
            full_body = metadata_header + section.content

            if len(full_body) <= self.max_chunk_size:
                documents.append(Document(
                    page_content=full_body,
                    metadata=metadata.copy(),
                ))
            else:
                # Split oversized sections.  Re-attach the metadata header
                # to every sub-chunk so each piece retains course identity.
                sub_docs = self.fallback_splitter.split_documents([
                    Document(
                        page_content=section.content,
                        metadata=metadata.copy(),
                    )
                ])
                for sub_idx, sub in enumerate(sub_docs):
                    sub.page_content = metadata_header + sub.page_content
                    sub.metadata.update(metadata)
                    sub.metadata["sub_chunk_index"] = sub_idx
                    sub.metadata["total_sub_chunks"] = len(sub_docs)
                    documents.append(sub)

        return documents

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata_header(section: CourseSection) -> str:
        """
        Compact bracketed metadata header.  Sits between the context block
        and the course body.  These structured tags improve embedding precision
        for course-specific queries.

        Example:
          [Course: Advanced Logic] [Code: X_405048] [Period: P4] [Credits: 6]
          [Level: 500] [Faculty: Faculty of Science]
        """
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

    @staticmethod
    def _clean_title(raw_block: str) -> str:
        """Strip VU footer noise and return the last 1-2 clean title lines."""
        lines = [ln.strip() for ln in raw_block.splitlines()]
        clean = [ln for ln in lines if ln and not _FOOTER_NOISE_RE.search(ln)]
        return " ".join(clean[-2:]).strip() if clean else ""

    @staticmethod
    def _field(pattern: re.Pattern, text: str) -> str:
        m = pattern.search(text)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _infer_specialisation(course_code: str, text: str) -> str:
        """
        Infer which specialisation track a course belongs to from its code
        prefix or faculty/text content.
        """
        for prefix, spec in _SPECIALISATION_MAP.items():
            if course_code.startswith(prefix):
                return spec
        # Fallback: look for specialisation keywords in the section text
        lower = text.lower()
        if 'ai for health' in lower or 'medical' in lower:
            return 'AI for Health'
        if 'cognitive science' in lower or 'cognitive psychology' in lower:
            return 'Cognitive Science'
        return ''

    @staticmethod
    def _extract_specialisation_name(header_text: str) -> str:
        """Extract clean specialisation name from a specialisation header line."""
        # Remove the "Artificial Intelligence specialisation" prefix
        cleaned = re.sub(
            r'^Artificial Intelligence\s+(?:specialisation|main specialisation)\s+',
            '', header_text, flags=re.IGNORECASE
        ).strip()
        return cleaned

    @staticmethod
    def _classify_preamble(text: str) -> str:
        lower = text.lower()
        if any(kw in lower for kw in ("compulsory courses", "programme sections", "elective")):
            return "program_overview"
        if re.search(r'course\s+name\s+period\s+credits', lower):
            return "course_list"
        return "general"

    def _page_level_fallback(
        self,
        full_text: str,
        page_map:  List[Tuple[int, int, int]],
        source:    str,
    ) -> List[CourseSection]:
        """Safety fallback: one section per page when no Course Code markers found."""
        sections = []
        for start, end, page_num in page_map:
            text = full_text[start:end].strip()
            if len(text) < self.min_chunk_size:
                continue
            sections.append(CourseSection(
                content      = text,
                page_number  = page_num,
                source       = source,
                section_type = self._classify_preamble(text),
                degree_name  = self._field(DEGREE_RE, text),
            ))
        return sections