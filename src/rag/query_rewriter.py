# src/rag/query_rewriter.py
"""
Query Rewriter Module

Expands user queries before embedding and BM25 search to handle:
1. Synonym expansion: "mandatory" → also search for "compulsory", "obligatory"
2. Abbreviation expansion: "NLP" → "NLP OR Natural Language Processing"
3. Metadata extraction: Detects course names, codes, and degree programs
   from the query to enable metadata-first filtering.

This compensates for BM25's inability to match synonyms (different stems)
and improves precision by extracting structured filters from natural language.
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RewrittenQuery:
    """Result of query analysis and rewriting."""
    original_query: str
    rewritten_query: str           # Enhanced query for BM25/embedding
    detected_course_name: Optional[str] = None
    detected_course_code: Optional[str] = None
    detected_degree_name: Optional[str] = None


# Synonym groups — each tuple is (canonical, [synonyms]) 
# BM25 uses stemming, so we need semantically equivalent terms with DIFFERENT stems
SYNONYM_MAP = {
    "mandatory": ["compulsory", "obligatory", "required"],
    "compulsory": ["mandatory", "obligatory", "required"],
    "obligatory": ["mandatory", "compulsory", "required"],
    "required": ["mandatory", "compulsory", "obligatory"],
    "teacher": ["instructor", "professor", "lecturer", "teaching staff", "coordinator"],
    "instructor": ["teacher", "professor", "lecturer", "teaching staff", "coordinator"],
    "professor": ["teacher", "instructor", "lecturer", "teaching staff", "coordinator"],
    "lecturer": ["teacher", "instructor", "professor", "teaching staff", "coordinator"],
    "credits": ["EC", "ECTS", "credit points"],
    "exam": ["assessment", "examination", "test"],
    "prerequisite": ["background knowledge", "prior knowledge", "recommended background"],
    "elective": ["optional", "choice"],
    "optional": ["elective", "choice"],
    "schedule": ["period", "timetable", "when"],
}

# Known abbreviations -> full forms
ABBREVIATION_MAP = {
    "NLP": "Natural Language Processing",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "DL": "Deep Learning",
    "RL": "Reinforcement Learning",
    "CV": "Computer Vision",
    "HCI": "Human Computer Interaction",
    "MAS": "Multi-Agent Systems",
    "KR": "Knowledge Representation",
}

# Known course names for exact matching
KNOWN_COURSES = [
    "Advanced NLP",
    "Natural Language Processing",
    "Deep Learning",
    "Evolutionary Computing",
    "Knowledge Representation",
    "Multi-Agent Systems",
    "AI and Society",
    "Experimental Design and Data Analysis",
    "Machine Learning for NLP",
    "Applied Text Mining",
    "Advanced Machine Learning",
    "Conversational AI and Robots",
    "Social Robotics",
    "Human-interactive Agent Learning",
    "Subjectivity Mining",
    "Language as Data",
    "Advanced Logic",
    "AI for Medical Imaging",
    "AI for Security",
    "Reinforcement Learning",
    "Computational Intelligence",
    "Computer Vision",
]

# Known degree programs
KNOWN_DEGREES = [
    "Artificial Intelligence",
    "Computer Science",
    "Business Analytics",
    "Mathematics",
    "Bioinformatics",
]

# Course code pattern
COURSE_CODE_RE = re.compile(r'\b([A-Z][A-Z_]*\d+[A-Z_\d]*)\b')


class QueryRewriter:
    """Analyzes and rewrites user queries for better retrieval."""
    
    def rewrite(self, query: str) -> RewrittenQuery:
        """
        Analyze a user query and produce an enhanced version.
        
        Steps:
        1. Detect any course name, code, or degree mentioned
        2. Expand synonyms for BM25-unfriendly terms
        3. Expand abbreviations
        
        Args:
            query: The raw user question.
            
        Returns:
            RewrittenQuery with enhanced text and extracted metadata.
        """
        result = RewrittenQuery(
            original_query=query,
            rewritten_query=query,
        )
        
        # Detect structured entities
        result.detected_course_name = self._detect_course_name(query)
        result.detected_course_code = self._detect_course_code(query)
        result.detected_degree_name = self._detect_degree_name(query)
        
        # Expand synonyms
        expanded = self._expand_synonyms(query)
        
        # Expand abbreviations (append full form if abbreviation found)
        expanded = self._expand_abbreviations(expanded)
        
        result.rewritten_query = expanded
        return result
    
    def _detect_course_name(self, query: str) -> Optional[str]:
        """Check if the query mentions a known course name."""
        query_lower = query.lower()
        
        # Sort by length descending to match longest first
        # e.g., "Advanced NLP" before "NLP"
        for course in sorted(KNOWN_COURSES, key=len, reverse=True):
            if course.lower() in query_lower:
                return course
        
        return None
    
    def _detect_course_code(self, query: str) -> Optional[str]:
        """Check if the query contains a course code pattern."""
        match = COURSE_CODE_RE.search(query)
        if match:
            return match.group(1)
        return None
    
    def _detect_degree_name(self, query: str) -> Optional[str]:
        """Check if the query mentions a degree program."""
        query_lower = query.lower()
        
        # Check for explicit degree mentions
        degree_patterns = [
            r"master(?:'?s?)?\s+(?:of\s+|in\s+)?(\w[\w\s]+)",
            r"bachelor(?:'?s?)?\s+(?:of\s+|in\s+)?(\w[\w\s]+)",
            r"(MSc|BSc|MA|BA)\s+(?:in\s+)?(\w[\w\s]+)",
        ]
        
        for pattern in degree_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Check if it matches a known degree
                potential = match.group(1).strip() if match.lastindex == 1 else match.group(2).strip()
                for degree in KNOWN_DEGREES:
                    if degree.lower() in potential.lower() or potential.lower() in degree.lower():
                        return degree
        
        return None
    
    def _expand_synonyms(self, query: str) -> str:
        """
        Expand synonyms in the query.
        E.g., "mandatory courses" → "mandatory compulsory obligatory required courses"
        """
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            clean_word = word.lower().strip("?.,!;:")
            if clean_word in SYNONYM_MAP:
                # Add synonyms (without duplicating the original)
                for synonym in SYNONYM_MAP[clean_word]:
                    if synonym.lower() not in query.lower():
                        expanded_words.append(synonym)
        
        return " ".join(expanded_words)
    
    def _expand_abbreviations(self, query: str) -> str:
        """
        If the query contains an abbreviation, append the full form.
        E.g., "Who teaches NLP?" → "Who teaches NLP Natural Language Processing?"
        """
        additions = []
        
        for abbrev, full_form in ABBREVIATION_MAP.items():
            # Match as whole word
            if re.search(r'\b' + re.escape(abbrev) + r'\b', query):
                if full_form.lower() not in query.lower():
                    additions.append(full_form)
        
        if additions:
            return query + " " + " ".join(additions)
        
        return query
