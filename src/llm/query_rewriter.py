# src/rag/query_rewriter.py
"""
Query Rewriter Module

Expands user queries before embedding and BM25 search to handle:
1. Synonym expansion: "mandatory" → also search for "compulsory", "obligatory"
2. Abbreviation expansion: "NLP" → "NLP OR Natural Language Processing"

This compensates for BM25's inability to match synonyms (different stems).
Program/degree detection is handled separately by ProgramExtractor.
"""

import re
from dataclasses import dataclass


@dataclass
class RewrittenQuery:
    """Result of query analysis and rewriting."""
    original_query: str
    rewritten_query: str           # Enhanced query for BM25/embedding


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

class QueryRewriter:
    """Analyzes and rewrites user queries for better retrieval."""
    
    def rewrite(self, query: str) -> RewrittenQuery:
        """
        Analyze a user query and produce an enhanced version.
        Expands synonyms and abbreviations for better BM25 recall.

        Args:
            query: The raw user question.

        Returns:
            RewrittenQuery with enhanced text.
        """
        result = RewrittenQuery(
            original_query=query,
            rewritten_query=query,
        )

        # Expand synonyms
        expanded = self._expand_synonyms(query)

        # Expand abbreviations (append full form if abbreviation found)
        expanded = self._expand_abbreviations(expanded)

        result.rewritten_query = expanded
        return result
    
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
