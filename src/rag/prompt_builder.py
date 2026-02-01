# rag_system/prompt_builder.py
"""
Prompt Builder Module

Responsibility: Assemble a final prompt string from retrieved context chunks
and the user's question.

Why a dedicated module?
    The prompt template is the single biggest lever for controlling LLM
    behaviour.  Keeping it in its own file means it can be edited, versioned,
    or swapped independently of everything else in the pipeline.
"""

from typing import List


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# The template uses a strict instruction to constrain the LLM to the
# provided context only.  This is the primary mechanism that prevents
# hallucination or drift outside the PDF content.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly
based on the provided context. If the answer is not contained in the context,
say "I don't have enough information to answer that question."

Context:
{context}

Question:
{question}

Answer:"""


class PromptBuilder:
    """Builds a complete prompt from retrieved chunks and a user question."""

    @staticmethod
    def build(question: str, retrieved_chunks: List[dict]) -> str:
        """
        Construct the final prompt string.

        Args:
            question:          The user's question.
            retrieved_chunks:  List of chunk dicts as returned by Retriever.
                               Each must have a "text" key.

        Returns:
            A fully-formed prompt string ready to be fed to the LLM.
        """
        # Concatenate chunk texts, labelled by source so the LLM (and any
        # human reader of the prompt) can see where each piece came from.
        context_parts: List[str] = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "unknown")
            context_parts.append(f"[Source: {source}, Excerpt {i}]\n{chunk['text']}")

        context = "\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        return prompt