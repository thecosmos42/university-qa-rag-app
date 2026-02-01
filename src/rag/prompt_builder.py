# src/rag/prompt_builder.py
"""
Prompt Builder Module

Responsibility: Assemble a final prompt string from retrieved context chunks,
conversation history, and the user's question.

Why a dedicated module?
    The prompt template is the single biggest lever for controlling LLM
    behaviour.  Keeping it in its own file means it can be edited, versioned,
    or swapped independently of everything else in the pipeline.
"""

from typing import List


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# Key design decisions in this template:
#   1. "Answer ONLY ... then stop."  — prevents the model from continuing
#      past a complete answer and re-entering instruction-following mode,
#      which was causing the contradictory disclaimer suffix.
#   2. {history} block — gives the model previous turns so it can resolve
#      pronouns and references like "this" or "that course".
#   3. "Cite the source"  — instructs the model to attribute its answer,
#      so the user sees which document and excerpt the answer came from.
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions strictly
based on the provided context. If the answer is not found in the context,
say exactly: "I don't have enough information to answer that question."
Answer ONLY the question asked, cite the source excerpt(s) you used, then stop.
Do not add any extra commentary after your answer.

{history_block}
Context:
{context}

Question:
{question}

Answer (cite the source excerpt number):"""


class PromptBuilder:
    """Builds a complete prompt from retrieved chunks, history, and a question."""

    @staticmethod
    def build(
        question: str,
        retrieved_chunks: List[dict],
        history: List[dict] | None = None,
    ) -> str:
        """
        Construct the final prompt string.

        Args:
            question:          The user's current question.
            retrieved_chunks:  List of chunk dicts as returned by Retriever.
                               Each must have "text" and "source" keys; "page"
                               is included if available.
            history:           Optional list of previous conversation turns.
                               Each entry is {"question": str, "answer": str}.
                               Most recent turn last.  Pass None or [] if this
                               is the first question.

        Returns:
            A fully-formed prompt string ready to be fed to the LLM.
        """
        # --- context block: numbered excerpts with full provenance --------
        context_parts: List[str] = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.get("source", "unknown")
            page = chunk.get("page", "?")
            context_parts.append(
                f"[Excerpt {i} — {source}, page {page}]\n{chunk['text']}"
            )
        context = "\n\n".join(context_parts)

        # --- history block: recent turns so the LLM can resolve refs ------
        history_block = ""
        if history:
            lines: List[str] = ["Previous conversation:"]
            for turn in history:
                lines.append(f"  User: {turn['question']}")
                lines.append(f"  Assistant: {turn['answer']}")
            history_block = "\n".join(lines) + "\n\n"

        prompt = PROMPT_TEMPLATE.format(
            history_block=history_block,
            context=context,
            question=question,
        )
        return prompt