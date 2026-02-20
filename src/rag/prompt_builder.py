# src/rag/prompt_builder.py
from src.config.config import ENABLE_HISTORY

PROMPT_TEMPLATE = """You are a helpful University Assistant. Your goal is to answer questions strictly using the provided SOURCES.

### RULES:
1. If the answer is not in the SOURCES, say: "I don't have enough information to answer that."
2. ALWAYS cite the source number and Page number(if present) in brackets for every claim you make, e.g., "The exam is on Friday [SOURCE 1]."
3. If multiple sources support a claim, list them: [SOURCE 1, SOURCE 2].
4. Answer concisely and do not add information from outside the sources.

{history_block}
### SOURCES:
{context}

---
Question: {question}
Answer:"""

class PromptBuilder:
    @staticmethod
    def build(question: str, retrieved_chunks: list[dict], history: list[dict] | None = None) -> str:
        # Format and Number the Context
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            # Clean text: Remove extra newlines and trim whitespace
            clean_text = " ".join(chunk['text'].split())
            
            # Create a clear labeled block
            header = f"SOURCE {i} (File: {chunk['source']}, Page: {chunk['page']})"
            context_parts.append(f"[{header}]\n{clean_text}")
        
        context_str = "\n\n".join(context_parts)

        # 2. Handle History Toggle
        history_block = ""
        if ENABLE_HISTORY and history:
            lines = ["### PREVIOUS CONVERSATION:"]
            for turn in history[-3:]:  # Limit to last 3 turns for performance
                lines.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
            history_block = "\n".join(lines) + "\n\n"

        # 3. Assemble Final Prompt
        return PROMPT_TEMPLATE.format(
            history_block=history_block,
            context=context_str,
            question=question
        )