from src.llm.prompt_builder import PromptBuilder
from src.llm.local_llm import LLM
from src.llm.gemini_llm import ask_gemini
from src.config.logger import logger

class LLMWrapper:
    def __init__(self):
        pass

    async def generate_answer(self, question: str, search_results: list[dict]) -> dict:
        # TRANSFORM (Map DB keys to match what PromptBuilder expects)
        formatted_chunks = [
            {
                "text": res["content"], 
                "source": res["filename"], 
                "page": res["page_number"],
                # Program-level metadata so PromptBuilder can label sources correctly
                "doc_type": res.get("chunk_doc_type", ""),
                "program_type": res.get("chunk_program_type", ""),
                "program_name": res.get("chunk_program_name", ""),
            } for res in search_results.data
        ]

        # BUILD PROMPT
        full_prompt = PromptBuilder.build(
            question=question,
            retrieved_chunks=formatted_chunks,
            history=[])
        
        logger.info(f"The final prompt is {full_prompt}")
        
        # Returns {"text": ..., "tokens": {"prompt": N, "response": N, "total": N}}
        return await ask_gemini(full_prompt)