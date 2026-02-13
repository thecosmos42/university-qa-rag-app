from src.rag.prompt_builder import PromptBuilder
from src.llm.local_llm import LLM
from src.llm.gemini_llm import client, ask_gemini
from src.config.logger import logger

# Local
# llm = LLM()



class RAGGenerator:
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name

    def format_context(self, search_results: list[dict]) -> str:
        """Converts DB results into a formatted string for the LLM."""
        context_blocks = []
        for i, res in enumerate(search_results):
            # removing \n
            clean_text = " ".join(res["content"].split())
            block = f"Source {i+1} [{res['filename']}, Page {res['page_number']}]: {clean_text}"
            context_blocks.append(block)
        
        return "\n\n".join(context_blocks)

    def generate_answer(self, question: str, search_results: list[dict]) -> dict:
        # context = self.format_context(search_results)
        
        # TRANSFORM (Map DB keys to match what PromptBuilder expects)
        formatted_chunks = [
            {
                "text": res["content"], 
                "source": res["filename"], 
                "page": res["page_number"]
            } for res in search_results.data
        ]

        # 3. BUILD PROMPT
        # Note: For now, you can pass history=[] until you implement session logic
        full_prompt = PromptBuilder.build(
            question=question,
            retrieved_chunks=formatted_chunks,
            history=[])
        
        logger.info(f"The final prompt is {full_prompt}")
        
        # llm_response = llm.generate(full_prompt)
        llm_response = ask_gemini(client, full_prompt)

        return llm_response