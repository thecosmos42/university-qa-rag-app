# rag_system/llm.py
"""
LLM Module

Responsibility: Load a local Hugging Face model and generate text from a
prompt.

This module is the only place that knows about the transformers library.
Swapping to a different inference backend (e.g. llama.cpp, vLLM) only
requires changes here.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.config import LLM_MODEL_NAME, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE


class LLM:
    """Thin wrapper around a Hugging Face causal language model."""

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        """
        Download (if needed) and load the model and tokenizer into memory.

        Args:
            model_name: A Hugging Face model identifier.
                        Defaults to the value in config.py.

        Note:
            On the first run this will download the model weights to the
            local HF cache (~5 GB for Gemma 2B).  Subsequent runs load from
            cache and are fast.
        """
        print(f"[LLM] Loading model '{model_name}' — this may take a minute...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use float16 on GPU for speed; fall back to float32 on CPU.
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype
        )

        self.model = self.model.to(device)

        self.device = device
        print(f"[LLM] Model loaded on {self.device}.")



    def generate(self, prompt: str) -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The full prompt string (as produced by PromptBuilder).

        Returns:
            The model's generated text with the original prompt stripped
            from the front so only the new answer is returned.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                temperature=LLM_TEMPERATURE,
                do_sample=LLM_TEMPERATURE > 0,  # greedy when temp == 0
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude the prompt).
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return answer.strip()