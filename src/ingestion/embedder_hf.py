import os
from typing import List
from huggingface_hub import InferenceClient

class Embedder:
    """
    Uses Hugging Face InferenceClient to produce text embeddings with Qwen3-Embedding-0.6B.
    """
    def __init__(self, model_id: str = "intfloat/multilingual-e5-large-instruct", hf_token: str = None):
        self.model_id = model_id
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.hf_token,
        )
        if not self.hf_token:
            raise ValueError("Hugging Face token (HF_TOKEN) is required.")

    def encode(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        import numpy as np
        embeddings = []
        for text in texts:
            result = self.client.feature_extraction(
                text,
                model=self.model_id,
            )
            # Convert numpy arrays to lists if needed
            if isinstance(result, np.ndarray):
                result = result.tolist()
            embeddings.append(result)
        return embeddings

    def save_embeddings(self, embeddings, filename="embeddings_backup.json"):
        import json
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(embeddings, f)

    def load_embeddings(self, filename="embeddings_backup.json"):
        import json
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
