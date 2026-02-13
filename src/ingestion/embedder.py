import os
from typing import List, Optional
from google import genai
from google.genai import types
from src.config.config import GOOGLE_API_KEY, EMBEDDING_MODEL_NAME
import time
from google.genai.errors import ClientError

class Embedder:
    """Wraps Google Gemini API to produce text embeddings."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        """
        Initialize the Gemini GenAI client.
        
        Args:
            model_name: The Gemini embedding model identifier.
                        Defaults to 'gemini-embedding-001'.
        """
        print(f"[Embedder] Initializing Gemini client with model '{model_name}'...")
        # api_key can be passed explicitly or read from environment variable
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model_name
        print("[Embedder] Client initialized.")

    def encode(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        if not texts:
            return []

        all_embeddings = []
        batch_size = 100 
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            
            # Retry logic for the 429 error
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=768
                            )
                    )
                    all_embeddings.extend([e.values for e in response.embeddings])
                    
                    # IMPORTANT: Wait 2-5 seconds between batches to stay under the 100 RPM limit
                    print(f"Batch {i//batch_size + 1} successful. Cooling down...")
                    time.sleep(2) 
                    break 
                    
                except ClientError as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = 30  # If hit, wait 30s as suggested by the error message
                        print(f"Quota exceeded. Sleeping for {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise e

        return all_embeddings