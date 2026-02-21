import os
from typing import List, Optional
from google import genai
from google.genai import types
from src.config.config import GOOGLE_API_KEYS
import time
from google.genai.errors import ClientError

class Embedder:
    """Wraps Google Gemini API to produce text embeddings."""

    def __init__(self, model_name: str = "gemini-embedding-001"):
        """
        Initialize one Gemini client per API key for quota rotation.
        
        Args:
            model_name: The Gemini embedding model identifier.
                        Defaults to 'gemini-embedding-001'.
        """
        print(f"[Embedder] Initializing {len(GOOGLE_API_KEYS)} Gemini client(s) with model '{model_name}'...")
        self._clients = [genai.Client(api_key=k) for k in GOOGLE_API_KEYS]
        self._key_index = 0
        self.model_name = model_name
        print("[Embedder] Client(s) initialized.")

    @property
    def client(self) -> genai.Client:
        return self._clients[self._key_index]

    def _rotate_key(self) -> bool:
        """Advance to the next key. Returns False if all keys are exhausted."""
        next_index = (self._key_index + 1) % len(self._clients)
        if next_index == self._key_index:
            return False  # only one key available
        self._key_index = next_index
        print(f"  [Embedder] Rotated to API key {self._key_index + 1}/{len(self._clients)}")
        return True

    def encode(self, texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
        if not texts:
            return []

        all_embeddings = []
        batch_size = 20
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            
            max_retries = len(self._clients) * 3  # retries across all keys
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
                    
                    print(f"  Batch {batch_num}/{total_batches} done ({len(batch)} texts)")
                    if batch_num < total_batches:
                        time.sleep(8)
                    break 
                    
                except ClientError as e:
                    if "429" in str(e):
                        # Try rotating to another key first
                        if self._rotate_key():
                            print(f"  Retrying batch {batch_num} with new key…")
                            continue
                        # All keys tried — fall back to timed wait for RPM limits
                        if attempt < max_retries - 1:
                            wait_time = 60
                            print(f"  All keys exhausted. Waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                            time.sleep(wait_time)
                        else:
                            raise
                    else:
                        raise

        return all_embeddings