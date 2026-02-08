"""
Central configuration module.

All tunable parameters for the RAG pipeline live here.
Every other module imports what it needs from this file
so that changing behaviour never requires hunting through code.
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Directory where user PDF files are placed for ingestion.
PDF_INPUT_DIR: Path = Path("./data/pdfs")

# Root directory where ChromaDB will persist its vector store on disk.
CHROMA_PERSIST_DIR: Path = Path("./src/db")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Number of characters per chunk (before overlap).
CHUNK_SIZE: int = 800

# Number of overlapping characters between consecutive chunks.
# Overlap ensures that context at chunk boundaries is not lost during retrieval.
CHUNK_OVERLAP: int = 100

# Ordered list of separator strings that RecursiveCharacterTextSplitter tries
# before falling back to the next one.  It attempts to split on double-newlines
# (paragraphs) first, then single newlines, then spaces, then individual
# characters.  Keeping natural boundaries intact improves retrieval quality.
CHUNK_SEPARATORS: list[str] = ["\n\n", "\n", " ", ""]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Sentence-Transformers model name used to encode text chunks into vectors.
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Vector Store (ChromaDB)
# ---------------------------------------------------------------------------
# Name of the ChromaDB collection that stores embedded chunks.
CHROMA_COLLECTION_NAME: str = "ms_ai_program"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# How many chunks to retrieve for a single query.
TOP_K_RESULTS: int = 5


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
# Hugging Face model identifier.  Must fit in local VRAM / RAM.
# "google/gemma-2b-it" is a good default; swap for any instruction-tuned model.
LLM_MODEL_NAME: str = "google/gemma-2b-it"

# Maximum number of new tokens the LLM is allowed to generate per answer.
LLM_MAX_NEW_TOKENS: int = 512

# Sampling temperature.  0 = greedy (deterministic); higher = more creative.
LLM_TEMPERATURE: float = 0.1