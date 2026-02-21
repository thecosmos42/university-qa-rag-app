# Root-level entry point
# Allows running: uvicorn main:app --reload
# from the rag-app/ directory without the src. prefix.
from src.main import app  # noqa: F401
