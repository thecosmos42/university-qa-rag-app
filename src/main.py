# src/main.py
"""
Query Pipeline

Accept questions from the user, retrieve relevant context,
build a grounded prompt, generate an answer, and print it.

This is the user-facing entry point for the RAG system.  Run it after
the ingestion pipeline has populated the vector store.

Usage:
    python -m src.ingestion.ingestion_pipeline  # Run once to ingest
    uvicorn main:app --reload
"""
import os
import uvicorn
from fastapi import FastAPI
from src.api.query import router as query_router
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

# Read allowed origins from env — comma-separated list.
# Example .env entry:  ALLOWED_ORIGINS=https://your-app.vercel.app,http://localhost:3000
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app = FastAPI(
    title="University RAG backend",
    version="0.1.0"
)

# Add CORS middleware BEFORE your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

@app.get('/health')
def health():
    return {'status':'ok'}

app.include_router(query_router)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
