# rag_system/main.py
"""
Query Pipeline

Accept questions from the user, retrieve relevant context,
build a grounded prompt, generate an answer, and print it.

This is the user-facing entry point for the RAG system.  Run it after
`ingest.py` has populated the vector store.

Usage:
    python -m src.ingestion.ingestion_pipeline # Run it once
    uvicorn src.main:app --reload
"""
from fastapi import FastAPI
from src.api.query import router as query_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title = "University RAG backend",
    version="0.1.0"
)

# Add CORS middleware BEFORE your routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

@app.get('/health')
def health():
    return {'status':'ok'}

app.include_router(query_router)
