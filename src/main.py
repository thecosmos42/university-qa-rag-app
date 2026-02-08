# rag_system/main.py
"""
Query Pipeline

Responsibility: Accept questions from the user, retrieve relevant context,
build a grounded prompt, generate an answer, and print it.

This is the user-facing entry point for the RAG system.  Run it after
`ingest.py` has populated the vector store.

Usage:
    python -m src.ingestion.ingestion_pipeline # Run it once
    python main.py
"""

from src.ingestion.embedder import Embedder
from src.vector_store.chroma_db_local import VectorStore
from src.rag.retriever import Retriever
from src.rag.prompt_builder import PromptBuilder
from src.llm.llm import LLM

def _format_sources(retrieved_chunks: list[dict]) -> str:
    """
    Build a human-readable source citation block from retrieved chunks.

    Deduplicates by (source, page) so the same page isn't listed twice
    even if multiple chunks came from it.

    Args:
        retrieved_chunks: The chunks that were fed into the prompt.

    Returns:
        A multi-line string like:
            📄 Sources:
               → report.pdf, page 3
               → report.pdf, page 7
    """
    seen: set[tuple[str, int]] = set()
    lines: list[str] = []

    for chunk in retrieved_chunks:
        key = (chunk.get("source", "unknown"), chunk.get("page", 0))
        if key not in seen:
            seen.add(key)
            lines.append(f"   → {key[0]}, page {key[1] + 1}")  # 1-indexed for humans

    return "📄 Sources:\n" + "\n".join(lines)


def run_query_loop() -> None:
    """Start the interactive question-answering loop."""

    # ---------------------------------------------------------------------------
    # Initialise all components once, then reuse them for every question.
    # Loading models is expensive — we do it here, not inside the loop.
    # ---------------------------------------------------------------------------
    print("Initialising RAG system...\n")

    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    llm = LLM()

    print("\n" + "=" * 60)
    print("  RAG System Ready")
    print("  Ask a question based on your ingested PDFs.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")


    while True:
        question = input("You: ").strip()

        # Exit conditions
        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        # Step 1 — Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(question)

        if not retrieved_chunks:
            print("\nAssistant: I don't have enough information to answer that question.\n")
            continue

        # Step 2 — Build the prompt
        prompt = PromptBuilder.build(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        print(retrieved_chunks)

        # Step 3 — Generate the answer
        answer = llm.generate(prompt)

        # Step 4 — Print answer + source citations
        print(f"\nAssistant: {answer}")
        print(f"\n{_format_sources(retrieved_chunks)}\n")

from fastapi import FastAPI
from src.api.query import router as query_router

# uvicorn src.main:app --reload


app = FastAPI(
    title = "University RAG backend",
    version="0.1.0"
)

@app.get('/health')
def health():
    return {'status':'ok'}

app.include_router(query_router)
