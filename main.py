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
from src.vector_store.persist_data import VectorStore
from src.rag.retriever import Retriever
from src.rag.prompt_builder import PromptBuilder
from src.llm.llm import LLM


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

        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    run_query_loop()