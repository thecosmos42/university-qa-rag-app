# tests/test_rag.py
"""
RAG System Test Suite

Three levels of testing, each targeting a different part of the pipeline:

    1. DIAGNOSTIC TOOL  (run_diagnostic)
       Prints exactly what the retriever returns for a given question —
       chunk text, source, page, and cosine distance.  Use this first
       whenever an answer is wrong: if the right chunk isn't in the output,
       the problem is retrieval.  If it is, the problem is the LLM or prompt.

    2. RETRIEVAL SMOKE TESTS  (test_retrieval)
       A small set of (question, expected_keyword) pairs.  Passes if the
       expected keyword appears in at least one of the top-k chunks.
       Catches regressions after chunking or embedding changes without
       needing to run the LLM.

    3. GOLDEN-SET END-TO-END TESTS  (test_end_to_end)
       Runs the full pipeline (retrieve → prompt → generate) for a small
       set of questions and checks that certain keywords appear in the
       answer.  Slow (LLM inference per question) but catches the full
       chain.

Usage:
    # Run the diagnostic tool on a single question:
    python -m tests.test_rag diagnostic "What is the method of assessment?"

    # Run retrieval smoke tests only (fast, no LLM):
    python -m tests.test_rag retrieval

    # Run full end-to-end golden-set tests (slow, needs LLM):
    python -m tests.test_rag e2e

    # Run everything:
    python -m tests.test_rag all
"""

import sys
from typing import List

from ingestion.embedder import Embedder
from vector_store.persist_data import VectorStore
from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder
from llm.llm import LLM


# ---------------------------------------------------------------------------
# Test data — edit these to match your actual PDFs.
# ---------------------------------------------------------------------------

# Each entry: (question, keyword that MUST appear in at least one retrieved chunk)
RETRIEVAL_SMOKE_TESTS: List[tuple[str, str]] = [
    # Replace these with question/keyword pairs drawn from your own PDFs.
    # The keyword just needs to appear somewhere in a chunk that comes back.
    ("What is the method of assessment for Evo Comp?", "assessment"),
    ("How many credits does the program offer?", "credits"),
    ("What are the admission requirements?", "admission"),
]

# Each entry: (question, keyword that MUST appear in the LLM's answer)
GOLDEN_SET_E2E_TESTS: List[tuple[str, str]] = [
    # Same idea — keyword should appear in the generated answer if the
    # pipeline is working correctly end-to-end.
    ("What is the method of assessment for Evo Comp?", "assessment"),
    ("How many credits does the program offer?", "credits"),
]


# ---------------------------------------------------------------------------
# Shared initialisation (lazy — only loads what's needed)
# ---------------------------------------------------------------------------

def _init_retrieval_components() -> tuple[Embedder, VectorStore, Retriever]:
    """Load only the components needed for retrieval tests."""
    embedder = Embedder()
    vector_store = VectorStore()
    retriever = Retriever(embedder=embedder, vector_store=vector_store)
    return embedder, vector_store, retriever


# ---------------------------------------------------------------------------
# 1. Diagnostic tool
# ---------------------------------------------------------------------------

def run_diagnostic(question: str) -> None:
    """
    Print exactly what the retriever returns for a given question.

    This is the single most useful debugging tool in the system.  Before
    blaming the LLM, check whether the right material is even being
    retrieved.

    Args:
        question: The question to diagnose.
    """
    print(f"\n{'=' * 60}")
    print(f"  DIAGNOSTIC: '{question}'")
    print(f"{'=' * 60}\n")

    _, _, retriever = _init_retrieval_components()
    chunks = retriever.retrieve(question)

    if not chunks:
        print("  ⚠ No chunks returned (all filtered by distance threshold).\n")
        print("  Possible causes:")
        print("    • DISTANCE_THRESHOLD in config.py is too low.")
        print("    • The vector store is empty — have you run ingestion?")
        print("    • The question topic has no coverage in your PDFs.\n")
        return

    for i, chunk in enumerate(chunks, start=1):
        print(f"  ─── Chunk {i} ───")
        print(f"  Source   : {chunk['source']}")
        print(f"  Page     : {chunk.get('page', '?')}")
        print(f"  Distance : {chunk['distance']:.4f}")
        print(f"  Text     : {chunk['text'][:200]}...")
        print()


# ---------------------------------------------------------------------------
# 2. Retrieval smoke tests
# ---------------------------------------------------------------------------

def test_retrieval() -> bool:
    """
    Run all RETRIEVAL_SMOKE_TESTS.

    Returns:
        True if every test passed, False if any failed.
    """
    print(f"\n{'=' * 60}")
    print("  RETRIEVAL SMOKE TESTS")
    print(f"{'=' * 60}\n")

    _, _, retriever = _init_retrieval_components()

    passed = 0
    failed = 0

    for question, expected_keyword in RETRIEVAL_SMOKE_TESTS:
        chunks = retriever.retrieve(question)

        # Check if expected_keyword appears in any retrieved chunk's text
        found = any(
            expected_keyword.lower() in chunk["text"].lower()
            for chunk in chunks
        )

        if found:
            print(f"  ✔ PASS — '{question}'")
            print(f"         keyword '{expected_keyword}' found in retrieved chunks.")
            passed += 1
        else:
            print(f"  ✘ FAIL — '{question}'")
            print(f"         keyword '{expected_keyword}' NOT found in any chunk.")
            if chunks:
                print(f"         Got {len(chunks)} chunk(s), first 100 chars of each:")
                for i, c in enumerate(chunks, 1):
                    print(f"           [{i}] {c['text'][:100]}...")
            else:
                print("         Got 0 chunks (all filtered or store empty).")
            failed += 1
        print()

    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed} tests.\n")
    return failed == 0


# ---------------------------------------------------------------------------
# 3. Golden-set end-to-end tests
# ---------------------------------------------------------------------------

def test_end_to_end() -> bool:
    """
    Run all GOLDEN_SET_E2E_TESTS through the full pipeline.

    Returns:
        True if every test passed, False if any failed.
    """
    print(f"\n{'=' * 60}")
    print("  GOLDEN-SET END-TO-END TESTS")
    print(f"{'=' * 60}\n")

    _, _, retriever = _init_retrieval_components()
    llm = LLM()

    passed = 0
    failed = 0

    for question, expected_keyword in GOLDEN_SET_E2E_TESTS:
        print(f"  Testing: '{question}'")

        chunks = retriever.retrieve(question)
        if not chunks:
            print(f"  ✘ FAIL — no chunks retrieved, cannot generate answer.")
            failed += 1
            print()
            continue

        prompt = PromptBuilder.build(
            question=question,
            retrieved_chunks=chunks,
            history=[],
        )
        answer = llm.generate(prompt)

        if expected_keyword.lower() in answer.lower():
            print(f"  ✔ PASS — keyword '{expected_keyword}' found in answer.")
            print(f"         Answer: {answer[:200]}...")
            passed += 1
        else:
            print(f"  ✘ FAIL — keyword '{expected_keyword}' NOT in answer.")
            print(f"         Answer: {answer[:200]}...")
            failed += 1
        print()

    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed} tests.\n")
    return failed == 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Route to the appropriate test mode based on the command-line argument."""
    if len(sys.argv) < 2:
        print("Usage:")
        print('  python -m tests.test_rag diagnostic "your question here"')
        print("  python -m tests.test_rag retrieval")
        print("  python -m tests.test_rag e2e")
        print("  python -m tests.test_rag all")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "diagnostic":
        if len(sys.argv) < 3:
            print("Error: diagnostic mode requires a question as the second argument.")
            print('  Example: python -m tests.test_rag diagnostic "What is assessment?"')
            sys.exit(1)
        run_diagnostic(" ".join(sys.argv[2:]))

    elif mode == "retrieval":
        success = test_retrieval()
        sys.exit(0 if success else 1)

    elif mode == "e2e":
        success = test_end_to_end()
        sys.exit(0 if success else 1)

    elif mode == "all":
        r1 = test_retrieval()
        r2 = test_end_to_end()
        sys.exit(0 if (r1 and r2) else 1)

    else:
        print(f"Unknown mode: '{mode}'. Use: diagnostic | retrieval | e2e | all")
        sys.exit(1)


if __name__ == "__main__":
    main()