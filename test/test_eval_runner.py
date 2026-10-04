"""End-to-end check of the benchmark runner with the DB / LLM stubbed out."""
import argparse
import asyncio
import sys
import types
from types import SimpleNamespace

from eval.run_benchmark import evaluate_item, load_dataset, DEFAULT_DATASET
from eval.metrics import aggregate


def _fake_query_module(chunks):
    mod = types.ModuleType("src.api.query")
    mod.retrieve_chunks = lambda question, limit=10, **kw: (
        SimpleNamespace(data=chunks), None, {"embed_s": 0.01, "search_s": 0.02})

    async def generate_answer(question, db_results):
        return {"text": "Eight weeks [SOURCE 1].", "tokens": {"prompt": 120, "response": 9, "total": 129}}

    mod.generator = SimpleNamespace(generate_answer=generate_answer)
    return mod


def test_runner_collects_metrics_latency_and_tokens(monkeypatch):
    chunks = [{"filename": "TER_Masters_ArtificialIntelligence.pdf", "page_number": 5,
               "content": "consist of eight weeks, and the final period consists of four weeks",
               "chunk_program_name": "artificial_intelligence"}]
    monkeypatch.setitem(sys.modules, "src.api.query", _fake_query_module(chunks))
    import src.api
    monkeypatch.setattr(src.api, "query", sys.modules["src.api.query"], raising=False)

    item = next(i for i in load_dataset(DEFAULT_DATASET) if i["id"] == "ai-ter-periods")
    args = argparse.Namespace(generate=True, judge=False, k=10)
    rec = asyncio.run(evaluate_item(item, args, None))

    assert rec["retrieval"]["recall@1"] == 1.0
    assert rec["fact_coverage"] == 0.5  # "four weeks" missing from the stub answer
    assert rec["citation"]["precision"] == 1.0
    assert rec["tokens"]["total"] == 129
    assert {"embed_s", "search_s", "llm_s", "total_s"} <= rec["latency"].keys()
    assert aggregate([rec])["tokens_prompt_sum"] == 120


def test_dataset_is_well_formed():
    items = load_dataset(DEFAULT_DATASET)
    assert len(items) >= 20
    for it in items:
        if it.get("answerable", True):
            assert it["gold_files"] and it["gold_pages"] and it["gold_snippets"] and it["reference_answer"]
