import asyncio

from eval.judge import judge_answer
from eval.metrics import (
    aggregate, citation_metrics, fact_coverage, is_abstention, percentile, retrieval_metrics,
)
from src.api.citations import parse_cited_indices

ITEM = {
    "id": "x", "gold_files": ["A.pdf"], "gold_pages": [5], "gold_snippets": ["eight weeks"],
    "answerable": True,
}


def chunk(filename="A.pdf", page=5, content="the first two periods consist of eight weeks"):
    return {"filename": filename, "page_number": page, "content": content}


def test_parse_cited_indices_grouped_and_pages():
    assert parse_cited_indices("a [SOURCE 1] b [SOURCE 2, SOURCE 4] c [SOURCE 3, Page 9]") == {1, 2, 3, 4}
    assert parse_cited_indices("no citations") == set()


def test_retrieval_metrics_rank_and_mrr():
    chunks = [chunk("B.pdf"), chunk("A.pdf", page=20, content="unrelated"), chunk()]
    m = retrieval_metrics(chunks, ITEM)
    assert m["first_relevant_rank"] == 3
    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0
    assert m["mrr"] == 1 / 3


def test_retrieval_without_snippets_uses_page_tolerance():
    item = {"gold_files": ["A.pdf"], "gold_pages": [5]}
    assert retrieval_metrics([chunk(page=6, content="x")], item)["recall@1"] == 1.0
    assert retrieval_metrics([chunk(page=8, content="x")], item)["recall@1"] == 0.0


def test_fact_coverage_alternatives():
    assert fact_coverage("It is 15 days and Eight Weeks", [["fifteen", "15"], "eight weeks"]) == 1.0
    assert fact_coverage("15", [["fifteen", "15"], "eight weeks"]) == 0.5
    assert fact_coverage("anything", []) is None


def test_abstention_and_citation_metrics():
    assert is_abstention("I don't have enough information to answer that.")
    flags = [True, False]
    m = citation_metrics("x [SOURCE 1][SOURCE 2][SOURCE 9]", {1, 2, 9}, [chunk(), chunk()], flags)
    assert m["valid_rate"] == 2 / 3 and m["precision"] == 0.5
    assert citation_metrics("none", set(), [chunk()], [True])["has_citation"] is False


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 95) is None


def test_aggregate_latency_tokens_and_split():
    ans = {
        "item": ITEM, "retrieval": retrieval_metrics([chunk()], ITEM), "answer": "eight weeks [SOURCE 1]",
        "fact_coverage": 1.0, "citation": citation_metrics("[SOURCE 1]", {1}, [chunk()], [True]),
        "latency": {"embed_s": 0.1, "search_s": 0.2, "llm_s": 1.0, "total_s": 1.3},
        "tokens": {"prompt": 100, "response": 10, "total": 110}, "judge_tokens": 50,
    }
    un_item = {"id": "u", "answerable": False}
    un = {
        "item": un_item, "retrieval": retrieval_metrics([], un_item),
        "answer": "I don't have enough information to answer that.",
        "latency": {"embed_s": 0.1, "search_s": 0.2, "llm_s": 0.5, "total_s": 0.8},
        "tokens": {"prompt": 80, "response": 8, "total": 88},
        "citation": citation_metrics("", set(), [], []),
    }
    s = aggregate([ans, un])
    assert s["recall@1"] == 1.0 and s["abstention_accuracy"] == 1.0
    assert s["tokens_total_sum"] == 198 and s["judge_tokens_sum"] == 50
    assert abs(s["latency_total_s_mean"] - 1.05) < 1e-9


def test_judge_parses_json_and_scores_faithfulness():
    async def ask(prompt):
        assert "[SOURCE 1]" in prompt
        return {"text": 'ok ```{"correct": true, "claims_total": 4, "claims_supported": 3, "reason": "r"}```',
                "tokens": {"total": 77}}

    r = asyncio.run(judge_answer(ask, "q", "ref", "ans", [{"index": 1, "content": "text"}]))
    assert r["correct"] is True and r["faithfulness"] == 0.75 and r["tokens"] == 77
