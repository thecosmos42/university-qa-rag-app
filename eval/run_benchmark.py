"""Run the labeled RAG benchmark.

    python -m eval.run_benchmark                      # retrieval only (no LLM calls)
    python -m eval.run_benchmark --generate           # + answers, latency, token usage
    python -m eval.run_benchmark --generate --judge   # + LLM-judged correctness/faithfulness
    python -m eval.run_benchmark --limit 5 --category admission

Requires the same .env as the API (Supabase + Gemini + embedding backend).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from src.api.citations import parse_cited_indices
from eval.metrics import (
    aggregate, by_category, citation_metrics, fact_coverage, retrieval_metrics,
)

EVAL_DIR = Path(__file__).parent
DEFAULT_DATASET = EVAL_DIR / "dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"


def load_dataset(path: Path) -> list[dict]:
    items = [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in dataset")
    return items


def _chunks_of(db_results) -> list[dict]:
    return list(db_results.data or [])


async def evaluate_item(item: dict, args, ask_judge) -> dict:
    # Imported lazily so metrics/dataset tooling works without DB credentials.
    from src.api import query as q

    t0 = time.perf_counter()
    db_results, _prog, timings = q.retrieve_chunks(item["question"], limit=args.k)
    chunks = _chunks_of(db_results)
    rec: dict = {
        "item": item,
        "retrieval": retrieval_metrics(chunks, item),
        "retrieved": [
            {"rank": i + 1, "filename": c["filename"], "page": c.get("page_number"),
             "program": c.get("chunk_program_name"), "snippet": c["content"][:160]}
            for i, c in enumerate(chunks)
        ],
        "answer": None,
    }
    latency = {"embed_s": timings["embed_s"], "search_s": timings["search_s"]}

    if args.generate:
        t_llm = time.perf_counter()
        llm = await q.generator.generate_answer(item["question"], db_results)
        latency["llm_s"] = time.perf_counter() - t_llm
        answer = llm["text"] or ""
        rec["answer"] = answer
        rec["tokens"] = llm["tokens"]
        cited = parse_cited_indices(answer)
        rec["cited"] = sorted(cited)
        rec["fact_coverage"] = fact_coverage(answer, item.get("must_contain", []))
        rec["citation"] = citation_metrics(answer, cited, chunks, rec["retrieval"]["relevant_flags"])

    # Pipeline latency ends here; judging below is evaluation overhead, not system latency.
    latency["total_s"] = time.perf_counter() - t0
    rec["latency"] = latency

    if args.generate:
        if args.judge and item.get("answerable", True) and item.get("reference_answer"):
            cited_chunks = [
                {"index": i, "content": chunks[i - 1]["content"]} for i in sorted(cited) if 1 <= i <= len(chunks)
            ]
            try:
                j = await _judge(ask_judge, item, answer, cited_chunks)
                rec.update(judge_correct=j["correct"], faithfulness=j["faithfulness"],
                           judge_reason=j["reason"], judge_tokens=j["tokens"])
            except Exception as e:  # keep the run alive; record the failure
                rec["judge_error"] = str(e)

    return rec


async def _judge(ask_judge, item, answer, cited_chunks):
    from eval.judge import judge_answer
    return await judge_answer(ask_judge, item["question"], item["reference_answer"], answer, cited_chunks)


def _fmt(v, pct=False):
    if v is None:
        return "  n/a"
    return f"{v*100:5.1f}%" if pct else f"{v:7.2f}"


def print_summary(s: dict) -> None:
    print(f"\nItems: {s['n_items']} ({s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable)")
    print("\nRetrieval (answerable items)")
    for k in ("recall@1", "recall@3", "recall@5", "recall@10"):
        print(f"  {k:<22}{_fmt(s[k], True)}")
    print(f"  {'MRR':<22}{_fmt(s['mrr'])}")
    if "fact_coverage" in s:
        print("\nAnswer correctness")
        print(f"  {'fact coverage':<22}{_fmt(s['fact_coverage'], True)}")
        print(f"  {'judge correct':<22}{_fmt(s.get('judge_correct_rate'), True)}")
        print(f"  {'wrongly abstained':<22}{_fmt(s['abstained_on_answerable'], True)}")
        print(f"  {'abstention accuracy':<22}{_fmt(s.get('abstention_accuracy'), True)}")
        print("\nCitations")
        print(f"  {'present':<22}{_fmt(s['citation_present_rate'], True)}")
        print(f"  {'valid index':<22}{_fmt(s['citation_valid_rate'], True)}")
        print(f"  {'precision vs gold':<22}{_fmt(s['citation_precision'], True)}")
        print(f"  {'faithfulness (judge)':<22}{_fmt(s.get('faithfulness'), True)}")
        print("\nLatency (seconds)            mean     p50     p95")
        for stage in ("embed_s", "search_s", "llm_s", "total_s"):
            print(f"  {stage:<22}{_fmt(s[f'latency_{stage}_mean'])}{_fmt(s[f'latency_{stage}_p50'])}{_fmt(s[f'latency_{stage}_p95'])}")
        print("\nTokens (pipeline)           mean     total")
        for kind in ("prompt", "response", "total"):
            tot = s[f"tokens_{kind}_sum"]
            print(f"  {kind:<22}{_fmt(s[f'tokens_{kind}_mean'])}{(tot if tot is not None else 0):>9}")
        if s.get("judge_tokens_sum"):
            print(f"  judge tokens (separate) {s['judge_tokens_sum']:>8}")
    else:
        print("\nLatency (seconds)            mean     p50     p95")
        for stage in ("embed_s", "search_s", "total_s"):
            print(f"  {stage:<22}{_fmt(s[f'latency_{stage}_mean'])}{_fmt(s[f'latency_{stage}_p50'])}{_fmt(s[f'latency_{stage}_p95'])}")


async def main_async(args) -> None:
    items = load_dataset(Path(args.dataset))
    if args.category:
        items = [i for i in items if i.get("category") == args.category]
    if args.limit:
        items = items[: args.limit]

    from src.api import query as q
    q._init_resources()
    ask_judge = None
    if args.judge:
        from src.llm.gemini_llm import ask_gemini
        ask_judge = ask_gemini

    results = []
    for n, item in enumerate(items, 1):
        r = await evaluate_item(item, args, ask_judge)
        results.append(r)
        print(f"[{n}/{len(items)}] {item['id']}: first_relevant_rank={r['retrieval']['first_relevant_rank']}"
              + (f" total={r['latency']['total_s']:.1f}s" if args.generate else ""))
        if args.generate and args.sleep:
            await asyncio.sleep(args.sleep)

    summary = aggregate(results)
    print_summary(summary)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(
        {"args": vars(args), "summary": summary, "by_category": by_category(results), "results": results},
        indent=2, default=str), encoding="utf-8")
    print(f"\nSaved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--generate", action="store_true", help="also generate answers (LLM calls)")
    ap.add_argument("--judge", action="store_true", help="LLM-judge correctness + faithfulness (implies --generate)")
    ap.add_argument("--k", type=int, default=10, help="retrieval depth (matches API default)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--category", default=None)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between items (rate limits)")
    args = ap.parse_args()
    if args.judge:
        args.generate = True
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
