"""Pure metric functions for the RAG benchmark (no network, no DB, no LLM)."""
from __future__ import annotations

import math
import re
from statistics import mean

ABSTAIN_PHRASE = "i don't have enough information"
PAGE_TOLERANCE = 1


def norm(text: str) -> str:
    """Lowercase, unify quotes/dashes and collapse whitespace for robust matching."""
    text = (text or "").replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(text.lower().split())


def chunk_matches_gold(chunk: dict, item: dict) -> bool:
    """A retrieved chunk is relevant if it comes from a gold file and, when the item
    lists evidence snippets, contains one of them; otherwise if its page is within
    PAGE_TOLERANCE of a gold page."""
    if chunk.get("filename") not in item.get("gold_files", []):
        return False
    snippets = item.get("gold_snippets") or []
    content = norm(chunk.get("content", ""))
    if any(norm(s) in content for s in snippets):
        return True
    page = chunk.get("page_number")
    pages = item.get("gold_pages") or []
    if not pages:
        return True
    return page is not None and any(abs(page - p) <= PAGE_TOLERANCE for p in pages)


def retrieval_metrics(chunks: list[dict], item: dict, ks=(1, 3, 5, 10)) -> dict:
    """Recall@k (hit-rate: any relevant chunk in top k), MRR and the first relevant rank."""
    flags = [chunk_matches_gold(c, item) for c in chunks]
    first = next((i + 1 for i, f in enumerate(flags) if f), None)
    out = {f"recall@{k}": float(any(flags[:k])) for k in ks}
    out["mrr"] = 1.0 / first if first else 0.0
    out["first_relevant_rank"] = first
    out["relevant_flags"] = flags
    return out


def fact_coverage(answer: str, must_contain: list) -> float | None:
    """Fraction of required facts present. Each entry is a string or a list of
    acceptable alternatives. Returns None when the item lists no facts."""
    if not must_contain:
        return None
    ans = norm(answer)
    hits = 0
    for fact in must_contain:
        alts = fact if isinstance(fact, list) else [fact]
        if any(norm(a) in ans for a in alts):
            hits += 1
    return hits / len(must_contain)


def is_abstention(answer: str) -> bool:
    return ABSTAIN_PHRASE in norm(answer)


def citation_metrics(answer: str, cited: set[int], chunks: list[dict], relevant_flags: list[bool]) -> dict:
    """Structural citation checks.

    valid_rate:      cited indices that point to a retrieved chunk
    precision:       valid cited chunks that are gold-relevant / valid cited chunks
    has_citation:    answer cites at least one source
    """
    n = len(chunks)
    valid = {i for i in cited if 1 <= i <= n}
    return {
        "has_citation": bool(cited),
        "n_cited": len(cited),
        "valid_rate": (len(valid) / len(cited)) if cited else None,
        "precision": (sum(relevant_flags[i - 1] for i in valid) / len(valid)) if valid else None,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def aggregate(results: list[dict]) -> dict:
    """Roll per-item results up into summary numbers. Retrieval metrics use answerable
    items only; abstention accuracy uses unanswerable items only."""
    answerable = [r for r in results if r["item"].get("answerable", True)]
    unanswerable = [r for r in results if not r["item"].get("answerable", True)]
    gen = [r for r in answerable if r.get("answer") is not None]

    summary: dict = {"n_items": len(results), "n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}
    for key in ("recall@1", "recall@3", "recall@5", "recall@10", "mrr"):
        summary[key] = _avg(r["retrieval"][key] for r in answerable)

    if gen:
        summary["fact_coverage"] = _avg(r["fact_coverage"] for r in gen)
        summary["judge_correct_rate"] = _avg(
            None if r.get("judge_correct") is None else float(r["judge_correct"]) for r in gen
        )
        summary["abstained_on_answerable"] = _avg(float(is_abstention(r["answer"])) for r in gen)
        summary["citation_present_rate"] = _avg(float(r["citation"]["has_citation"]) for r in gen)
        summary["citation_valid_rate"] = _avg(r["citation"]["valid_rate"] for r in gen)
        summary["citation_precision"] = _avg(r["citation"]["precision"] for r in gen)
        summary["faithfulness"] = _avg(r.get("faithfulness") for r in gen)
    gen_un = [r for r in unanswerable if r.get("answer") is not None]
    if gen_un:
        summary["abstention_accuracy"] = _avg(float(is_abstention(r["answer"])) for r in gen_un)

    lat = [r["latency"] for r in results if r.get("latency")]
    tok = [r["tokens"] for r in results if r.get("tokens")]
    for stage in ("embed_s", "search_s", "llm_s", "total_s"):
        vals = [l[stage] for l in lat if l.get(stage) is not None]
        summary[f"latency_{stage}_mean"] = _avg(vals)
        summary[f"latency_{stage}_p50"] = percentile(vals, 50)
        summary[f"latency_{stage}_p95"] = percentile(vals, 95)
    for kind in ("prompt", "response", "total"):
        vals = [t[kind] for t in tok if t.get(kind) is not None]
        summary[f"tokens_{kind}_mean"] = _avg(vals)
        summary[f"tokens_{kind}_sum"] = sum(vals) if vals else None
    summary["judge_tokens_sum"] = sum(r.get("judge_tokens", 0) for r in results)
    return summary


def by_category(results: list[dict]) -> dict:
    cats: dict[str, list[dict]] = {}
    for r in results:
        cats.setdefault(r["item"].get("category", "uncategorised"), []).append(r)
    return {c: aggregate(rs) for c, rs in sorted(cats.items())}
