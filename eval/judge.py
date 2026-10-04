"""LLM-as-judge for answer correctness and citation faithfulness (Gemini)."""
from __future__ import annotations

import json
import re

JUDGE_PROMPT = """You are a strict evaluator of a university question-answering system.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{reference}

SYSTEM ANSWER:
{answer}

CITED SOURCES (the only text the system's citations point to):
{cited_sources}

Evaluate two things and reply with ONLY a JSON object:
{{
  "correct": true|false,        // the system answer agrees with the reference answer on every key fact; extra correct detail is fine, any contradiction or missing key fact is false
  "claims_total": <int>,        // number of distinct factual claims in the system answer
  "claims_supported": <int>,    // claims explicitly supported by the CITED SOURCES text above (not by outside knowledge)
  "reason": "<one short sentence>"
}}"""


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise ValueError(f"judge returned no JSON: {text!r}")
    return json.loads(m.group(0))


async def judge_answer(ask, question: str, reference: str, answer: str, cited_chunks: list[dict]) -> dict:
    """`ask` is an async callable (prompt) -> {"text", "tokens"}; injected for testability."""
    sources = "\n\n".join(
        f"[SOURCE {c['index']}] {' '.join(c['content'].split())}" for c in cited_chunks
    ) or "(the answer cites no sources)"
    res = await ask(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer, cited_sources=sources))
    data = _parse_json(res["text"])
    total = int(data.get("claims_total", 0))
    supported = int(data.get("claims_supported", 0))
    return {
        "correct": bool(data.get("correct")),
        "faithfulness": (min(supported, total) / total) if total > 0 else None,
        "reason": data.get("reason", ""),
        "tokens": res["tokens"]["total"],
    }
