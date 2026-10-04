# University RAG Evaluation Benchmark Results

**Run Date:** 2026-10-04 04:22:20
**Results File:** `eval/results/run_20261004_041003.json`
**Embedding Backend:** Gemini
**LLM for Answers:** Google Gemini
**Items Evaluated:** 3

## How to Run the Evaluation

### Prerequisites
- Python venv activated: `.ag-app\Scriptsctivate` (Windows)
- `.env` configured with `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, and `GOOGLE_API_KEY`
- Supabase instance populated with chunked documents and embeddings

### Command Used for This Run
```bash
python -m eval.run_benchmark --limit 3 --generate
```

### Common Commands
| Command | Purpose |
|---------|---------|
| `python -m eval.run_benchmark --limit 3` | Retrieval-only (no answers) on 3 items |
| `python -m eval.run_benchmark --limit 3 --generate` | Retrieval + Gemini answers on 3 items |
| `python -m eval.run_benchmark` | Full benchmark (all 23 items), retrieval only |
| `python -m eval.run_benchmark --generate` | Full benchmark + Gemini answers |
| `python -m eval.run_benchmark --generate --judge` | Full benchmark + answers + LLM judge |
| `python -m eval.run_benchmark --generate --judge --sleep 2` | Full benchmark + judge (rate-limited) |
| `python -m eval.run_benchmark --category regulations` | Only "regulations" category |
| `python -m eval.run_benchmark --k 20` | Retrieve top 20 (default: 10) |

### Output Location
Results are saved to `eval/results/run_YYYYMMDD_HHMMSS.json` and printed to stdout.
Parse with: `python -c "import json; print(json.dumps(json.load(open('eval/results/<filename>.json')), indent=2))"`

## Summary Statistics

### Retrieval Quality
- **Recall@1:** 0.0% (gold file in top 1)
- **Recall@3:** 0.0% (gold file in top 3)
- **Recall@5:** 33.3% (gold file in top 5)
- **Recall@10:** 66.7% (gold file in top 10)
- **MRR (Mean Reciprocal Rank):** 0.1167 (average rank of first relevant result)
- **Fact Coverage:** 33.3% (answers contain all required facts)

### Citation Quality
- **Citation Present Rate:** 66.7% (answers include citations)
- **Citation Valid Rate:** 100.0% (cited sources are valid)
- **Citation Precision:** 0.0% (cited chunks are relevant)

### Judge Evaluation (LLM-as-Judge)
- **Correctness Rate:** N/A (judge abstained on 66.7% of answerable items)
- **Faithfulness Rate:** N/A (no judge evaluation performed)
- **Judge Tokens Used:** 0

### Latency Breakdown (seconds)

| Component | Mean | P50 | P95 |
|-----------|------|-----|-----|
| Embedding Inference | 0.283s | 0.266s | 0.342s |
| Vector Search | 0.103s | 0.092s | 0.129s |
| LLM Generation | 2.918s | 3.089s | 3.120s |
| **Total (E2E)** | **3.305s** | **3.449s** | **3.592s** |

### Token Usage

- **Total Prompt Tokens:** 13,490
- **Average Prompt per Item:** 4497 tokens
- **Total Response Tokens:** 94
- **Average Response per Item:** 31 tokens
- **Total Tokens (all):** 15,069
- **Average Tokens per Item:** 5023 tokens

## Results by Category

### REGULATIONS

- **Items:** 3 (answerable: 3)
- **Recall@5:** 33.3% | **Recall@10:** 66.7%
- **MRR:** 0.1167 | **Fact Coverage:** 33.3%
- **Avg Total Latency:** 3.305s
- **Avg Tokens:** 5023 tokens


## Per-Item Results

### 1. ai-ter-periods
**Question:** How many weeks are the periods in a semester of the Master Artificial Intelligence?
**Category:** regulations

**Retrieval:**
- Recall@5: 0.0% | Recall@10: 0.0%
- MRR: 0.0000 | First Relevant Rank: None

**Quality Metrics:**
- Fact Coverage: 0.0%
- Citations Present: Yes
- Citation Precision: 0.0%

**Performance:**
- Latency: 3.608s (embed: 0.350s, search: 0.133s, llm: 3.123s)
- Tokens: 4,545 (prompt: 3949, response: 64)

### 2. ai-ter-grading-deadline
**Question:** Within how many working days must an examiner determine the result of a written exam in the Master Artificial Intelligence?
**Category:** regulations

**Retrieval:**
- Recall@5: 0.0% | Recall@10: 100.0%
- MRR: 0.1000 | First Relevant Rank: 10

**Quality Metrics:**
- Fact Coverage: 0.0%
- Citations Present: No

**Performance:**
- Latency: 3.449s (embed: 0.266s, search: 0.092s, llm: 3.089s)
- Tokens: 4,282 (prompt: 3777, response: 11)

### 3. ai-ter-thesis-deadline
**Question:** What is the marking deadline for theses in the Master Artificial Intelligence?
**Category:** regulations

**Retrieval:**
- Recall@5: 100.0% | Recall@10: 100.0%
- MRR: 0.2500 | First Relevant Rank: 4

**Quality Metrics:**
- Fact Coverage: 100.0%
- Citations Present: Yes
- Citation Precision: 0.0%

**Performance:**
- Latency: 2.859s (embed: 0.233s, search: 0.083s, llm: 2.541s)
- Tokens: 6,242 (prompt: 5764, response: 19)
