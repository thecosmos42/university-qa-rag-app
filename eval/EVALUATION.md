# RAG Evaluation Benchmark: How It Works

This document describes the logic of the benchmark in `eval/`: what it measures, how the gold labels are built and verified, how a run is carried out, and how every number is computed.

## 1. Purpose

The system under test is the `/query` pipeline:

```
question -> program extraction -> embed -> hybrid_search (pgvector + BM25, RRF) -> top-k chunks
         -> prompt with numbered SOURCEs -> Gemini 2.5 Flash -> answer with [SOURCE n] citations
```

The benchmark separates failures by stage, so a bad answer can be traced to its cause:

| Question | Dimension | Stage tested |
|---|---|---|
| Did we fetch the right evidence? | Retrieval recall | embed + hybrid search |
| Is the answer right? | Answer correctness | prompt + LLM |
| Is the answer backed by what it cites? | Citation faithfulness | LLM + citation behaviour |
| Is it fast and affordable? | Latency, token usage | whole pipeline |

If recall is low, fix retrieval/chunking. If recall is high but correctness is low, fix the prompt or the LLM.

## 2. File map

| File | Role |
|---|---|
| `dataset.jsonl` | The labeled questions (gold labels). |
| `validate_dataset.py` | Checks the labels against the source PDFs. |
| `run_benchmark.py` | CLI runner: executes the pipeline per item, collects results and prints a summary. |
| `metrics.py` | Pure functions computing every metric (no network or DB). |
| `judge.py` | LLM-as-judge prompt and parsing. |
| `results/` | One JSON file per run (gitignored). |
| `../src/api/query.py` | `retrieve_chunks()`: the retrieval step shared by `/query` and the benchmark. |
| `../src/api/citations.py` | `parse_cited_indices()`: extracts `[SOURCE n]` numbers, shared by `/query` and the benchmark. |
| `../test/test_eval_metrics.py`, `../test/test_eval_runner.py` | Offline unit tests (no DB or LLM needed). |

The benchmark calls the **same** `retrieve_chunks()`, `generator.generate_answer()` and `parse_cited_indices()` that the API uses, so it measures what production does rather than a copy of it.

## 3. The dataset

One JSON object per line in `dataset.jsonl`.

| Field | Meaning |
|---|---|
| `id` | Unique id (the loader rejects duplicates). |
| `category` | `regulations`, `admission`, `programme-info`, `course-lookup` or `unanswerable`. Used for the per-category breakdown. |
| `question` | The user question. |
| `answerable` | Defaults to `true`. `false` marks questions the corpus cannot answer. |
| `gold_files` | PDF filenames that contain the evidence. |
| `gold_pages` | 1-based page numbers of the evidence. |
| `gold_snippets` | Short verbatim strings from the evidence. The retrieval relevance test uses these. |
| `reference_answer` | The ground-truth answer. Used by the judge. |
| `must_contain` | Facts the answer must include. Each entry is a string, or a list of acceptable alternatives (for example `["fifteen", "15"]`). |
| `absent_terms`, `programmes` | Unanswerable items only: terms that must not appear in the named programmes' PDFs. |

### Why snippets rather than just page numbers

Chunk boundaries change whenever the chunkers change. Labelling by chunk id would break on every re-ingest. Labelling by file and verbatim text stays valid whatever the chunking is. Page numbers are only a fallback (see 5.1).

### Label validation (`python -m eval.validate_dataset`)

The labels are checked against the source PDFs:

- **Answerable items:** every gold snippet must occur on one of the item's gold pages, in one of its gold files. Text is normalized first (lowercase, curly quotes and dashes unified, whitespace collapsed).
- **Unanswerable items:** none of `absent_terms` may appear anywhere in the named programmes' PDFs, so abstaining really is the correct behaviour.
- **Structure:** answerable items must have gold files, pages and snippets.

It exits non-zero on any error. Run it after editing the dataset. It has already caught a label that failed because of a curly quote in the PDF.

## 4. How a run is carried out

`python -m eval.run_benchmark [--generate] [--judge] [--k N] [--limit N] [--category C] [--sleep S]`

| Flag | Effect |
|---|---|
| (none) | Retrieval only. No LLM calls, so it costs nothing. |
| `--generate` | Also generate answers, and record latency and tokens for the LLM stage. |
| `--judge` | Also run LLM-judged correctness and faithfulness (implies `--generate`). |
| `--k` | Retrieval depth (default 10, the same as the API). |
| `--sleep` | Pause between items to respect Gemini rate limits. |
| `--limit`, `--category` | Run a subset. |

### Per-item flow (`evaluate_item`)

1. **Retrieve.** Call `retrieve_chunks(question, limit=k)`. This runs the program extractor, embeds the question, calls the `hybrid_search` RPC, and falls back to an unfiltered search if the filtered one returns nothing. It returns the ranked chunks and the embed and search timings.
2. **Score retrieval.** Compare each ranked chunk to the gold labels (section 5.1).
3. **Generate** (with `--generate`). Call `generator.generate_answer()`, the same call as `/query`. It records the answer, Gemini's token counts and the LLM latency.
4. **Parse citations.** `parse_cited_indices(answer)` extracts the cited `SOURCE` numbers. These are 1-based ranks into the retrieved list, the same numbering the prompt used.
5. **Score the answer.** Compute fact coverage and the structural citation metrics (5.2, 5.3).
6. **Stop the pipeline clock.** `total_s` is captured here, before judging.
7. **Judge** (with `--judge`, answerable items with a reference answer only). Send the question, reference answer, system answer and the text of the cited chunks to Gemini (5.2, 5.3). Judging failures are recorded as `judge_error` and do not abort the run.

After all items, `aggregate()` computes the summary and `by_category()` computes the same summary per category. Everything is written to `eval/results/run_<timestamp>.json`, including the per-item retrieved chunks, answers, citations and judge reasons, so any individual failure can be inspected.

## 5. Metric definitions

### 5.1 Retrieval recall

A chunk is **relevant** (`chunk_matches_gold`) if:

1. Its `filename` is in `gold_files`, **and**
2. if the item has `gold_snippets`: the normalized chunk text contains at least one snippet (live chunks are rewritten with contextual headers and re-split, so snippets are not always preserved), **or** its page is within ±1 of a gold page.

Per answerable item, over the ranked list:

- **recall@k** (k = 1, 3, 5, 10): 1 if any relevant chunk is in the top k, else 0. This is a hit rate, which fits questions that have one piece of evidence. The summary averages it over answerable items.
- **MRR**: 1 / rank of the first relevant chunk, or 0 if none.
- **first_relevant_rank** and per-chunk **relevant_flags**: kept for debugging and reused by citation precision.

Unanswerable items are excluded from retrieval metrics because there is nothing to retrieve.

### 5.2 Answer correctness

- **Fact coverage:** the fraction of `must_contain` entries found in the normalized answer. A list entry counts if any alternative matches. This is cheap, deterministic and free of LLM bias, but it is literal: a correct answer worded differently can miss.
- **Judge correct rate:** the judge sees the question, reference answer and system answer, and returns `correct` (true only if all key facts agree with the reference and nothing contradicts it). It covers paraphrases that fact coverage misses.
- **Wrongly abstained:** the share of answerable questions where the answer contains "I don't have enough information". This separates "refused" from "answered wrongly".
- **Abstention accuracy:** the share of unanswerable questions where the system abstained. A low value means it hallucinates when it should refuse.

### 5.3 Citation faithfulness

Structural checks (no LLM), per generated answer:

- **has_citation:** the answer cites at least one source. Summary: citation present rate.
- **valid_rate:** cited numbers that point to an existing retrieved chunk, divided by cited numbers. A value below 1 means the model cited a source that does not exist.
- **precision:** among valid cited chunks, the share that is gold-relevant (5.1). It checks that the model cites the right evidence, not just some evidence.

Semantic check (LLM judge, `--judge`):

- The judge is shown **only the text of the cited chunks**, not the whole context. It counts the answer's distinct factual claims (`claims_total`) and how many of them the cited text explicitly supports (`claims_supported`).
- **faithfulness** = supported / total, capped at 1, averaged over items.

Answers that cite nothing are judged against "(the answer cites no sources)", so they score as unsupported rather than being excused.

### 5.4 Latency

Measured with `perf_counter` per item:

| Stage | Covers |
|---|---|
| `embed_s` | Program extraction plus question embedding (from `retrieve_chunks`). |
| `search_s` | The `hybrid_search` RPC (and the fallback search, if it ran). |
| `llm_s` | The Gemini call, including key rotation and any rate-limit backoff. |
| `total_s` | Retrieval plus generation. Excludes judging. |

The summary reports mean, p50 and p95 for each stage. Retrieval-only runs report embed, search and total.

### 5.5 Token usage

Taken from Gemini's usage metadata on the answer call: prompt, response and total tokens, reported as per-item mean and run total. Judge tokens are summed separately as `judge_tokens_sum` so that evaluation overhead does not inflate the system's own cost. Retrieval-only runs use no tokens.

## 6. Reading the results

| Pattern | Likely cause |
|---|---|
| Low recall@10 | Chunking, embeddings or the program filter dropped the evidence. |
| Recall@10 high, recall@1 low | Ranking and fusion weights. Increase `k` in the prompt or tune the semantic/keyword weights. |
| High recall, low correctness | Prompt or LLM problem. Check `wrongly abstained`. |
| High fact coverage, low judge correct rate | Surface-matching passes but meaning is wrong. Read `judge_reason` in the JSON. |
| Low citation precision, high valid rate | The model cites real but irrelevant sources. |
| Low faithfulness | Claims are not supported by the cited text, which is a hallucination or mis-citation risk. |
| Low abstention accuracy | The prompt does not make the model refuse reliably. |
| High p95 latency | Check for rate-limit backoff in `llm_s`. |

Use the same dataset and settings when comparing runs, and compare `by_category` as well as the overall numbers.

## 7. Limitations

- **Small seed set.** The dataset has 23 items across the AI, Computer Science and Mathematics programmes. Recall on this size is noisy, so grow it before drawing fine conclusions.
- **Hit-rate recall.** Each item has one evidence location. Questions that need several chunks would need a stricter definition, such as the fraction of gold snippets retrieved.
- **Judge bias.** The judge is Gemini, the same family as the generator, so it may favour its own phrasing. Judge scores are an estimate. Pair them with the deterministic fact coverage and spot-check `judge_reason`.
- **Literal fact matching.** Fact coverage does not understand paraphrase or numerals written in unusual forms.
- **Not yet run live.** The offline tests stub the DB and LLM. The runner has not been exercised against Supabase and Gemini.
- **Run-to-run variation.** LLM output and latency vary. For a stable comparison, average several runs.

## 8. Extending the benchmark

1. Read the PDF page and pick a verbatim sentence as the `gold_snippets` entry.
2. Write the question, the `reference_answer` and the `must_contain` facts.
3. For unanswerable questions, set `"answerable": false`, add `absent_terms`, and add a `programmes` list. Entries use the lowercase, underscore-free programme name from the PDF filename, for example `artificialintelligence`.
4. Run `python -m eval.validate_dataset` until it reports OK.
5. Run the offline tests: `pytest test/test_eval_metrics.py test/test_eval_runner.py`.
