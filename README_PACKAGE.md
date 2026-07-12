# ScholarGym One-Pass Baseline + Shadow Postprocessors

This directory is an independent ScholarGym checkout pinned to upstream baseline
commit `f426fd15e3ff28ee11ddeafc253dffd73ef88500`.

For every benchmark query, one command runs:

```text
complete baseline query trajectory
  -> optional per-retrieval-page per-subquery graph rerank + Selector replay
  -> optional all-retrieved global graph rerank + Selector replay
  -> next benchmark query
```

The two replays never write into baseline memory. The benchmark is not run three
times. Per-subquery replay keeps the actual baseline page size so only the
candidate papers change. Global replay uses the baseline query-level retrieval
total (the sum of actual results across all subqueries/pages) as Selector top-k;
`--results_per_query` does not cap the global Selector input.

## Method definitions

Per-subquery shadow replay uses only the newly retrieved page as seeds:

```text
0.30 * query_score_normalized
+ 0.40 * subquery_score_normalized
+ 0.15 * intent_score
+ 0.15 * path_count_normalized
```

Global replay uses all baseline-retrieved seeds for the completed query:

```text
0.5 * query_score_normalized
+ 0.5 * max(all_executed_subquery_scores_normalized)
```

For this named phase3 method, each component is normalized as
`raw_score / max_positive_raw_score` (or all zero when the maximum is not
positive), matching the previous phase3 implementation. The per-subquery new
formula continues to use closed-pool min-max features.

After reranking, the global Selector receives the top `baseline retrieved_count`
papers, where `retrieved_count` is the sum of actual baseline retrieval records
across the complete query. A seed retrieved by multiple subqueries contributes
once per retrieval occurrence to this budget, while the candidate papers are
deduplicated by arXiv ID and retain all provenance edges. If the deduplicated
seed+expanded pool is smaller than the budget, all available candidates are
passed.

Only expanded papers present in the local paper DB, with a non-missing paper
month no later than the query/subquery cutoff month, enter scoring and artifacts.
Baseline seeds are trusted because retrieval already applied the cutoff.

## Install

```bash
conda create -n scholargym-graph python=3.10 -y
conda activate scholargym-graph
pip install -r requirements.txt
```

On macOS, the requirements select `faiss-cpu` instead of `faiss-gpu`.

Set the LLM and S2 credentials required by your config, for example:

```bash
export DASHSCOPE_API_KEY="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export S2_API_KEY="..."
```

## Sparse/BM25 run

```bash
python code/eval.py \
  --config configs/config_qwen30b_api.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl data/scholargym_test_fast.jsonl \
  --bm25_path data/bm25_index.pkl \
  --output_dir eval_results_onepass \
  --run_label testfast_run1 \
  --workflow deep_research \
  --search_method bm25 \
  --max_iterations 5 \
  --results_per_query 10 \
  --browser_mode NONE \
  --save_level full \
  --run_per_subquery_postprocess \
  --run_global_postprocess \
  --graph_method citations_references \
  --graph_expansion_limit 100 \
  --graph_cache_dir cache/s2_graph_oracle \
  --graph_rate_limit_rps 4.0
```

Use a new `--run_label` for a new experiment. ScholarGym resume is keyed by the
existing `detailed_results.jsonl`; reusing a completed output directory skips
the baseline query and therefore also skips its replay.

Artifact writes are query-transactional without changing the baseline agents.
During one query, baseline/replay rows go to `onepass_artifacts/.staging/` and
the canonical JSONLs remain unchanged. Only after the baseline plus both enabled
postprocessors finish are the staged rows flushed to the existing flat JSONLs;
then ScholarGym appends the query to `detailed_results.jsonl`. On every resume,
`detailed_results.jsonl` is the commit source of truth: stale staging directories,
malformed tail rows, and rows belonging to uncommitted benchmark indices/query
IDs are removed atomically before the unfinished query is rerun. Reconciliation
statistics are saved in `onepass_artifacts/resume_reconciliation.json`.

Resume remains query-level. An interruption inside a query restarts that query's
baseline trajectory from iteration 1; it does not resume an individual Planner,
retrieval, or Selector call.

Disable either replay independently:

```bash
--no-run_per_subquery_postprocess
--no-run_global_postprocess
```

Replay requires `--browser_mode NONE` and `ENABLE_SUMMARIZATION=False`; this
keeps the replay Selector call identical to the baseline abstract-only Selector
except for candidate IDs and their rerank scores.

## Dense run with Ollama + Qdrant

Start Ollama and Qdrant, then pull the exact model:

```bash
ollama pull qwen3-embedding:0.6b
```

Build the collection with the same backend/model used by evaluation:

```bash
python code/build_vector_db_configurable.py \
  --paper_db data/scholargym_paper_db.json \
  --qdrant_url http://localhost:6333 \
  --qdrant_collection paper_qwen3_06b \
  --embedding_backend ollama \
  --embedding_model qwen3-embedding:0.6b \
  --embedding_base_url http://localhost:11434 \
  --batch_size 64 \
  --recreate
```

Run retrieval and local reranking with that same model:

```bash
python code/eval.py \
  --config path/to/config.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl data/scholargym_test_fast.jsonl \
  --output_dir eval_results_onepass_dense \
  --workflow deep_research \
  --search_method vector \
  --results_per_query 10 \
  --max_iterations 5 \
  --browser_mode NONE \
  --save_level full \
  --embedding_backend ollama \
  --embedding_service_model qwen3-embedding:0.6b \
  --embedding_base_url http://localhost:11434 \
  --qdrant_url http://localhost:6333 \
  --qdrant_collection paper_qwen3_06b
```

## OpenAI-compatible embedding API / OpenRouter

The corpus collection and evaluation must use the exact same API model and
dimension. OpenRouter currently does not expose Qwen3-Embedding-0.6B; using its
4B/8B model is a different dense experiment and requires rebuilding Qdrant.

```bash
export EMBEDDING_API_KEY="$OPENROUTER_API_KEY"

python code/build_vector_db_configurable.py \
  --paper_db data/scholargym_paper_db.json \
  --qdrant_collection paper_openrouter_qwen3_4b \
  --embedding_backend api \
  --embedding_model qwen/qwen3-embedding-4b \
  --embedding_base_url https://openrouter.ai/api/v1 \
  --recreate

python code/eval.py \
  ... \
  --search_method vector \
  --embedding_backend api \
  --embedding_service_model qwen/qwen3-embedding-4b \
  --embedding_base_url https://openrouter.ai/api/v1 \
  --embedding_api_key_env EMBEDDING_API_KEY \
  --qdrant_collection paper_openrouter_qwen3_4b
```

## Save levels and outputs

`minimal` saves manifests, query metrics, complete final candidate/selected
arXiv ID lists, warnings, and errors. `full` also
saves all analysis rows. No prompt, title, abstract, author, API key, or
date-filtered paper ID is written to analysis artifacts.

Main full files under `<run>/onepass_artifacts/`:

```text
baseline/planner_events.jsonl
baseline/paper_rows.jsonl
baseline/selector_decisions.jsonl
per_subquery/paper_rows.jsonl
per_subquery/expansion_edges.jsonl
per_subquery/selector_decisions.jsonl
global/subquery_paper_scores.jsonl
global/final_paper_rows.jsonl
global/expansion_edges.jsonl
global/selector_decisions.jsonl
query_results.jsonl
run_manifest.json
resume_reconciliation.json
```

The outer `<output_dir>/evaluation_summary.jsonl` also contains
`postprocess_overall_metrics.per_subquery` and
`postprocess_overall_metrics.global`. Each block reports macro averages over
successful unique benchmark queries, micro recall/precision, total GT/candidate/
selected counts, and failed or missing query counts. Here `candidate_*` means the
deduplicated rerank top-k actually passed to that shadow Selector, not the larger
pre-top-k expansion pool. Resume/retry duplicates are collapsed by benchmark
`idx`, with the last detailed result treated as authoritative.

Planner events include the structured pre-call `ResearchMemory`, every prior
subquery page state, selected/retrieved arXiv IDs, overview/checklist, retrieval
exclusion IDs, and stable planner/retrieval event IDs. Raw prompts are not saved.

`observed_retrieval_score/rank` is the actual baseline page result and therefore
exists only for seeds. `retrieval_score/rank` is recomputed with the same backend
inside the closed seed+expanded pool and therefore exists for seeds and expanded
papers. `retrieval_rank_scope` prevents these two rank meanings from being mixed.

One expanded paper may have several source seeds. `paper_rows` contains the
aggregated `source_seed_arxiv_ids`; `expansion_edges` stores one exact
`(query_id, subquery_id, seed_arxiv_id, expanded_arxiv_id)` edge per provenance.
The global component file stores every `(query, scoring_subquery, paper)` score,
not only the maximum.

## Important parameters

```text
--save_level minimal|full
--run_label LABEL
--limit N
--results_per_query N
--run_per_subquery_postprocess / --no-run_per_subquery_postprocess
--run_global_postprocess / --no-run_global_postprocess
--graph_method citations|references|citations_references
--graph_expansion_limit N
--graph_cache_dir PATH
--graph_rate_limit_rps FLOAT
--graph_offline_cache_only
--global_alpha FLOAT
--embedding_backend ollama|api
--embedding_service_model MODEL
--embedding_base_url URL
--embedding_api_key_env ENV_NAME
--qdrant_url URL
--qdrant_collection NAME
```

When both independent packages run simultaneously with the same S2 key, the
rate limit is per process. Set the two `--graph_rate_limit_rps` values so their
sum stays within the key quota. Cache writes use atomic replacement, but two
processes may still issue the same uncached request once.
