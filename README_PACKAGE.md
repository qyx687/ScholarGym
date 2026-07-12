# ScholarGym Online Per-Subquery Graph Rerank

This is an independent ScholarGym checkout pinned to upstream baseline commit
`f426fd15e3ff28ee11ddeafc253dffd73ef88500`.

Unlike the shadow replay package, this method changes the live trajectory:

```text
Planner
  -> retrieve one new page for each subquery
  -> S2 citation/reference expansion
  -> local paper-DB membership and date cutoff
  -> same-backend rerank
  -> rerank top K, where K is the actual raw retrieval-page size
  -> unchanged baseline Selector prompt
  -> write retrieved/selected papers into SubQueryState and memory
  -> next Planner iteration observes the changed state
```

No graph context is added to the Selector prompt. A continued subquery processes
only its newly retrieved page; previous online selections already influence the
next iteration through normal ScholarGym memory.

The fixed formula is:

```text
0.30 * query_score_normalized
+ 0.40 * subquery_score_normalized
+ 0.15 * intent_score
+ 0.15 * path_count_normalized
```

## Install

```bash
conda create -n scholargym-graph python=3.10 -y
conda activate scholargym-graph
pip install -r requirements.txt

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
  --output_dir eval_results_online \
  --run_label testfast_run1 \
  --workflow deep_research \
  --search_method bm25 \
  --max_iterations 5 \
  --results_per_query 10 \
  --browser_mode NONE \
  --enable_per_subquery_graph \
  --save_level full \
  --graph_method citations_references \
  --graph_expansion_limit 100 \
  --graph_cache_dir cache/s2_graph_oracle \
  --graph_rate_limit_rps 4.0
```

Use a distinct `--run_label` for each experiment; the original ScholarGym
checkpoint logic skips query indices already present in `detailed_results.jsonl`.

`--no-enable_per_subquery_graph` provides a baseline-only diagnostic run, but
the intended package method keeps it enabled.

## Dense/Ollama run

Build the Qdrant collection with the same model used during retrieval/rerank:

```bash
ollama pull qwen3-embedding:0.6b

python code/build_vector_db_configurable.py \
  --paper_db data/scholargym_paper_db.json \
  --qdrant_url http://localhost:6333 \
  --qdrant_collection paper_qwen3_06b \
  --embedding_backend ollama \
  --embedding_model qwen3-embedding:0.6b \
  --embedding_base_url http://localhost:11434 \
  --recreate

python code/eval.py \
  --config path/to/config.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl data/scholargym_test_fast.jsonl \
  --output_dir eval_results_online_dense \
  --workflow deep_research \
  --search_method vector \
  --max_iterations 5 \
  --results_per_query 10 \
  --save_level full \
  --embedding_backend ollama \
  --embedding_service_model qwen3-embedding:0.6b \
  --embedding_base_url http://localhost:11434 \
  --qdrant_url http://localhost:6333 \
  --qdrant_collection paper_qwen3_06b
```

## OpenAI-compatible embedding API / OpenRouter

API retrieval and rerank are supported, but the Qdrant collection must be
built with that exact API model. OpenRouter Qwen3-Embedding-4B is not the same
experiment as Ollama Qwen3-Embedding-0.6B.

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

## Full outputs

Under `<run>/online_artifacts/`:

```text
run_manifest.json
planner_events.jsonl
raw_retrieval_rows.jsonl
paper_rows.jsonl
expansion_edges.jsonl
selector_decisions.jsonl
selector_passes.jsonl
memory_transitions.jsonl
query_results.jsonl
```

Every `paper_rows` row is one `(query, subquery, paper)` candidate and contains:

```text
query_id/query/iteration_idx/subquery_id/subquery/page/offset
paper_arxiv_id and seed|expanded|seed_and_expanded
source_seed_arxiv_ids/source_subquery_ids/edge_types
observed baseline page score/rank for seeds
same-backend closed-pool retrieval score/rank for every seed and expanded paper
query raw/normalized score and component rank
subquery raw/normalized score and component rank
intent labels/score
path count/normalized path count
fixed feature weights and final rerank score/rank
Selector top-k/selected/reason flags
written_to_retrieved_memory/written_to_selected_memory
affects_next_iteration=true
```

`planner_events` contains the structured state seen before every Planner call:
`ResearchMemory`, all prior per-page subquery states, retrieved/selected arXiv
IDs, overview/checklist, and retrieval exclusions. `selector_passes` separately
records the initial and optional post-browsing Selector inputs/outputs without
paper text or raw prompts. Pagination offsets use raw retrieval page counts,
not the size of the graph-reranked pool.

`expansion_edges` stores every exact
`(query_id, subquery_id, seed_arxiv_id, expanded_arxiv_id)` provenance, including
S2 paper IDs, citation/reference edge type and rank, intents, influential flag,
citation/reference counts, cache/API status, retry count, and cutoff context.

The per-query `detailed_results.jsonl` keeps two deliberately separate rank
views for every iteration:

```text
gt_rank / avg_distance
  = original ScholarGym baseline-retriever rank_dict after its date cutoff and
    prior-selection exclusion, before graph expansion

local_gt_rank / local_avg_distance
  = rank inside the date-valid seed+expanded closed pool after graph reranking
```

The two views use independent cross-iteration best-rank trackers. The outer
`evaluation_summary.jsonl` writes their iteration means as
`avg_distance_iter_N` and `local_avg_distance_iter_N`. A run made by an older
package revision stored the local view under `gt_rank/avg_distance`; those old
records cannot reconstruct the original global retriever rank, so use a new
`--run_label` rather than resuming them into this schema.

If one seed is retrieved by several subqueries, each retrieval event remains a
separate row. If one expanded paper comes from several seeds, its candidate row
aggregates all source seed IDs and the edge file retains all individual edges.

No prompt, paper title/abstract/author, API key, or date-filtered paper ID is
written. `minimal` omits intermediate rows and keeps manifests/query summaries,
including complete final candidate/selected arXiv ID lists;
`full` writes every file above.

## Important parameters

```text
--enable_per_subquery_graph / --no-enable_per_subquery_graph
--save_level minimal|full
--run_label LABEL
--limit N
--results_per_query N
--graph_method citations|references|citations_references
--graph_expansion_limit N
--graph_cache_dir PATH
--graph_rate_limit_rps FLOAT
--graph_offline_cache_only
--embedding_backend ollama|api
--embedding_service_model MODEL
--embedding_base_url URL
--embedding_api_key_env ENV_NAME
--qdrant_url URL
--qdrant_collection NAME
```

When this package and the one-pass package run simultaneously with the same S2
key, `--graph_rate_limit_rps` is per process. Keep the sum within the key quota.
