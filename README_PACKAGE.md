# ScholarGym One-Pass Baseline + Shadow Postprocessors

This directory is an independent ScholarGym checkout pinned to upstream baseline
commit `f426fd15e3ff28ee11ddeafc253dffd73ef88500`.

For every benchmark query, one command runs:

```text
complete baseline query trajectory
  -> optional per-retrieval-page per-subquery graph rerank + Selector replay
  -> optional event/offset-matched text deep retrieval + rerank + Selector replay
  -> optional stable-subquery merged-budget text deep retrieval + rerank + Selector replay
  -> next benchmark query
```

The three shadows never write into baseline memory. The benchmark is run once,
not once per method. Deep-retrieval budgets are taken from the already computed
per-subquery graph pools, so the deep controls do not call Semantic Scholar.

## Method definitions

Per-subquery shadow replay uses only the newly retrieved page as seeds:

```text
0.30 * query_score_normalized
+ 0.40 * subquery_score_normalized
+ 0.15 * intent_score
+ 0.15 * path_count_normalized
```

Both deep controls use the same weights, with graph-only terms fixed to zero:

```text
0.30 * query_score_normalized
+ 0.40 * subquery_score_normalized
+ 0.15 * 0
+ 0.15 * 0
```

All Q/SQ components use closed-pool min-max normalization. Retrieval and local
reranking use the same mode as baseline: BM25 with BM25, or Qdrant plus the fixed
Ollama `qwen3-embedding:0.6b` model for dense runs.

`deep_event_offset_matched` operates once per baseline retrieval event. Its
budget `N_i` is the corresponding graph local-pool size; it applies that event's
frozen exclusion first, then the saved offset, retrieves `N_i`, reranks the pool,
and passes the event's actual `selector_top_k`.

`deep_merged_subquery_sum_budget` groups all continue events with the same stable
`subquery_id`. Its budget is `N = sum_i N_i`; it freezes the first event's
exclusion, starts from offset 0, retrieves/reranks once, then gives Selector
chronological, disjoint slices `[0:k1)`, `[k1:k1+k2)`, etc. Each slice uses its
own event checklist and iteration; selections are not fed to later slices.

When both controls are enabled, the full retriever ranking for one stable
subquery is computed/fetched once and the two controls materialize their own
exclusion/offset slices from it. Their rerank normalization scopes remain
independent: one closed pool per event for scheme 1 and one merged closed pool
per stable subquery for scheme 2.

Within each completed baseline query, independent graph events and deep reranks
use a bounded worker pool. Shadow Selector calls use bounded async concurrency,
while output rows are committed in the original retrieval-event order. The
default limits are 4 event workers, 4 Selector calls, and 1 postprocess Ollama
embedding call. S2 requests still share the global `--graph_rate_limit_rps`;
identical concurrent cache keys use single-flight so a shared seed is fetched
only once. These settings change scheduling, not candidate pools, formulas,
Selector inputs, or baseline feedback.

For the graph shadow, only expanded papers present in the local paper DB, with a non-missing paper
month no later than the query/subquery cutoff month, enter scoring and artifacts.
Baseline seeds are trusted because retrieval already applied the cutoff. Deep
retrieval likewise keeps only canonical arXiv papers with a non-missing date no
later than the corresponding subquery cutoff.

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

`DASHSCOPE_*` is used only by the Qwen 30B Planner/Selector LLM. Embedding does
not call a remote/cloud embedding API; it is served only by local Ollama.

Large baseline assets are intentionally not committed in this package. Before
running, provide `scholargym_paper_db.json`, the BM25 pickle for sparse runs, or
the original `paper_knowledge_base` Qdrant storage for dense runs. They may be
copied/symlinked into `data/`, or supplied with absolute `--paper_db` and
`--bm25_path` paths. The baseline Qdrant archive is
`qdrant_vector_index_qwen3_embedding_0.6b.tgz`; it already contains collection
`paper_knowledge_base` and should be preferred when strict baseline retrieval
comparability is required.

For the original archive, the expected SHA-256 is
`329b681a0b98b9523af08df413bdd1adfe82e3947f3e63f699812336563043ea`.
It was built with Qdrant `1.18.0`; restore and expose it consistently, for
example:

```bash
sha256sum /path/to/qdrant_vector_index_qwen3_embedding_0.6b.tgz
tar -xzf /path/to/qdrant_vector_index_qwen3_embedding_0.6b.tgz -C .
docker run --rm \
  -p 6433:6333 -p 6434:6334 \
  -v "$PWD/data/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant:v1.18.0
```

The original Ollama model blob digest was
`sha256-06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.

## Sparse/BM25 run

```bash
python code/eval.py \
  --config configs/config_qwen30b_api.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl data/scholargym_bench.jsonl \
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
  --run_deep_event_postprocess \
  --run_deep_merged_postprocess \
  --graph_method citations_references \
  --graph_expansion_limit 100 \
  --graph_cache_dir cache/s2_graph_oracle \
  --graph_rate_limit_rps 4.0 \
  --postprocess_event_workers 4 \
  --postprocess_selector_concurrency 4 \
  --postprocess_embedding_concurrency 1
```

Use a new `--run_label` for a new experiment. ScholarGym resume is keyed by the
existing `detailed_results.jsonl`; reusing a completed output directory skips
the baseline query and therefore also skips its replay.

Before resume, `run_manifest.json` verifies a signature over the package code,
config, benchmark content, local corpus/index file metadata, Qdrant/Ollama
endpoint plus collection/model locator, baseline parameters, graph/deep arms,
and fixed embedding settings. It cannot hash a live Qdrant collection, so this
is not a content-level check of the remote index; strict dense reproducibility
still relies on the archive/model digests above. A mismatch, unreadable
manifest, or checkpoint rows without a manifest are rejected instead of mixing
old checkpoints with a new experiment; use a new `--run_label` in that case.
For an already-started run upgraded only to this audited parallel scheduler,
restart with the original command plus
`--allow_resume_compatible_code_change`. The override still requires every
recorded data, model, method, and experiment parameter to match; it ignores only
the package source hash and the three postprocess parallelism settings, and is
recorded in the replacement manifest. Do not use it for formula, prompt, model,
dataset, or retrieval changes.

Artifact writes are query-transactional without changing the baseline agents.
During one query, baseline/shadow rows go to `onepass_artifacts/.staging/` and
the canonical JSONLs remain unchanged. Only after the baseline and all enabled
shadows finish are the staged rows flushed to the existing flat JSONLs;
then ScholarGym appends the query to `detailed_results.jsonl`. On every resume,
`detailed_results.jsonl` is the commit source of truth: stale staging directories,
malformed tail rows, and rows belonging to uncommitted benchmark indices/query
IDs are removed atomically before the unfinished query is rerun. Reconciliation
statistics are saved in `onepass_artifacts/resume_reconciliation.json`.
An incomplete final line in `detailed_results.jsonl` is also atomically
truncated; corruption before later valid rows is rejected instead of guessed.

Resume remains query-level. An interruption inside a query restarts that query's
baseline trajectory from iteration 1; it does not resume an individual Planner,
retrieval, or Selector call.

Disable either deep control independently:

```bash
--no-run_deep_event_postprocess
--no-run_deep_merged_postprocess
```

The deep controls require `--run_per_subquery_postprocess`, because its local
graph pools define `N_i`. Disable all three shadows with all three `--no-run_*`
flags when a baseline-only run is needed.

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
  --qdrant_url http://localhost:6433 \
  --qdrant_collection paper_knowledge_base \
  --embedding_base_url http://localhost:11434 \
  --batch_size 64 \
  --recreate
```

Run retrieval and local reranking with that same model:

```bash
python code/eval.py \
  --config configs/config_qwen30b_api.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl data/scholargym_bench.jsonl \
  --output_dir eval_results_onepass_dense \
  --workflow deep_research \
  --search_method vector \
  --results_per_query 10 \
  --max_iterations 5 \
  --browser_mode NONE \
  --save_level full \
  --embedding_base_url http://localhost:11434 \
  --qdrant_url http://localhost:6433 \
  --qdrant_collection paper_knowledge_base \
  --postprocess_event_workers 4 \
  --postprocess_selector_concurrency 4 \
  --postprocess_embedding_concurrency 1
```

The configurable builder and dense local reranker use the baseline
`OllamaEmbeddings` class, fixed model, cosine distance, and exact baseline
`title: ...\n abstract: ...` paper serialization. Configurable fields are limited
to service URL, collection name, and batching. A newly built collection is operationally
compatible, but restoring the original collection is the stronger choice when
exact ranking reproducibility matters.

## Save levels and outputs

`minimal` saves manifests, query metrics, graph/deep compact pool records with
complete arXiv ID/score/rank lists, comparisons, warnings, and errors. `full`
also saves flat per-paper analysis rows, exact expansion edges and Selector
decision rows. No rendered prompt, title, abstract, author, API key, or paper
rejected by the date cutoff is written to analysis artifacts.

Main full files under `<run>/onepass_artifacts/`:

```text
baseline/planner_events.jsonl
baseline/paper_rows.jsonl
baseline/selector_decisions.jsonl
per_subquery/paper_rows.jsonl
per_subquery/pool_records.jsonl
per_subquery/expansion_edges.jsonl
per_subquery/selector_decisions.jsonl
deep_event/pool_records.jsonl
deep_event/paper_rows.jsonl
deep_event/comparisons.jsonl
deep_event/selector_decisions.jsonl
deep_merged/pool_records.jsonl
deep_merged/paper_rows.jsonl
deep_merged/comparisons.jsonl
deep_merged/selector_decisions.jsonl
query_results.jsonl
run_manifest.json
resume_reconciliation.json
```

The outer `<output_dir>/evaluation_summary.jsonl` also contains
`postprocess_overall_metrics.per_subquery`, `.deep_event`, and `.deep_merged`.
Each block reports macro averages over
successful unique benchmark queries, micro recall/precision, total GT/candidate/
selected counts, and failed or missing query counts. Here `candidate_*` means the
deduplicated rerank top-k actually passed to that shadow Selector, not the larger
pre-top-k expansion pool. Resume/retry duplicates are collapsed by benchmark
`idx`, with the last detailed result treated as authoritative.

Planner events include the structured pre-call `ResearchMemory`, every prior
subquery page state, selected/retrieved arXiv IDs, overview/checklist, retrieval
exclusion IDs, and stable planner/retrieval event IDs. Raw prompts are not saved.

`observed_retrieval_score/rank` is the actual baseline page result and therefore
exists only for seeds. `observed_retrieval_rank_after_exclusion` adds the saved
offset after baseline's frozen exclusion; it is not a pre-exclusion global rank.
The legacy `observed_retrieval_absolute_rank` is retained as an explicitly scoped
alias. `retrieval_score/rank` is recomputed with the same backend inside the
closed seed+expanded pool and therefore exists for seeds and expanded papers.

Every deep paper is keyed by `(query_id, subquery_id, retrieval_event_id,
paper_arxiv_id)`. Both `deep_*/pool_records.jsonl` files are written in minimal
and full mode and contain, for every deep-pool paper:

- retriever raw score, global date-valid rank, exclusion-adjusted rank, and local-pool rank;
- Q/SQ raw score, min-max score, and component rank;
- text-only rerank score/rank, with intent/path explicitly zero;
- Selector membership/input rank/selection fields and graph-pool overlap flags.

Full mode additionally emits one flat `paper_rows.jsonl` row per paper. The
comparison files directly save graph/deep intersection, graph-only, deep-only,
Jaccard, and ordered arXiv ID lists for the complete local pool and rerank top-k.
For merged mode they also save every event's `N_i`, graph local/top-k IDs and the
chronological Selector slices, so graph event top-k can be compared with its
matching deep slice without reconstructing provenance. Overlapping full rows
also carry the matching graph rerank features: scheme 1 has direct graph
score/rank fields, while scheme 2 stores a list keyed by source event because one
paper may occur in several continue-event graph pools.

One expanded paper may have several source seeds. `paper_rows` contains the
aggregated `source_seed_arxiv_ids`; `expansion_edges` stores one exact
`(query_id, subquery_id, seed_arxiv_id, expanded_arxiv_id)` edge per provenance.

## Important parameters

```text
--save_level minimal|full
--run_label LABEL
--limit N
--results_per_query N
--run_per_subquery_postprocess / --no-run_per_subquery_postprocess
--run_deep_event_postprocess / --no-run_deep_event_postprocess
--run_deep_merged_postprocess / --no-run_deep_merged_postprocess
--graph_method citations|references|citations_references
--graph_expansion_limit N
--graph_cache_dir PATH
--graph_rate_limit_rps FLOAT
--graph_offline_cache_only
--embedding_base_url OLLAMA_URL
--qdrant_url URL
--qdrant_collection NAME
--postprocess_event_workers N
--postprocess_selector_concurrency N
--postprocess_embedding_concurrency N
--allow_resume_compatible_code_change
```

Dense model selection is intentionally not a CLI option. Fixed settings are
`OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b`, local-rerank batch size `64`, and
adaptive deep-retrieval fetch cap `20000`; all are written to the run manifest.
Parallelism defaults are event workers `4`, remote shadow Selector concurrency
`4`, and postprocess Ollama embedding concurrency `1`. For a 3060, keep the
embedding value at `1` initially; graph and remote Selector concurrency do not
consume GPU memory.
If date filtering, collection exhaustion, or that cap prevents a requested
budget from being filled, the actual count and fulfillment rate remain explicit
in each pool record rather than being silently treated as matched.

When both independent packages run simultaneously with the same S2 key, the
rate limit is per process. Set the two `--graph_rate_limit_rps` values so their
sum stays within the key quota. Cache writes use atomic replacement, but two
processes may still issue the same uncached request once.
