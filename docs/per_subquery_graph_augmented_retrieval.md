# Per-subquery Graph-Augmented Retrieval

This branch implements a ScholarGym-compatible retrieval method that inserts a graph expansion and local reranking step between the original Retriever and Selector for every subquery. The rest of the ScholarGym workflow remains intentionally close to the upstream project.

## Motivation

The original ScholarGym deep-research pipeline is:

```text
Planner -> Retriever topK -> Selector -> SubQueryState -> next Planner
```

The new method changes only the candidate set shown to the Selector:

```text
Planner -> Retriever topK seeds
        -> S2 citation/reference expansion
        -> paper-DB/date cutoff filtering
        -> local seed+expanded corpus
        -> local rerank
        -> rerank topK to Selector
        -> SubQueryState -> next Planner
```

The Selector still receives the same number of papers controlled by `--results_per_query`. The difference is that those papers are no longer the raw Retriever topK. They are the topK after local graph expansion and reranking.

## Difference From The Baseline

Baseline ScholarGym:

- Retrieves papers directly from the global BM25/vector/hybrid retriever.
- Sends the retriever topK to the Selector.
- Stores these raw retrieved papers in `SubQueryState.retrieved_papers`.
- Planner reads `retrieved_count`, `selected_count`, and selector overview in later iterations.

Per-subquery graph method:

- Uses the original retriever topK only as graph seeds.
- Expands each seed through Semantic Scholar citations/references.
- Keeps only papers that are in the local ScholarGym paper database.
- Applies date cutoff to expanded papers using paper-DB dates.
- Builds a local candidate pool from seeds plus valid expanded papers.
- Reranks the local pool and sends only rerank topK to the Selector.
- Stores rerank topK in `SubQueryState.retrieved_papers`, so Planner and evaluation see the Selector-facing candidates.

Important metric caveat:

- `retrieval_recall` and `retrieval_precision` are based on the Selector-facing rerank topK candidates.
- `avg_distance` in the per-subquery method is graph-local rerank distance, not the same global-retriever distance as the original baseline. It is useful as an internal diagnostic, but should not be treated as perfectly identical to baseline `Avg.Distance`.

## Main Implementation Files

- `code/per_subquery_graph.py`
  - Semantic Scholar cache/client/rate limiting.
  - Seed extraction.
  - Citation/reference expansion.
  - Paper database membership filtering.
  - Date cutoff for expanded papers.
  - Local BM25 and feature-weighted rerank modes.

- `code/deeprag.py`
  - Calls the graph augmenter after retrieval and before selector.
  - Replaces `papers_for_selection` and `SubQueryState.retrieved_papers` with rerank topK.
  - Preserves `raw_retriever_papers` for analysis.
  - Writes per-subquery graph traces into detailed results.

- `code/eval.py`
  - Adds CLI/config switches for per-subquery graph rerank.
  - Records method settings in summary output.
  - Uses shortened output directory naming and run labels.

- `code/config.py`
  - Adds default graph method settings.

- `code/s2_rate_probe.py`
  - Utility for probing Semantic Scholar API rate limits.

## Rerank Modes

The initial mode used for first40 experiments was:

```text
original_current_subquery_weighted
```

It combines normalized BM25 scores from the original user query and the current subquery:

```text
score = alpha * BM25(original_query) + (1 - alpha) * BM25(current_subquery)
```

The full200 run used:

```text
query_subquery_intent_path_weighted
```

with weights:

```json
{
  "bm25_query_norm": 0.2,
  "bm25_subquery_norm": 0.3,
  "intent_score": 0.1,
  "path_count_norm": 0.4
}
```

This mode adds graph/path evidence to the local rerank score.

## Main Results

The lightweight result table is in:

```text
docs/results/per_subquery_main_results.csv
```

### Full Test-Fast Comparison

The historical full baseline run completed 184 of 200 queries. The missing queries failed because the Planner returned no valid subqueries and the original eval code skips failed queries instead of writing failed records. Therefore the fairest full-set comparison is on the common 184 query ids.

| Metric | Baseline common184 | Per-subquery common184 | Delta |
|---|---:|---:|---:|
| Selection Recall | 0.5581 | 0.6464 | +0.0883 |
| Selection Precision | 0.1204 | 0.1892 | +0.0687 |
| Retrieval Recall | 0.5885 | 0.7307 | +0.1422 |
| Retrieval Precision | 0.0189 | 0.0127 | -0.0062 |
| Graph-local Avg.Distance | 0.6480 | 0.6981 | +0.0502 |
| Retrieved total | 11254 | 20174 | +8920 |
| Retrieved avg/query | 61.16 | 109.64 | +48.48 |
| Selected total | 3927 | 2609 | -1318 |
| Selected avg/query | 21.34 | 14.18 | -7.16 |
| Retrieved GT | 179 / 342 | 248 / 342 | +69 |
| Selected GT | 172 / 342 | 212 / 342 | +40 |

The full per-subquery run over all 200 Test-Fast queries achieved:

| Metric | Per-subquery full200 |
|---|---:|
| Selection Recall | 0.6495 |
| Selection Precision | 0.1902 |
| Retrieval Recall | 0.7338 |
| Retrieval Precision | 0.0127 |
| Graph-local Avg.Distance | 0.7046 |
| Retrieved total / avg | 21877 / 109.39 |
| Selected total / avg | 2796 / 13.98 |
| Retrieved GT | 268 / 374 |
| Selected GT | 230 / 374 |

### First40 Stability Runs

Three first40 runs with `original_current_subquery_weighted` show run-to-run variation but a consistent graph contribution.

| Run | Selection Recall | Selection Precision | Retrieval Recall | Retrieved avg/query | Selected avg/query | Retrieved GT | Selected GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20260620-175844 | 0.5711 | 0.1140 | 0.6982 | 103.55 | 14.78 | 46 / 68 | 37 / 68 |
| 20260621-014147 | 0.6185 | 0.1392 | 0.6595 | 100.63 | 14.78 | 48 / 68 | 43 / 68 |
| 20260622-014404 | 0.6107 | 0.1235 | 0.6732 | 98.23 | 12.75 | 46 / 68 | 43 / 68 |

Reference first40 baselines:

| Baseline | Selection Recall | Selection Precision | Retrieval Recall | Retrieved avg/query | Selected avg/query | Retrieved GT |
|---|---:|---:|---:|---:|---:|---:|
| ScholarGym_trace stage1_v3 | 0.6045 | 0.1589 | 0.6420 | 114.58 | 14.98 | 43 / 68 |
| third_party dashscope_graph stage1 | 0.5774 | 0.1214 | 0.6149 | NA | NA | NA |

The trace summary for `stage1_v3` reports `enable_reasoning=true`, but the saved config and API wrapper disabled Qwen thinking. Treat it as a no-thinking/instruct baseline.

## Interpretation

The per-subquery method improves retrieval coverage by using graph neighborhoods around the raw retriever seeds. On the common184 comparison, it retrieves 69 more ground-truth papers and ultimately selects 40 more ground-truth papers than the baseline.

The method also changes the candidate distribution. It retrieves more total candidates than the baseline, so retrieval precision drops. However, the Selector selects fewer papers overall while selecting more ground-truth papers. This suggests the graph-augmented topK provides better evidence for Selector decisions even though the pre-selector candidate set is larger.

The most important evidence is not just higher retrieval recall. The selected set improves:

```text
baseline common184 selected GT:      172 / 342
per-subquery common184 selected GT:  212 / 342
```

This is the central result for the method.

## Reproduction Command

For feature-weighted rerank, the weights are read from the config file rather
than from an eval CLI flag. The full200 run used:

```python
PER_SUBQUERY_GRAPH_RERANK_FEATURE_WEIGHTS = {
    "bm25_query_norm": 0.20,
    "bm25_subquery_norm": 0.30,
    "intent_score": 0.10,
    "path_count_norm": 0.40,
}
```

Example full200 run:

```bash
cd /home/quan/projects/hybrid_frame/hybrid-paper-graph-search/Per_Subquery_Expand
conda activate scholargym-official

export DASHSCOPE_API_KEY="..."
export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export S2_API_KEY="..."
export SCHOLARGYM_MODEL="qwen3-30b-a3b-instruct-2507"

python code/eval.py \
  --config cache/config_dashscope_per_subquery.py \
  --paper_db data/scholargym_paper_db.json \
  --benchmark_jsonl ../third_party/ScholarGym/data/scholargym_test_fast.jsonl \
  --bm25_path data/bm25_index.pkl \
  --output_dir eval_results_per_subquery \
  --run_label testfast_full200 \
  --llm_model "$SCHOLARGYM_MODEL" \
  --workflow deep_research \
  --search_method bm25 \
  --prompt_type complex \
  --max_iterations 5 \
  --results_per_query 10 \
  --browser_mode NONE \
  --enable_per_subquery_graph_rerank true \
  --per_subquery_graph_method citations_references \
  --per_subquery_graph_expansion_limit 100 \
  --per_subquery_graph_rate_limit_rps 4.0 \
  --per_subquery_graph_rerank_mode query_subquery_intent_path_weighted \
  --per_subquery_graph_cache_dir cache/s2_graph_oracle \
  --per_subquery_graph_fail_fast false
```

Use `--per_subquery_graph_rerank_mode original_current_subquery_weighted --per_subquery_graph_rerank_alpha 0.5` to reproduce the earlier first40 runs.

## Notes

- Full detailed result files are intentionally not committed because they are large and are ignored by `.gitignore`.
- Semantic Scholar cache files are local artifacts and are not committed.
- The baseline full200 historical run had 16 failed queries. Use the common184 comparison for a clean apples-to-apples result until a complete 200-query baseline is rerun.
