# Baseline-budget Ret rerank analysis

`analyze_baseline_budget_ret_rerank.py` replays Graph and Deep merged with a
configurable four-factor formula:

```text
query_weight * query_score_normalized
+ subquery_weight * subquery_score_normalized
+ intent_weight * intent_score
+ path_weight * path_count_normalized
```

The weights must be non-negative and sum to 1. Missing feature values are
treated as zero. The defaults reproduce
`0.30Q + 0.40SQ + 0.15I + 0.15P`.

It is an offline analysis over completed OnePass artifacts. It does not call
retrieval, embeddings, Planner, Selector, or Codex annotation.

## Budget definition

For each Baseline retrieval event:

```text
K_i = len(baseline selector_decision.candidate_rows)
```

Graph ranks the matching complete local pool and retains `K_i` rows. Deep
merged ranks each merged subquery pool once, then divides the ranked list into
chronological source-event slices with the corresponding Baseline `K_i`
lengths. Within each query, retained event occurrences are unioned and paper
IDs are deduplicated before metrics are computed.

The saved pipeline Selector inputs are replayed in parallel as
`stored_pipeline`. Reproducing the saved query-level candidate IDs is a hard
validation condition.

## Run

```bash
/home/quan/miniconda3/bin/python \
  scripts/analyze_baseline_budget_ret_rerank.py \
  --run-dir 'eval_results_onepass_dense_pasa_realscholar_full/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_pasa_realscholar_dense_rtx3060_full_run1' \
  --annotation-work-dir 'eval_candidate_annotations_dense_pasa_full_v1' \
  --refinement-work-dir 'eval_candidate_semantic_relation_refinement_dense_pasa_full_v1' \
  --output-dir 'eval_candidate_ret_rerank_q030_sq040_i015_p015_baseline_k_v1' \
  --bootstrap-samples 2000
```

For the four-factor replay used in the current analysis:

```bash
/home/quan/miniconda3/bin/python \
  scripts/analyze_baseline_budget_ret_rerank.py \
  --run-dir 'eval_results_onepass_dense_pasa_realscholar_full/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_pasa_realscholar_dense_rtx3060_full_run1' \
  --annotation-work-dir 'eval_candidate_annotations_dense_pasa_full_v1' \
  --refinement-work-dir 'eval_candidate_semantic_relation_refinement_dense_pasa_full_v1' \
  --output-dir 'eval_candidate_ret_rerank_q055_sq030_i010_p005_baseline_k_v1' \
  --formula-id 'hybrid_55_30_10_05' \
  --query-weight 0.55 \
  --subquery-weight 0.30 \
  --intent-weight 0.10 \
  --path-weight 0.05 \
  --bootstrap-samples 2000
```

The run summary includes feature diagnostics. In the current saved Deep
merged artifacts, `intent_score` and `path_count_normalized` are both zero, so
the Deep ordering for this example is effectively `0.6471Q + 0.3529SQ`.

Run this command from `ScholarGym_OnePass_Postprocess/`.

## Output interpretation

Two notions of exclusivity are deliberately kept separate:

1. **Top-K selection exclusivity**: selected by one arm but not the other after
   truncation. This can be caused solely by ranking and does not prove unique
   retrieval coverage.
2. **Full-pool provenance exclusivity**: selected by one arm and absent from
   the other arm's complete saved candidate pool. This is the strict measure
   of source-specific retrieval complement.

The completed second-stage relation labels originally covered full-pool
exclusive candidates. Therefore:

- `provenance_fine_relation_summary.*` is the complete strict analysis;
- `fine_relation_summary.*` reports explicit coverage for Top-K selection
  exclusivity;
- `fine_relation_backfill_manifest.jsonl` lists selection-exclusive papers
  that would need an additional blinded second-stage pass if that secondary
  ranking-only comparison is required.

Main artifacts:

```text
summary.json
report.md
event_ret_selections.jsonl
query_ret_results.jsonl
retrieval_summary.{jsonl,csv}
partition_summary.{jsonl,csv}
formula_churn.{jsonl,csv}
annotated_formula_churn.{jsonl,csv}
semantic_group_summary.{jsonl,csv}
semantic_complement.{jsonl,csv}
provenance_semantic_summary.{jsonl,csv}
provenance_fine_relation_summary.{jsonl,csv}
fine_relation_summary.{jsonl,csv}
fine_relation_backfill_manifest.jsonl
```

## Compare Graph-only and Deep-only annotation profiles

After the rerank replay, run the complete first-stage label comparison with
query-paired bootstrap intervals:

```bash
/home/quan/miniconda3/bin/python \
  scripts/analyze_ret_label_differences.py \
  --workspace 'eval_candidate_ret_rerank_q055_sq030_i010_p005_baseline_k_v1/annotation_views/hybrid_55_30_10_05' \
  --output-dir 'eval_candidate_ret_rerank_q055_sq030_i010_p005_baseline_k_v1/label_difference_analysis/hybrid_55_30_10_05' \
  --bootstrap-samples 10000
```

This produces full distributions for relevance, semantic distance, paper type,
scholarly role, information contribution, exclusion reason, rubric-aspect
coverage, and continuous annotation properties. The primary comparison is
Graph-only versus Deep-merged-only; inclusive sets are retained as a
high-overlap sensitivity comparison.

## Metrics expected to move

The event occurrence budget must remain fixed. Candidate count can change
after query-local deduplication. Recall, precision, F1, Graph/Deep overlap,
strict provenance-unique coverage, and the semantic profile of retained papers
can change with the rerank formula.

## Diagnose Graph-only GT papers that miss Ret Top-K

To inspect every subquery occurrence of full-pool-provenance Graph-only GT
papers, including Q/SQ/intent/path contributions, production rank, a same-Q/SQ
semantic control, maximum-structure counterfactual, cutoff gap, and tie-break
status:

```bash
/home/quan/miniconda3/bin/python \
  scripts/analyze_graph_unique_gt_rank_failures.py \
  --run-dir 'eval_results_onepass_dense_pasa_realscholar_full/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_pasa_realscholar_dense_rtx3060_full_run1' \
  --annotation-work-dir 'eval_candidate_annotations_dense_pasa_full_v1' \
  --output-dir 'eval_candidate_ret_rerank_q055_sq030_i010_p005_baseline_k_v1/graph_unique_gt_rank_diagnostics'
```

The output contains `occurrence_diagnostics.*` for the complete per-subquery
view, `paper_summary.*` for one row per Graph-only GT, `summary.json`, and a
Chinese report in `report_zh.md`.
