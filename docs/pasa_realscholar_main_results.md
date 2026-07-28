# PaSa-RealScholar main comparison

Current checkpoint after tuning on `tune100`. Values are percentages from one
complete 50-query run per method. They are not yet mean ± standard deviation
across repeated runs, so no significance marker is reported.

## K = 10

| Track | Method | R@10 | nDCG@10 | MAP@10 | Sel. R | Sel. P | Sel. F1 | GT Conv. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Controlled | Semantic | 5.56 | 8.79 | 4.26 | 34.44 | 19.07 | 24.54 | 76.12 |
| Controlled | Static-Fusion | 5.04 | 9.12 | 4.78 | 30.68 | 17.02 | 21.89 | 78.99 |
| Controlled | QuDAR-Rerank | 5.65 | 9.61 | 4.94 | 34.76 | 17.44 | 23.22 | 78.01 |
| Controlled | LLM-Semantic-Rerank | 4.73 | 8.99 | 4.69 | 29.91 | 17.02 | 21.69 | 78.80 |
| Controlled | Ours | 5.84 | 9.66 | 5.09 | 33.86 | 17.35 | 22.94 | 79.00 |
| Controlled | Oracle-Policy | -- | -- | -- | -- | -- | -- | -- |
| Native | QuDAR | 4.84 | 8.63 | 4.53 | 38.02 | 16.83 | 23.33 | 84.92 |
| Native | LLM-guided retrieval | 4.36 | 8.18 | 4.12 | 33.75 | 18.83 | 24.17 | 84.25 |
| Native | Ours | 5.23 | 8.63 | 4.41 | 39.61 | 17.09 | 23.88 | 87.42 |

## K = 20

| Track | Method | R@20 | nDCG@20 | MAP@20 | Sel. R | Sel. P | Sel. F1 | GT Conv. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Controlled | Semantic | 8.84 | 9.23 | 3.78 | 34.44 | 19.07 | 24.54 | 76.12 |
| Controlled | Static-Fusion | 5.04 | 7.01 | 3.13 | 30.68 | 17.02 | 21.89 | 78.99 |
| Controlled | QuDAR-Rerank | 5.65 | 7.47 | 3.32 | 34.76 | 17.44 | 23.22 | 78.01 |
| Controlled | LLM-Semantic-Rerank | 4.73 | 6.82 | 3.05 | 29.91 | 17.02 | 21.69 | 78.80 |
| Controlled | Ours | 5.84 | 7.59 | 3.52 | 33.86 | 17.35 | 22.94 | 79.00 |
| Controlled | Oracle-Policy | -- | -- | -- | -- | -- | -- | -- |
| Native | QuDAR | 8.26 | 8.99 | 3.91 | 38.02 | 16.83 | 23.33 | 84.92 |
| Native | LLM-guided retrieval | 7.42 | 8.32 | 3.45 | 33.75 | 18.83 | 24.17 | 84.25 |
| Native | Ours | 8.53 | 8.96 | 3.87 | 39.61 | 17.09 | 23.88 | 87.42 |

## Method mapping

- `Controlled` is OnePass: the original ScholarGym query/subquery trajectory
  is frozen, and reranking is post-processing that never writes back to
  Planner or memory.
- `Semantic` is the OnePass `deep_event_offset_matched` deep semantic
  retrieval arm, with no graph expansion. Each retrieval event uses the
  graph-local-pool-sized deep-retrieval budget and its static semantic order.
- `Static-Fusion` is citation/reference graph expansion plus static fusion.
- `QuDAR-Rerank` is graph expansion plus QuDAR-Confidence-QSQ.
- `LLM-Semantic-Rerank` is graph expansion plus SemRank classifier-only with
  a strict-date-safe auxiliary Top-1000 and Top-100 topic feedback.
- Controlled `Ours` is graph expansion plus the fresh-policy S2-native dynamic
  reranker.
- Native `LLM-guided retrieval` is the closed-loop Online version of the same
  SemRank classifier-only Top-1000 method.
- Native `QuDAR` and `Ours` are their closed-loop Online versions.

SemRank full is not included. Both SemRank rows use classifier topics only,
Qwen3-Embedding-0.6B for per-concept vectors, one query-profile task per
original query, and zero paper-level LLM calls. Top-107 is sensitivity-only and
is excluded from this table.

## Metric note

All ranking metrics now use one evaluator and the following fixed aggregation:

1. compute R@K, nDCG@K, and AP@K independently for every retrieval event,
   using the complete GT set of that event's original query;
2. average all events with equal weight inside each original query;
3. average the resulting 50 original-query values with equal weight.

AP@K uses `min(number of complete-query GT, K)` as its denominator, and MAP@K
is the final macro average of AP@K.

Native artifacts and Controlled `Semantic` have at least 20 ranked candidates
for every event. The four Controlled graph-postprocess arms preserve the
actual Selector input depth: 90.43% of events have 10 candidates, the
remainder have 5--9, and no event has 20. For those four rows, K=20 therefore
treats unavailable ranks 11--20 as non-hits: R@20 equals R@10, while nDCG@20
and MAP@20 use the K=20 ideal/denominator. These are valid metrics of the
saved truncated lists, but a fair exported Top-20 comparison among all
Controlled graph arms requires rerunning those four arms at output depth 20.

Selection recall, precision, and F1 are macro averages over the 50 original
queries. GT conversion is `Selected GT / Retrieved GT`.

## Result provenance

- Controlled Semantic:
  `eval_results_onepass_dense_pasa_realscholar_full/.../onepass_artifacts/deep_event`
  (`method=deep_event_offset_matched`, `rank_field=rerank_rank`).
- Controlled Static-Fusion and QuDAR:
  `comparisons/external_rerank_onepass_pasa_semrank_top107_v1/report/main_table.json`
  (the non-SemRank rows are independent of the Top-107 sensitivity setting).
- Controlled SemRank Top-1000:
  `comparisons/external_rerank_onepass_pasa_v1/report/main_table.json`.
- Controlled Ours:
  `eval_dynamic_rerank/s2_native/pasa_s2_native_v4_sem090_fresh_policy_20260727_173351`
  plus its full Selector replay.
- Native SemRank Top-1000:
  `../ScholarGym_PerSubquery_Online_SemRank/comparisons/semrank_classifier_only_full_analysis/summary.json`.
- Native QuDAR:
  `../ScholarGym_PerSubquery_Online_QuDAR/comparisons/qudar_full_analysis/summary.json`.
- Native Ours:
  `../ScholarGym_PerSubquery_Online/eval_results_online_dynamic_pasa_s2_native_v4/...pasa_dynamic_rerank_s2_native_v4_run1`
  (the earlier of the two S2-native runs).

Exact machine-readable values are in
`docs/pasa_realscholar_main_results.json`. The evaluator's complete audit and
per-query values are under
`docs/pasa_realscholar_event_query_macro_top1000/`.
