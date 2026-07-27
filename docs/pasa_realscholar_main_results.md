# PaSa-RealScholar main comparison

Current checkpoint after tuning on `tune100`. Values are percentages from one
complete 50-query run per method. They are not yet mean ± standard deviation
across repeated runs, so no significance marker is reported.

| Track | Method | R@10 | R@20 | nDCG@10 | nDCG@20 | MAP@10 | MAP@20 | Sel. R | Sel. P | Sel. F1 | GT Conv. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Controlled | Semantic | -- | -- | -- | -- | -- | -- | 37.66 | 14.78 | 21.23 | 81.91 |
| Controlled | Static-Fusion | -- | -- | -- | -- | -- | -- | 30.68 | 17.02 | 21.89 | 78.99 |
| Controlled | QuDAR-Rerank | -- | -- | -- | -- | -- | -- | 34.76 | 17.44 | 23.22 | 78.01 |
| Controlled | LLM-Semantic-Rerank | -- | -- | -- | -- | -- | -- | 29.91 | 17.02 | 21.69 | 78.80 |
| Controlled | Ours | -- | -- | -- | -- | -- | -- | 33.86 | 17.35 | 22.94 | 79.00 |
| Controlled | Oracle-Policy | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Native | QuDAR | 16.05 | 28.10 | 13.10 | 16.98 | 8.39 | 9.67 | 38.02 | 16.83 | 23.33 | 84.92 |
| Native | LLM-guided retrieval | 15.04 | 26.07 | 12.12 | 15.69 | 7.29 | 8.46 | 33.75 | 18.83 | 24.17 | 84.25 |
| Native | Ours | 17.79 | 29.03 | 13.86 | 17.74 | 8.76 | 10.18 | 39.17 | 17.20 | 23.91 | 83.60 |

## Method mapping

- `Controlled` is OnePass: the original ScholarGym query/subquery trajectory
  is frozen, and reranking is post-processing that never writes back to
  Planner or memory.
- `Semantic` is deep semantic retrieval with the baseline static ranking.
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

`R@10`, `R@20`, `nDCG@10`, `nDCG@20`, `MAP@10`, and `MAP@20` in the Native
track use one event-macro evaluator. Controlled ranking cells remain pending:
the existing OnePass replay summary instead collapses each query's event Top-K
union and therefore is not numerically interchangeable with the Native
event-macro results.

Selection recall, precision, and F1 are macro averages over the 50 original
queries. GT conversion is `Selected GT / Retrieved GT`.

## Result provenance

- Controlled baseline, Static-Fusion, and QuDAR:
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
  `../ScholarGym_PerSubquery_Online/eval_results_online_s2_native_v4_20260727`.

Exact machine-readable values are in
`docs/pasa_realscholar_main_results.json`.
