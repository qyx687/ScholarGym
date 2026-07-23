# S2-native-v4 results: Online per-subquery

This note records the compact, reviewable PASA-Realscholar result for the
closed-loop Online implementation. Large generated artifacts remain in ignored
evaluation directories and are not vendored into git.

## Method contract

- Qwen generates one query-conditioned rerank policy per original query.
- The policy affects the current Selector input, memory, and all later Planner
  iterations.
- Candidate paper types come only from Semantic Scholar
  `publicationTypes`; Qwen does not classify candidate papers.
- Policies use the 13 native S2 labels directly with `prefer`, `avoid`, and
  `exclude`. Positive requirements compile to strong preferences.
- Native `exclude` uses direct set membership. Missing S2 type metadata remains
  unknown and is never hard-filtered.
- Static and dynamic runs use the same paper embedding serialization:
  `scholargym_baseline_title_newline_space_abstract_v1`.

## PASA-Realscholar result

The dynamic run is compared with the completed static run after the embedding
serialization fix.

| method | Sel R | Sel P | Sel F1 | Ret R | Ret P | retrieved GT | selected GT | gap | GT conversion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static, embedding-matched | 0.369447 | 0.168209 | 0.231168 | 0.429396 | 0.069754 | 314 | 259 | 55 | 0.8248 |
| Online S2-native-v4 | 0.396092 | 0.170897 | 0.238773 | 0.450802 | 0.075035 | 310 | 271 | 39 | 0.8742 |

Main-table Selector F1 improves by `+0.007605` (`+3.29%`). Selected GT
increases by 12 even though retrieved GT decreases by 4; the retrieval-to-
selection gap shrinks by 16 and GT conversion increases by 4.94 percentage
points.

This is a closed-loop comparison, so the two arms can take different later
retrieval and planning trajectories. The result is an end-to-end method
comparison, not a paired causal estimate of rerank alone; no paired bootstrap
claim is made here.

## Type-query attribution

All 50 query policies completed without fallback. Only
`RealScholarQuery_3` emitted a type rule, exactly `exclude Review`.

For that query, static selected 6 of 39 GT papers from 30 selected papers,
while S2-native-v4 selected 11 of 39 GT papers from 34 selected papers.
Query-level Selector F1 changes from `0.173913` to `0.301370`
(`+0.127457`). The other 49 queries have no paper-type rule and still make a
positive aggregate contribution, so the full-run improvement is not solely a
paper-type effect. Because the method is closed-loop, even the type query's
gain cannot be attributed only to the hard filter.

## Run provenance

Dynamic output root:

```text
eval_results_online_dynamic_pasa_s2_native_v4/
```

Dynamic run label:

```text
pasa_dynamic_rerank_s2_native_v4_run1
```

Matching static output root:

```text
eval_results_persubquery_online_dense_pasa_realscholar_embser_v1_full/
```

The dynamic run contains 50 `query_results` rows and 50
`query_rerank_policies` rows. The compact provenance is in
`online_artifacts/run_manifest.json`; API credentials are not persisted.
