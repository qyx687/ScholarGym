# S2-native-v4 results: OnePass replay

This note records the compact, reviewable results for the final native
Semantic Scholar paper-type implementation. Generated JSONL files, API caches,
and Selector checkpoints remain in ignored experiment directories and are not
vendored into git.

## Method contract

- Qwen generates one rerank policy from each original query and that policy is
  reused for all of the query's events.
- Candidate paper types come only from Semantic Scholar
  `publicationTypes`; Qwen does not classify candidate papers.
- Policies operate directly on the 13 native S2 labels. There is no mapping
  back to the earlier functional type taxonomy.
- Type actions are `prefer`, `avoid`, and `exclude`. A positive query
  requirement is compiled to a strong `prefer` rule because native S2 labels
  do not provide reliable negative evidence for a hard `require`.
- `exclude` is direct set membership. A paper is hard-filtered only when its
  returned native labels contain the excluded label; missing metadata remains
  unknown and is not filtered.
- All non-type features, score normalization, stable tie-breaking, and the
  static fallback formula are unchanged.

## Tune100 parameter selection

The semantic-mass floor was selected only on tune100 using a seven-point sweep.
The winning floor was `0.90`.

| method | candidate F1 | recall | precision | MRR | nDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| static | 0.041725 | 0.569963 | 0.021655 | 0.151284 | 0.184306 |
| native S2, semantic floor 0.90 | 0.044400 | 0.586844 | 0.023073 | 0.187995 | 0.214929 |

The candidate-F1 delta is `+0.002674` (`+6.41%`). A 20,000-sample paired
query bootstrap gives a 95% interval of `[+0.000158, +0.005395]` and
`P(dynamic > static) = 0.9816`.

The tuning artifact is named `tune100_s2_native_v3_sem090` because it predates
the final rule-schema bump. V4 retains the selected scoring hyperparameters
and changes the paper-type rule semantics described above.

## PASA-Realscholar candidate rerank

The frozen configuration was transferred to all 50 PASA-Realscholar queries.

| method | candidate recall | candidate precision | candidate F1 | retrieved GT | MRR | nDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 0.367278 | 0.062755 | 0.107194 | 257 | 0.296560 | 0.150998 |
| S2-native-v4 | 0.404961 | 0.070070 | 0.119468 | 280 | 0.306447 | 0.169299 |

Candidate F1 improves by `+0.012274` (`+11.45%`). The paired 20,000-sample
bootstrap interval is `[+0.003140, +0.021759]`, with
`P(dynamic > static) = 0.99635`. All 50 policies completed without fallback.

## Frozen Selector replay

This is an open-loop rerank-to-Selector comparison. Query, subquery, Planner
checklist, iteration, Top-K budget, Selector prompt, model, and generation
configuration are fixed. Each arm supplies its own reranked papers and its own
raw rerank scores.

| method | Sel R | Sel P | Sel F1 | Ret R | Ret P | retrieved GT | selected GT | gap | GT conversion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 0.306777 | 0.170385 | 0.219088 | 0.367278 | 0.062755 | 257 | 203 | 54 | 0.7899 |
| S2-native-v4 | 0.341596 | 0.172228 | 0.228998 | 0.404961 | 0.070070 | 280 | 222 | 58 | 0.7929 |

Selector F1 improves by `+0.009910` (`+4.52%`), selected GT increases by 19,
and query win/tie/loss is `22/3/25`. The paired bootstrap interval for the
main-table Selector F1 delta is `[-0.018854, +0.041399]`, with
`P(delta > 0) = 0.739`; this Selector-level point estimate is not
statistically conclusive at 50 queries. All 1,798 method-event decisions
completed with zero failures.

## Type-query attribution

Only `RealScholarQuery_3` emitted a paper-type rule:
`exclude Review`. Across its events the native filter removed 284 candidate
occurrences, and the final dynamic candidate set had zero exclusion
violations, versus two for static.

Constraint satisfaction did not improve this query's relevance F1. Candidate
GT changed from 15 to 13; both Selectors retained 9 GT, while selected paper
count changed from 29 to 40. Query-level Selector F1 therefore changed from
`0.264706` to `0.227848`. The aggregate OnePass gain is consequently driven
by non-type rerank-policy changes rather than the single type query.

## Local source artifacts

Candidate rerank:

```text
eval_dynamic_rerank/s2_native/pasa_s2_native_v4_sem090/
```

Frozen Selector replay:

```text
eval_dynamic_rerank_selector/pasa_s2_native_v4_full_pipeline/
```

The authoritative compact files are `summary.json`,
`bootstrap_rerank_delta.json`, `bootstrap_selection_delta.json`, and
`run_manifest.json`. API credentials are not persisted.
