# Dynamic rerank to frozen OnePass Selector replay

## Experiment contract

The replay keeps the completed OnePass planning trajectory fixed. For each
retrieval event, both arms use the same original query, subquery, Planner
checklist, iteration, Top-K budget, Qwen Selector prompt, and generation
configuration. The only changing prompt block is the candidate paper list:

- static arm: static rerank Top-K with static rerank scores;
- dynamic arm: query-conditioned rerank Top-K with dynamic rerank scores.

The paper metadata format is unchanged (`id`, score, year, title, abstract).
Planner and its memory are not rerun, so this is an open-loop causal comparison
of rerank-to-Selector behavior rather than a closed-loop agent evaluation.

## Implementation and validation

The entrypoint is `scripts/replay_dynamic_rerank_selector.py`.

- Selector calls are independently checkpointed and resumable.
- Candidate ID/order/score/checklist equality is required before a historical
  static Selector decision can be reused.
- Qwen malformed JSON is conservatively repaired or retried. An all-empty
  legacy parse is never silently accepted as a valid zero-selection decision.
- Candidate metrics reproduced the input rerank summaries exactly: zero count,
  hit-count, and F1 mismatch on both datasets and both arms.
- tune100 completed 3,516/3,516 requests and 100/100 paired queries.
- PASA-realscholar completed 1,798/1,798 method-event results and 50/50 paired
  queries. It reused 894 strictly validated static results and called Qwen for
  all 899 dynamic events plus five malformed historical static events.
- No API credential is persisted.

## Results

The main F1 is the harmonic mean of macro-average query Recall and Precision.

| dataset | arm | candidate F1 | Selector F1 | Recall | Precision | mean-query F1 | micro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tune100 | static | 0.041725 | 0.274390 | 0.491425 | 0.190331 | 0.232961 | 0.139918 |
| tune100 | dynamic | 0.045072 | 0.273461 | 0.492394 | 0.189295 | 0.228893 | 0.130092 |
| PASA-realscholar | static | 0.107194 | 0.219088 | 0.306777 | 0.170385 | 0.186397 | 0.201489 |
| PASA-realscholar | dynamic | 0.120657 | 0.235133 | 0.360895 | 0.174369 | 0.197537 | 0.216191 |

Dynamic minus static after Selector:

| dataset | F1 delta | relative F1 | Recall delta | Precision delta | selected count delta | selected-GT delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tune100 | -0.000928 | -0.34% | +0.000969 | -0.001036 | +64 | -3 |
| PASA-realscholar | +0.016045 | +7.32% | +0.054118 | +0.003985 | +122 | +28 |

The PASA point estimate shows that Selector retains part of the dynamic rerank
benefit. The tune result shows that this is not automatic: Selector can erase a
candidate-ranking gain by accepting additional false positives.

## Paired uncertainty

Bootstrap unit: query; 20,000 paired resamples; seed 20260719.

| dataset | point F1 delta | 95% interval | P(delta > 0) | two-sided bootstrap p | wins/ties/losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| tune100 | -0.000928 | [-0.033515, +0.033983] | 0.46655 | 0.9331 | 22/52/26 |
| PASA-realscholar | +0.016045 | [-0.012331, +0.047040] | 0.85995 | 0.2801 | 22/5/23 |

The PASA trend is positive but its confidence interval crosses zero at n=50.
It should not yet be described as statistically significant.

## Motivating exclusion query

`RealScholarQuery_3` asks for visual/audio multimodal foundation-model papers
and explicitly excludes surveys.

- The policy selects a hard `survey_review` exclusion and uses only query and
  subquery semantic dimensions for positive scoring.
- The static candidate union contains two S2 `Review` papers; dynamic contains
  zero. Selector also discards those two in the static arm.
- No ground-truth paper is removed by the survey hard filter.
- Static candidate metrics: 91 papers, 15 GT, F1 0.230769.
- Dynamic candidate metrics: 81 papers, 13 GT, F1 0.216667.
- Both Selectors keep 9 GT; static keeps 29 total and dynamic keeps 36 total.
- Final F1 therefore changes from 0.264706 to 0.240000.

The constraint is enforced correctly, but it is redundant with the saved
Planner checklist and Selector. The remaining semantic reorder loses useful
candidate slots and the dynamic-score distribution causes Selector to keep
more false positives. This motivates calibrating scores across policies and/or
making Selector aware of the policy rationale in a future, separate ablation.

## Outputs

- `eval_dynamic_rerank_selector/tune100_dynamic_v19_full_pipeline/`
- `eval_dynamic_rerank_selector/pasa_dynamic_v4_full_pipeline/`

The decision JSONL files support per-query and per-event case analysis without
calling the API again.
