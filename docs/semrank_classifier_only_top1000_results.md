# SemRank classifier-only Top-1000 results

The canonical paper baseline uses the official SemRank multi-label topic
classifier and label space, but no per-paper LLM filtering or keyphrase
extraction:

```text
initial_top_m = 1000
feedback_top_n = 100
C(p) = T_classifier(p)
concept encoder = qwen3-embedding:0.6b
paper-level LLM = 0
query-level LLM = 1 task/original query
```

The Top-1000 auxiliary retrieval is strict-date-safe and builds one cached
query profile per original query. It does not add papers to the ScholarGym
candidate pool. The Top-107 variant is retained only as a sensitivity
experiment and is excluded from the main paper table.

## Current single-run checkpoints

All values are percentages. These are one complete 50-query run per track,
not mean ± standard deviation across seeds.

| Track | R@10 | R@20 | nDCG@10 | nDCG@20 | MAP@10 | MAP@20 | Sel. R | Sel. P | Sel. F1 | GT Conv. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Controlled / OnePass | -- | -- | -- | -- | -- | -- | 29.91 | 17.02 | 21.69 | 78.80 |
| Native / Online | 15.04 | 26.07 | 12.12 | 15.69 | 7.29 | 8.46 | 33.75 | 18.83 | 24.17 | 84.25 |

Controlled ranking metrics remain pending because the existing OnePass summary
uses a query-level de-duplicated-union ranking schema, whereas the Native
summary uses event-macro rerank metrics. They must not be placed in the same
columns until one evaluator is applied to both tracks.

## Audit status

- Online run: 50 queries and 913 rerank events.
- Scientific invariant audit: all checks passed.
- Query profile identity: one profile per original query.
- Concept vectors: Qwen3-only cache namespace and signature.
- Paper concepts: classifier topics only; zero paper-level LLM calls.
- Query concepts: cached Top-1000 profile; three queries used the explicit
  empty-selection fallback recorded by the run.

Canonical run command:

```bash
bash scripts/run_semrank_pasa.sh classifier-only-full
```
