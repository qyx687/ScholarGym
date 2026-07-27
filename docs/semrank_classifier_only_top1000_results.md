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

### K = 10

| Track | R@10 | nDCG@10 | MAP@10 | Sel. R | Sel. P | Sel. F1 | GT Conv. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Controlled / OnePass | 4.73 | 8.99 | 4.69 | 29.91 | 17.02 | 21.69 | 78.80 |
| Native / Online | 4.36 | 8.18 | 4.12 | 33.75 | 18.83 | 24.17 | 84.25 |

### K = 20

| Track | R@20 | nDCG@20 | MAP@20 | Sel. R | Sel. P | Sel. F1 | GT Conv. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Controlled / OnePass | 4.73 | 6.82 | 3.05 | 29.91 | 17.02 | 21.69 | 78.80 |
| Native / Online | 7.42 | 8.32 | 3.45 | 33.75 | 18.83 | 24.17 | 84.25 |

Ranking metrics are computed independently per retrieval event against the
complete GT set of its original query. Events are averaged with equal weight
inside each original query, then the 50 query means are averaged equally.
AP@K uses `min(|GT(query)|, K)` as its denominator.

The saved Controlled rankings have at most 10 entries, so their K=20 values
treat unavailable positions 11--20 as non-hits. Native rankings all contain at
least 20 entries.

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
