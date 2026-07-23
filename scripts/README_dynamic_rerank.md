# Query-conditioned rerank replay

The same policy/scorer is also integrated into the OnePass `code/eval.py`
full pipeline. Runtime selection is explicit:

```bash
--no-dynamic_rerank
--dynamic_rerank --paper_type_cache <native-s2-cache.jsonl>
```

The first command is the exact static arm. Dynamic rerank has one candidate
type contract: the unmapped Semantic Scholar `publicationTypes` catalog. Qwen
still generates the query policy once per original query, but never classifies
candidate papers. The S2 cache is positive-only; a miss remains unknown. The
rest of this document describes the saved-pool replay used to tune and validate
the method.

This experiment reuses materialized OnePass graph pools. It does not rerun the
retriever, graph expansion, date filtering, Planner, or Selector.

For each original query, Qwen is called once to produce a fixed policy. The
same policy is reused by every iteration, subquery, and pool event for that
query. The policy model sees only the original query. Candidate papers are
scored deterministically from saved features.

## Methods compared

`legacy_static` uses the exact existing formula:

```text
0.30 * query_score_normalized
+ 0.40 * subquery_score_normalized
+ 0.15 * intent_score
+ 0.15 * path_count_normalized
```

`dynamic_policy` compiles Qwen's discrete levels over this fixed catalog:

```text
query_similarity, subquery_similarity,
intent_background, intent_method, intent_result,
path_count, paper_type_alignment
```

Positive levels are normalized to total mass 1. A tune100-only sweep selected
0.90 as the semantic-mass floor; this is the frozen default for PASA transfer.
The remaining mass is a conservative query-conditioned graph/type residual.
A negative dimension has nominal weight -0.15 and total negative mass is
capped at 0.30. `methodology` and `method` citation labels both activate
`intent_method`.

Prompt v3 distinguishes citation-edge intent from paper topic. Generic method
or result searches do not automatically activate graph dimensions. It also
uses an empty paper-type example, so type rules are generated only for an
explicit genre/type requirement rather than copied from the prompt.

For the PASA query ending in "Please exclude survey papers", Qwen generated
the following query-level structure (abridged):

```json
{
  "weight_levels": {
    "query_similarity": "very_high",
    "subquery_similarity": "high",
    "intent_background": "off",
    "intent_method": "off",
    "intent_result": "off",
    "path_count": "off",
    "paper_type_alignment": "off"
  },
  "paper_type_rules": [
    {"types": ["Review"], "action": "exclude", "logic": "any"}
  ],
  "confidence": 0.95
}
```

This compiles the semantic score to `0.571429 * query_similarity +
0.428571 * subquery_similarity`, with a separate native-tag survey hard
constraint. There is no query-specific survey branch in code.

Paper-type rules support `prefer`, `avoid`, and `exclude` over the fixed native
S2 catalog. Positive constraints written as “require”, “must”, or “only” in the
query compile to a high-strength `prefer` rule; native S2 does not provide the
negative evidence needed for a hard `require` action. Missing S2 metadata never
causes a hard drop. An `exclude` rule is enforced by direct membership in the
paper's discrete `publicationTypes` set, even when the policy leaves the soft
`paper_type_alignment` dimension off. There is no configurable classifier-
confidence or match threshold for native S2 exclusion. The query-policy cache
key includes `--rerank_min_confidence` and the paper-type prompt/schema version,
so the v4 schema cannot silently reuse a v3 policy.

## Native S2 publication types

Semantic Scholar's `publicationTypes` metadata can be fetched in batches of at
most 500 IDs and stored with explicit provenance:

```bash
python scripts/build_s2_paper_type_cache.py \
  --pool_records "$POOL_RECORDS" \
  --query_policies <output>/query_rerank_policies.jsonl \
  --output eval_dynamic_rerank/cache/paper_type_s2.jsonl \
  --env_file "$ENV_FILE" \
  --batch_size 500 \
  --rate_limit_rps 1
```

Policies and scored rows use the 13 S2 labels directly:

```text
Review, JournalArticle, CaseReport, ClinicalTrial, Conference, Dataset,
Editorial, LettersAndComments, MetaAnalysis, News, Study, Book, BookSection
```

The three actions (`prefer`, `avoid`, `exclude`) work over those labels. S2 is
positive-only evidence: a returned label supports a rule, while a missing label
remains unknown. Therefore the presence of `Review` directly triggers an
`exclude Review` rule, but absence of `Conference` only means that the paper
does not receive a `prefer Conference` reward. Native `Dataset` means a dataset
record, not a paper that introduces or evaluates a dataset/benchmark. The
native prompt explicitly prevents that subject-matter confusion and also
prevents a query about a system that writes surveys from becoming a `Review`
preference.

Native replay is selected with:

```bash
python scripts/replay_dynamic_rerank.py \
  --pool_records "$POOL_RECORDS" \
  --benchmark "$BENCHMARK" \
  --paper_type_cache eval_dynamic_rerank/cache/paper_type_s2.jsonl \
  --policy_cache eval_dynamic_rerank/cache/query_rerank_policy_s2_native_v4.jsonl \
  --output eval_dynamic_rerank/s2_native/run_sem090 \
  --model qwen3-30b-a3b-instruct-2507 \
  --env_file "$ENV_FILE" \
  --semantic_min_mass 0.90 \
  --artifact_level selected
```

The policy cache identity includes the fixed native namespace and prompt-profile
version, so a legacy mapped policy cannot be silently reused.

## API environment

Use a user-owned file that is not committed, for example:

```bash
mkdir -p ~/.config/hybrid-paper-graph-search
chmod 700 ~/.config/hybrid-paper-graph-search
# Edit qwen.env and add: export DASHSCOPE_API_KEY='...'
chmod 600 ~/.config/hybrid-paper-graph-search/qwen.env
```

Pass it with `--env_file`. Secret values are loaded without being logged.
The loader also supports the historical ScholarGym environment where a
DashScope endpoint/key was stored under `DEEPSEEK_*`; the alias is enabled only
when the configured hostname is actually `dashscope.aliyuncs.com`.

## Tune100 commands

Set paths to the completed materialization run:

```bash
POOL_RECORDS="eval_results_stage_a_dense_tune100/<run>/onepass_artifacts/per_subquery/pool_records.jsonl"
BENCHMARK="../third_party/ScholarGym/data/scholargym_tune_non_test_100_seed20260629.jsonl"
PAPER_DB="../third_party/ScholarGym/data/hf_scholargym/scholargym_paper_db.json"
ENV_FILE="$HOME/.config/hybrid-paper-graph-search/qwen.env"
```

No-API static reconstruction:

```bash
python scripts/replay_dynamic_rerank.py \
  --pool_records "$POOL_RECORDS" \
  --benchmark "$BENCHMARK" \
  --output eval_dynamic_rerank/tune100_static \
  --artifact_level selected \
  --no-generate_policies
```

First dynamic pass, without paper-type cache:

```bash
python scripts/replay_dynamic_rerank.py \
  --pool_records "$POOL_RECORDS" \
  --benchmark "$BENCHMARK" \
  --output eval_dynamic_rerank/tune100_dynamic_prompt_v3_semantic90 \
  --policy_cache eval_dynamic_rerank/cache/query_rerank_policy_tune100_v3.jsonl \
  --model qwen3-30b-a3b-instruct-2507 \
  --env_file "$ENV_FILE" \
  --semantic_min_mass 0.90 \
  --artifact_level selected \
  --compare_legacy
```

Build the native S2 paper-type cache. The command is append-only and resumable;
it resolves each unique pool paper once:

```bash
python scripts/build_s2_paper_type_cache.py \
  --pool_records "$POOL_RECORDS" \
  --query_policies eval_dynamic_rerank/tune100_dynamic_prompt_v3_semantic90/query_rerank_policies.jsonl \
  --output eval_dynamic_rerank/cache/paper_type_s2_tune100_v3.jsonl \
  --env_file "$ENV_FILE" \
  --batch_size 500 \
  --rate_limit_rps 1 \
  --resume
```

`--query_policies` is an exact cost optimization: papers are resolved only
for queries whose validated policy contains at least one paper-type rule.
Queries without such rules cannot consume paper-type features.

Replay with type alignment and hard constraints:

```bash
python scripts/replay_dynamic_rerank.py \
  --pool_records "$POOL_RECORDS" \
  --benchmark "$BENCHMARK" \
  --paper_type_cache eval_dynamic_rerank/cache/paper_type_s2_tune100_v3.jsonl \
  --policy_cache eval_dynamic_rerank/cache/query_rerank_policy_tune100_v3.jsonl \
  --output eval_dynamic_rerank/tune100_dynamic_prompt_v3_semantic90_types \
  --model qwen3-30b-a3b-instruct-2507 \
  --env_file "$ENV_FILE" \
  --compare_legacy
```

## PASA-realscholar transfer

Freeze the prompt version, compiler constants, and chosen tune100 configuration
before running PASA-realscholar. Point the same replay at the PASA graph
`pool_records.jsonl` and `scholargym_pasa_realscholar.jsonl`. Older PASA pools
do not store split intent labels in `pool_records`; the replay automatically
backfills them from the sibling `paper_rows.jsonl`. It never infers them from
ground truth or paper text.

For PASA, build or resume its native S2 cache directly. `--exclude_cache` can
skip IDs already present in a validated base S2 cache:

```bash
python scripts/build_s2_paper_type_cache.py \
  --pool_records "$PASA_POOL_RECORDS" \
  --query_policies "$PASA_POLICIES" \
  --exclude_cache eval_dynamic_rerank/cache/paper_type_s2_tune100_v3.jsonl \
  --output eval_dynamic_rerank/cache/paper_type_s2_pasa_v3.jsonl \
  --env_file "$ENV_FILE" \
  --batch_size 500 \
  --rate_limit_rps 1 \
  --resume
```

The builder is append-only while running and atomically compacts duplicate IDs
on completion.

## Outputs and F1 definitions

Each output directory contains:

```text
summary.json
per_query_results.jsonl
query_rerank_policies.jsonl
event_results.jsonl
ranked_candidates.jsonl
```

`main_table_candidate_f1` is the repository's main-table definition: the
harmonic mean of macro-average candidate Recall and Precision.
`mean_query_candidate_f1` is the mean of per-query F1 values, retained for case
analysis. `micro_candidate_f1` is also reported. Candidate scope is the
deduplicated query-level union of every event's reranked original
`selector_top_k` slice.

With `--artifact_level selected`, `ranked_candidates.jsonl` contains the
event top-k rows plus every hard-filtered row. Filtered rows retain their
action/reason and `selected_at_event_top_k=false`, so exclusions remain
auditable without allowing those papers into Selector.

Measured static reconstruction baselines for the saved pools are:

| dataset | main-table candidate F1 | macro Recall | macro Precision |
| --- | ---: | ---: | ---: |
| tune100 | 0.041725 | 0.569963 | 0.021655 |
| PASA-realscholar | 0.107194 | 0.367278 | 0.062755 |

The PASA static replay reproduces all 899 stored event top-k sets exactly.

Frozen prompt v3 + semantic floor 0.90 measured on tune100 as follows:

| method | main-table F1 | macro Recall | macro Precision | MRR | nDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy static | 0.041725 | 0.569963 | 0.021655 | 0.151284 | 0.184306 |
| dynamic v3, no types | 0.045072 | 0.591075 | 0.023429 | 0.181103 | 0.206894 |
| dynamic v3, Qwen types | 0.045062 | 0.591075 | 0.023424 | 0.182589 | 0.207975 |
| dynamic v3, S2 types | 0.045072 | 0.591075 | 0.023429 | 0.181103 | 0.206894 |

The dynamic row improves main-table F1 by 8.02%. Qwen activated at least one
graph dimension for 45/100 queries and left all graph dimensions off for
55/100. The graph-on group improved mean per-query F1 by 0.007394 (24 improved,
4 degraded, 17 tied); the graph-off group was effectively flat (+0.000007).

For the final S2-type tune100 configuration, a paired query bootstrap with
20,000 samples gives F1 delta +0.003346, 95% percentile interval
[+0.000606, +0.006286], and probability 0.9921 that dynamic exceeds static.
In the historical v3 run, the two frozen tune type rules requested functional
types for which S2 supplied no reliable negative evidence, so they did not
cause a hard drop and the result exactly matched the no-type row. Schema v4
makes that safeguard explicit by representing positive constraints as
`prefer`, not `require`; the table above remains the historical v3 result.

The frozen configuration transferred to PASA-realscholar without further
tuning:

| method | main-table F1 | macro Recall | macro Precision | MRR | nDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy static | 0.107194 | 0.367278 | 0.062755 | 0.296560 | 0.150998 |
| dynamic v3, no types | 0.120650 | 0.418294 | 0.070491 | 0.314983 | 0.175130 |
| dynamic v3, Qwen types | 0.120541 | 0.418294 | 0.070417 | 0.322165 | 0.178511 |
| dynamic v3, S2 types | 0.120657 | 0.418294 | 0.070496 | 0.317165 | 0.177113 |

The final S2-type method improves PASA main-table F1 by 12.56%. Its paired
20,000-sample bootstrap delta is +0.013463, with 95% percentile interval
[+0.004042, +0.023201] and probability 0.99795 that dynamic exceeds static.
There were zero policy fallbacks. Qwen left every graph dimension off for
42/50 queries, used `path_count` for only 1/50, and generated paper-type rules
for 3/50.

S2 type evidence is the final default: it has the best PASA F1, removes both
survey exclusions violated by the static result, and needs only 14 batch API
calls for the 6,877 relevant candidates. For the motivating
`RealScholarQuery_3`, the excluded papers are arXiv 2307.13721 (a survey in its
title) and 2407.15672 (an overview of computer-audition foundation models).
S2 and Qwen produce identical query-level F1, MRR, and nDCG@20 for this case;
nDCG@20 rises from 0.249043 without types to 0.288380 with either source.

On `RealScholarQuery_46`, the policy asks to prefer `dataset_benchmark`.
Because S2 `Dataset` is not equivalent, capability-aware compilation disables
the unsupported type weight and exactly recovers the no-type ranking, avoiding
the small F1 decrease observed in the historical Qwen-type ablation. That
candidate-classifier path is no longer exposed by the current native-S2 method.

Generate the saved report with:

```bash
python scripts/bootstrap_rerank_delta.py \
  <output>/per_query_results.jsonl \
  --samples 20000 \
  --seed 20260719 \
  --output <output>/bootstrap_rerank_delta.json
```

Run the focused test suite with:

```bash
python -m pytest -q \
  tests/test_dynamic_rerank_skill.py \
  tests/test_dynamic_rerank_replay.py \
  tests/test_online_paper_type.py
```

The final repository-wide run passes 125 tests.

## Native S2 frozen replay (2026-07-19)

The native prompt was tuned only on tune100. A seven-point semantic-floor
sweep (`0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 1.00`) selected 0.90 by the
main-table F1; 0.875 was nearly tied but had lower MRR and nDCG@20. The
configuration was then frozen before PASA-Realscholar.

| dataset/method | main-table F1 | macro Recall | macro Precision | MRR | nDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tune100 static | 0.041725 | 0.569963 | 0.021655 | 0.151284 | 0.184306 |
| tune100 native S2 | 0.044400 | 0.586844 | 0.023073 | 0.187995 | 0.214929 |
| PASA static | 0.107194 | 0.367278 | 0.062755 | 0.296560 | 0.150998 |
| PASA native S2 (frozen) | 0.120038 | 0.417269 | 0.070102 | 0.304484 | 0.168865 |

Tune100 improves by +0.002674 (+6.41%). Its 20,000-sample paired bootstrap
interval is [+0.000158, +0.005395], with probability 0.9816 that dynamic is
better. PASA improves by +0.012844 (+11.98%); the frozen-transfer interval is
[+0.003064, +0.022770], with probability 0.9958 that dynamic is better.

Tune100 generated 100 policies with zero fallback and no type rules. PASA
generated 50 policies with zero fallback; only `RealScholarQuery_3` had a type
rule, exactly `Review exclude`. It produced 284 hard-filtered candidate
occurrences and reduced selected exclusion violations from two to zero.
`RealScholarQuery_20`, which discusses an LLM that writes surveys, correctly
kept an empty type-rule list. The prior canonical-mapped S2 run has PASA F1
0.120657, 0.000620 above the native run, but its independently generated Qwen
weights differ; that uncontrolled difference must not be attributed to the
type namespace alone.

The native reranked candidates were also replayed through the frozen OnePass
Selector. The original query, subquery, iteration, Planner checklist, prompt,
model, and event Top-K budget were preserved; each arm carried its own rerank
score.

| dataset/method | candidate F1 | Selector F1 | Selector Recall | Selector Precision | micro Selector F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tune100 static (previously completed arm) | 0.041725 | 0.274390 | 0.491425 | 0.190331 | 0.139918 |
| tune100 native S2 | 0.044400 | 0.280528 | 0.516537 | 0.192550 | 0.129360 |
| PASA static | 0.107194 | 0.219088 | 0.306777 | 0.170385 | 0.201489 |
| PASA native S2 | 0.120038 | 0.236734 | 0.353740 | 0.177892 | 0.213510 |

Tune100 main-table Selector F1 improves by +0.006138 (+2.24%). Its paired
bootstrap interval is [-0.023013, +0.039569], with probability 0.6362 that the
delta is positive. PASA improves by +0.017646 (+8.05%); the 50-query interval
is [-0.011242, +0.049676], with probability 0.8760 that the delta is positive.
Both point estimates improve, but neither Selector-level interval excludes
zero. PASA query win/tie/loss is 24/4/22.

For `RealScholarQuery_3`, native hard filtering successfully removes both S2
`Review` candidates, but the accompanying semantic reorder reduces candidate
ground-truth hits from 15 to 11. Selector ground-truth hits fall from 9 to 7,
so query-level Selector F1 changes from 0.264706 to 0.191781. This confirms
that satisfying an explicit exclusion is independently valuable but does not
guarantee higher relevance F1 for that query.

## Frozen OnePass Selector replay

`replay_dynamic_rerank_selector.py` measures the complete rerank-to-Selector
pipeline without rerunning Planner, retrieval, graph expansion, embeddings, or
Semantic Scholar. For every saved retrieval event it freezes the original
query, subquery, iteration, Planner checklist, Selector prompt/model settings,
and `selector_top_k`. It replaces the candidate block with either the static or
dynamic rerank Top-K, and each arm exposes its own raw `rerank_score` to
Selector.

The replay is event-checkpointed and safe to resume. A saved OnePass Selector
arm may be supplied with `--precomputed-decisions METHOD=PATH`; every candidate
ID, order, score, subquery, and checklist is checked before reuse. Legacy
all-empty decisions caused by malformed JSON are rejected and called again.
The parser conservatively recovers Qwen's occasional raw control characters,
invalid LaTeX-like backslashes, and trailing commas; unrecovered responses are
retried and prevent a partial run from being reported as complete.

Example:

```bash
python scripts/replay_dynamic_rerank_selector.py \
  --ranked-candidates <dynamic-rerank-output>/ranked_candidates.jsonl \
  --pool-records <onepass-artifacts>/per_subquery/pool_records.jsonl \
  --benchmark <benchmark.jsonl> \
  --paper-db <scholargym_paper_db.json> \
  --config configs/config_qwen30b_api.py \
  --env-file ~/.config/hybrid-paper-graph-search/qwen.env \
  --precomputed-decisions \
    legacy_static=<onepass-artifacts>/per_subquery/selector_decisions.jsonl \
  --output-dir <selector-replay-output> \
  --concurrency 8
```

The frozen full-pipeline results are:

| dataset | method | candidate F1 | Selector F1 | Selector Recall | Selector Precision | micro Selector F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tune100 | static | 0.041725 | 0.274390 | 0.491425 | 0.190331 | 0.139918 |
| tune100 | dynamic | 0.045072 | 0.273461 | 0.492394 | 0.189295 | 0.130092 |
| PASA-realscholar | static | 0.107194 | 0.219088 | 0.306777 | 0.170385 | 0.201489 |
| PASA-realscholar | dynamic | 0.120657 | 0.235133 | 0.360895 | 0.174369 | 0.216191 |

On tune100, dynamic candidate F1 improves 8.02%, but Selector F1 is effectively
flat/slightly lower: -0.000928 absolute (-0.34% relative). The paired
20,000-sample query bootstrap interval is [-0.033515, +0.033983], p=0.9331.

On PASA-realscholar, the dynamic point estimate retains a substantial part of
the pre-Selector gain: Selector F1 improves by +0.016045 absolute (+7.32%
relative), Recall by +0.054118, and Precision by +0.003985. The 50-query
bootstrap interval is [-0.012331, +0.047040], with probability 0.85995 that the
delta is positive and two-sided bootstrap p=0.2801. This is a positive but not
yet statistically conclusive trend. Query-level selection-F1 counts are 22
wins, 5 ties, and 23 losses; the aggregate gain comes from larger wins.

For the motivating multimodal visual/audio query (`RealScholarQuery_3`), the
dynamic paper-type rule correctly excludes surveys. The static query-level
candidate union contains two S2 `Review` papers (2307.13721 and 2407.15672),
while the dynamic union contains none; neither arm's Selector ultimately keeps
them. Dynamic hard filtering removes no ground-truth paper for this query.
Nevertheless, its semantic reorder reduces candidate GT from 15 to 13 and the
Selector keeps 9 GT in both arms while keeping more non-GT papers (36 versus
29). Therefore this query's Selector F1 decreases from 0.264706 to 0.240000.
The case shows that an explicit exclusion already present in the checklist can
be redundant with Selector, and that policy-level constraint success does not
guarantee a relevance-F1 improvement.

Selector replay outputs are saved under:

```text
eval_dynamic_rerank_selector/tune100_dynamic_v19_full_pipeline/
eval_dynamic_rerank_selector/pasa_dynamic_v4_full_pipeline/
```

Each contains `summary.json`, `per_query_results.jsonl`,
`selector_decisions.jsonl`, `bootstrap_selection_delta.json`, `run_manifest.json`,
and resumable per-event checkpoints.
