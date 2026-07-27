# SemRank-QSQ external reranking baseline

This fork keeps ScholarGym retrieval, citation/reference expansion, closed
corpus filtering, publication cutoff, Selector, memory, Planner, browser mode,
and per-event Selector K unchanged. It replaces only the ranking applied to the
already-built graph candidate rows.

## Method

For one original query `q`, `C(q)` is constructed once and reused across every
subquery, page, and iteration. An auxiliary original-query vector retrieval
progressively over-fetches until it has 1,000 papers *after* strict date
filtering. Its papers are used only for concept feedback and never enter the
Agent candidate pool or memory.

Every paper has one query-independent cached profile:

```text
official SPECTER2 multi-label topic classifier
  -> LLM topic filtering + keyphrase extraction
  -> C(p) = selected topics union extracted keyphrases
```

The strings in `C(q)` and `C(p)` are embedded by the same
`qwen3-embedding:0.6b` Ollama `EmbeddingProvider` instance used by dense
retrieval and closed-pool semantic scoring. Inputs are normalized concept
strings; outputs are float32 and L2-normalized. Startup rejects any mismatch
between the SemRank concept backend/model/base URL and the dense provider.
The encoder receives each selected concept string independently. It never
receives a paper title/abstract serialization, so a whole-paper dense vector
cannot substitute for the required per-concept vectors.

For each date-valid graph candidate pool:

```text
base(p) = 0.4 * query_norm(p) + 0.6 * subquery_norm(p)
concept(p) = mean[cq in C(q)] max[cp in C(p)] cosine(cq, cp)
final(p) = z_pool(base(p)) + z_pool(concept(p))
```

Population standard deviation (`ddof=0`) and epsilon `1e-12` are fixed. If
`C(q)` is empty, the explicit fallback is `final = z_pool(base)`. Citation
intent, path count, graph provenance, seed count, paper type, policy signals,
and GT labels are not inputs to this formula.

## Upstream provenance

- Paper: *Scientific Paper Retrieval with LLM-Guided Semantic-Based Ranking*,
  Findings of EMNLP 2025.
- Public implementation inspected at
  `https://github.com/yzhan238/SemRank`, revision
  `4e723ea2615da07f253672583ab8bed748a695e2`.
- The public repository supplies `classifier/labels.txt` and links its trained
  SPECTER2 topic-classifier checkpoint from its README.
- No license file was present in that inspected revision. Consequently this
  fork does not vendor upstream source or weights. It provides an independently
  written compatible loader and keeps the checkout, labels, and checkpoint
  under the ignored `../third_party/SemRank` directory.
- The public notebook contains two apparent variable-name defects
  (`selected_terms` and `new_results`). This implementation follows the
  published algorithm and declared data flow rather than copying those defects.
- The main comparison adapts SemRank's concept similarity encoder to the
  experiment-wide `qwen3-embedding:0.6b` Ollama service. This isolates the
  reranking method from a concept-encoder mismatch with the dense baselines.
- The official SemRank topic classifier remains unchanged: its
  `allenai/specter2_base` backbone is pinned to Hugging Face revision
  `3447645e1def9117997203454fa4495937bfbd83` and loads the official learned
  checkpoint.
- The experiment launcher fixes topic-classifier inference batches at 4 on
  the available 6 GiB GPU. Concept embeddings use the already-configured
  Ollama Qwen3 provider rather than loading a second SPECTER2 model.
- The compatible classifier loader matches the official inference contract:
  SPECTER2 `pooler_output`, 512-token truncation, `padding=max_length`, and the
  learned bilinear matrix applied as `doc @ W @ label.T`.

Required external files:

```text
../third_party/SemRank/classifier/topic_classifier_specter2.pt
../third_party/SemRank/classifier/labels.txt
```

Assets used for this experiment were verified as:

```text
checkpoint SHA-256  7f2f3e48e2d6c6e195e4bac2c45cee6c59c5d887b2e65e4145d7b0b53bbf306a
labels SHA-256      1b3c471856be9e28fed14135afa0d96f6e97208e055620db69aeb5e8c40e65f2
```

Missing classifier assets cause a clear startup error; the main method never
silently changes to LLM-only topic generation.

Transient LLM failures are stored with `status=failed` for audit, but are not
treated as successful cache hits on a later build or resume. Only those
explicit failures are retried. Successful profiles and valid `empty` profiles
remain immutable cache hits; retry behavior is included in the run signature.
If an otherwise complete saved response is truncated at its final JSON
delimiter, the parser can recover only fully decoded list strings from that
saved raw response. It discards any unfinished final string and still applies
the original classifier-vocabulary and title/abstract occurrence checks. This
local repair invokes neither the classifier nor the LLM and invents no text.
Query-profile construction fails closed if any of its Top-100 feedback paper
profiles has that transient failure status. The warm-up command exits nonzero
while failed records remain, so repeating it repairs only those records before
formal evaluation.

## Canonical classifier-only main comparison

The paper's canonical SemRank comparison is `classifier-only-full` with
`initial_top_m=1000`, `feedback_top_n=100`, and zero paper-level LLM calls.
The date-safe auxiliary Top-1000 is used only to construct one fixed query
profile per original query; it never enters the ScholarGym candidate pool,
Selector, memory, or Planner trajectory.

```bash
# Closed-loop run: SemRank Top-K affects Selector, memory, and later planning.
bash scripts/run_semrank_pasa.sh classifier-only-full
```

## Classifier-only Top-107 sensitivity variant

The PASA-Realscholar sensitivity experiment additionally fixes
`initial_top_m=107`, matching the rounded ScholarGym baseline mean of 5,326
deduplicated retrieved papers over 50 queries (`106.52/query`). The official
`feedback_top_n=100` is unchanged: query topics are aggregated from the first
100 papers in the strict-date-safe Top-107 auxiliary retrieval. The remaining
seven papers establish the retrieval exposure cap but do not enter topic
frequency aggregation.

Top-107 query profiles use a distinct identity and SQLite cache directory.
Paper classifier topics and Qwen concept vectors are copied from the audited
classifier-only cache, but no Top-1000 query profile can satisfy a Top-107
lookup. The saved Top-1000 date-safe retrieval is used only as an ordered
retriever-equivalent source; the Top-107 profile consumes its exact prefix.
This variant is retained for sensitivity analysis and is not used in the main
paper table.

```bash
bash scripts/prepare_semrank_classifier_only_top107_cache.sh

# Closed-loop run: SemRank Top-K affects Selector, memory, and later planning.
bash scripts/run_semrank_pasa.sh classifier-only-top107-full
```

## Commands

All commands run from this fork.

```bash
# Unit/integration tests
/home/quan/miniconda3/envs/scholargym-official/bin/python -m pytest -q

# One-query end-to-end smoke
bash scripts/run_semrank_pasa.sh smoke

# Migrate successful/failed text profiles from an archived SPECTER2-era cache.
# This copies zero concept-vector rows and invokes neither classifier nor LLM.
python scripts/migrate_semrank_text_profiles.py \
  --source_cache cache/archived/semrank.sqlite3 \
  --target_cache cache/semrank_smoke/semrank.sqlite3 \
  --report_json cache/migrations/text_only.json

# Re-encode exactly the cached selected query concepts, selected paper topics,
# and extracted paper keyphrases with Qwen3. No classifier/LLM is constructed.
python scripts/reencode_semrank_cached_concepts.py \
  --cache_path cache/semrank_smoke/semrank.sqlite3 \
  --report_json cache/migrations/qwen_reencode.json

# Seed another run with already-verified Qwen vectors. The clone refuses any
# source containing a SPECTER2 or otherwise unexpected vector namespace.
python scripts/clone_semrank_qwen_cache.py \
  --source_cache cache/semrank_smoke/semrank.sqlite3 \
  --target_cache cache/semrank_pasa_full/semrank.sqlite3 \
  --report_json cache/migrations/qwen_clone.json

# Prebuild only text profiles for the de-duplicated union found in one or more
# graph-run artifacts. Existing selected_topics/keyphrases are cache hits.
# Concurrency remains 32; the 1,024 queue batch only amortizes slow-request
# barriers. No concept vectors are written in this phase.
SEMRANK_WARMUP_DEFER_CONCEPT_ENCODING=1 \
SEMRANK_WARMUP_LLM_WORKERS=32 \
SEMRANK_WARMUP_BATCH_SIZE=1024 \
  bash scripts/run_semrank_pasa.sh warmup /path/to/run_a /path/to/run_b

# After text warm-up, encode the exact selected query concepts, selected paper
# topics, and extracted paper keyphrases into the Qwen-only vector namespace.
python scripts/reencode_semrank_cached_concepts.py \
  --cache_path cache/semrank_pasa_full/semrank.sqlite3 \
  --report_json cache/migrations/qwen_reencode_after_warmup.json

# Full SemRank with paper-level topic filtering/keyphrase extraction.
# This is not the classifier-only main-table method.
bash scripts/run_semrank_pasa.sh full

# Canonical classifier-only 50-query paper run: Top-1000 auxiliary retrieval,
# Top-100 topic feedback, one query-level LLM task, and zero paper-level LLM.
bash scripts/run_semrank_pasa.sh classifier-only-full

# Full cache-only run (fails on any missing profile)
bash scripts/run_semrank_pasa.sh cache-only

# Compare end-to-end online runs
/home/quan/miniconda3/envs/scholargym-official/bin/python \
  scripts/analyze_semrank_qsq.py \
  --semrank_run /path/to/semrank/run \
  --comparison static=/path/to/static/run \
  --comparison ours=/path/to/s2-native-v4/run \
  --output_dir comparisons/semrank_qsq

# Apply the fixed SemRank C(q) profiles to any saved candidate trajectory
/home/quan/miniconda3/envs/scholargym-official/bin/python \
  scripts/replay_semrank_on_frozen_pools.py \
  --candidate_run /path/to/qudar/run \
  --semrank_run /path/to/semrank/run \
  --paper_db ../third_party/ScholarGym/data/hf_scholargym/scholargym_paper_db.json \
  --output_jsonl comparisons/semrank_on_qudar_pools.jsonl
```

`scripts/build_semrank_paper_concepts.py` also accepts paper-ID files, multiple
artifacts, an explicit limit, or `--all_corpus`. The last option is deliberately
explicit because it invokes the official classifier and extraction LLM for all
uncached papers.

## Persistent cache and identities

The smoke run uses `cache/semrank_smoke/semrank.sqlite3`; the full run,
warmup, cache-only verification, and frozen replay use
`cache/semrank_pasa_full/semrank.sqlite3`. The full cache is deliberately
seeded with migrated text profiles and verified Qwen concept vectors, so formal
run cost reporting separates cache reuse from newly constructed profiles; it
is not described as a cold-cache measurement.
Each database contains:

- paper profiles keyed by arXiv ID, title/abstract hash, checkpoint SHA-256,
  label-space SHA-256, paper prompt, extraction LLM, and text pipeline;
- query profiles keyed by qid, query hash, date cutoff, retriever/index
  identity, Top-M/N/K settings, the complete paper pipeline identity, query
  prompt/LLM, but not the vector encoder;
- float32 concept vectors keyed by normalized concept plus the exact
  Qwen3 provider identity.

Text-profile keys are deliberately encoder-independent because topic
classification, topic filtering, keyphrase extraction, and query-concept
selection operate on text. On a text-profile cache hit, the selected strings
are passed through the active concept encoder; classifier and LLM are not
called. Vector keys include backend, model, base URL, batch identity,
normalization, dtype, and L2 contract. Qwen3 and SPECTER2 therefore occupy
different namespaces and cannot satisfy each other's lookups. The active main
cache contains only the Qwen3 namespace; the SPECTER2 vectors remain in the
archived source database.

The run manifest includes the full active Qwen3 method identity, source hash,
checkpoint and label checksums, resume signature, and output-directory
signature. Incompatible resume is refused.

## Artifacts

Under `online_artifacts/`:

- `run_manifest.json`: complete configuration and provenance.
- `semrank_query_profiles.jsonl`: one `C(q)` per original query.
- `semrank_paper_concepts.jsonl`: one emitted cache metadata record per paper
  profile used in the run.
- `semrank_event_profiles.jsonl`: per-pool distribution, fallback, runtime, and
  candidate-preservation record.
- `paper_rows.jsonl`: all candidates with raw/normalized query and subquery
  scores, base/concept/z/final scores, rank, candidate-pool signature, graph
  provenance, and Selector outcomes. Concept vectors are not duplicated here.
- `query_results.jsonl`: end-to-end outcomes and per-query cost/cache deltas.
- `*_raw.jsonl`: raw LLM responses, only when `save_level=full`.

The analysis script emits `summary.json`, `query_level.jsonl`,
`query_level.csv`, and `report.md`, including an exact formula audit.
## Safe resume after an upstream API failure

ScholarGym can append event artifacts before a query reaches
`detailed_results.jsonl`.  Preview the affected rows after a failed pass:

```bash
python scripts/clean_failed_attempt_artifacts.py \
  --run "<formal-run-directory>" \
  --expected_count 50
```

While no evaluator process is writing that run, repeat with `--apply`.  The
utility first copies every affected JSONL into a timestamped
`resume_backups/` directory, then atomically removes rows belonging only to
uncheckpointed benchmark indices.  Resume with the unchanged formal command.
