# Full candidate-pool annotation with Codex

This pipeline annotates every deduplicated **query-paper pair** in the saved
post-process candidate artifacts. It is intentionally independent of the
rerank formula and Selector cutoff.

## Candidate scope

The default universe is the query-paper union of:

- `onepass_artifacts/baseline/paper_rows.jsonl`
- `onepass_artifacts/per_subquery/paper_rows.jsonl` (`graph`)
- `onepass_artifacts/deep_event/paper_rows.jsonl`
- `onepass_artifacts/deep_merged/paper_rows.jsonl`

Rows are deduplicated by `(query_id, paper_arxiv_id)`. A paper that appears for
two benchmark queries receives two annotations because its relationship and
semantic contribution are query dependent.

No `rerank_rank`, `in_selector_topk`, or `selector_selected` field determines
membership. The normalized occurrence file retains those fields only so the
same annotations can later be sliced at different Top-K levels.

For the dense PaSa RealScholar full run prepared on 2026-07-15, this produces:

| Item | Count |
|---|---:|
| Queries | 50 |
| Deduplicated query-paper candidates | 179,143 |
| Unique papers | 75,803 |
| Graph-only query-paper candidates | 82,627 |
| Deep-only query-paper candidates | 70,155 |
| Graph-and-deep overlap | 26,361 |
| Missing title and abstract | 0 |
| Codex batches at 20 papers per batch | 8,982 |

Here `deep` means membership in either `deep_event` or `deep_merged`.

## Annotation blinding

Codex receives only:

- the benchmark query and date;
- one fixed, query-level rubric derived from the query, planner checklist, and
  subqueries;
- each candidate's opaque ID, title, abstract, date, and categories.

Codex does **not** receive the paper's arXiv ID, retrieval source, graph path,
similarity/rerank score, rank, Selector decision, or GT membership. These fields
are reattached only during aggregation. This prevents source-aware labeling
bias such as treating graph candidates as structurally useful in advance.

## Per-paper labels

Each query-paper annotation contains:

| Field | Purpose |
|---|---|
| `relationship` | `direct`, `partial`, `contextual`, `unrelated`, or `insufficient_evidence` |
| `semantic_distance` | `exact`, `near`, `adjacent`, `distant`, or `unknown` |
| `paper_type` | Empirical, method/theory, dataset/benchmark, survey, application, and related types |
| `matched_aspect_ids` | Atomic query-rubric aspects supported by the title/abstract |
| `relation_roles` | Direct target, method component, task, dataset, evaluation, evidence, foundation, etc. |
| `contribution_types` | Method variant, application, dataset, metric, theory, evidence, comparison, limitation, synthesis, or historical context |
| `semantic_contribution` | One concise sentence stating what query-relevant information the paper adds |
| `key_concepts` | At most five concepts for later qualitative clustering |
| `exclusion_reasons` | Controlled reasons for non-direct relevance |
| `evidence_phrases` | At most two short title/abstract evidence phrases |
| `needs_full_text` | Whether title/abstract evidence is insufficient |
| `confidence` | Calibrated value in `{0.25, 0.5, 0.75, 1.0}` |

`contextual` is deliberately strict: a paper must explicitly connect to at
least one query-specific aspect and materially help answer the query. Merely
being in the same broad field, using a generic ML toolkit, or sharing keywords
is `unrelated`.

## Commands

Set the repository root to `ScholarGym_OnePass_Postprocess`.

### 1. Prepare the full candidate manifest

```bash
/home/quan/miniconda3/bin/python \
  scripts/annotate_full_candidate_pool_with_codex.py prepare \
  --run-dir 'eval_results_onepass_dense_pasa_realscholar_full/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_pasa_realscholar_dense_rtx3060_full_run1' \
  --paper-db '../third_party/ScholarGym/data/hf_scholargym/scholargym_paper_db.json' \
  --work-dir 'eval_candidate_annotations_dense_pasa_full_v1' \
  --batch-size 20
```

The 820 MB paper DB is scanned once with a low-memory streaming parser. Batch
jobs subsequently use small prepared input files and do not reload it.

### 2. Create one rubric per query

```bash
/home/quan/miniconda3/bin/python \
  scripts/annotate_full_candidate_pool_with_codex.py annotate-rubrics \
  --work-dir 'eval_candidate_annotations_dense_pasa_full_v1' \
  --workers 4 \
  --reasoning-effort medium
```

### 3. Annotate all candidate batches

```bash
/home/quan/miniconda3/bin/python \
  scripts/annotate_full_candidate_pool_with_codex.py annotate-candidates \
  --work-dir 'eval_candidate_annotations_dense_pasa_full_v1' \
  --workers 4 \
  --reasoning-effort medium
```

The command is resumable. A completed batch is skipped only when its input
hash, prompt version, candidate IDs, output enums, and rubric aspect IDs all
validate. Use `--limit N` for a smoke test and `--query-id ID` to restrict a
run. Each failed batch is retried up to three times by default.

Codex is invoked non-interactively with:

- saved CLI authentication;
- an ephemeral session;
- a read-only sandbox;
- an isolated temporary working directory;
- JSON Schema constrained final output;
- JSONL event logging, including token usage.

The script discovers Codex in `PATH` and common NVM installations. Pass an
absolute path with `--codex-bin` if necessary. `~/.codex/config.toml` is ignored
by default while saved authentication is retained; add `--load-user-config`
only when a custom provider/config is required.

### 4. Check progress

```bash
/home/quan/miniconda3/bin/python \
  scripts/annotate_full_candidate_pool_with_codex.py status \
  --work-dir 'eval_candidate_annotations_dense_pasa_full_v1'
```

### 5. Aggregate overall candidate properties

```bash
/home/quan/miniconda3/bin/python \
  scripts/annotate_full_candidate_pool_with_codex.py aggregate \
  --work-dir 'eval_candidate_annotations_dense_pasa_full_v1'
```

By default aggregation refuses incomplete or invalid batches. Use
`--allow-incomplete` only for an explicitly labeled progress snapshot.

For a detached run, strict aggregation can be chained to the annotation Linux
PID without polling by hand:

```bash
/home/quan/miniconda3/bin/python \
  scripts/annotate_full_candidate_pool_with_codex.py wait-and-aggregate \
  --work-dir 'eval_candidate_annotations_dense_pasa_full_v1' \
  --pid ANNOTATION_PID \
  --poll-seconds 60
```

The watcher runs normal strict aggregation after that PID exits. If any batch
is still missing or invalid, aggregation fails visibly instead of emitting a
misleading complete report.

### Prompt revisions without rescanning the paper DB

After intentionally changing a versioned prompt, refresh all input signatures:

```bash
/home/quan/miniconda3/bin/python \
  scripts/annotate_full_candidate_pool_with_codex.py refresh-prompts \
  --work-dir 'eval_candidate_annotations_dense_pasa_full_v1'
```

Old outputs remain on disk for audit but fail the new signature and are not
reused.

## Output layout

```text
eval_candidate_annotations_dense_pasa_full_v1/
├── run_config.json
├── prompts/
├── schemas/
├── inputs/
│   ├── rubrics/
│   └── candidates/
├── manifest/
│   ├── prepare_summary.json
│   ├── queries.jsonl
│   ├── candidates.jsonl
│   ├── occurrences.jsonl
│   ├── rubric_jobs.jsonl
│   └── candidate_jobs.jsonl
├── outputs/
│   ├── rubrics/
│   ├── candidates/
│   └── meta/
├── logs/
└── analysis/
    ├── annotations.jsonl
    ├── summary.json
    ├── source_comparison.{jsonl,csv}
    ├── label_distribution.{jsonl,csv}
    └── query_summary.jsonl
```

`manifest/candidates.jsonl` is the auditable query-paper table with full source
provenance and paper metadata. `manifest/occurrences.jsonl` preserves event-level
rank, score, seed/expanded state, edge types, source seeds, and Selector fields.
`analysis/annotations.jsonl` is a compact join of semantic labels with source
membership and GT flags.

## Overall analysis before Top-K

`source_comparison.csv` reports both micro and query-macro properties for:

- all candidates;
- graph, deep-event, deep-merged, deep-any, and baseline source membership;
- graph-only, deep-only, and graph-and-deep partitions.

The primary comparison is graph-only versus deep-only:

- direct and direct-or-partial rates;
- information-bearing rate (`direct + partial + contextual`);
- mean ordinal relevance score;
- semantic-distance and paper-type distributions;
- contribution-type, relation-role, exclusion-reason, and matched-aspect rates;
- GT rate as an external calibration variable, never as model input.

Later Top-K analysis should join `analysis/annotations.jsonl` to
`manifest/occurrences.jsonl` and apply rank cutoffs there. No paper needs to be
re-annotated when the rerank formula or K changes.

## Graph versus Deep merged primary analysis

After strict aggregation is complete, generate the two-arm full-pool report:

```bash
/home/quan/miniconda3/bin/python \
  scripts/analyze_graph_vs_deep_merged.py \
  --work-dir 'eval_candidate_annotations_dense_pasa_full_v1'
```

This analysis defines its universe as `Graph ∪ Deep merged` and recomputes
membership directly from `manifest/candidates.jsonl`. Other saved retrieval
sources do not participate in the partitions or metrics. The canonical groups
are `graph_only`, `deep_merged_only`, and `graph_and_deep_merged`.

Outputs are written under `analysis/deep_merged_primary/` and include:

- source-group relevance and direct task/method-match rates;
- semantic-complement shares and whole-pool yields;
- graph edge/path/support quality at candidate level;
- marginal coverage when graph-only is added to inclusive Deep merged;
- query-local rubric-aspect novelty;
- query-paired macro comparisons and equal-candidate-budget sensitivity;
- a self-contained `report.md` plus machine-readable JSONL/CSV/JSON files.

The current labels provide only proxies for historical predecessors, implicit
mechanisms, and cross-domain applications. Strict claims for those three
concepts require a targeted secondary annotation; the metric definitions and
report preserve this limitation explicitly.

## Second-stage semantic relation refinement

Refine the three coarse contribution labels on information-bearing exclusive
Graph/Deep-merged candidates without modifying the first-stage annotations:

```bash
PARENT='eval_candidate_annotations_dense_pasa_full_v1'
REFINED='eval_candidate_semantic_relation_refinement_dense_pasa_full_v1'

/home/quan/miniconda3/bin/python \
  scripts/annotate_semantic_relation_refinement_with_codex.py prepare \
  --parent-work-dir "$PARENT" \
  --work-dir "$REFINED" \
  --batch-size 10

/home/quan/miniconda3/bin/python \
  scripts/annotate_semantic_relation_refinement_with_codex.py annotate \
  --work-dir "$REFINED" \
  --model gpt-5.5 \
  --reasoning-effort medium \
  --workers 12

/home/quan/miniconda3/bin/python \
  scripts/annotate_semantic_relation_refinement_with_codex.py status \
  --work-dir "$REFINED"

/home/quan/miniconda3/bin/python \
  scripts/annotate_semantic_relation_refinement_with_codex.py aggregate \
  --work-dir "$REFINED"
```

The three fixed axes are:

- `historical_relation`: direct predecessor, enabling foundation, historical
  background, retrospective/survey, or insufficient evidence;
- `mechanism_relation`: explicit target mechanism, implicit explanatory
  mechanism, generic theory, or insufficient evidence;
- `domain_relation`: same domain, adjacent domain, cross-domain transfer,
  unrelated domain drift, or insufficient evidence.

Only an axis triggered by its first-stage coarse label is classified; all other
axes are fixed to `not_applicable`. For every triggered axis,
`insufficient_evidence` is a valid result and must set `needs_full_text=true`
with low confidence. Codex inputs contain no retrieval source, partition, graph
path, rank, ground-truth flag, or paper ID.
