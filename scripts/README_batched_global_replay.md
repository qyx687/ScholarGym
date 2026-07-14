# Experimental global rerank + batched Selector replay

This is a legacy standalone replay for completed runs that still contain the
old `global/` artifacts. The current integrated pipeline no longer creates
those artifacts, so this script is not an input path for new runs.

It reuses a completed OnePass `full` run and never reruns the baseline retriever,
Planner, or Semantic Scholar expansion. `detailed_results.jsonl` alone is not
sufficient: the script also requires the saved baseline seed rows, global final
candidate rows, per-subquery raw component scores, and expansion edges.

## Controlled arms

Both methods use the same date-valid global candidate pool, the same Selector
budget (`global_selector_top_k`, equal to the baseline query retrieval occurrence
count), independent contiguous batches of 10, the default global checklist, and
an empty `old_overview` for every batch. Selected IDs are unioned across batches.

1. `global_original_plus_all_subqueries_max_batched_selector`
   preserves the saved global rerank score and changes only one-shot Selector to
   batched Selector.
2. `global_query_subquery_intent_path_weighted_batched_selector` uses:

   ```text
   0.30 * global-pool minmax(query raw score)
   + 0.40 * max over global-pool minmax(each executed subquery raw score)
   + 0.15 * intent score
   + 0.15 * global-pool normalized path count
   ```

The second arm changes only reranking relative to the first batched arm.

## Smoke test

Run one committed query first. This still makes roughly 2 × ceil(baseline
retrieval occurrences / 10) Selector calls because both methods are enabled.

```bash
python scripts/replay_global_batched_selector.py \
  --baseline_run_dir path/to/completed_onepass_run \
  --paper_db ../third_party/ScholarGym/data/hf_scholargym/scholargym_paper_db.json \
  --config configs/config_qwen30b_api.py \
  --output_dir eval_replay_global_batched_smoke \
  --batch_size 10 \
  --save_level full \
  --limit 1
```

Remove `--limit 1` and use a new output directory for the complete run. Query
files are written atomically under `<method>/queries/`; an interrupted replay can
resume with the same output directory and skips existing valid query files.

## Outputs

```text
run_manifest.json
evaluation_summary.json
<method>/queries/000000.json
<method>/query_results.jsonl
<method>/summary.json
```

Full query files retain rerank features/ranks, batch assignment, Selector input
IDs, selected IDs, reasons, and overview. Paper text and raw prompts are not
saved.
