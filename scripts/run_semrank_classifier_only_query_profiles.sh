#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

PYTHON_BIN="${SCHOLARGYM_PYTHON:-/home/quan/miniconda3/envs/scholargym-official/bin/python}"
ENV_FILE="${SCHOLARGYM_ENV_FILE:-${HOME}/.config/hybrid-paper-graph-search/qwen.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

FULL_RUN="eval_results_semrank_pasa_realscholar/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_per_subquery_online_semrank_qsq_sig-30e7074f4a81_semrank_qsq_qwen3_pasa_full_run1"
INITIAL_TOP_M="${SEMRANK_INITIAL_TOP_M:-1000}"
FEEDBACK_TOP_N="${SEMRANK_FEEDBACK_TOP_N:-100}"
TARGET_CACHE_DIR="${SEMRANK_CLASSIFIER_ONLY_CACHE_DIR:-cache/semrank_classifier_only_pasa_full}"
REPORT_JSON="${SEMRANK_QUERY_PROFILE_REPORT:-cache/migrations/semrank_classifier_only_query_profiles.json}"

exec "$PYTHON_BIN" scripts/build_semrank_classifier_only_query_profiles.py \
  --source_semrank_run "$FULL_RUN" \
  --paper_db ../third_party/ScholarGym/data/hf_scholargym/scholargym_paper_db.json \
  --target_cache "${TARGET_CACHE_DIR}/semrank.sqlite3" \
  --report_json "$REPORT_JSON" \
  --llm_model qwen3-30b-a3b-instruct-2507 \
  --no-llm_is_local \
  --workers "${SEMRANK_QUERY_PROFILE_WORKERS:-8}" \
  --initial_top_m "$INITIAL_TOP_M" \
  --feedback_top_n "$FEEDBACK_TOP_N" \
  --prompt_top_papers 50 \
  --candidate_topic_k 50 \
  --candidate_phrase_k 50 \
  --classifier_topic_k 100
