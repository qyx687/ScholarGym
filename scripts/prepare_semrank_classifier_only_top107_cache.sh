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

SOURCE_CACHE="cache/semrank_classifier_only_pasa_full/semrank.sqlite3"
TARGET_DIR="cache/semrank_classifier_only_pasa_top107"
TARGET_CACHE="${TARGET_DIR}/semrank.sqlite3"
DERIVE_REPORT="cache/migrations/semrank_classifier_only_top107_derive.json"
PROFILE_REPORT="cache/migrations/semrank_classifier_only_top107_query_profiles.json"
AUDIT_REPORT="cache/migrations/semrank_classifier_only_top107_audit.json"
LOG_PATH="cache/migrations/semrank_classifier_only_top107_prepare.log"
STATUS_PATH="cache/migrations/semrank_classifier_only_top107_prepare.status"

mkdir -p cache/migrations
exec >>"$LOG_PATH" 2>&1

write_status() {
  printf "%s\n" "$1" >"$STATUS_PATH"
  printf "[%s] %s\n" "$(date --iso-8601=seconds)" "$1"
}

fail() {
  local exit_code=$?
  write_status "failed:${exit_code}"
  exit "$exit_code"
}
trap fail ERR

if [[ ! -f "$TARGET_CACHE" ]]; then
  write_status "derive_paper_cache"
  "$PYTHON_BIN" scripts/derive_semrank_classifier_only_cache.py \
    --source_cache "$SOURCE_CACHE" \
    --target_cache "$TARGET_CACHE" \
    --report_json "$DERIVE_REPORT"
fi

write_status "build_query_profiles"
SEMRANK_INITIAL_TOP_M=107 \
SEMRANK_FEEDBACK_TOP_N=100 \
SEMRANK_CLASSIFIER_ONLY_CACHE_DIR="$TARGET_DIR" \
SEMRANK_QUERY_PROFILE_REPORT="$PROFILE_REPORT" \
  bash scripts/run_semrank_classifier_only_query_profiles.sh

write_status "audit"
"$PYTHON_BIN" scripts/audit_semrank_classifier_only_cache.py \
  --cache_path "$TARGET_CACHE" \
  --report_json "$AUDIT_REPORT"

write_status "complete"
