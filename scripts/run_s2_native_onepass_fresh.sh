#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_TAG" >&2
  exit 2
fi

RUN_TAG="$1"
ROOT="/home/quan/projects/hybrid_frame/hybrid-paper-graph-search/ScholarGym_OnePass_Postprocess"
PYTHON="/home/quan/miniconda3/envs/scholargym-official/bin/python"
BASELINE_RUN="${ROOT}/eval_results_onepass_dense_pasa_realscholar_full/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_pasa_realscholar_dense_rtx3060_full_run1"
POOL_RECORDS="${BASELINE_RUN}/onepass_artifacts/per_subquery/pool_records.jsonl"
PAPER_ROWS="${BASELINE_RUN}/onepass_artifacts/per_subquery/paper_rows.jsonl"
STATIC_SELECTOR="${BASELINE_RUN}/onepass_artifacts/per_subquery/selector_decisions.jsonl"
BENCHMARK="/home/quan/projects/hybrid_frame/hybrid-paper-graph-search/third_party/ScholarGym/data/scholargym_pasa_realscholar.jsonl"
PAPER_DB="/home/quan/projects/hybrid_frame/hybrid-paper-graph-search/third_party/ScholarGym/data/hf_scholargym/scholargym_paper_db.json"
ENV_FILE="/home/quan/.config/hybrid-paper-graph-search/qwen.env"
BASE_TYPE_CACHE="${ROOT}/eval_dynamic_rerank/cache/paper_type_s2_pasa_v1.jsonl"

RUN_NAME="pasa_s2_native_v4_sem090_fresh_policy_${RUN_TAG}"
BOOTSTRAP_OUT="${ROOT}/eval_dynamic_rerank/s2_native/${RUN_NAME}_policy_bootstrap"
FINAL_OUT="${ROOT}/eval_dynamic_rerank/s2_native/${RUN_NAME}"
SELECTOR_OUT="${ROOT}/eval_dynamic_rerank_selector/${RUN_NAME}_full_pipeline"
POLICY_CACHE="${ROOT}/eval_dynamic_rerank/cache/query_rerank_policy_pasa_s2_native_v4_fresh_${RUN_TAG}.jsonl"
TYPE_CACHE="${ROOT}/eval_dynamic_rerank/cache/paper_type_s2_pasa_native_fresh_${RUN_TAG}.jsonl"
LOG_PATH="${FINAL_OUT}/pipeline.log"
STATUS_PATH="${FINAL_OUT}/pipeline.status"

mkdir -p "${BOOTSTRAP_OUT}" "${FINAL_OUT}" "${SELECTOR_OUT}"
exec >>"${LOG_PATH}" 2>&1

write_status() {
  printf "%s\n" "$1" >"${STATUS_PATH}"
  printf "[%s] %s\n" "$(date --iso-8601=seconds)" "$1"
}

fail() {
  local exit_code=$?
  write_status "failed:${exit_code}"
  exit "${exit_code}"
}
trap fail ERR

if [[ -e "${POLICY_CACHE}" || -e "${TYPE_CACHE}" ]]; then
  echo "fresh cache path already exists for RUN_TAG=${RUN_TAG}" >&2
  exit 1
fi

write_status "policy_generation"
"${PYTHON}" "${ROOT}/scripts/replay_dynamic_rerank.py" \
  --pool_records "${POOL_RECORDS}" \
  --benchmark "${BENCHMARK}" \
  --paper_rows "${PAPER_ROWS}" \
  --output "${BOOTSTRAP_OUT}" \
  --model qwen3-30b-a3b-instruct-2507 \
  --env_file "${ENV_FILE}" \
  --policy_cache "${POLICY_CACHE}" \
  --semantic_min_mass 0.90 \
  --artifact_level selected \
  --generate_policies \
  --no-retry_cached_fallbacks

write_status "paper_type_cache"
cp --reflink=auto "${BASE_TYPE_CACHE}" "${TYPE_CACHE}"
"${PYTHON}" "${ROOT}/scripts/build_s2_paper_type_cache.py" \
  --pool_records "${POOL_RECORDS}" \
  --query_policies "${BOOTSTRAP_OUT}/query_rerank_policies.jsonl" \
  --output "${TYPE_CACHE}" \
  --env_file "${ENV_FILE}" \
  --batch_size 500 \
  --rate_limit_rps 1 \
  --resume

write_status "final_rerank"
"${PYTHON}" "${ROOT}/scripts/replay_dynamic_rerank.py" \
  --pool_records "${POOL_RECORDS}" \
  --benchmark "${BENCHMARK}" \
  --paper_rows "${PAPER_ROWS}" \
  --paper_type_cache "${TYPE_CACHE}" \
  --output "${FINAL_OUT}" \
  --model qwen3-30b-a3b-instruct-2507 \
  --env_file "${ENV_FILE}" \
  --policy_cache "${POLICY_CACHE}" \
  --semantic_min_mass 0.90 \
  --artifact_level selected \
  --generate_policies \
  --no-retry_cached_fallbacks

"${PYTHON}" "${ROOT}/scripts/bootstrap_rerank_delta.py" \
  "${FINAL_OUT}/per_query_results.jsonl" \
  --samples 20000 \
  --seed 20260719 \
  --output "${FINAL_OUT}/bootstrap_rerank_delta.json"

write_status "selector"
"${PYTHON}" "${ROOT}/scripts/replay_dynamic_rerank_selector.py" \
  --ranked-candidates "${FINAL_OUT}/ranked_candidates.jsonl" \
  --pool-records "${POOL_RECORDS}" \
  --benchmark "${BENCHMARK}" \
  --paper-db "${PAPER_DB}" \
  --config "${ROOT}/configs/config_qwen30b_api.py" \
  --env-file "${ENV_FILE}" \
  --precomputed-decisions "legacy_static=${STATIC_SELECTOR}" \
  --output-dir "${SELECTOR_OUT}" \
  --methods legacy_static dynamic_policy \
  --concurrency 8 \
  --max-attempts 3

write_status "complete"
