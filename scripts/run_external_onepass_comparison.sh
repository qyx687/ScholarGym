#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "usage: $0 {qudar-scores|qudar-artifact|qudar-selector|semrank-scores|semrank-artifact|semrank-selector|report}" >&2
  exit 2
fi

REPO_ROOT="/home/quan/projects/hybrid_frame/hybrid-paper-graph-search"
ONEPASS_ROOT="${REPO_ROOT}/ScholarGym_OnePass_Postprocess"
ONEPASS_RUN="${ONEPASS_ROOT}/eval_results_onepass_dense_pasa_realscholar_full/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_pasa_realscholar_dense_rtx3060_full_run1"
ARTIFACTS="${ONEPASS_RUN}/onepass_artifacts"
GRAPH_POOL="${ARTIFACTS}/per_subquery"
S2_RUN="${ONEPASS_ROOT}/eval_dynamic_rerank/s2_native/pasa_s2_native_v4_sem090_fresh_policy_20260727_173351"
STATIC_RANKED="${S2_RUN}/ranked_candidates.jsonl"
S2_SELECTOR_SUMMARY="${ONEPASS_ROOT}/eval_dynamic_rerank_selector/pasa_s2_native_v4_sem090_fresh_policy_20260727_173351_full_pipeline/summary.json"
COMPARE="${EXTERNAL_COMPARE_DIR:-${ONEPASS_ROOT}/comparisons/external_rerank_onepass_pasa_v1}"
QUDAR_COMPARE="${EXTERNAL_QUDAR_COMPARE_DIR:-${COMPARE}}"
PYTHON_BIN="/home/quan/miniconda3/envs/scholargym-official/bin/python"
BENCHMARK="${REPO_ROOT}/third_party/ScholarGym/data/scholargym_pasa_realscholar.jsonl"
PAPER_DB="${REPO_ROOT}/third_party/ScholarGym/data/hf_scholargym/scholargym_paper_db.json"
QWEN_ENV_FILE="${QWEN_ENV_FILE:-/home/quan/.config/hybrid-paper-graph-search/qwen.env}"
SEMRANK_ROOT="${REPO_ROOT}/ScholarGym_PerSubquery_Online_SemRank"
SEMRANK_CACHE="${SEMRANK_CACHE_DIR:-${SEMRANK_ROOT}/cache/semrank_classifier_only_pasa_full}"
SEMRANK_PROFILE_REPORT="${SEMRANK_QUERY_PROFILE_REPORT:-${SEMRANK_ROOT}/cache/migrations/semrank_classifier_only_query_profiles.json}"
SEMRANK_POLICY_ID="${SEMRANK_EXTERNAL_POLICY_ID:-semrank_classifier_only_qsq_qwen3_selector_minmax_v1}"

mkdir -p "${COMPARE}"

case "${MODE}" in
  qudar-scores)
    exec "${PYTHON_BIN}" \
      "${REPO_ROOT}/ScholarGym_PerSubquery_Online_QuDAR/scripts/export_qudar_frozen_pool_scores.py" \
      --candidate_run "${GRAPH_POOL}" \
      --output_jsonl "${COMPARE}/qudar_scores.jsonl" \
      --tau 2.0
    ;;
  qudar-artifact)
    exec "${PYTHON_BIN}" \
      "${ONEPASS_ROOT}/scripts/build_external_rerank_selector_artifact.py" \
      --static-ranked-candidates "${STATIC_RANKED}" \
      --pool-records "${GRAPH_POOL}/pool_records.jsonl" \
      --external-scores "${COMPARE}/qudar_scores.jsonl" \
      --external-policy-id qudar_confidence_qsq_v1_tau2 \
      --selector-score-transform none \
      --output "${COMPARE}/qudar_vs_static_ranked_candidates.jsonl"
    ;;
  qudar-selector)
    exec "${PYTHON_BIN}" \
      "${ONEPASS_ROOT}/scripts/replay_dynamic_rerank_selector.py" \
      --ranked-candidates "${COMPARE}/qudar_vs_static_ranked_candidates.jsonl" \
      --pool-records "${GRAPH_POOL}/pool_records.jsonl" \
      --benchmark "${BENCHMARK}" \
      --paper-db "${PAPER_DB}" \
      --output-dir "${COMPARE}/selector_qudar_vs_static" \
      --config "${ONEPASS_ROOT}/configs/config_qwen30b_api.py" \
      --env-file "${QWEN_ENV_FILE}" \
      --methods legacy_static dynamic_policy \
      --precomputed-decisions \
        "legacy_static=${GRAPH_POOL}/selector_decisions.jsonl" \
      --concurrency 8 \
      --max-attempts 3
    ;;
  semrank-scores)
    exec "${PYTHON_BIN}" \
      "${SEMRANK_ROOT}/scripts/replay_semrank_on_frozen_pools.py" \
      --candidate_run "${GRAPH_POOL}" \
      --query_profile_cache "${SEMRANK_CACHE}/semrank.sqlite3" \
      --paper_db "${PAPER_DB}" \
      --output_jsonl "${COMPARE}/semrank_classifier_only_scores.jsonl" \
      --checkpoint \
        "${REPO_ROOT}/third_party/SemRank/classifier/topic_classifier_specter2.pt" \
      --labels "${REPO_ROOT}/third_party/SemRank/classifier/labels.txt" \
      --concept_encoder_backend ollama \
      --concept_encoder qwen3-embedding:0.6b \
      --concept_encoder_base_url http://127.0.0.1:11434 \
      --device cuda:0 \
      --classifier_batch_size 4 \
      --encoder_batch_size 64 \
      --paper_concept_mode classifier_only \
      --cache_only \
      --cache_dir "${SEMRANK_CACHE}" \
      --llm_workers 8
    ;;
  semrank-artifact)
    exec "${PYTHON_BIN}" \
      "${ONEPASS_ROOT}/scripts/build_external_rerank_selector_artifact.py" \
      --static-ranked-candidates "${STATIC_RANKED}" \
      --pool-records "${GRAPH_POOL}/pool_records.jsonl" \
      --external-scores "${COMPARE}/semrank_classifier_only_scores.jsonl" \
      --external-policy-id "${SEMRANK_POLICY_ID}" \
      --selector-score-transform minmax \
      --output \
        "${COMPARE}/semrank_classifier_only_vs_static_minmax_ranked_candidates.jsonl"
    ;;
  semrank-selector)
    exec "${PYTHON_BIN}" \
      "${ONEPASS_ROOT}/scripts/replay_dynamic_rerank_selector.py" \
      --ranked-candidates \
        "${COMPARE}/semrank_classifier_only_vs_static_minmax_ranked_candidates.jsonl" \
      --pool-records "${GRAPH_POOL}/pool_records.jsonl" \
      --benchmark "${BENCHMARK}" \
      --paper-db "${PAPER_DB}" \
      --output-dir \
        "${COMPARE}/selector_semrank_classifier_only_minmax_vs_static" \
      --config "${ONEPASS_ROOT}/configs/config_qwen30b_api.py" \
      --env-file "${QWEN_ENV_FILE}" \
      --methods legacy_static dynamic_policy \
      --precomputed-decisions \
        "legacy_static=${GRAPH_POOL}/selector_decisions.jsonl" \
      --concurrency 8 \
      --max-attempts 3
    ;;
  report)
    exec "${PYTHON_BIN}" \
      "${ONEPASS_ROOT}/scripts/build_external_onepass_main_table.py" \
      --baseline-details "${ONEPASS_RUN}/detailed_results.jsonl" \
      --s2-ranked-candidates "${STATIC_RANKED}" \
      --s2-selector-summary \
        "${S2_SELECTOR_SUMMARY}" \
      --qudar-selector-summary \
        "${QUDAR_COMPARE}/selector_qudar_vs_static/summary.json" \
      --semrank-selector-summary \
        "${COMPARE}/selector_semrank_classifier_only_minmax_vs_static/summary.json" \
      --qudar-score-summary \
        "${QUDAR_COMPARE}/qudar_scores.jsonl.summary.json" \
      --semrank-score-summary \
        "${COMPARE}/semrank_classifier_only_scores.jsonl.summary.json" \
      --qudar-artifact-audit \
        "${QUDAR_COMPARE}/qudar_vs_static_ranked_candidates.jsonl.summary.json" \
      --semrank-artifact-audit \
        "${COMPARE}/semrank_classifier_only_vs_static_minmax_ranked_candidates.jsonl.summary.json" \
      --semrank-query-profile-report "${SEMRANK_PROFILE_REPORT}" \
      --output-dir "${COMPARE}/report"
    ;;
  *)
    echo "unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
