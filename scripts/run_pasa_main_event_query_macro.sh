#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${HYBRID_REPO_ROOT:-/home/quan/projects/hybrid_frame/hybrid-paper-graph-search}"
ONEPASS_ROOT="${REPO_ROOT}/ScholarGym_OnePass_Postprocess"
PYTHON_BIN="${SCHOLARGYM_PYTHON:-/home/quan/miniconda3/envs/scholargym-official/bin/python}"
BENCHMARK="${REPO_ROOT}/third_party/ScholarGym/data/scholargym_pasa_realscholar.jsonl"

BASELINE_RUN="${ONEPASS_ROOT}/eval_results_onepass_dense_pasa_realscholar_full/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_pasa_realscholar_dense_rtx3060_full_run1"
SEMANTIC_ROWS="${BASELINE_RUN}/onepass_artifacts/deep_event/paper_rows.jsonl"
S2_RANKED="${ONEPASS_ROOT}/eval_dynamic_rerank/s2_native/pasa_s2_native_v4_sem090_fresh_policy_20260727_173351/ranked_candidates.jsonl"
QUDAR_RANKED="${ONEPASS_ROOT}/comparisons/external_rerank_onepass_pasa_v1/qudar_vs_static_ranked_candidates.jsonl"
SEMRANK_RANKED="${ONEPASS_ROOT}/comparisons/external_rerank_onepass_pasa_v1/semrank_classifier_only_vs_static_minmax_ranked_candidates.jsonl"

QUDAR_NATIVE="${REPO_ROOT}/ScholarGym_PerSubquery_Online_QuDAR/eval_results_qudar_pasa_realscholar/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_per_subquery_online_qudar_confidence_qsq_tau-2_qudar_confidence_qsq_pasa_full_run1/online_artifacts/paper_rows.jsonl"
SEMRANK_NATIVE="${REPO_ROOT}/ScholarGym_PerSubquery_Online_SemRank/eval_results_semrank_classifier_only_pasa_realscholar/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_per_subquery_online_semrank_qsq_sig-048176e801a7_semrank_qsq_classifier_only_qwen3_pasa_full_run1/online_artifacts/paper_rows.jsonl"
OURS_NATIVE="${REPO_ROOT}/ScholarGym_PerSubquery_Online/eval_results_online_dynamic_pasa_s2_native_v4/qwen3-30b-a3b-instruct-2507_complex_vector_deep_research_topk-[5, 10, 20]_maxq-10_instruct_non-structured_NONE_per_subquery_online_dynamic_rerank_v1_type-s2-native_pasa_dynamic_rerank_s2_native_v4_run1/online_artifacts/paper_rows.jsonl"

OUTPUT_DIR="${EVENT_QUERY_MACRO_OUTPUT_DIR:-${ONEPASS_ROOT}/docs/pasa_realscholar_event_query_macro_top1000}"

exec "${PYTHON_BIN}" \
  "${ONEPASS_ROOT}/scripts/analyze_event_query_macro_rankings.py" \
  --benchmark "${BENCHMARK}" \
  --flat "Controlled/Semantic" "${SEMANTIC_ROWS}" rerank_rank \
  --nested "Controlled/Static-Fusion" "${S2_RANKED}" legacy_static \
  --nested "Controlled/Ours" "${S2_RANKED}" dynamic_policy \
  --nested "Controlled/QuDAR-Rerank" "${QUDAR_RANKED}" dynamic_policy \
  --nested \
    "Controlled/LLM-Semantic-Rerank" \
    "${SEMRANK_RANKED}" \
    dynamic_policy \
  --flat "Native/QuDAR" "${QUDAR_NATIVE}" rerank_rank \
  --flat \
    "Native/LLM-guided-retrieval" \
    "${SEMRANK_NATIVE}" \
    rerank_rank \
  --flat "Native/Ours" "${OURS_NATIVE}" rerank_rank \
  --cutoff 10 \
  --cutoff 20 \
  --expected_query_count 50 \
  --output_dir "${OUTPUT_DIR}"
