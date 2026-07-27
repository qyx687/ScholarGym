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

PAPER_DB="../third_party/ScholarGym/data/hf_scholargym/scholargym_paper_db.json"
BENCHMARK="../third_party/ScholarGym/data/scholargym_pasa_realscholar.jsonl"
CHECKPOINT="../third_party/SemRank/classifier/topic_classifier_specter2.pt"
LABELS="../third_party/SemRank/classifier/labels.txt"
MODE="${1:-smoke}"
DEFAULT_INITIAL_TOP_M=1000
if [[ "$MODE" == "classifier-only-top107-full" ]]; then
  DEFAULT_INITIAL_TOP_M=107
fi
SEMRANK_INITIAL_TOP_M_VALUE="${SEMRANK_INITIAL_TOP_M:-${DEFAULT_INITIAL_TOP_M}}"
SEMRANK_FEEDBACK_TOP_N_VALUE="${SEMRANK_FEEDBACK_TOP_N:-100}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing official SemRank checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$LABELS" ]]; then
  echo "Missing official SemRank labels: $LABELS" >&2
  exit 2
fi

COMMON=(
  scripts/run_semrank_qsq.py
  --config configs/config_qwen30b_api.py
  --paper_db "$PAPER_DB"
  --benchmark_jsonl "$BENCHMARK"
  --workflow deep_research
  --search_method vector
  --max_iterations 5
  --results_per_query 10
  --browser_mode NONE
  --save_level full
  --enable_per_subquery_graph
  --graph_rerank_method semrank_qsq
  --graph_method citations_references
  --graph_expansion_limit 100
  --graph_cache_dir cache/s2_graph_oracle
  --graph_rate_limit_rps 2.0
  --embedding_backend ollama
  --embedding_service_model qwen3-embedding:0.6b
  --embedding_base_url http://127.0.0.1:11434
  --embedding_batch_size 64
  --qdrant_url http://127.0.0.1:6433
  --qdrant_collection paper_knowledge_base
  --qdrant_timeout_seconds 60
  --semrank_initial_top_m "$SEMRANK_INITIAL_TOP_M_VALUE"
  --semrank_feedback_top_n "$SEMRANK_FEEDBACK_TOP_N_VALUE"
  --semrank_prompt_top_papers 50
  --semrank_candidate_topic_k 50
  --semrank_candidate_phrase_k 50
  --semrank_classifier_topic_k 100
  --semrank_base_query_weight 0.4
  --semrank_base_subquery_weight 0.6
  --semrank_concept_encoder_backend ollama
  --semrank_concept_encoder qwen3-embedding:0.6b
  --semrank_concept_encoder_base_url http://127.0.0.1:11434
  --semrank_concept_encoder_batch_size 64
  --semrank_topic_classifier_checkpoint "$CHECKPOINT"
  --semrank_topic_labels_path "$LABELS"
  --semrank_topic_classifier_encoder allenai/specter2_base
  --semrank_topic_classifier_encoder_revision 3447645e1def9117997203454fa4495937bfbd83
  --semrank_topic_classifier_device cuda:0
  --semrank_topic_classifier_batch_size 4
  --semrank_llm_model qwen3-30b-a3b-instruct-2507
  --no-semrank_llm_is_local
  --semrank_allow_lazy_paper_concepts
  --semrank_retry_failed_query_profiles
  --semrank_retry_failed_paper_concepts
)

case "$MODE" in
  smoke)
    exec "$PYTHON_BIN" "${COMMON[@]}" \
      --semrank_paper_concept_mode full \
      --semrank_cache_dir cache/semrank_smoke \
      --limit 1 \
      --output_dir eval_results_semrank_smoke \
      --run_label semrank_qsq_qwen3_pasa_smoke1
    ;;
  full)
    exec "$PYTHON_BIN" "${COMMON[@]}" \
      --semrank_paper_concept_mode full \
      --semrank_cache_dir cache/semrank_pasa_full \
      --output_dir eval_results_semrank_pasa_realscholar \
      --run_label semrank_qsq_qwen3_pasa_full_run1
    ;;
  cache-only)
    exec "$PYTHON_BIN" "${COMMON[@]}" \
      --semrank_paper_concept_mode full \
      --semrank_cache_dir cache/semrank_pasa_full \
      --semrank_cache_only \
      --no-semrank_allow_lazy_paper_concepts \
      --output_dir eval_results_semrank_pasa_realscholar_cache_only \
      --run_label semrank_qsq_qwen3_pasa_cache_only_run1
    ;;
  classifier-only-smoke)
    exec "$PYTHON_BIN" "${COMMON[@]}" \
      --semrank_paper_concept_mode classifier_only \
      --semrank_cache_dir cache/semrank_classifier_only_smoke \
      --limit 1 \
      --output_dir eval_results_semrank_classifier_only_smoke \
      --run_label semrank_qsq_classifier_only_qwen3_pasa_smoke1
    ;;
  classifier-only-full)
    exec "$PYTHON_BIN" "${COMMON[@]}" \
      --semrank_paper_concept_mode classifier_only \
      --semrank_cache_dir cache/semrank_classifier_only_pasa_full \
      --output_dir eval_results_semrank_classifier_only_pasa_realscholar \
      --run_label semrank_qsq_classifier_only_qwen3_pasa_full_run1
    ;;
  classifier-only-top107-full)
    exec "$PYTHON_BIN" "${COMMON[@]}" \
      --semrank_paper_concept_mode classifier_only \
      --semrank_cache_dir cache/semrank_classifier_only_pasa_top107 \
      --output_dir \
        eval_results_semrank_classifier_only_top107_pasa_realscholar \
      --run_label semrank_qsq_classifier_only_qwen3_pasa_top107_full_run1
    ;;
  warmup)
    shift
    if [[ "$#" -eq 0 ]]; then
      echo "usage: $0 warmup PATH_TO_RUN_OR_PAPER_ROWS [...]" >&2
      exit 2
    fi
    ARTIFACT_ARGS=()
    for artifact in "$@"; do
      ARTIFACT_ARGS+=(--artifact "$artifact")
    done
    WARMUP_LLM_WORKERS="${SEMRANK_WARMUP_LLM_WORKERS:-32}"
    WARMUP_BATCH_SIZE="${SEMRANK_WARMUP_BATCH_SIZE:-1024}"
    WARMUP_SHARD_COUNT="${SEMRANK_WARMUP_SHARD_COUNT:-1}"
    WARMUP_SHARD_INDEX="${SEMRANK_WARMUP_SHARD_INDEX:-0}"
    WARMUP_DEFER_ARGS=()
    if [[ "${SEMRANK_WARMUP_DEFER_CONCEPT_ENCODING:-0}" == "1" ]]; then
      WARMUP_DEFER_ARGS+=(--defer_concept_encoding)
    fi
    exec "$PYTHON_BIN" scripts/build_semrank_paper_concepts.py \
      --paper_db "$PAPER_DB" \
      "${ARTIFACT_ARGS[@]}" \
      --checkpoint "$CHECKPOINT" \
      --labels "$LABELS" \
      --concept_encoder_backend ollama \
      --concept_encoder qwen3-embedding:0.6b \
      --concept_encoder_base_url http://127.0.0.1:11434 \
      --encoder_batch_size 64 \
      --topic_classifier_encoder allenai/specter2_base \
      --topic_classifier_encoder_revision 3447645e1def9117997203454fa4495937bfbd83 \
      --device cuda:0 \
      --llm_model qwen3-30b-a3b-instruct-2507 \
      --llm_workers "$WARMUP_LLM_WORKERS" \
      --batch_size "$WARMUP_BATCH_SIZE" \
      --shard_count "$WARMUP_SHARD_COUNT" \
      --shard_index "$WARMUP_SHARD_INDEX" \
      "${WARMUP_DEFER_ARGS[@]}" \
      --cache_dir cache/semrank_pasa_full
    ;;
  retry-failed)
    shift
    if [[ "$#" -eq 0 ]]; then
      echo "usage: $0 retry-failed PAPER_ID [...]" >&2
      exit 2
    fi
    PAPER_ID_ARGS=()
    for paper_id in "$@"; do
      PAPER_ID_ARGS+=(--paper_id "$paper_id")
    done
    RETRY_CACHE_DIR="${SEMRANK_RETRY_CACHE_DIR:-cache/semrank_smoke}"
    exec "$PYTHON_BIN" scripts/build_semrank_paper_concepts.py \
      --paper_db "$PAPER_DB" \
      "${PAPER_ID_ARGS[@]}" \
      --checkpoint "$CHECKPOINT" \
      --labels "$LABELS" \
      --concept_encoder_backend ollama \
      --concept_encoder qwen3-embedding:0.6b \
      --concept_encoder_base_url http://127.0.0.1:11434 \
      --encoder_batch_size 64 \
      --topic_classifier_encoder allenai/specter2_base \
      --topic_classifier_encoder_revision 3447645e1def9117997203454fa4495937bfbd83 \
      --device cuda:0 \
      --llm_model qwen3-30b-a3b-instruct-2507 \
      --llm_workers 1 \
      --cache_dir "$RETRY_CACHE_DIR"
    ;;
  *)
    echo "usage: $0 {smoke|full|cache-only|classifier-only-smoke|classifier-only-full|classifier-only-top107-full|warmup PATH [...]|retry-failed ID [...]}" >&2
    exit 2
    ;;
esac
