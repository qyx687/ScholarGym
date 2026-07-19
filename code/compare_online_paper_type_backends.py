#!/usr/bin/env python3
"""Compare closed-loop dynamic runs using S2 versus Qwen paper types."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from compare_online_rerank import (
    _aggregate,
    _artifact_dir,
    _dynamic_audit,
    _latest_by_query,
    _load_jsonl,
    _load_manifest,
    _paired_delta,
    _safe_div,
)
from dimension_catalog import PAPER_TYPES
from paper_type import S2_SUPPORTED_CANONICAL_TYPES


COMMON_MANIFEST_KEYS = (
    "benchmark_jsonl_path",
    "paper_db_path",
    "llm_model",
    "prompt_type",
    "search_method",
    "scoring_backend",
    "embedding_backend",
    "embedding_model",
    "paper_embedding_serialization_id",
    "embedding_base_url",
    "qdrant_url",
    "qdrant_collection",
    "graph_method",
    "graph_expansion_limit",
    "results_per_query",
    "max_iterations",
    "browser_mode",
    "limit",
    "dynamic_rerank_requested",
    "rerank_policy_model",
    "rerank_policy_is_local",
    "rerank_min_confidence",
    "rerank_semantic_min_mass",
    "rerank_negative_weight",
    "rerank_max_negative_mass",
    "rerank_catalog_version",
    "rerank_prompt_version",
    "package_source_sha256",
)


def _policy_audit(artifact_dir: Path) -> Dict[str, Any]:
    policies = _latest_by_query(
        _load_jsonl(artifact_dir / "query_rerank_policies.jsonl")
    )
    adjustment_counts: Counter[str] = Counter()
    type_alignment_enabled = 0
    type_rule_queries = 0
    for record in policies.values():
        compiled = record.get("compiled_policy") or {}
        adjustment_counts.update(compiled.get("adjustments") or [])
        type_alignment_enabled += bool(
            compiled.get("paper_type_alignment_enabled")
        )
        type_rule_queries += bool(compiled.get("paper_type_rules") or [])
    return {
        "query_policy_count": len(policies),
        "type_rule_query_count": type_rule_queries,
        "type_alignment_enabled_query_count": type_alignment_enabled,
        "compiler_adjustment_counts": dict(sorted(adjustment_counts.items())),
    }


def _latest_type_rows(path: Path) -> tuple[Dict[str, Dict[str, Any]], int]:
    records: Dict[str, Dict[str, Any]] = {}
    occurrence_count = 0
    if not path.exists():
        return records, occurrence_count
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            paper_id = str(row.get("paper_arxiv_id") or "")
            source = str(row.get("paper_type_evidence_source") or "")
            if not paper_id or not source:
                continue
            occurrence_count += 1
            records[paper_id] = {
                "paper_arxiv_id": paper_id,
                "evidence_source": source,
                "type_probs": dict(row.get("paper_type_probs") or {}),
                "confidence": float(
                    row.get("paper_type_classifier_confidence") or 0.0
                ),
                "publication_types": list(
                    row.get("paper_type_publication_types") or []
                ),
                "supported_types": list(
                    row.get("paper_type_supported_types") or []
                ),
            }
    return records, occurrence_count


def _type_evidence_comparison(
    s2_dir: Path,
    qwen_dir: Path,
    *,
    positive_threshold: float,
) -> Dict[str, Any]:
    s2, s2_occurrences = _latest_type_rows(s2_dir / "paper_rows.jsonl")
    qwen, qwen_occurrences = _latest_type_rows(qwen_dir / "paper_rows.jsonl")
    shared = sorted(set(s2) & set(qwen))
    per_type: Dict[str, Any] = {}
    for type_name in PAPER_TYPES:
        s2_supported = type_name in S2_SUPPORTED_CANONICAL_TYPES
        both_positive = 0
        s2_positive_qwen_below = 0
        qwen_positive_s2_no_positive = 0
        neither_positive = 0
        qwen_positive = 0
        s2_positive = 0
        for paper_id in shared:
            s2_value = float(s2[paper_id]["type_probs"].get(type_name, 0.0) or 0.0)
            qwen_value = float(
                qwen[paper_id]["type_probs"].get(type_name, 0.0) or 0.0
            )
            s2_is_positive = s2_supported and s2_value >= positive_threshold
            qwen_is_positive = qwen_value >= positive_threshold
            s2_positive += s2_is_positive
            qwen_positive += qwen_is_positive
            if s2_is_positive and qwen_is_positive:
                both_positive += 1
            elif s2_is_positive:
                s2_positive_qwen_below += 1
            elif qwen_is_positive:
                qwen_positive_s2_no_positive += 1
            else:
                neither_positive += 1
        union_positive = (
            both_positive
            + s2_positive_qwen_below
            + qwen_positive_s2_no_positive
        )
        per_type[type_name] = {
            "s2_supports_type": s2_supported,
            "s2_positive_count": s2_positive if s2_supported else None,
            "qwen_positive_count": qwen_positive,
            "both_positive_count": both_positive if s2_supported else None,
            "s2_positive_qwen_below_threshold_count": (
                s2_positive_qwen_below if s2_supported else None
            ),
            # S2 is positive-only, so this is deliberately not called a false
            # negative or disagreement.
            "qwen_positive_s2_no_positive_evidence_count": (
                qwen_positive_s2_no_positive if s2_supported else None
            ),
            "neither_positive_count": neither_positive if s2_supported else None,
            "positive_set_jaccard": (
                _safe_div(both_positive, union_positive)
                if s2_supported
                else None
            ),
        }
    return {
        "positive_threshold": positive_threshold,
        "s2_typed_occurrence_count": s2_occurrences,
        "qwen_typed_occurrence_count": qwen_occurrences,
        "s2_unique_typed_paper_count": len(s2),
        "qwen_unique_typed_paper_count": len(qwen),
        "shared_typed_paper_count": len(shared),
        "s2_only_typed_paper_count": len(set(s2) - set(qwen)),
        "qwen_only_typed_paper_count": len(set(qwen) - set(s2)),
        "s2_supported_types": list(S2_SUPPORTED_CANONICAL_TYPES),
        "qwen_supported_types": list(PAPER_TYPES),
        "per_type_on_shared_papers": per_type,
    }


def compare(
    s2_run: str | Path,
    qwen_run: str | Path,
    *,
    bootstrap_samples: int = 10000,
    seed: int = 20260719,
    type_positive_threshold: float = 0.5,
    allow_legacy_s2_manifest: bool = False,
) -> Dict[str, Any]:
    if not 0.0 <= float(type_positive_threshold) <= 1.0:
        raise ValueError("type_positive_threshold must be in [0, 1]")
    s2_dir = _artifact_dir(s2_run)
    qwen_dir = _artifact_dir(qwen_run)
    s2_manifest = _load_manifest(s2_dir)
    qwen_manifest = _load_manifest(qwen_dir)
    s2_results = _latest_by_query(_load_jsonl(s2_dir / "query_results.jsonl"))
    qwen_results = _latest_by_query(_load_jsonl(qwen_dir / "query_results.jsonl"))
    query_ids = sorted(set(s2_results) & set(qwen_results))
    if not query_ids:
        raise ValueError("S2 and Qwen runs have no aligned query_id values")

    inferred_s2_backend = s2_manifest.get("paper_type_backend")
    legacy_s2_manifest = (
        inferred_s2_backend is None
        and s2_manifest.get("paper_type_source")
        in {"semantic_scholar", "semantic_scholar_publicationTypes"}
    )
    if legacy_s2_manifest:
        inferred_s2_backend = "s2"
    comparison_keys = list(COMMON_MANIFEST_KEYS)
    accepted_legacy_differences: Dict[str, Any] = {}
    if allow_legacy_s2_manifest and legacy_s2_manifest:
        comparison_keys.remove("package_source_sha256")
        accepted_legacy_differences = {
            "legacy_s2_manifest_without_backend_field": True,
            "s2_package_source_sha256": s2_manifest.get("package_source_sha256"),
            "qwen_package_source_sha256": qwen_manifest.get(
                "package_source_sha256"
            ),
            "reason": (
                "S2 run predates the explicit backend switch; its S2 scoring "
                "path is algorithmically unchanged, while new source changes "
                "add Qwen/provenance code."
            ),
        }
    mismatches = {
        key: {"s2": s2_manifest.get(key), "qwen": qwen_manifest.get(key)}
        for key in comparison_keys
        if s2_manifest.get(key) != qwen_manifest.get(key)
    }
    s2_only = sorted(set(s2_results) - set(qwen_results))
    qwen_only = sorted(set(qwen_results) - set(s2_results))
    s2_policies = _latest_by_query(
        _load_jsonl(s2_dir / "query_rerank_policies.jsonl")
    )
    qwen_policies = _latest_by_query(
        _load_jsonl(qwen_dir / "query_rerank_policies.jsonl")
    )
    policy_id_mismatches = [
        query_id
        for query_id in query_ids
        if (s2_policies.get(query_id) or {}).get("rerank_policy_id")
        != (qwen_policies.get(query_id) or {}).get("rerank_policy_id")
    ]
    mode_check = {
        "s2_run_is_dynamic": bool(s2_manifest.get("dynamic_rerank_requested")),
        "qwen_run_is_dynamic": bool(qwen_manifest.get("dynamic_rerank_requested")),
        "s2_backend_is_s2": inferred_s2_backend == "s2",
        "qwen_backend_is_qwen": qwen_manifest.get("paper_type_backend") == "qwen",
        "legacy_s2_manifest_explicitly_allowed": (
            not legacy_s2_manifest or allow_legacy_s2_manifest
        ),
    }
    result: Dict[str, Any] = {
        "s2_artifacts": str(s2_dir),
        "qwen_artifacts": str(qwen_dir),
        "aligned_query_count": len(query_ids),
        "s2_only_query_ids": s2_only,
        "qwen_only_query_ids": qwen_only,
        "policy_id_mismatch_query_ids": policy_id_mismatches,
        "comparability": {
            "comparable_config": (
                not mismatches
                and not s2_only
                and not qwen_only
                and not policy_id_mismatches
                and all(mode_check.values())
            ),
            "mismatches": mismatches,
            "mode_check": mode_check,
            "accepted_legacy_differences": accepted_legacy_differences,
            "intentionally_different_manifest_keys": [
                "paper_type_backend",
                "paper_type_source",
                "paper_type_classifier_version",
                "paper_type_model",
                "paper_type_supported_types",
                "paper_type_cache",
                "paper_type_rate_limit_rps",
                "paper_type_qwen_batch_size",
                "paper_type_qwen_is_local",
                "run_label",
            ],
        },
        "s2_audit": {**_dynamic_audit(s2_dir), **_policy_audit(s2_dir)},
        "qwen_audit": {**_dynamic_audit(qwen_dir), **_policy_audit(qwen_dir)},
        "type_evidence": _type_evidence_comparison(
            s2_dir,
            qwen_dir,
            positive_threshold=float(type_positive_threshold),
        ),
    }
    for prefix in ("candidate", "selection"):
        s2_metrics = _aggregate(s2_results, query_ids, prefix)
        qwen_metrics = _aggregate(qwen_results, query_ids, prefix)
        result[prefix] = {
            "s2": s2_metrics,
            "qwen": qwen_metrics,
            "qwen_minus_s2": _paired_delta(
                s2_results,
                qwen_results,
                query_ids,
                prefix,
                samples=bootstrap_samples,
                seed=seed,
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2_run", required=True)
    parser.add_argument("--qwen_run", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--type_positive_threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow_legacy_s2_manifest",
        action="store_true",
        help=(
            "Accept an audited pre-switch S2 run whose manifest identifies "
            "semantic_scholar_publicationTypes but lacks paper_type_backend"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(
        args.s2_run,
        args.qwen_run,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        type_positive_threshold=args.type_positive_threshold,
        allow_legacy_s2_manifest=args.allow_legacy_s2_manifest,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
