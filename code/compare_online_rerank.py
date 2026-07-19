#!/usr/bin/env python3
"""Compare static and dynamic closed-loop Online runs on aligned queries."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


def _artifact_dir(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / "query_results.jsonl").exists():
        return path
    if (path / "online_artifacts" / "query_results.jsonl").exists():
        return path / "online_artifacts"
    raise FileNotFoundError(
        f"cannot find query_results.jsonl below {path}; pass a run or online_artifacts directory"
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if isinstance(value, dict):
                output.append(value)
    return output


def _latest_by_query(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for record in records:
        query_id = str(record.get("query_id") or "")
        if query_id:
            output[query_id] = dict(record)
    return output


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def _query_metrics(record: Mapping[str, Any], prefix: str) -> Dict[str, float | int]:
    if prefix == "candidate":
        predicted = int(record.get("candidate_count") or 0)
        hit_ids = record.get("candidate_gt_ids") or []
    elif prefix == "selection":
        predicted = int(record.get("selected_count") or 0)
        hit_ids = record.get("selected_gt_ids") or []
    else:
        raise ValueError(prefix)
    hits = len(set(str(value) for value in hit_ids))
    gt_count = int(record.get("gt_count") or 0)
    precision = _safe_div(hits, predicted)
    recall = _safe_div(hits, gt_count)
    return {
        "hits": hits,
        "predicted": predicted,
        "gt": gt_count,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _aggregate(
    records: Mapping[str, Mapping[str, Any]],
    query_ids: Sequence[str],
    prefix: str,
) -> Dict[str, float | int]:
    rows = [_query_metrics(records[query_id], prefix) for query_id in query_ids]
    hits = sum(int(row["hits"]) for row in rows)
    predicted = sum(int(row["predicted"]) for row in rows)
    gt = sum(int(row["gt"]) for row in rows)
    micro_precision = _safe_div(hits, predicted)
    micro_recall = _safe_div(hits, gt)
    macro_precision = mean(float(row["precision"]) for row in rows)
    macro_recall = mean(float(row["recall"]) for row in rows)
    return {
        "query_count": len(rows),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        # ScholarGym main-table convention: take the harmonic mean only
        # after macro-averaging precision and recall.
        "macro_f1": _f1(macro_precision, macro_recall),
        "mean_per_query_f1": mean(float(row["f1"]) for row in rows),
        "micro_hits": hits,
        "micro_predicted": predicted,
        "micro_gt": gt,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": _f1(micro_precision, micro_recall),
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_delta(
    static: Mapping[str, Mapping[str, Any]],
    dynamic: Mapping[str, Mapping[str, Any]],
    query_ids: Sequence[str],
    prefix: str,
    *,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    per_query_deltas = [
        float(_query_metrics(dynamic[query_id], prefix)["f1"])
        - float(_query_metrics(static[query_id], prefix)["f1"])
        for query_id in query_ids
    ]
    observed_delta = float(_aggregate(dynamic, query_ids, prefix)["macro_f1"]) - float(
        _aggregate(static, query_ids, prefix)["macro_f1"]
    )
    rng = random.Random(seed)
    bootstrap = []
    if query_ids:
        for _ in range(max(0, int(samples))):
            sampled_ids = [rng.choice(query_ids) for _ in query_ids]
            bootstrap.append(
                float(_aggregate(dynamic, sampled_ids, prefix)["macro_f1"])
                - float(_aggregate(static, sampled_ids, prefix)["macro_f1"])
            )
    improved = sum(delta > 0.0 for delta in per_query_deltas)
    tied = sum(abs(delta) <= 1e-12 for delta in per_query_deltas)
    return {
        "macro_f1_delta": observed_delta,
        "paired_bootstrap_95_ci": (
            [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)]
            if bootstrap
            else [0.0, 0.0]
        ),
        "improved_query_count": improved,
        "tied_query_count": tied,
        "degraded_query_count": len(per_query_deltas) - improved - tied,
        "bootstrap_samples": max(0, int(samples)),
        "bootstrap_seed": seed,
    }


def _load_manifest(artifact_dir: Path) -> Dict[str, Any]:
    path = artifact_dir / "run_manifest.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _comparability(
    static_manifest: Mapping[str, Any], dynamic_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    keys = (
        "benchmark_jsonl_path",
        "paper_db_path",
        "llm_model",
        "search_method",
        "scoring_backend",
        "embedding_model",
        "qdrant_collection",
        "graph_method",
        "graph_expansion_limit",
        "results_per_query",
        "max_iterations",
        "browser_mode",
    )
    mismatches = {
        key: {"static": static_manifest.get(key), "dynamic": dynamic_manifest.get(key)}
        for key in keys
        if static_manifest.get(key) != dynamic_manifest.get(key)
    }
    static_is_dynamic = bool(
        static_manifest.get(
            "dynamic_rerank_requested",
            static_manifest.get("rerank_formula_id") == "dynamic_rerank_v1",
        )
    )
    dynamic_is_dynamic = bool(
        dynamic_manifest.get(
            "dynamic_rerank_requested",
            dynamic_manifest.get("rerank_formula_id") == "dynamic_rerank_v1",
        )
    )
    mode_check = {
        "static_run_uses_static_formula": not static_is_dynamic,
        "dynamic_run_requests_dynamic_formula": dynamic_is_dynamic,
    }
    return {
        "comparable_config": not mismatches and all(mode_check.values()),
        "mismatches": mismatches,
        "mode_check": mode_check,
    }


def _dynamic_audit(artifact_dir: Path) -> Dict[str, Any]:
    policies = _latest_by_query(
        _load_jsonl(artifact_dir / "query_rerank_policies.jsonl")
    )
    filters = _load_jsonl(artifact_dir / "filter_stats.jsonl")
    return {
        "policy_count": len(policies),
        "policy_fallback_count": sum(
            bool(record.get("used_fallback")) for record in policies.values()
        ),
        "policy_fallback_rate": _safe_div(
            sum(bool(record.get("used_fallback")) for record in policies.values()),
            len(policies),
        ),
        "retrieval_event_count": len(filters),
        "hard_filtered_candidate_count": sum(
            int(record.get("hard_filtered_candidate_count") or 0)
            for record in filters
        ),
        "paper_type_record_count": sum(
            int(record.get("paper_type_record_count") or 0) for record in filters
        ),
    }


def compare(
    static_run: str | Path,
    dynamic_run: str | Path,
    *,
    bootstrap_samples: int = 10000,
    seed: int = 20260719,
) -> Dict[str, Any]:
    static_dir = _artifact_dir(static_run)
    dynamic_dir = _artifact_dir(dynamic_run)
    static = _latest_by_query(_load_jsonl(static_dir / "query_results.jsonl"))
    dynamic = _latest_by_query(_load_jsonl(dynamic_dir / "query_results.jsonl"))
    query_ids = sorted(set(static) & set(dynamic))
    if not query_ids:
        raise ValueError("static and dynamic runs have no aligned query_id values")
    result: Dict[str, Any] = {
        "static_artifacts": str(static_dir),
        "dynamic_artifacts": str(dynamic_dir),
        "aligned_query_count": len(query_ids),
        "static_only_query_ids": sorted(set(static) - set(dynamic)),
        "dynamic_only_query_ids": sorted(set(dynamic) - set(static)),
        "comparability": _comparability(
            _load_manifest(static_dir), _load_manifest(dynamic_dir)
        ),
        "dynamic_audit": _dynamic_audit(dynamic_dir),
    }
    for prefix in ("candidate", "selection"):
        static_metrics = _aggregate(static, query_ids, prefix)
        dynamic_metrics = _aggregate(dynamic, query_ids, prefix)
        result[prefix] = {
            "static": static_metrics,
            "dynamic": dynamic_metrics,
            "dynamic_minus_static": _paired_delta(
                static,
                dynamic,
                query_ids,
                prefix,
                samples=bootstrap_samples,
                seed=seed,
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static_run", required=True)
    parser.add_argument("--dynamic_run", required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare(
        args.static_run,
        args.dynamic_run,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
