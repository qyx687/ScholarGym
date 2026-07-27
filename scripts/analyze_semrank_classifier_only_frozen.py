#!/usr/bin/env python3
"""Paired query-cluster bootstrap for the classifier-only frozen ablation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np


METRICS = (
    "recall@5",
    "recall@10",
    "recall@20",
    "mrr",
    "map@20",
    "ndcg@20",
    "hit@20",
)
COMPARISONS = {
    "full_minus_classifier_only": (
        "SemRankFull",
        "SemRankClassifierOnly",
    ),
    "classifier_only_minus_s2_native": (
        "SemRankClassifierOnly",
        "S2Native",
    ),
    "classifier_only_minus_qudar": (
        "SemRankClassifierOnly",
        "qudar_confidence_qsq",
    ),
    "classifier_only_minus_static": (
        "SemRankClassifierOnly",
        "static",
    ),
}


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"expected JSON object at {path}:{line_number}"
                )
            yield value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event_metrics", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    path = Path(args.event_metrics).expanduser().resolve()
    values: Dict[str, Dict[str, Dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    event_counts: Dict[str, int] = defaultdict(int)
    for row in iter_jsonl(path):
        method = str(row.get("method") or "")
        query_id = str(row.get("query_id") or "")
        if not method or not query_id:
            raise ValueError("event metric row lacks method or query_id")
        event_counts[method] += 1
        for metric in METRICS:
            values[method][query_id][metric].append(float(row[metric]))

    query_means: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, by_query in values.items():
        query_means[method] = {
            query_id: {
                metric: float(np.mean(metric_values[metric]))
                for metric in METRICS
            }
            for query_id, metric_values in by_query.items()
        }

    rng = np.random.default_rng(args.seed)
    output_comparisons: Dict[str, Any] = {}
    for name, (left, right) in COMPARISONS.items():
        left_queries = set(query_means.get(left, {}))
        right_queries = set(query_means.get(right, {}))
        if left_queries != right_queries or len(left_queries) != 50:
            raise AssertionError(
                f"{name} requires the same 50 queries; "
                f"left={len(left_queries)}, right={len(right_queries)}"
            )
        query_ids = sorted(left_queries)
        indices = rng.integers(
            0,
            len(query_ids),
            size=(max(1, args.samples), len(query_ids)),
        )
        metric_output: Dict[str, Any] = {}
        for metric in METRICS:
            left_values = np.asarray(
                [query_means[left][query_id][metric] for query_id in query_ids],
                dtype=np.float64,
            )
            right_values = np.asarray(
                [query_means[right][query_id][metric] for query_id in query_ids],
                dtype=np.float64,
            )
            differences = left_values - right_values
            bootstrapped = differences[indices].mean(axis=1)
            metric_output[metric] = {
                "left_query_macro": float(left_values.mean()),
                "right_query_macro": float(right_values.mean()),
                "query_macro_delta": float(differences.mean()),
                "bootstrap_95pct_ci": [
                    float(np.quantile(bootstrapped, 0.025)),
                    float(np.quantile(bootstrapped, 0.975)),
                ],
            }
        output_comparisons[name] = {
            "left": left,
            "right": right,
            "query_count": len(query_ids),
            "metrics": metric_output,
        }

    report = {
        "analysis": "paired_query_cluster_bootstrap_v1",
        "event_metrics": str(path),
        "bootstrap_samples": max(1, args.samples),
        "random_seed": args.seed,
        "event_counts": dict(sorted(event_counts.items())),
        "comparisons": output_comparisons,
    }
    output = Path(args.output_json).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
