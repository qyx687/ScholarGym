#!/usr/bin/env python3
"""Paired query bootstrap for legacy-vs-dynamic rerank metrics."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Sequence


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _f1(recall: float, precision: float) -> float:
    return 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _main_table_f1(rows: Sequence[dict], method: str, indices: Iterable[int]) -> float:
    chosen = [rows[index]["methods"][method] for index in indices]
    return _f1(
        _mean([float(row["candidate_recall"]) for row in chosen]),
        _mean([float(row["candidate_precision"]) for row in chosen]),
    )


def paired_bootstrap(path: Path, *, samples: int, seed: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("per-query result file is empty")
    count = len(rows)
    indices = list(range(count))
    legacy = _main_table_f1(rows, "legacy_static", indices)
    dynamic = _main_table_f1(rows, "dynamic_policy", indices)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        sampled = [rng.randrange(count) for _ in range(count)]
        deltas.append(
            _main_table_f1(rows, "dynamic_policy", sampled)
            - _main_table_f1(rows, "legacy_static", sampled)
        )
    ordered = sorted(deltas)
    return {
        "per_query_results": str(path.resolve()),
        "query_count": count,
        "bootstrap_samples": samples,
        "seed": seed,
        "legacy_main_table_candidate_f1": legacy,
        "dynamic_main_table_candidate_f1": dynamic,
        "observed_delta": dynamic - legacy,
        "relative_improvement": (dynamic - legacy) / legacy if legacy else None,
        "paired_bootstrap_delta_ci95": [
            _percentile(ordered, 0.025),
            _percentile(ordered, 0.975),
        ],
        "bootstrap_probability_dynamic_gt_legacy": (
            sum(delta > 0.0 for delta in deltas) / len(deltas) if deltas else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_query_results", type=Path)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the reproducible bootstrap report.",
    )
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    report = paired_bootstrap(
        args.per_query_results,
        samples=args.samples,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
