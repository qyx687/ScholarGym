#!/usr/bin/env python3
"""Compare annotation profiles of Graph and Deep-merged Ret selections.

The primary comparison is Graph-only versus Deep-merged-only after query-local
Ret union/deduplication.  Inclusive Graph versus Deep merged is also emitted as
a sensitivity view, but it is expected to be diluted by their shared papers.

All comparisons are candidate-level and query-paired.  Bootstrap intervals
resample queries, not papers.  They are descriptive intervals over completed
model annotations and are not adjusted for multiple label comparisons.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


IMPLEMENTATION_VERSION = "1.0"
INFORMATION_GRADES = {"direct", "partial", "contextual"}
DIRECT_OR_PARTIAL = {"direct", "partial"}
RELEVANCE_SCORES = {
    "direct": 1.0,
    "partial": 2.0 / 3.0,
    "contextual": 1.0 / 3.0,
    "unrelated": 0.0,
    "insufficient_evidence": 0.0,
}


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _safe_rate(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator) / float(denominator) if denominator else None


def _annotation_field(
    annotation: Mapping[str, Any], canonical: str, legacy: str
) -> Any:
    return annotation[canonical] if canonical in annotation else annotation.get(legacy)


def _grade(annotation: Mapping[str, Any]) -> str:
    return str(_annotation_field(annotation, "relevance_grade", "relationship") or "unknown")


def _roles(annotation: Mapping[str, Any]) -> List[str]:
    return [
        str(value)
        for value in (
            _annotation_field(annotation, "scholarly_roles", "relation_roles") or []
        )
    ]


def _information(annotation: Mapping[str, Any]) -> List[str]:
    return [
        str(value)
        for value in (
            _annotation_field(annotation, "information_added", "contribution_types")
            or []
        )
    ]


def _partition(sources: Iterable[str]) -> str:
    source_set = set(sources)
    graph = "graph" in source_set
    deep = "deep_merged" in source_set
    if graph and deep:
        return "graph_and_deep_merged"
    if graph:
        return "graph_only"
    if deep:
        return "deep_merged_only"
    raise ValueError(f"candidate has no Graph/Deep-merged source: {sorted(source_set)}")


def _groups(sources: Iterable[str]) -> List[str]:
    source_set = set(sources)
    groups = ["union", _partition(source_set)]
    if "graph" in source_set:
        groups.append("graph")
    if "deep_merged" in source_set:
        groups.append("deep_merged")
    return groups


def _matched_aspect_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    return "6+"


def annotation_features(
    candidate: Mapping[str, Any], annotation: Mapping[str, Any]
) -> Set[Tuple[str, str]]:
    """Return deduplicated binary labels whose denominator is all candidates."""

    grade = _grade(annotation)
    roles = set(_roles(annotation))
    information = set(_information(annotation))
    matched_aspects = set(str(value) for value in annotation.get("matched_aspect_ids") or [])
    features: Set[Tuple[str, str]] = {
        ("relevance_grade", grade),
        ("semantic_distance", str(annotation.get("semantic_distance") or "unknown")),
        ("paper_type", str(annotation.get("paper_type") or "unknown")),
        ("matched_aspect_count", _matched_aspect_bucket(len(matched_aspects))),
    }
    features.update(("scholarly_role", value) for value in roles)
    features.update(("information_added", value) for value in information)
    features.update(
        ("exclusion_reason", str(value))
        for value in annotation.get("exclusion_reasons") or []
    )
    if grade in DIRECT_OR_PARTIAL:
        features.add(("derived", "direct_or_partial"))
    if grade in INFORMATION_GRADES:
        features.add(("derived", "information_bearing"))
    if str(annotation.get("semantic_distance") or "") in {"exact", "near"}:
        features.add(("derived", "exact_or_near"))
    if matched_aspects:
        features.add(("derived", "matched_aspect_any"))
    if bool(annotation.get("needs_full_text")):
        features.add(("derived", "needs_full_text"))
    if bool(candidate.get("is_ground_truth")):
        features.add(("derived", "ground_truth"))
    if grade == "direct" and roles & {
        "direct_target",
        "method_component",
        "task_or_application",
    }:
        features.add(("derived", "direct_task_method_match"))
    return features


def numeric_features(annotation: Mapping[str, Any]) -> Mapping[str, float]:
    grade = _grade(annotation)
    return {
        "relevance_score": RELEVANCE_SCORES.get(grade, 0.0),
        "matched_aspect_count": float(len(set(annotation.get("matched_aspect_ids") or []))),
        "confidence": float(annotation.get("confidence") or 0.0),
    }


def load_counts(
    workspace: Path,
) -> Tuple[
    Counter[str],
    Counter[Tuple[str, str, str]],
    Counter[Tuple[str, str]],
    Counter[Tuple[str, str, str, str]],
    Dict[Tuple[str, str], List[float]],
    Dict[Tuple[str, str, str], List[float]],
]:
    group_totals: Counter[str] = Counter()
    feature_counts: Counter[Tuple[str, str, str]] = Counter()
    query_totals: Counter[Tuple[str, str]] = Counter()
    query_feature_counts: Counter[Tuple[str, str, str, str]] = Counter()
    numeric_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    query_numeric_values: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)

    candidate_iter = _iter_jsonl(workspace / "manifest" / "candidates.jsonl")
    annotation_iter = _iter_jsonl(workspace / "analysis" / "annotations.jsonl")
    for candidate, annotation_row in itertools.zip_longest(candidate_iter, annotation_iter):
        if candidate is None or annotation_row is None:
            raise ValueError("candidate and annotation files differ in length")
        if candidate.get("candidate_id") != annotation_row.get("candidate_id"):
            raise ValueError("candidate and annotation order/IDs do not match")
        annotation = annotation_row.get("annotation")
        if not isinstance(annotation, dict):
            raise ValueError(f"missing annotation for {candidate.get('candidate_id')}")
        query_id = str(candidate.get("query_id") or "")
        features = annotation_features(candidate, annotation)
        numeric = numeric_features(annotation)
        for group in _groups(candidate.get("sources") or []):
            group_totals[group] += 1
            query_totals[(group, query_id)] += 1
            for dimension, label in features:
                feature_counts[(group, dimension, label)] += 1
                query_feature_counts[(group, query_id, dimension, label)] += 1
            for metric, value in numeric.items():
                numeric_values[(group, metric)].append(value)
                query_numeric_values[(group, query_id, metric)].append(value)
    return (
        group_totals,
        feature_counts,
        query_totals,
        query_feature_counts,
        numeric_values,
        query_numeric_values,
    )


def _bootstrap_ci(
    differences: Sequence[float], *, samples: int, seed: int
) -> Optional[List[float]]:
    if not differences or samples <= 0:
        return None
    rng = random.Random(seed)
    size = len(differences)
    means = [
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    ]
    means.sort()
    return [
        means[max(0, int(samples * 0.025))],
        means[min(samples - 1, max(0, int(samples * 0.975) - 1))],
    ]


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def distribution_rows(
    group_totals: Mapping[str, int],
    feature_counts: Mapping[Tuple[str, str, str], int],
) -> List[Dict[str, Any]]:
    labels = sorted({(dimension, label) for _, dimension, label in feature_counts})
    groups = (
        "union",
        "graph",
        "deep_merged",
        "graph_only",
        "deep_merged_only",
        "graph_and_deep_merged",
    )
    rows = []
    for group in groups:
        denominator = int(group_totals.get(group, 0))
        for dimension, label in labels:
            count = int(feature_counts.get((group, dimension, label), 0))
            rows.append(
                {
                    "group": group,
                    "dimension": dimension,
                    "label": label,
                    "candidate_count": denominator,
                    "paper_count": count,
                    "candidate_rate": _safe_rate(count, denominator),
                }
            )
    return rows


def comparison_rows(
    group_totals: Mapping[str, int],
    feature_counts: Mapping[Tuple[str, str, str], int],
    query_totals: Mapping[Tuple[str, str], int],
    query_feature_counts: Mapping[Tuple[str, str, str, str], int],
    *,
    bootstrap_samples: int,
) -> List[Dict[str, Any]]:
    labels = sorted({(dimension, label) for _, dimension, label in feature_counts})
    query_ids = sorted({query_id for _, query_id in query_totals})
    comparisons = (
        ("selection_exclusive", "graph_only", "deep_merged_only"),
        ("inclusive", "graph", "deep_merged"),
    )
    rows = []
    for scope, left_group, right_group in comparisons:
        for dimension, label in labels:
            left_count = int(feature_counts.get((left_group, dimension, label), 0))
            right_count = int(feature_counts.get((right_group, dimension, label), 0))
            left_total = int(group_totals.get(left_group, 0))
            right_total = int(group_totals.get(right_group, 0))
            paired: List[Tuple[float, float]] = []
            for query_id in query_ids:
                left_query_total = int(query_totals.get((left_group, query_id), 0))
                right_query_total = int(query_totals.get((right_group, query_id), 0))
                if not left_query_total or not right_query_total:
                    continue
                paired.append(
                    (
                        int(
                            query_feature_counts.get(
                                (left_group, query_id, dimension, label), 0
                            )
                        )
                        / left_query_total,
                        int(
                            query_feature_counts.get(
                                (right_group, query_id, dimension, label), 0
                            )
                        )
                        / right_query_total,
                    )
                )
            differences = [left - right for left, right in paired]
            ci = _bootstrap_ci(
                differences,
                samples=bootstrap_samples,
                seed=_stable_seed(f"{scope}:{dimension}:{label}"),
            )
            left_micro = _safe_rate(left_count, left_total)
            right_micro = _safe_rate(right_count, right_total)
            left_macro = (
                sum(left for left, _ in paired) / len(paired) if paired else None
            )
            right_macro = (
                sum(right for _, right in paired) / len(paired) if paired else None
            )
            macro_difference = (
                sum(differences) / len(differences) if differences else None
            )
            rows.append(
                {
                    "scope": scope,
                    "left_group": left_group,
                    "right_group": right_group,
                    "dimension": dimension,
                    "label": label,
                    "left_candidate_count": left_total,
                    "right_candidate_count": right_total,
                    "left_count": left_count,
                    "right_count": right_count,
                    "left_micro_rate": left_micro,
                    "right_micro_rate": right_micro,
                    "micro_rate_difference": (
                        left_micro - right_micro
                        if left_micro is not None and right_micro is not None
                        else None
                    ),
                    "left_over_right_rate_ratio": (
                        left_micro / right_micro
                        if left_micro is not None and right_micro not in (None, 0)
                        else None
                    ),
                    "paired_query_count": len(paired),
                    "left_macro_rate": left_macro,
                    "right_macro_rate": right_macro,
                    "macro_rate_difference": macro_difference,
                    "query_bootstrap_95_ci": ci,
                    "interval_excludes_zero": bool(
                        ci and (ci[0] > 0.0 or ci[1] < 0.0)
                    ),
                    "left_wins": sum(left > right for left, right in paired),
                    "right_wins": sum(left < right for left, right in paired),
                    "ties": sum(left == right for left, right in paired),
                }
            )
    return rows


def numeric_summary_rows(
    numeric_values: Mapping[Tuple[str, str], Sequence[float]],
) -> List[Dict[str, Any]]:
    rows = []
    for (group, metric), values in sorted(numeric_values.items()):
        rows.append(
            {
                "group": group,
                "metric": metric,
                "candidate_count": len(values),
                "mean": sum(values) / len(values) if values else None,
            }
        )
    return rows


def numeric_comparison_rows(
    query_numeric_values: Mapping[Tuple[str, str, str], Sequence[float]],
    *,
    bootstrap_samples: int,
) -> List[Dict[str, Any]]:
    query_ids = sorted({query_id for _, query_id, _ in query_numeric_values})
    metrics = sorted({metric for _, _, metric in query_numeric_values})
    rows = []
    for scope, left_group, right_group in (
        ("selection_exclusive", "graph_only", "deep_merged_only"),
        ("inclusive", "graph", "deep_merged"),
    ):
        for metric in metrics:
            paired = []
            for query_id in query_ids:
                left_values = query_numeric_values.get((left_group, query_id, metric), [])
                right_values = query_numeric_values.get((right_group, query_id, metric), [])
                if not left_values or not right_values:
                    continue
                paired.append(
                    (
                        sum(left_values) / len(left_values),
                        sum(right_values) / len(right_values),
                    )
                )
            differences = [left - right for left, right in paired]
            ci = _bootstrap_ci(
                differences,
                samples=bootstrap_samples,
                seed=_stable_seed(f"numeric:{scope}:{metric}"),
            )
            rows.append(
                {
                    "scope": scope,
                    "left_group": left_group,
                    "right_group": right_group,
                    "metric": metric,
                    "paired_query_count": len(paired),
                    "left_macro_mean": (
                        sum(left for left, _ in paired) / len(paired) if paired else None
                    ),
                    "right_macro_mean": (
                        sum(right for _, right in paired) / len(paired) if paired else None
                    ),
                    "macro_mean_difference": (
                        sum(differences) / len(differences) if differences else None
                    ),
                    "query_bootstrap_95_ci": ci,
                    "interval_excludes_zero": bool(
                        ci and (ci[0] > 0.0 or ci[1] < 0.0)
                    ),
                }
            )
    return rows


def _pct(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def report_markdown(
    group_totals: Mapping[str, int],
    comparison: Sequence[Mapping[str, Any]],
    numeric_comparison: Sequence[Mapping[str, Any]],
    formula_id: str,
) -> str:
    exclusive = [row for row in comparison if row["scope"] == "selection_exclusive"]
    index = {(row["dimension"], row["label"]): row for row in exclusive}
    lines = [
        "# Annotation differences: Graph vs Deep merged Ret selections",
        "",
        f"Primary unit: query-local deduplicated candidate under `{formula_id}`.",
        "Primary contrast: Graph-only versus Deep-merged-only after Ret truncation.",
        "",
        "| Group | Candidates |",
        "|---|---:|",
    ]
    for group in (
        "graph",
        "deep_merged",
        "graph_only",
        "deep_merged_only",
        "graph_and_deep_merged",
    ):
        lines.append(f"| {group} | {int(group_totals.get(group, 0)):,} |")

    lines.extend(
        [
            "",
            "## Relevance and semantic distance",
            "",
            "| Dimension | Label | Graph-only | Deep-only | Macro Graph−Deep | 95% query-bootstrap interval |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    preferred = (
        ("relevance_grade", "direct"),
        ("relevance_grade", "partial"),
        ("relevance_grade", "contextual"),
        ("relevance_grade", "unrelated"),
        ("semantic_distance", "exact"),
        ("semantic_distance", "near"),
        ("semantic_distance", "adjacent"),
        ("semantic_distance", "distant"),
    )
    for key in preferred:
        row = index[key]
        ci = row["query_bootstrap_95_ci"]
        lines.append(
            f"| {key[0]} | {key[1]} | {_pct(row['left_micro_rate'])} | "
            f"{_pct(row['right_micro_rate'])} | {_pct(row['macro_rate_difference'])} | "
            f"[{_pct(ci[0])}, {_pct(ci[1])}] |"
        )

    significant = [
        row
        for row in exclusive
        if row["interval_excludes_zero"]
        and row["dimension"]
        in {
            "paper_type",
            "scholarly_role",
            "information_added",
            "derived",
        }
        and row["left_count"] + row["right_count"] >= 10
    ]
    significant.sort(
        key=lambda row: abs(float(row["macro_rate_difference"] or 0.0)), reverse=True
    )
    lines.extend(
        [
            "",
            "## Largest query-consistent label differences",
            "",
            "| Dimension | Label | Graph-only | Deep-only | Macro Graph−Deep | 95% interval | Direction |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in significant[:15]:
        ci = row["query_bootstrap_95_ci"]
        direction = "Graph-only" if row["macro_rate_difference"] > 0 else "Deep-only"
        lines.append(
            f"| {row['dimension']} | {row['label']} | {_pct(row['left_micro_rate'])} | "
            f"{_pct(row['right_micro_rate'])} | {_pct(row['macro_rate_difference'])} | "
            f"[{_pct(ci[0])}, {_pct(ci[1])}] | {direction} |"
        )

    lines.extend(
        [
            "",
            "## Query-paired numeric properties",
            "",
            "| Metric | Graph-only | Deep-only | Graph−Deep | 95% interval |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in numeric_comparison:
        if row["scope"] != "selection_exclusive":
            continue
        ci = row["query_bootstrap_95_ci"]
        formatter = _pct if row["metric"] in {"relevance_score", "confidence"} else lambda value: f"{float(value):.3f}"
        lines.append(
            f"| {row['metric']} | {formatter(row['left_macro_mean'])} | "
            f"{formatter(row['right_macro_mean'])} | "
            f"{formatter(row['macro_mean_difference'])} | "
            f"[{formatter(ci[0])}, {formatter(ci[1])}] |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Intervals resample the 50 queries and are descriptive; they are not adjusted for "
            "multiple labels. Labels are model annotations. Top-K exclusivity measures ranking "
            "differences and must not be interpreted as full-pool retrieval provenance.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    workspace: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int,
) -> Dict[str, Any]:
    workspace = workspace.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        group_totals,
        feature_counts,
        query_totals,
        query_feature_counts,
        numeric_values,
        query_numeric_values,
    ) = load_counts(workspace)
    distributions = distribution_rows(group_totals, feature_counts)
    comparisons = comparison_rows(
        group_totals,
        feature_counts,
        query_totals,
        query_feature_counts,
        bootstrap_samples=bootstrap_samples,
    )
    numeric_summaries = numeric_summary_rows(numeric_values)
    numeric_comparisons = numeric_comparison_rows(
        query_numeric_values, bootstrap_samples=bootstrap_samples
    )
    _atomic_write_jsonl(output_dir / "label_distribution.jsonl", distributions)
    _atomic_write_csv(output_dir / "label_distribution.csv", distributions)
    _atomic_write_jsonl(output_dir / "label_comparison.jsonl", comparisons)
    _atomic_write_csv(output_dir / "label_comparison.csv", comparisons)
    _atomic_write_jsonl(output_dir / "numeric_summary.jsonl", numeric_summaries)
    _atomic_write_csv(output_dir / "numeric_summary.csv", numeric_summaries)
    _atomic_write_jsonl(output_dir / "numeric_comparison.jsonl", numeric_comparisons)
    _atomic_write_csv(output_dir / "numeric_comparison.csv", numeric_comparisons)
    report = report_markdown(
        group_totals,
        comparisons,
        numeric_comparisons,
        formula_id=workspace.name,
    )
    temporary = output_dir / f".report.md.tmp-{os.getpid()}"
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, output_dir / "report.md")
    summary = {
        "complete": True,
        "implementation_version": IMPLEMENTATION_VERSION,
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "bootstrap_samples": bootstrap_samples,
        "group_candidate_counts": dict(group_totals),
        "primary_comparison": "graph_only_minus_deep_merged_only",
        "formula_id": workspace.name,
        "query_bootstrap_unit": "query",
        "multiplicity_adjustment": False,
        "label_comparison_count": len(comparisons),
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze(
        Path(args.workspace),
        Path(args.output_dir),
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
