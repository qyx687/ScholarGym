#!/usr/bin/env python3
"""Build three-way semantic tables after Baseline-budget Ret reranking.

This script distinguishes Ret-selection exclusivity from full-pool provenance
exclusivity. It combines the original exclusive-source fine annotations with
the separately blinded full-pool-overlap refinement, so every selected
information-bearing candidate has complete historical/mechanism/domain labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import build_three_way_candidate_analysis as base  # noqa: E402


Key = Tuple[str, str]
METHODS = ("graph", "deep_merged")
TRIGGERS = {
    "historical_relation": "historical_context",
    "mechanism_relation": "mechanism_or_theory",
    "domain_relation": "application_domain",
}
RELEVANCE_SCORES = {
    "direct": 1.0,
    "partial": 2.0 / 3.0,
    "contextual": 1.0 / 3.0,
    "unrelated": 0.0,
}
SET_DISPLAY = {
    "graph": "Graph",
    "deep_merged": "Deep",
    "graph_only": "Graph-only",
    "deep_merged_only": "Deep-only",
    "graph_and_deep_merged": "两者交集",
}


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    yield from base._iter_jsonl(path)


def _load_partition_sets(path: Path, formula: str) -> Dict[str, Set[Key]]:
    sets = {
        "graph": set(),
        "deep_merged": set(),
        "graph_only": set(),
        "deep_merged_only": set(),
        "graph_and_deep_merged": set(),
    }
    query_count = 0
    for row in _iter_jsonl(path):
        if str(row.get("formula")) != formula:
            continue
        query_count += 1
        query_id = str(row["query_id"])
        graph_only = {(query_id, str(paper_id)) for paper_id in row["graph_only_ids"]}
        deep_only = {(query_id, str(paper_id)) for paper_id in row["deep_merged_only_ids"]}
        overlap = {(query_id, str(paper_id)) for paper_id in row["overlap_ids"]}
        if len(graph_only) != int(row["graph_only_count"]):
            raise ValueError(f"Graph-only count mismatch for {query_id}")
        if len(deep_only) != int(row["deep_merged_only_count"]):
            raise ValueError(f"Deep-only count mismatch for {query_id}")
        if len(overlap) != int(row["overlap_count"]):
            raise ValueError(f"overlap count mismatch for {query_id}")
        if graph_only & deep_only or graph_only & overlap or deep_only & overlap:
            raise ValueError(f"Ret partitions are not disjoint for {query_id}")
        sets["graph_only"].update(graph_only)
        sets["deep_merged_only"].update(deep_only)
        sets["graph_and_deep_merged"].update(overlap)
        sets["graph"].update(graph_only | overlap)
        sets["deep_merged"].update(deep_only | overlap)
    if query_count == 0:
        raise ValueError(f"formula {formula!r} not found in {path}")
    return sets


def _load_selected_annotations(path: Path, expected: Set[Key]) -> Dict[Key, Dict[str, Any]]:
    annotations: Dict[Key, Dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        key = (str(row["query_id"]), str(row["paper_id"]))
        if key in annotations:
            raise ValueError(f"duplicate selected annotation: {key}")
        if not row.get("annotation"):
            raise ValueError(f"missing selected annotation: {key}")
        annotations[key] = row
    if set(annotations) != expected:
        raise ValueError(
            f"selected annotation universe mismatch: annotations={len(annotations)}, "
            f"expected={len(expected)}"
        )
    return annotations


def _load_original_provenance(
    path: Path, expected: Set[Key]
) -> Dict[Key, Dict[str, Any]]:
    provenance: Dict[Key, Dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        key = (str(row["query_id"]), str(row["paper_id"]))
        if key not in expected:
            continue
        if key in provenance:
            raise ValueError(f"duplicate original provenance row: {key}")
        provenance[key] = {
            "partition": base._partition(row.get("sources") or []),
            "sources": tuple(row.get("sources") or []),
            "is_ground_truth": bool(row.get("is_ground_truth")),
        }
    if set(provenance) != expected:
        raise ValueError(
            f"original provenance coverage mismatch: found={len(provenance)}, "
            f"expected={len(expected)}"
        )
    return provenance


def _grade(row: Mapping[str, Any]) -> str:
    annotation = row["annotation"]
    return str(annotation.get("relevance_grade", annotation.get("relationship", "")))


def _group_rows(
    partition_sets: Mapping[str, Set[Key]],
    annotations: Mapping[Key, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    output: List[Dict[str, Any]] = []
    by_partition: Dict[str, Dict[str, Any]] = {}
    for partition, display in base.PARTITIONS:
        keys = partition_sets[partition]
        grades = Counter(_grade(annotations[key]) for key in keys)
        count = len(keys)
        information_count = sum(grades[grade] for grade in base.INFORMATION_GRADES)
        scored_count = sum(grades[grade] for grade in RELEVANCE_SCORES)
        relevance_total = sum(
            RELEVANCE_SCORES.get(_grade(annotations[key]), 0.0) for key in keys
        )
        gt_count = sum(bool(annotations[key].get("is_ground_truth")) for key in keys)
        row = {
            "partition": partition,
            "display": display,
            "candidate_count": count,
            "ground_truth_count": gt_count,
            "direct_count": grades["direct"],
            "partial_count": grades["partial"],
            "contextual_count": grades["contextual"],
            "information_bearing_count": information_count,
            "unrelated_count": grades["unrelated"],
            "insufficient_evidence_count": count
            - grades["direct"]
            - grades["partial"]
            - grades["contextual"]
            - grades["unrelated"],
            "direct_rate": grades["direct"] / count if count else None,
            "partial_rate": grades["partial"] / count if count else None,
            "contextual_rate": grades["contextual"] / count if count else None,
            "information_bearing_rate": information_count / count if count else None,
            "unrelated_rate": grades["unrelated"] / count if count else None,
            "insufficient_evidence_rate": (
                (
                    count
                    - grades["direct"]
                    - grades["partial"]
                    - grades["contextual"]
                    - grades["unrelated"]
                )
                / count
                if count
                else None
            ),
            "mean_relevance_score": (
                relevance_total / scored_count if scored_count else None
            ),
        }
        output.append(row)
        by_partition[partition] = row
    return output, by_partition


def _semantic_rows(
    partition_sets: Mapping[str, Set[Key]],
    annotations: Mapping[Key, Mapping[str, Any]],
    group_index: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for semantic_key, semantic_label_zh, lookup in base.SEMANTIC_ROWS:
        if lookup is None:
            field, label = "information_added", "dataset_or_population"
        else:
            field, label = lookup
        for partition, _ in base.PARTITIONS:
            relevant_keys = {
                key
                for key in partition_sets[partition]
                if _grade(annotations[key]) in base.INFORMATION_GRADES
            }
            count = 0
            for key in relevant_keys:
                annotation = annotations[key]["annotation"]
                values = annotation.get(field) or []
                if label in set(str(value) for value in values):
                    count += 1
            denominator = int(group_index[partition]["information_bearing_count"])
            if len(relevant_keys) != denominator:
                raise ValueError(f"information denominator mismatch for {partition}")
            output.append(
                {
                    "semantic_key": semantic_key,
                    "semantic_label_zh": semantic_label_zh,
                    "partition": partition,
                    "candidate_count": count,
                    "information_bearing_denominator": denominator,
                    "rate_among_information_bearing": count / denominator if denominator else None,
                }
            )
    return output


def _load_fine_annotations(paths: Sequence[Path]) -> Dict[Key, Dict[str, Any]]:
    output: Dict[Key, Dict[str, Any]] = {}
    for path in paths:
        summary = base._load_json(path.parent / "summary.json")
        if not summary.get("complete"):
            raise RuntimeError(f"fine annotation aggregate is incomplete: {path.parent}")
        for row in _iter_jsonl(path):
            key = (str(row["query_id"]), str(row["paper_id"]))
            if key in output:
                raise ValueError(f"duplicate fine annotation across workspaces: {key}")
            if row.get("annotation_status") != "codex" or not row.get("annotation"):
                raise ValueError(f"missing fine annotation: {key}")
            output[key] = row
    return output


def _fine_rows(
    partition_sets: Mapping[str, Set[Key]],
    annotations: Mapping[Key, Mapping[str, Any]],
    fine_annotations: Mapping[Key, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for axis, labels in base.AXIS_LABELS.items():
        trigger = TRIGGERS[axis]
        for partition, _ in base.PARTITIONS:
            applicable_keys: Set[Key] = set()
            label_counts: Counter[str] = Counter()
            for key in partition_sets[partition]:
                first = annotations[key]["annotation"]
                if _grade(annotations[key]) not in base.INFORMATION_GRADES:
                    continue
                if trigger not in set(first.get("information_added") or []):
                    continue
                applicable_keys.add(key)
                fine = fine_annotations.get(key)
                if fine is None:
                    raise ValueError(f"selected triggered candidate lacks fine label: {key}/{axis}")
                if axis not in set(fine.get("applicable_axes") or []):
                    raise ValueError(f"fine label does not mark axis applicable: {key}/{axis}")
                label = str(fine["annotation"][axis]["label"])
                if label not in {value for value, _ in labels}:
                    raise ValueError(f"invalid fine label {label!r}: {key}/{axis}")
                label_counts[label] += 1
            denominator = len(applicable_keys)
            if sum(label_counts.values()) != denominator:
                raise ValueError(f"fine denominator not conserved for {partition}/{axis}")
            for label, label_zh in labels:
                count = label_counts[label]
                output.append(
                    {
                        "axis": axis,
                        "axis_zh": base.AXIS_NAMES[axis],
                        "label": label,
                        "label_zh": label_zh,
                        "partition": partition,
                        "applicable_count": denominator,
                        "paper_count": count,
                        "rate_among_applicable": count / denominator if denominator else None,
                    }
                )
    return output


def _set_rows(
    partition_sets: Mapping[str, Set[Key]],
    annotations: Mapping[Key, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    order = ("graph", "deep_merged", "graph_only", "deep_merged_only", "graph_and_deep_merged")
    return [
        {
            "set": name,
            "display": SET_DISPLAY[name],
            "candidate_count": len(partition_sets[name]),
            "ground_truth_count": sum(
                bool(annotations[key].get("is_ground_truth"))
                for key in partition_sets[name]
            ),
        }
        for name in order
    ]


def _provenance_retention_rows(
    partition_sets: Mapping[str, Set[Key]],
    original_provenance: Mapping[Key, Mapping[str, Any]],
    first_stage_analysis: Path,
) -> List[Dict[str, Any]]:
    full_groups = {
        row["group"]: row
        for row in base._load_csv(first_stage_analysis / "deep_merged_primary" / "group_summary.csv")
    }
    specs = (
        ("graph_only", "Graph-only source", "graph"),
        ("deep_merged_only", "Deep-only source", "deep_merged"),
    )
    rows = []
    for provenance, display, method in specs:
        selected = {
            key
            for key in partition_sets[method]
            if original_provenance[key]["partition"] == provenance
        }
        selected_gt = sum(
            bool(original_provenance[key]["is_ground_truth"]) for key in selected
        )
        full_candidate_count = int(full_groups[provenance]["candidate_count"])
        full_gt_count = int(full_groups[provenance]["ground_truth_count"])
        rows.append(
            {
                "provenance_partition": provenance,
                "display": display,
                "full_pool_candidate_count": full_candidate_count,
                "full_pool_gt_count": full_gt_count,
                "retained_candidate_count": len(selected),
                "retained_gt_count": selected_gt,
                "candidate_retention_rate": len(selected) / full_candidate_count,
                "gt_retention_rate": selected_gt / full_gt_count if full_gt_count else None,
            }
        )
    return rows


def _ret_provenance_rows(
    partition_sets: Mapping[str, Set[Key]],
    original_provenance: Mapping[Key, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for ret_partition, _ in base.PARTITIONS:
        for provenance_partition, _ in base.PARTITIONS:
            keys = {
                key
                for key in partition_sets[ret_partition]
                if original_provenance[key]["partition"] == provenance_partition
            }
            rows.append(
                {
                    "ret_partition": ret_partition,
                    "original_provenance_partition": provenance_partition,
                    "candidate_count": len(keys),
                    "ground_truth_count": sum(
                        bool(original_provenance[key]["is_ground_truth"])
                        for key in keys
                    ),
                }
            )
    return rows


def _report(
    formula: str,
    set_rows: Sequence[Mapping[str, Any]],
    retention: Sequence[Mapping[str, Any]],
    ret_provenance: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    semantic: Sequence[Mapping[str, Any]],
    fine: Sequence[Mapping[str, Any]],
) -> str:
    group_index = {row["partition"]: row for row in groups}
    semantic_index = {(row["semantic_key"], row["partition"]): row for row in semantic}
    fine_index = {(row["axis"], row["label"], row["partition"]): row for row in fine}
    provenance_index = {
        (row["ret_partition"], row["original_provenance_partition"]): row
        for row in ret_provenance
    }
    headers = [display for _, display in base.PARTITIONS]
    lines = [
        "# OnePass Ret rerank 后三分区语义分析",
        "",
        f"Formula: `{formula}`。单位是 query–paper；每个 retrieval event 使用 Baseline Selector-input K_i，未运行 Selector。",
        "",
        "## Ret 集合规模",
        "",
        "| 集合 | 候选 | GT |",
        "|---|---:|---:|",
    ]
    for row in set_rows:
        lines.append(f"| {row['display']} | {row['candidate_count']:,} | {row['ground_truth_count']:,} |")
    lines.extend(
        [
            "",
            "## 完整候选库来源独有项的 Ret 保留",
            "",
            "这里的 only 按完整候选库来源定义，不是 Ret 截断后的选择差集。",
            "",
            "| 完整池来源 | 完整池候选 | 完整池 GT | Ret 保留候选 | Ret 保留 GT | GT 保留率 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in retention:
        lines.append(
            f"| {row['display']} | {row['full_pool_candidate_count']:,} | "
            f"{row['full_pool_gt_count']:,} | {row['retained_candidate_count']:,} | "
            f"{row['retained_gt_count']:,} | {base._pct(row['gt_retention_rate'])} |"
        )
    lines.extend(
        [
            "",
            "### Ret-only GT 的原始来源分解",
            "",
            "| Ret 分区 | 原 Graph-only GT | 原完整池交集 GT | 原 Deep-only GT | Ret 分区 GT 合计 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for ret_partition, display in base.PARTITIONS:
        graph_gt = int(provenance_index[(ret_partition, "graph_only")]["ground_truth_count"])
        overlap_gt = int(provenance_index[(ret_partition, "graph_and_deep_merged")]["ground_truth_count"])
        deep_gt = int(provenance_index[(ret_partition, "deep_merged_only")]["ground_truth_count"])
        lines.append(
            f"| {display} | {graph_gt:,} | {overlap_gt:,} | {deep_gt:,} | "
            f"{graph_gt + overlap_gt + deep_gt:,} |"
        )
    lines.extend(
        [
            "",
            "## ① 相关性",
            "",
            f"| 指标 | {headers[0]}（{group_index['graph_only']['candidate_count']:,}） | "
            f"{headers[1]}（{group_index['graph_and_deep_merged']['candidate_count']:,}） | "
            f"{headers[2]}（{group_index['deep_merged_only']['candidate_count']:,}） |",
            "|---|---:|---:|---:|",
        ]
    )
    relevance_specs = (
        ("Direct", "direct_count", "direct_rate"),
        ("Partial", "partial_count", "partial_rate"),
        ("Contextual", "contextual_count", "contextual_rate"),
        ("D + P + C", "information_bearing_count", "information_bearing_rate"),
        ("Unrelated", "unrelated_count", "unrelated_rate"),
        ("Insufficient evidence", "insufficient_evidence_count", "insufficient_evidence_rate"),
    )
    for display, count_key, rate_key in relevance_specs:
        cells = [
            base._cell(int(group_index[p][count_key]), group_index[p][rate_key])
            for p, _ in base.PARTITIONS
        ]
        lines.append(f"| {display} | {' | '.join(cells)} |")
    cells = [f"{float(group_index[p]['mean_relevance_score']):.4f}" for p, _ in base.PARTITIONS]
    lines.append(f"| 加权相关性 | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## ② 学术角色与语义补充",
            "",
            "比例以各 Ret 分区的 D+P+C 为分母；标签可多选。",
            "",
            f"| 补充类型 | {headers[0]} | {headers[1]} | {headers[2]} |",
            "|---|---:|---:|---:|",
        ]
    )
    for semantic_key, display, _ in base.SEMANTIC_ROWS:
        cells = []
        for partition, _ in base.PARTITIONS:
            row = semantic_index[(semantic_key, partition)]
            cells.append(base._cell(int(row["candidate_count"]), row["rate_among_information_bearing"]))
        lines.append(f"| {display} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## ③ 历史、机制、领域二阶段细分",
            "",
            "分母是相应 Ret 分区中被一阶段标签触发的候选；`insufficient_evidence` 保留在分母。",
        ]
    )
    for axis, labels in base.AXIS_LABELS.items():
        denoms = [fine_index[(axis, labels[0][0], p)]["applicable_count"] for p, _ in base.PARTITIONS]
        lines.extend(
            [
                "",
                f"### {base.AXIS_NAMES[axis]}",
                "",
                f"| 类别 | {headers[0]}（n={denoms[0]:,}） | {headers[1]}（n={denoms[1]:,}） | {headers[2]}（n={denoms[2]:,}） |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, label_zh in labels:
            cells = []
            for partition, _ in base.PARTITIONS:
                row = fine_index[(axis, label, partition)]
                cells.append(base._cell(int(row["paper_count"]), row["rate_among_applicable"]))
            lines.append(f"| {label_zh} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- Ret-only 表示经过各自 rerank 和 K_i 截断后只被一边保留，可能来自完整池交集。",
            "- Source-only 表示论文在完整候选库中本来只由一个检索源发现；它才衡量来源独有 GT 的保留。",
            "- 本分析未使用 Selector；标签来自相同的一阶段标注和完整覆盖的盲化二阶段标注。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ret_dir = Path(args.ret_run_dir).resolve()
    first_analysis = Path(args.first_stage_analysis_dir).resolve()
    exclusive_analysis = Path(args.exclusive_refinement_analysis_dir).resolve()
    overlap_analysis = Path(args.overlap_refinement_analysis_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    formula = str(args.formula)
    partition_sets = _load_partition_sets(ret_dir / "query_graph_deep_partitions.jsonl", formula)
    union = partition_sets["graph"] | partition_sets["deep_merged"]
    annotations = _load_selected_annotations(
        ret_dir / "annotation_views" / formula / "analysis" / "annotations.jsonl",
        union,
    )
    original_provenance = _load_original_provenance(
        first_analysis / "annotations.jsonl", union
    )
    fine_annotations = _load_fine_annotations(
        (
            exclusive_analysis / "annotations.jsonl",
            overlap_analysis / "annotations.jsonl",
        )
    )
    groups, group_index = _group_rows(partition_sets, annotations)
    semantic = _semantic_rows(partition_sets, annotations, group_index)
    fine = _fine_rows(partition_sets, annotations, fine_annotations)
    sets = _set_rows(partition_sets, annotations)
    retention = _provenance_retention_rows(
        partition_sets, original_provenance, first_analysis
    )
    ret_provenance = _ret_provenance_rows(partition_sets, original_provenance)
    base._atomic_csv(output_dir / "set_summary.csv", sets)
    base._atomic_csv(output_dir / "provenance_retention.csv", retention)
    base._atomic_csv(output_dir / "ret_partition_provenance.csv", ret_provenance)
    base._atomic_csv(output_dir / "relatedness.csv", groups)
    base._atomic_csv(output_dir / "semantic_roles.csv", semantic)
    base._atomic_csv(output_dir / "fine_relations.csv", fine)
    summary = {
        "complete": True,
        "formula": formula,
        "unit": "query-paper after per-event Baseline K_i Ret truncation and query union/deduplication",
        "selector_used": False,
        "set_summary": sets,
        "provenance_retention": retention,
        "ret_partition_provenance": ret_provenance,
        "relatedness": groups,
        "semantic_roles": semantic,
        "fine_relations": fine,
    }
    base._atomic_json(output_dir / "summary.json", summary)
    base._atomic_text(
        output_dir / "report_zh.md",
        _report(formula, sets, retention, ret_provenance, groups, semantic, fine),
    )
    return {
        "complete": True,
        "formula": formula,
        "output_dir": str(output_dir),
        "set_summary": sets,
        "provenance_retention": retention,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ret-run-dir", required=True)
    parser.add_argument("--formula", required=True)
    parser.add_argument("--first-stage-analysis-dir", required=True)
    parser.add_argument("--exclusive-refinement-analysis-dir", required=True)
    parser.add_argument("--overlap-refinement-analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
