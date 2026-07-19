#!/usr/bin/env python3
"""Build a three-way full-candidate analysis for Graph, overlap, and Deep merged.

The unit is a query-paper pair from the complete OnePass candidate universe.
First-stage relevance and semantic-role labels come from the completed parent
annotation. Fine relations combine the original exclusive-arm refinement with
a separately blinded refinement of ``graph_and_deep_merged`` candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


PARTITIONS: Tuple[Tuple[str, str], ...] = (
    ("graph_only", "Graph-only"),
    ("graph_and_deep_merged", "Graph ∩ Deep merged"),
    ("deep_merged_only", "Deep-only"),
)
INFORMATION_GRADES = {"direct", "partial", "contextual"}
AXIS_LABELS: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "historical_relation": (
        ("direct_predecessor", "直接技术前身"),
        ("enabling_foundation", "基础/使能工作"),
        ("historical_background", "普通历史背景"),
        ("retrospective_or_survey", "回顾/综述脉络"),
        ("insufficient_evidence", "证据不足"),
    ),
    "mechanism_relation": (
        ("explicit_target_mechanism", "显式目标机制"),
        ("implicit_explanatory_mechanism", "隐式解释机制"),
        ("generic_theory", "一般/泛化理论"),
        ("insufficient_evidence", "证据不足"),
    ),
    "domain_relation": (
        ("same_domain", "同领域"),
        ("adjacent_domain", "邻近领域"),
        ("cross_domain_transfer", "跨领域迁移"),
        ("unrelated_domain_drift", "领域漂移"),
        ("insufficient_evidence", "证据不足"),
    ),
}
AXIS_NAMES = {
    "historical_relation": "历史关系",
    "mechanism_relation": "机制关系",
    "domain_relation": "领域关系",
}
SEMANTIC_ROWS: Tuple[Tuple[str, str, Optional[Tuple[str, str]]], ...] = (
    ("historical_context", "历史回顾/发展脉络", ("information_added", "historical_context")),
    ("background_or_foundation", "背景/基础工作", ("scholarly_roles", "background_or_foundation")),
    ("dataset_or_population", "数据集/研究对象", None),
    ("dataset_or_domain", "数据集/领域桥接", ("scholarly_roles", "dataset_or_domain")),
    ("survey_or_taxonomy", "综述/分类体系", ("scholarly_roles", "survey_or_taxonomy")),
    ("method_component", "方法组件", ("scholarly_roles", "method_component")),
    ("mechanism_or_theory", "机制/理论代理", ("information_added", "mechanism_or_theory")),
    ("application_domain", "应用领域代理", ("information_added", "application_domain")),
)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _partition(sources: Iterable[str]) -> Optional[str]:
    source_set = set(sources)
    graph = "graph" in source_set
    deep = "deep_merged" in source_set
    if graph and deep:
        return "graph_and_deep_merged"
    if graph:
        return "graph_only"
    if deep:
        return "deep_merged_only"
    return None


def _dataset_population_counts(annotation_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in _iter_jsonl(annotation_path):
        partition = _partition(row.get("sources") or [])
        if partition is None:
            continue
        annotation = row.get("annotation") or {}
        grade = str(annotation.get("relevance_grade", annotation.get("relationship", "")))
        if grade not in INFORMATION_GRADES:
            continue
        labels = set(annotation.get("information_added", annotation.get("contribution_types", [])) or [])
        if "dataset_or_population" in labels:
            counts[partition] += 1
    return counts


def _pct(rate: Optional[float]) -> str:
    if rate is None:
        return "—"
    value = 100.0 * rate
    decimals = 3 if 0 < abs(value) < 0.1 else 2
    return f"{value:.{decimals}f}%"


def _cell(count: int, rate: Optional[float]) -> str:
    return f"{count:,}（{_pct(rate)}）"


def _group_rows(first_analysis: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    raw = {row["group"]: row for row in _load_csv(first_analysis / "deep_merged_primary" / "group_summary.csv")}
    output: List[Dict[str, Any]] = []
    by_partition: Dict[str, Dict[str, Any]] = {}
    for partition, display in PARTITIONS:
        row = raw[partition]
        candidate_count = int(row["candidate_count"])
        direct = int(row["direct_count"])
        partial = int(row["partial_count"])
        contextual = int(row["contextual_count"])
        unrelated = int(row["unrelated_count"])
        insufficient = candidate_count - direct - partial - contextual - unrelated
        normalized = {
            "partition": partition,
            "display": display,
            "candidate_count": candidate_count,
            "ground_truth_count": int(row["ground_truth_count"]),
            "direct_count": direct,
            "partial_count": partial,
            "contextual_count": contextual,
            "information_bearing_count": direct + partial + contextual,
            "unrelated_count": unrelated,
            "insufficient_evidence_count": insufficient,
            "direct_rate": direct / candidate_count,
            "partial_rate": partial / candidate_count,
            "contextual_rate": contextual / candidate_count,
            "information_bearing_rate": (direct + partial + contextual) / candidate_count,
            "unrelated_rate": unrelated / candidate_count,
            "insufficient_evidence_rate": insufficient / candidate_count,
            "mean_relevance_score": float(row["mean_relevance_score"]),
        }
        output.append(normalized)
        by_partition[partition] = normalized
    return output, by_partition


def _semantic_rows(
    first_analysis: Path,
    groups: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    source = _load_csv(first_analysis / "deep_merged_primary" / "semantic_complement.csv")
    semantic_index = {
        (row["field"], row["label"]): row
        for row in source
    }
    dataset_counts = _dataset_population_counts(first_analysis / "annotations.jsonl")
    output: List[Dict[str, Any]] = []
    column_prefix = {
        "graph_only": "graph_only",
        "graph_and_deep_merged": "graph_and_deep_merged",
        "deep_merged_only": "deep_merged_only",
    }
    for key, display, lookup in SEMANTIC_ROWS:
        for partition, _ in PARTITIONS:
            denominator = int(groups[partition]["information_bearing_count"])
            if lookup is None:
                count = int(dataset_counts[partition])
                rate = count / denominator if denominator else None
            else:
                row = semantic_index[lookup]
                prefix = column_prefix[partition]
                count = int(row[f"{prefix}_count"])
                rate = float(row[f"{prefix}_rate"])
            output.append(
                {
                    "semantic_key": key,
                    "semantic_label_zh": display,
                    "partition": partition,
                    "candidate_count": count,
                    "information_bearing_denominator": denominator,
                    "rate_among_information_bearing": rate,
                }
            )
    return output


def _fine_rows(exclusive_analysis: Path, overlap_analysis: Path) -> List[Dict[str, Any]]:
    exclusive_summary = _load_json(exclusive_analysis / "summary.json")
    overlap_summary = _load_json(overlap_analysis / "summary.json")
    if not exclusive_summary.get("complete") or not overlap_summary.get("complete"):
        raise RuntimeError("Both fine-relation annotation aggregates must be complete")
    source = _load_csv(exclusive_analysis / "relation_distribution.csv")
    source.extend(_load_csv(overlap_analysis / "relation_distribution.csv"))
    index = {
        (row["group"], row["axis"], row["label"]): row
        for row in source
        if row["group"] != "union"
    }
    output: List[Dict[str, Any]] = []
    for axis, labels in AXIS_LABELS.items():
        applicable_by_partition: Dict[str, int] = {}
        for partition, _ in PARTITIONS:
            for label, label_zh in labels:
                row = index[(partition, axis, label)]
                applicable = int(row["applicable_count"])
                applicable_by_partition.setdefault(partition, applicable)
                if applicable_by_partition[partition] != applicable:
                    raise ValueError(f"inconsistent denominator for {partition}/{axis}")
                count = int(row["paper_count"])
                output.append(
                    {
                        "axis": axis,
                        "axis_zh": AXIS_NAMES[axis],
                        "label": label,
                        "label_zh": label_zh,
                        "partition": partition,
                        "applicable_count": applicable,
                        "paper_count": count,
                        "rate_among_applicable": count / applicable if applicable else None,
                    }
                )
        for partition, _ in PARTITIONS:
            partition_rows = [
                row
                for row in output
                if row["axis"] == axis and row["partition"] == partition
            ]
            if not partition_rows:
                raise ValueError(f"missing fine-relation rows for {partition}/{axis}")
            applicable = int(partition_rows[0]["applicable_count"])
            assigned = sum(int(row["paper_count"]) for row in partition_rows)
            if assigned != applicable:
                raise ValueError(
                    f"fine-relation labels do not conserve denominator for "
                    f"{partition}/{axis}: assigned={assigned}, applicable={applicable}"
                )
    return output


def _report(
    groups: Sequence[Mapping[str, Any]],
    semantic: Sequence[Mapping[str, Any]],
    fine: Sequence[Mapping[str, Any]],
) -> str:
    group_index = {row["partition"]: row for row in groups}
    semantic_index = {(row["semantic_key"], row["partition"]): row for row in semantic}
    fine_index = {(row["axis"], row["label"], row["partition"]): row for row in fine}
    headers = [display for _, display in PARTITIONS]
    lines = [
        "# OnePass 完整候选池三分区语义分析",
        "",
        "统计单位是 query–paper；Graph ∩ Deep merged 表示同一 query 下同时出现在两个完整候选池中的论文。",
        "",
        "## ① 相关性",
        "",
        f"| 指标 | {headers[0]}（{group_index['graph_only']['candidate_count']:,}篇） | {headers[1]}（{group_index['graph_and_deep_merged']['candidate_count']:,}篇） | {headers[2]}（{group_index['deep_merged_only']['candidate_count']:,}篇） |",
        "|---|---:|---:|---:|",
    ]
    relevance_rows = (
        ("Direct", "direct_count", "direct_rate"),
        ("Partial", "partial_count", "partial_rate"),
        ("Contextual", "contextual_count", "contextual_rate"),
        ("D + P + C", "information_bearing_count", "information_bearing_rate"),
        ("Unrelated", "unrelated_count", "unrelated_rate"),
        ("Insufficient evidence", "insufficient_evidence_count", "insufficient_evidence_rate"),
    )
    for label, count_key, rate_key in relevance_rows:
        cells = [_cell(int(group_index[p][count_key]), float(group_index[p][rate_key])) for p, _ in PARTITIONS]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    cells = [f"{float(group_index[p]['mean_relevance_score']):.4f}" for p, _ in PARTITIONS]
    lines.append(f"| 加权相关性 | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "交集是明显的共识/高相关区：它不是任一方法的边际补充，因此不能与 only 分区一样解释为独有贡献。",
            "",
            "## ② 学术角色与语义补充",
            "",
            "以下比例均以各分区 D+P+C 为分母。标签可多选，所以各行不会加总为 100%。",
            "",
            f"| 补充类型 | {headers[0]} | {headers[1]} | {headers[2]} |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, display, _ in SEMANTIC_ROWS:
        cells = []
        for partition, _ in PARTITIONS:
            row = semantic_index[(key, partition)]
            cells.append(_cell(int(row["candidate_count"]), row["rate_among_information_bearing"]))
        lines.append(f"| {display} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## ③ 历史、机制、领域二阶段细分",
            "",
            "每个轴的分母是该分区被相应一阶段标签触发的候选数；`insufficient_evidence` 保留在分母中。",
        ]
    )
    for axis, labels in AXIS_LABELS.items():
        denoms = [fine_index[(axis, labels[0][0], p)]["applicable_count"] for p, _ in PARTITIONS]
        lines.extend(
            [
                "",
                f"### {AXIS_NAMES[axis]}",
                "",
                f"| 类别 | {headers[0]}（n={denoms[0]:,}） | {headers[1]}（n={denoms[1]:,}） | {headers[2]}（n={denoms[2]:,}） |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, label_zh in labels:
            cells = []
            for partition, _ in PARTITIONS:
                row = fine_index[(axis, label, partition)]
                cells.append(_cell(int(row["paper_count"]), row["rate_among_applicable"]))
            lines.append(f"| {label_zh} | {' | '.join(cells)} |")
    total_candidates = sum(int(row["candidate_count"]) for row in groups)
    total_gt = sum(int(row["ground_truth_count"]) for row in groups)
    overlap = group_index["graph_and_deep_merged"]

    def fine_rate_sum(axis: str, labels: Sequence[str], partition: str) -> float:
        return sum(
            float(fine_index[(axis, label, partition)]["rate_among_applicable"] or 0.0)
            for label in labels
        )

    lines.extend(
        [
            "",
            "## 三分区新增结论",
            "",
            f"- 交集只占完整候选并集的 {_pct(int(overlap['candidate_count']) / total_candidates)}，"
            f"却包含 {_pct(int(overlap['ground_truth_count']) / total_gt)} 的候选 GT；"
            f"其 D+P+C 为 {_pct(float(overlap['information_bearing_rate']))}，说明双路命中本身是很强的相关性信号。",
            f"- 交集相关论文中的方法组件占 "
            f"{_pct(float(semantic_index[('method_component', 'graph_and_deep_merged')]['rate_among_information_bearing']))}，"
            "高于两个 only，表明双方共同发现的论文更集中于 query 的方法核心。",
            f"- 在历史触发项中，交集的直接前身+使能基础为 "
            f"{_pct(fine_rate_sum('historical_relation', ('direct_predecessor', 'enabling_foundation'), 'graph_and_deep_merged'))}；"
            "Graph-only 的历史优势仍主要来自回顾/综述，而交集的历史关系更技术化。",
            f"- 在机制触发项中，交集的显式+隐式解释机制合计 "
            f"{_pct(fine_rate_sum('mechanism_relation', ('explicit_target_mechanism', 'implicit_explanatory_mechanism'), 'graph_and_deep_merged'))}，"
            "明显高于两个 only；双方共同命中的机制论文更少停留在泛化理论层面。",
            f"- 交集的同领域比例为 "
            f"{_pct(float(fine_index[('domain_relation', 'same_domain', 'graph_and_deep_merged')]['rate_among_applicable']))}，"
            f"跨领域迁移为 "
            f"{_pct(float(fine_index[('domain_relation', 'cross_domain_transfer', 'graph_and_deep_merged')]['rate_among_applicable']))}。"
            "后者高于两个 only，但它是共识池性质，不能归因于某一个检索方法的独有增益。",
        ]
    )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- only 分区回答各方法独有候选补充了什么；交集回答两种方法共同发现的候选具有什么性质。",
            "- 这些是基于标题、摘要与 rubric 的模型标注描述，不证明图边的因果作用。",
            "- 二阶段标注对来源、GT、排名和图结构保持盲化；交集使用与两个 only 分区相同的 schema 与 prompt。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    first_analysis = Path(args.first_stage_analysis_dir).resolve()
    exclusive_analysis = Path(args.exclusive_refinement_analysis_dir).resolve()
    overlap_analysis = Path(args.overlap_refinement_analysis_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    groups, group_index = _group_rows(first_analysis)
    semantic = _semantic_rows(first_analysis, group_index)
    fine = _fine_rows(exclusive_analysis, overlap_analysis)
    _atomic_csv(output_dir / "relatedness.csv", groups)
    _atomic_csv(output_dir / "semantic_roles.csv", semantic)
    _atomic_csv(output_dir / "fine_relations.csv", fine)
    _atomic_json(
        output_dir / "summary.json",
        {
            "complete": True,
            "unit": "query-paper",
            "partitions": [partition for partition, _ in PARTITIONS],
            "first_stage_analysis_dir": str(first_analysis),
            "exclusive_refinement_analysis_dir": str(exclusive_analysis),
            "overlap_refinement_analysis_dir": str(overlap_analysis),
            "relatedness": groups,
            "semantic_roles": semantic,
            "fine_relations": fine,
        },
    )
    _atomic_text(output_dir / "report_zh.md", _report(groups, semantic, fine))
    return {
        "complete": True,
        "output_dir": str(output_dir),
        "candidate_counts": {
            partition: group_index[partition]["candidate_count"]
            for partition, _ in PARTITIONS
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
