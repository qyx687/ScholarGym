#!/usr/bin/env python3
"""Build Graph Sel-only / overlap / Deep-merged Sel-only semantic tables.

The selected paper sets come from the completed production OnePass
``query_results.jsonl`` artifacts.  The script verifies the saved rerank weights,
reuses the completed blinded first- and second-stage labels, and never calls the
Selector or an LLM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_baseline_budget_ret_rerank as rerank  # noqa: E402
import build_ret_three_way_candidate_analysis as ret_analysis  # noqa: E402
import build_three_way_candidate_analysis as table_base  # noqa: E402


IMPLEMENTATION_VERSION = "1.0"
Key = Tuple[str, str]
METHODS = ("graph", "deep_merged")
DISPLAY = {
    "graph": "Graph Sel",
    "deep_merged": "Deep merged Sel",
    "graph_only": "Graph Sel-only",
    "graph_and_deep_merged": "两者 Sel 交集",
    "deep_merged_only": "Deep Sel-only",
}


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    yield from rerank._iter_jsonl(path)


def load_selector_query_sets(path: Path) -> Dict[str, Set[str]]:
    output: Dict[str, Set[str]] = {}
    for row in _iter_jsonl(path):
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in output:
            raise ValueError(f"missing or duplicate Selector query result: {query_id!r}")
        paper_ids = set(rerank._ordered_unique(row.get("selected_arxiv_ids") or []))
        if len(paper_ids) != int(row.get("selected_count") or 0):
            raise ValueError(
                f"selected_count mismatch for {query_id}: ids={len(paper_ids)}, "
                f"saved={row.get('selected_count')}"
            )
        output[query_id] = paper_ids
    if not output:
        raise ValueError(f"no Selector query results in {path}")
    return output


def build_partition_sets(
    graph: Mapping[str, Set[str]],
    deep: Mapping[str, Set[str]],
) -> Dict[str, Set[Key]]:
    if set(graph) != set(deep):
        raise ValueError(
            f"Graph/Deep query coverage mismatch: graph={len(graph)}, deep={len(deep)}"
        )
    output: Dict[str, Set[Key]] = {
        "graph": set(),
        "deep_merged": set(),
        "graph_only": set(),
        "graph_and_deep_merged": set(),
        "deep_merged_only": set(),
    }
    for query_id in graph:
        graph_ids = set(graph[query_id])
        deep_ids = set(deep[query_id])
        graph_only = graph_ids - deep_ids
        overlap = graph_ids & deep_ids
        deep_only = deep_ids - graph_ids
        output["graph"].update((query_id, paper_id) for paper_id in graph_ids)
        output["deep_merged"].update((query_id, paper_id) for paper_id in deep_ids)
        output["graph_only"].update((query_id, paper_id) for paper_id in graph_only)
        output["graph_and_deep_merged"].update(
            (query_id, paper_id) for paper_id in overlap
        )
        output["deep_merged_only"].update((query_id, paper_id) for paper_id in deep_only)
    if output["graph_only"] & output["graph_and_deep_merged"]:
        raise ValueError("Graph Sel-only and Sel overlap are not disjoint")
    if output["deep_merged_only"] & output["graph_and_deep_merged"]:
        raise ValueError("Deep Sel-only and Sel overlap are not disjoint")
    return output


def load_selected_annotations(
    path: Path, expected: Set[Key]
) -> Dict[Key, Dict[str, Any]]:
    output: Dict[Key, Dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        key = (str(row.get("query_id") or ""), rerank._paper_id(row.get("paper_id")))
        if key not in expected:
            continue
        if key in output:
            raise ValueError(f"duplicate selected annotation: {key}")
        if not row.get("annotation"):
            raise ValueError(f"selected candidate lacks annotation: {key}")
        output[key] = row
    if set(output) != expected:
        missing = expected - set(output)
        raise ValueError(
            f"selected annotation coverage mismatch: found={len(output)}, "
            f"expected={len(expected)}, first_missing={next(iter(missing), None)}"
        )
    return output


def validate_formula(
    path: Path,
    *,
    query_weight: float,
    subquery_weight: float,
    intent_weight: float,
    path_weight: float,
) -> Dict[str, float]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    saved = manifest.get("feature_weights") or {}
    expected = {
        "query_score_normalized": float(query_weight),
        "subquery_score_normalized": float(subquery_weight),
        "intent_score": float(intent_weight),
        "path_count_normalized": float(path_weight),
    }
    for feature, value in expected.items():
        if not math.isclose(float(saved.get(feature, -1.0)), value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"saved production weight mismatch for {feature}: "
                f"saved={saved.get(feature)}, expected={value}"
            )
    return expected


def retrieval_rows(
    query_sets: Mapping[str, Mapping[str, Set[str]]],
    contexts: Mapping[str, rerank.QueryContext],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for method in METHODS:
        per_query: List[Dict[str, float]] = []
        total_gt = total_candidates = total_hits = 0
        for context in contexts.values():
            candidates = set(query_sets[method].get(context.query_id, set()))
            gt = set(context.ground_truth_ids)
            hits = candidates & gt
            recall = rerank._safe_rate(len(hits), len(gt))
            precision = rerank._safe_rate(len(hits), len(candidates))
            per_query.append(
                {
                    "recall": recall,
                    "precision": precision,
                    "f1": rerank._f1(recall, precision),
                }
            )
            total_gt += len(gt)
            total_candidates += len(candidates)
            total_hits += len(hits)
        micro_recall = rerank._safe_rate(total_hits, total_gt)
        micro_precision = rerank._safe_rate(total_hits, total_candidates)
        output.append(
            {
                "method": method,
                "display": DISPLAY[method],
                "query_count": len(per_query),
                "candidate_count": total_candidates,
                "ground_truth_count": total_hits,
                "micro_recall": micro_recall,
                "micro_precision": micro_precision,
                "micro_f1": rerank._f1(micro_recall, micro_precision),
                "macro_recall": sum(row["recall"] for row in per_query) / len(per_query),
                "macro_precision": sum(row["precision"] for row in per_query) / len(per_query),
                "macro_f1": sum(row["f1"] for row in per_query) / len(per_query),
            }
        )
    return output


def _set_rows(
    partition_sets: Mapping[str, Set[Key]],
    annotations: Mapping[Key, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    order = (
        "graph",
        "deep_merged",
        "graph_only",
        "deep_merged_only",
        "graph_and_deep_merged",
    )
    return [
        {
            "set": name,
            "display": DISPLAY[name],
            "candidate_count": len(partition_sets[name]),
            "ground_truth_count": sum(
                bool(annotations[key].get("is_ground_truth"))
                for key in partition_sets[name]
            ),
        }
        for name in order
    ]


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "—"
    pct = 100.0 * float(value)
    decimals = 3 if 0 < abs(pct) < 0.1 else 2
    return f"{pct:.{decimals}f}%"


def _cell(count: int, rate: Optional[float]) -> str:
    return f"{count:,}（{_pct(rate)}）"


def build_report(
    *,
    formula: str,
    sets: Sequence[Mapping[str, Any]],
    retrieval: Sequence[Mapping[str, Any]],
    relatedness: Sequence[Mapping[str, Any]],
    semantic: Sequence[Mapping[str, Any]],
    fine: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
) -> str:
    group_index = {row["partition"]: row for row in relatedness}
    semantic_index = {
        (row["semantic_key"], row["partition"]): row for row in semantic
    }
    fine_index = {
        (row["axis"], row["label"], row["partition"]): row for row in fine
    }
    provenance_index = {
        (row["ret_partition"], row["original_provenance_partition"]): row
        for row in provenance
    }
    partitions = tuple(table_base.PARTITIONS)
    headers = [DISPLAY[name] for name, _ in partitions]
    lines = [
        "# Graph Sel-only vs Sel overlap vs Deep merged Sel-only",
        "",
        f"生产 rerank 公式：`{formula}`。本报告使用原 pipeline 已保存的 Selector 输出，"
        "没有重新调用 Selector 或 LLM。",
        "",
        "## Sel 集合规模",
        "",
        "| 集合 | 论文 | GT |",
        "|---|---:|---:|",
    ]
    for row in sets:
        lines.append(
            f"| {row['display']} | {int(row['candidate_count']):,} | "
            f"{int(row['ground_truth_count']):,} |"
        )
    lines.extend(
        [
            "",
            "## Selector 后检索指标",
            "",
            "| 方法 | 候选 | GT | Macro Recall | Macro Precision | Macro F1 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in retrieval:
        lines.append(
            f"| {row['display']} | {int(row['candidate_count']):,} | "
            f"{int(row['ground_truth_count']):,} | {_pct(row['macro_recall'])} | "
            f"{_pct(row['macro_precision'])} | {_pct(row['macro_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Sel-only GT 的完整候选库来源",
            "",
            "Sel-only 是最终选择差集，不一定是检索来源独有。",
            "",
            "| Sel 分区 | 原 Graph-only GT | 原候选库交集 GT | 原 Deep-only GT | 合计 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for partition, _ in partitions:
        values = [
            int(provenance_index[(partition, provenance_name)]["ground_truth_count"])
            for provenance_name in (
                "graph_only",
                "graph_and_deep_merged",
                "deep_merged_only",
            )
        ]
        lines.append(
            f"| {DISPLAY[partition]} | {values[0]:,} | {values[1]:,} | "
            f"{values[2]:,} | {sum(values):,} |"
        )
    lines.extend(
        [
            "",
            "## 1. 相关性",
            "",
            f"| 指标 | {headers[0]} | {headers[1]} | {headers[2]} |",
            "|---|---:|---:|---:|",
        ]
    )
    rows = (
        ("Direct", "direct_count", "direct_rate"),
        ("Partial", "partial_count", "partial_rate"),
        ("Contextual", "contextual_count", "contextual_rate"),
        ("D + P + C", "information_bearing_count", "information_bearing_rate"),
        ("Unrelated", "unrelated_count", "unrelated_rate"),
        ("Insufficient evidence", "insufficient_evidence_count", "insufficient_evidence_rate"),
    )
    for label, count_key, rate_key in rows:
        cells = [
            _cell(int(group_index[partition][count_key]), group_index[partition][rate_key])
            for partition, _ in partitions
        ]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    lines.append(
        "| 加权相关性 | "
        + " | ".join(
            f"{float(group_index[partition]['mean_relevance_score']):.4f}"
            for partition, _ in partitions
        )
        + " |"
    )
    lines.extend(
        [
            "",
            "## 2. 学术角色与语义补充",
            "",
            "比例分母为各 Sel 分区中的 D+P+C；标签允许多选。",
            "",
            f"| 补充类型 | {headers[0]} | {headers[1]} | {headers[2]} |",
            "|---|---:|---:|---:|",
        ]
    )
    for semantic_key, label_zh, _ in table_base.SEMANTIC_ROWS:
        cells = []
        for partition, _ in partitions:
            row = semantic_index[(semantic_key, partition)]
            cells.append(
                _cell(
                    int(row["candidate_count"]),
                    row["rate_among_information_bearing"],
                )
            )
        lines.append(f"| {label_zh} | {' | '.join(cells)} |")
    lines.extend(["", "## 3. 二阶段细分"])
    for axis, labels in table_base.AXIS_LABELS.items():
        denoms = [
            int(fine_index[(axis, labels[0][0], partition)]["applicable_count"])
            for partition, _ in partitions
        ]
        lines.extend(
            [
                "",
                f"### {table_base.AXIS_NAMES[axis]}",
                "",
                f"| 类别 | {headers[0]}（n={denoms[0]:,}） | "
                f"{headers[1]}（n={denoms[1]:,}） | {headers[2]}（n={denoms[2]:,}） |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, label_zh in labels:
            cells = []
            for partition, _ in partitions:
                row = fine_index[(axis, label, partition)]
                cells.append(
                    _cell(int(row["paper_count"]), row["rate_among_applicable"])
                )
            lines.append(f"| {label_zh} | {' | '.join(cells)} |")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    annotation_work_dir = Path(args.annotation_work_dir).resolve()
    exclusive_refinement_dir = Path(args.exclusive_refinement_analysis_dir).resolve()
    overlap_refinement_dir = Path(args.overlap_refinement_analysis_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = validate_formula(
        run_dir / "onepass_artifacts" / "run_manifest.json",
        query_weight=float(args.query_weight),
        subquery_weight=float(args.subquery_weight),
        intent_weight=float(args.intent_weight),
        path_weight=float(args.path_weight),
    )
    formula = (
        f"{weights['query_score_normalized']:.2f}Q + "
        f"{weights['subquery_score_normalized']:.2f}SQ + "
        f"{weights['intent_score']:.2f}Intent + "
        f"{weights['path_count_normalized']:.2f}Path"
    )

    artifact_dir = run_dir / "onepass_artifacts"
    query_sets: Dict[str, Mapping[str, Set[str]]] = {
        "graph": load_selector_query_sets(
            artifact_dir / "per_subquery" / "query_results.jsonl"
        ),
        "deep_merged": load_selector_query_sets(
            artifact_dir / "deep_merged" / "query_results.jsonl"
        ),
    }
    partition_sets = build_partition_sets(
        query_sets["graph"], query_sets["deep_merged"]
    )
    union = partition_sets["graph"] | partition_sets["deep_merged"]
    annotations = load_selected_annotations(
        annotation_work_dir / "analysis" / "annotations.jsonl", union
    )
    contexts = rerank.load_query_contexts(
        annotation_work_dir / "manifest" / "queries.jsonl"
    )
    retrieval = retrieval_rows(query_sets, contexts)
    relatedness, group_index = ret_analysis._group_rows(partition_sets, annotations)
    semantic = ret_analysis._semantic_rows(partition_sets, annotations, group_index)
    fine_annotations = ret_analysis._load_fine_annotations(
        (
            exclusive_refinement_dir / "annotations.jsonl",
            overlap_refinement_dir / "annotations.jsonl",
        )
    )
    fine = ret_analysis._fine_rows(partition_sets, annotations, fine_annotations)
    sets = _set_rows(partition_sets, annotations)
    original_provenance = ret_analysis._load_original_provenance(
        annotation_work_dir / "analysis" / "annotations.jsonl", union
    )
    provenance = ret_analysis._ret_provenance_rows(
        partition_sets, original_provenance
    )
    provenance_retention = ret_analysis._provenance_retention_rows(
        partition_sets,
        original_provenance,
        annotation_work_dir / "analysis",
    )

    rerank._atomic_write_csv(output_dir / "set_summary.csv", sets)
    rerank._atomic_write_csv(output_dir / "retrieval_summary.csv", retrieval)
    rerank._atomic_write_csv(output_dir / "relatedness.csv", relatedness)
    rerank._atomic_write_csv(output_dir / "semantic_roles.csv", semantic)
    rerank._atomic_write_csv(output_dir / "fine_relations.csv", fine)
    rerank._atomic_write_csv(output_dir / "selector_partition_provenance.csv", provenance)
    rerank._atomic_write_csv(output_dir / "provenance_retention.csv", provenance_retention)

    report = build_report(
        formula=formula,
        sets=sets,
        retrieval=retrieval,
        relatedness=relatedness,
        semantic=semantic,
        fine=fine,
        provenance=provenance,
    )
    report_path = output_dir / "report_zh.md"
    temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)

    summary = {
        "complete": True,
        "implementation_version": IMPLEMENTATION_VERSION,
        "formula": formula,
        "weights": weights,
        "selector_source": "saved production OnePass query_results selected_arxiv_ids",
        "selector_rerun": False,
        "sets": sets,
        "retrieval_summary": retrieval,
        "relatedness": relatedness,
        "semantic_roles": semantic,
        "fine_relations": fine,
        "selector_partition_provenance": provenance,
        "provenance_retention": provenance_retention,
        "output_dir": str(output_dir),
    }
    rerank._atomic_write_json(output_dir / "summary.json", summary)
    rerank._atomic_write_json(
        output_dir / "run_config.json",
        {
            "implementation_version": IMPLEMENTATION_VERSION,
            "run_dir": str(run_dir),
            "annotation_work_dir": str(annotation_work_dir),
            "exclusive_refinement_analysis_dir": str(exclusive_refinement_dir),
            "overlap_refinement_analysis_dir": str(overlap_refinement_dir),
            "output_dir": str(output_dir),
            "weights": weights,
            "selector_rerun": False,
            "annotation_run": False,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--annotation-work-dir", required=True)
    parser.add_argument("--exclusive-refinement-analysis-dir", required=True)
    parser.add_argument("--overlap-refinement-analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-weight", type=float, default=0.30)
    parser.add_argument("--subquery-weight", type=float, default=0.40)
    parser.add_argument("--intent-weight", type=float, default=0.15)
    parser.add_argument("--path-weight", type=float, default=0.15)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
