#!/usr/bin/env python3
"""Rerank the Graph/Deep-merged candidate-pool intersection per stable subquery.

For every Deep-merged stable-subquery group, define the candidate pool as::

    I_s = (union of the group's Graph event pools) intersect DeepPool_s

Candidates are ranked once inside ``I_s`` and then sliced chronologically with
the actual Baseline Selector-input budgets ``K_i``.  No Selector, retriever,
embedding backend, or language model is called.

A paper can occur in more than one Graph event in the same stable-subquery
group.  Its canonical feature row is the occurrence with the best configured
four-factor score (with the production Graph tie breaks).  This preserves the
meaningful Graph ``intent_score`` and ``path_count_normalized`` fields, which
are zero-filled on the Deep rows in the saved production artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_baseline_budget_ret_rerank as rerank  # noqa: E402
import build_ret_three_way_candidate_analysis as ret_analysis  # noqa: E402
import build_three_way_candidate_analysis as table_base  # noqa: E402


IMPLEMENTATION_VERSION = "1.0"
METHOD = "subquery_pool_intersection"
PARTITION = "graph_and_deep_merged"
Key = Tuple[str, str]
TRIGGERS = ret_analysis.TRIGGERS
RELEVANCE_SCORES = ret_analysis.RELEVANCE_SCORES


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    yield from rerank._iter_jsonl(path)


def _key(query_id: Any, paper_id: Any) -> Key:
    return str(query_id or ""), rerank._paper_id(paper_id)


def _profile_query_sets(
    *,
    name: str,
    query_sets: Mapping[str, Set[str]],
    contexts: Mapping[str, rerank.QueryContext],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    per_query: List[Dict[str, Any]] = []
    total_gt = total_candidates = total_hits = 0
    for context in sorted(contexts.values(), key=lambda value: value.benchmark_idx):
        candidates = set(query_sets.get(context.query_id, set()))
        hits = candidates & set(context.ground_truth_ids)
        recall = rerank._safe_rate(len(hits), len(context.ground_truth_ids))
        precision = rerank._safe_rate(len(hits), len(candidates))
        per_query.append(
            {
                "set": name,
                "benchmark_idx": context.benchmark_idx,
                "query_id": context.query_id,
                "gt_count": len(context.ground_truth_ids),
                "candidate_count": len(candidates),
                "candidate_gt_count": len(hits),
                "candidate_recall": recall,
                "candidate_precision": precision,
                "candidate_f1": rerank._f1(recall, precision),
                "candidate_ids": sorted(candidates),
                "candidate_gt_ids": sorted(hits),
            }
        )
        total_gt += len(context.ground_truth_ids)
        total_candidates += len(candidates)
        total_hits += len(hits)
    micro_recall = rerank._safe_rate(total_hits, total_gt)
    micro_precision = rerank._safe_rate(total_hits, total_candidates)
    summary = {
        "set": name,
        "query_count": len(per_query),
        "candidate_count": total_candidates,
        "candidate_gt_count": total_hits,
        "micro_recall": micro_recall,
        "micro_precision": micro_precision,
        "micro_f1": rerank._f1(micro_recall, micro_precision),
        "macro_recall": sum(row["candidate_recall"] for row in per_query) / len(per_query),
        "macro_precision": sum(row["candidate_precision"] for row in per_query) / len(per_query),
        "macro_f1": sum(row["candidate_f1"] for row in per_query) / len(per_query),
    }
    return summary, per_query


def _minimal_graph_row(row: Mapping[str, Any], event_id: str) -> Dict[str, Any]:
    return {
        "paper_arxiv_id": rerank._paper_id(row.get("paper_arxiv_id")),
        "retrieval_event_id": event_id,
        "candidate_type": row.get("candidate_type"),
        "is_seed": rerank._graph_is_seed(row),
        "is_expanded": rerank._graph_is_expanded(row),
        "query_score_normalized": row.get("query_score_normalized"),
        "subquery_score_normalized": row.get("subquery_score_normalized"),
        "intent_score": row.get("intent_score"),
        "path_count": row.get("path_count"),
        "path_count_normalized": row.get("path_count_normalized"),
        "observed_retrieval_rank": row.get("observed_retrieval_rank"),
    }


def load_graph_event_rows(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    output: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for record in _iter_jsonl(path):
        event_id = str(record.get("retrieval_event_id") or "")
        if not event_id or event_id in output:
            raise ValueError(f"missing or duplicate Graph event: {event_id!r}")
        rows: Dict[str, Dict[str, Any]] = {}
        for raw in record.get("local_pool_rows") or []:
            row = _minimal_graph_row(raw, event_id)
            paper_id = row["paper_arxiv_id"]
            if not paper_id or paper_id in rows:
                raise ValueError(f"missing or duplicate Graph paper in {event_id}: {paper_id!r}")
            rows[paper_id] = row
        output[event_id] = rows
    if not output:
        raise ValueError(f"no Graph pools found in {path}")
    return output


def choose_best_graph_occurrences(
    occurrences: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    query_weight: float,
    subquery_weight: float,
    intent_weight: float,
    path_weight: float,
) -> List[Dict[str, Any]]:
    """Choose one deterministic best Graph feature occurrence per paper."""

    canonical: List[Dict[str, Any]] = []
    for paper_id, rows in occurrences.items():
        if not rows:
            raise ValueError(f"paper {paper_id!r} has no Graph feature occurrences")
        best = rerank.rank_graph_rows(
            rows,
            query_weight=query_weight,
            subquery_weight=subquery_weight,
            intent_weight=intent_weight,
            path_weight=path_weight,
        )[0]
        selected = dict(best)
        selected["paper_arxiv_id"] = paper_id
        selected["graph_occurrence_count"] = len(rows)
        selected["winning_graph_event_id"] = str(best.get("retrieval_event_id") or "")
        selected["rerank_score"] = rerank.semantic_score(
            best,
            query_weight=query_weight,
            subquery_weight=subquery_weight,
            intent_weight=intent_weight,
            path_weight=path_weight,
        )
        canonical.append(selected)
    return rerank.rank_graph_rows(
        canonical,
        query_weight=query_weight,
        subquery_weight=subquery_weight,
        intent_weight=intent_weight,
        path_weight=path_weight,
    )


def rerank_intersection_group(
    *,
    record: Mapping[str, Any],
    graph_event_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    budgets: Mapping[str, rerank.BaselineBudget],
    query_weight: float,
    subquery_weight: float,
    intent_weight: float,
    path_weight: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Set[str], Set[str]]:
    """Return group diagnostics, event slices, pool IDs, and selected IDs."""

    query_id = str(record.get("query_id") or "")
    slices = sorted(
        list(record.get("selector_slices") or []),
        key=lambda row: rerank._as_int(row.get("selector_slice_idx"), 10**12),
    )
    if not slices:
        raise ValueError(f"Deep merged group has no selector slices: query={query_id}")
    event_ids = [str(row.get("retrieval_event_id") or "") for row in slices]
    if len(event_ids) != len(set(event_ids)) or any(not event_id for event_id in event_ids):
        raise ValueError(f"invalid source events in Deep merged group: query={query_id}")

    graph_union: Set[str] = set()
    graph_occurrences: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for event_id in event_ids:
        if event_id not in budgets:
            raise ValueError(f"Deep merged source event lacks Baseline budget: {event_id}")
        if event_id not in graph_event_rows:
            raise ValueError(f"Deep merged source event lacks Graph pool: {event_id}")
        budget = budgets[event_id]
        if budget.query_id != query_id:
            raise ValueError(f"query mismatch for Deep merged source event {event_id}")
        for paper_id, graph_row in graph_event_rows[event_id].items():
            graph_union.add(paper_id)
            graph_occurrences[paper_id].append(graph_row)

    saved_graph_union = set(
        rerank._ordered_unique(record.get("source_graph_pool_union_arxiv_ids") or [])
    )
    if saved_graph_union and saved_graph_union != graph_union:
        raise ValueError(
            f"Graph union mismatch for query={query_id}, event={record.get('retrieval_event_id')}: "
            f"computed={len(graph_union)}, saved={len(saved_graph_union)}"
        )

    deep_rows = list(record.get("deep_pool_rows") or [])
    deep_ids = {
        rerank._paper_id(row.get("paper_arxiv_id"))
        for row in deep_rows
        if rerank._paper_id(row.get("paper_arxiv_id"))
    }
    if len(deep_ids) != len(deep_rows):
        raise ValueError(f"missing or duplicate Deep paper IDs for query={query_id}")
    intersection_ids = graph_union & deep_ids
    occurrences = {
        paper_id: graph_occurrences[paper_id] for paper_id in intersection_ids
    }
    ranked = choose_best_graph_occurrences(
        occurrences,
        query_weight=query_weight,
        subquery_weight=subquery_weight,
        intent_weight=intent_weight,
        path_weight=path_weight,
    )

    selected_ids: Set[str] = set()
    event_rows: List[Dict[str, Any]] = []
    offset = 0
    for slice_index, (slice_row, event_id) in enumerate(zip(slices, event_ids), start=1):
        budget = budgets[event_id]
        selected = ranked[offset : offset + budget.k]
        offset += budget.k
        selected_ids.update(
            rerank._paper_id(row.get("paper_arxiv_id")) for row in selected
        )
        event_rows.append(
            {
                "method": METHOD,
                "query_id": query_id,
                "benchmark_idx": budget.benchmark_idx,
                "stable_subquery_group_event_id": str(record.get("retrieval_event_id") or ""),
                "selector_slice_idx": slice_index,
                "retrieval_event_id": event_id,
                "iteration_idx": budget.iteration_idx,
                "subquery_id": budget.subquery_id,
                "subquery": budget.subquery,
                "baseline_selector_input_k": budget.k,
                "intersection_pool_size": len(intersection_ids),
                "selection_start_rank": min(offset - budget.k + 1, len(ranked) + 1),
                "selection_end_rank": min(offset, len(ranked)),
                "selected_count": len(selected),
                "budget_shortfall": budget.k - len(selected),
                "selected": [
                    {
                        "paper_arxiv_id": rerank._paper_id(row.get("paper_arxiv_id")),
                        "merged_rank": offset - budget.k + rank,
                        "rerank_score": row.get("rerank_score"),
                        "query_score_normalized": row.get("query_score_normalized"),
                        "subquery_score_normalized": row.get("subquery_score_normalized"),
                        "intent_score": row.get("intent_score"),
                        "path_count_normalized": row.get("path_count_normalized"),
                        "winning_graph_event_id": row.get("winning_graph_event_id"),
                        "graph_occurrence_count": row.get("graph_occurrence_count"),
                    }
                    for rank, row in enumerate(selected, start=1)
                ],
            }
        )

    required = sum(budgets[event_id].k for event_id in event_ids)
    group = {
        "method": METHOD,
        "query_id": query_id,
        "benchmark_idx": rerank._as_int(record.get("benchmark_idx"), -1),
        "stable_subquery_group_event_id": str(record.get("retrieval_event_id") or ""),
        "subquery_id": rerank._as_int(record.get("subquery_id"), -1),
        "subquery": str(record.get("subquery") or ""),
        "source_event_count": len(event_ids),
        "source_event_ids": event_ids,
        "graph_union_count": len(graph_union),
        "deep_pool_count": len(deep_ids),
        "intersection_pool_count": len(intersection_ids),
        "intersection_pool_ids": sorted(intersection_ids),
        "requested_occurrence_budget": required,
        "selected_occurrence_count": min(required, len(ranked)),
        "budget_shortfall": max(0, required - len(ranked)),
        "selected_ids": [
            rerank._paper_id(row.get("paper_arxiv_id"))
            for row in ranked[:required]
        ],
    }
    return group, event_rows, intersection_ids, selected_ids


def _load_reference_ret_sets(
    path: Path, formula: str
) -> Dict[str, MutableMapping[str, Set[str]]]:
    output: Dict[str, MutableMapping[str, Set[str]]] = {
        "graph": defaultdict(set),
        "deep_merged": defaultdict(set),
    }
    observed: Set[Tuple[str, str]] = set()
    for row in _iter_jsonl(path):
        if str(row.get("formula") or "") != formula:
            continue
        method = str(row.get("method") or "")
        if method not in output:
            continue
        query_id = str(row.get("query_id") or "")
        marker = (method, query_id)
        if marker in observed:
            raise ValueError(f"duplicate reference Ret result: {marker}")
        observed.add(marker)
        output[method][query_id] = set(
            rerank._ordered_unique(row.get("candidate_ids") or [])
        )
    if not observed:
        raise ValueError(f"formula {formula!r} not found in {path}")
    return output


def _load_annotations_and_full_overlap(
    path: Path,
    expected: Set[Key],
) -> Tuple[Dict[Key, Dict[str, Any]], Dict[str, Set[str]]]:
    annotations: Dict[Key, Dict[str, Any]] = {}
    full_overlap: Dict[str, Set[str]] = defaultdict(set)
    for row in _iter_jsonl(path):
        key = _key(row.get("query_id"), row.get("paper_id"))
        sources = set(str(value) for value in (row.get("sources") or []))
        if {"graph", "deep_merged"}.issubset(sources):
            full_overlap[key[0]].add(key[1])
        if key not in expected:
            continue
        if key in annotations:
            raise ValueError(f"duplicate first-stage annotation: {key}")
        if not row.get("annotation"):
            raise ValueError(f"missing first-stage annotation: {key}")
        annotations[key] = row
    if set(annotations) != expected:
        missing = expected - set(annotations)
        raise ValueError(
            f"selected annotation coverage mismatch: found={len(annotations)}, "
            f"expected={len(expected)}, first_missing={next(iter(missing), None)}"
        )
    return annotations, full_overlap


def _semantic_tables(
    keys: Set[Key],
    annotations: Mapping[Key, Mapping[str, Any]],
    fine_paths: Sequence[Path],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    partition_sets = {
        "graph_only": set(),
        PARTITION: keys,
        "deep_merged_only": set(),
    }
    groups, group_index = ret_analysis._group_rows(partition_sets, annotations)
    semantic = ret_analysis._semantic_rows(partition_sets, annotations, group_index)
    fine_annotations = ret_analysis._load_fine_annotations(fine_paths)
    fine = ret_analysis._fine_rows(partition_sets, annotations, fine_annotations)
    group = next(row for row in groups if row["partition"] == PARTITION)
    return (
        group,
        [row for row in semantic if row["partition"] == PARTITION],
        [row for row in fine if row["partition"] == PARTITION],
    )


def _membership_rows(
    selected: Mapping[str, Set[str]],
    references: Mapping[str, Mapping[str, Set[str]]],
    contexts: Mapping[str, rerank.QueryContext],
) -> List[Dict[str, Any]]:
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for context in contexts.values():
        graph = set(references["graph"].get(context.query_id, set()))
        deep = set(references["deep_merged"].get(context.query_id, set()))
        gt = set(context.ground_truth_ids)
        for paper_id in selected.get(context.query_id, set()):
            if paper_id in graph and paper_id in deep:
                membership = "in_both_reference_ret"
            elif paper_id in graph:
                membership = "in_graph_reference_ret_only"
            elif paper_id in deep:
                membership = "in_deep_reference_ret_only"
            else:
                membership = "in_neither_reference_ret"
            counts[membership]["candidate_count"] += 1
            counts[membership]["ground_truth_count"] += int(paper_id in gt)
    order = (
        "in_both_reference_ret",
        "in_graph_reference_ret_only",
        "in_deep_reference_ret_only",
        "in_neither_reference_ret",
    )
    return [
        {
            "membership": name,
            "candidate_count": counts[name]["candidate_count"],
            "ground_truth_count": counts[name]["ground_truth_count"],
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


def _report(
    *,
    formula: str,
    summaries: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    membership: Sequence[Mapping[str, Any]],
    group: Mapping[str, Any],
    semantic: Sequence[Mapping[str, Any]],
    fine: Sequence[Mapping[str, Any]],
) -> str:
    semantic_index = {row["semantic_key"]: row for row in semantic}
    fine_index = {(row["axis"], row["label"]): row for row in fine}
    lines = [
        "# Stable-subquery Graph ∩ Deep merged candidate-pool rerank",
        "",
        f"公式：`{formula}`。不运行 Selector。",
        "候选池口径：每个稳定 subquery 先取 Graph 事件池并集与 Deep merged 池的交集，"
        "在交集内排序一次，再按 Baseline selector-input K_i 顺序切片。",
        "多事件重复论文使用四因子得分最高的 Graph 事件特征。",
        "",
        "## 检索结果",
        "",
        "| 集合 | 候选 | GT | Macro Recall | Macro Precision | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['set']} | {int(row['candidate_count']):,} | "
            f"{int(row['candidate_gt_count']):,} | {_pct(row['macro_recall'])} | "
            f"{_pct(row['macro_precision'])} | {_pct(row['macro_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## 交集池与预算",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| 稳定 subquery 组 | {int(diagnostics['group_count']):,} |",
            f"| Baseline occurrence 预算 | {int(diagnostics['requested_occurrence_budget']):,} |",
            f"| 实际填充 occurrence | {int(diagnostics['selected_occurrence_count']):,} |",
            f"| 预算缺口 | {int(diagnostics['budget_shortfall']):,} |",
            f"| 交集池不足预算的组 | {int(diagnostics['groups_below_budget']):,} |",
            f"| 空交集组 | {int(diagnostics['empty_group_count']):,} |",
            f"| 交集池大小中位数 | {float(diagnostics['median_intersection_pool_size']):.1f} |",
            "",
            "## 新交集 Ret 与原 Graph/Deep Ret 的关系",
            "",
            "| 原 Ret 成员关系 | 候选 | GT |",
            "|---|---:|---:|",
        ]
    )
    for row in membership:
        lines.append(
            f"| {row['membership']} | {int(row['candidate_count']):,} | "
            f"{int(row['ground_truth_count']):,} |"
        )
    lines.extend(
        [
            "",
            "## 1. 相关性",
            "",
            "| 指标 | 交集池 Ret |",
            "|---|---:|",
            f"| Direct | {_cell(int(group['direct_count']), group['direct_rate'])} |",
            f"| Partial | {_cell(int(group['partial_count']), group['partial_rate'])} |",
            f"| Contextual | {_cell(int(group['contextual_count']), group['contextual_rate'])} |",
            f"| D + P + C | {_cell(int(group['information_bearing_count']), group['information_bearing_rate'])} |",
            f"| Unrelated | {_cell(int(group['unrelated_count']), group['unrelated_rate'])} |",
            f"| Insufficient evidence | {_cell(int(group['insufficient_evidence_count']), group['insufficient_evidence_rate'])} |",
            f"| 加权相关性 | {float(group['mean_relevance_score']):.4f} |",
            "",
            "## 2. 学术角色与语义补充",
            "",
            "比例分母为交集池 Ret 中的 D+P+C。",
            "",
            "| 补充类型 | 交集池 Ret |",
            "|---|---:|",
        ]
    )
    for semantic_key, label_zh, _ in table_base.SEMANTIC_ROWS:
        row = semantic_index[semantic_key]
        lines.append(
            f"| {label_zh} | {_cell(int(row['candidate_count']), row['rate_among_information_bearing'])} |"
        )
    lines.extend(["", "## 3. 二阶段细分"])
    for axis, labels in table_base.AXIS_LABELS.items():
        denominator = fine_index[(axis, labels[0][0])]["applicable_count"]
        lines.extend(
            [
                "",
                f"### {table_base.AXIS_NAMES[axis]}",
                "",
                f"| 类别 | 交集池 Ret（n={int(denominator):,}） |",
                "|---|---:|",
            ]
        )
        for label, label_zh in labels:
            row = fine_index[(axis, label)]
            lines.append(
                f"| {label_zh} | {_cell(int(row['paper_count']), row['rate_among_applicable'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    reference_ret_dir = Path(args.reference_ret_dir).resolve()
    annotation_work_dir = Path(args.annotation_work_dir).resolve()
    exclusive_refinement_dir = Path(args.exclusive_refinement_analysis_dir).resolve()
    overlap_refinement_dir = Path(args.overlap_refinement_analysis_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = (
        float(args.query_weight),
        float(args.subquery_weight),
        float(args.intent_weight),
        float(args.path_weight),
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("rerank weights must be non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"rerank weights must sum to one; got {sum(weights):.12g}")
    query_weight, subquery_weight, intent_weight, path_weight = weights
    formula = (
        f"{query_weight:.2f} * q_norm + {subquery_weight:.2f} * subq_norm + "
        f"{intent_weight:.2f} * intent_score + {path_weight:.2f} * path_count_norm"
    )

    artifact_dir = run_dir / "onepass_artifacts"
    budgets = rerank.load_baseline_budgets(
        artifact_dir / "baseline" / "selector_decisions.jsonl"
    )
    contexts = rerank.load_query_contexts(
        annotation_work_dir / "manifest" / "queries.jsonl"
    )
    graph_event_rows = load_graph_event_rows(
        artifact_dir / "per_subquery" / "pool_records.jsonl"
    )

    seen_events: Set[str] = set()
    group_rows: List[Dict[str, Any]] = []
    event_rows: List[Dict[str, Any]] = []
    pool_query_sets: Dict[str, Set[str]] = defaultdict(set)
    selected_query_sets: Dict[str, Set[str]] = defaultdict(set)
    for record in _iter_jsonl(artifact_dir / "deep_merged" / "pool_records.jsonl"):
        group, events, pool_ids, selected_ids = rerank_intersection_group(
            record=record,
            graph_event_rows=graph_event_rows,
            budgets=budgets,
            query_weight=query_weight,
            subquery_weight=subquery_weight,
            intent_weight=intent_weight,
            path_weight=path_weight,
        )
        group_event_ids = set(group["source_event_ids"])
        duplicate_events = seen_events & group_event_ids
        if duplicate_events:
            raise ValueError(f"Deep merged events repeated across groups: {sorted(duplicate_events)[:3]}")
        seen_events.update(group_event_ids)
        query_id = str(group["query_id"])
        pool_query_sets[query_id].update(pool_ids)
        selected_query_sets[query_id].update(selected_ids)
        group_rows.append(group)
        event_rows.extend(events)
    if seen_events != set(budgets):
        raise ValueError(
            f"Deep merged event coverage mismatch: missing={len(set(budgets) - seen_events)}, "
            f"extra={len(seen_events - set(budgets))}"
        )

    references = _load_reference_ret_sets(
        reference_ret_dir / "query_ret_results.jsonl", args.reference_formula
    )
    selected_keys = {
        (query_id, paper_id)
        for query_id, paper_ids in selected_query_sets.items()
        for paper_id in paper_ids
    }
    annotations, full_overlap_query_sets = _load_annotations_and_full_overlap(
        annotation_work_dir / "analysis" / "annotations.jsonl", selected_keys
    )
    if any(
        paper_id not in full_overlap_query_sets.get(query_id, set())
        for query_id, paper_id in selected_keys
    ):
        raise ValueError("same-subquery intersection selection contains a non-overlap provenance paper")

    summaries: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []
    set_specs = (
        ("Full query-level source overlap", full_overlap_query_sets),
        ("Same-subquery pool intersection", pool_query_sets),
        ("Intersection-pool Ret", selected_query_sets),
        ("Graph Ret", references["graph"]),
        ("Deep merged Ret", references["deep_merged"]),
    )
    for name, query_sets in set_specs:
        summary, rows = _profile_query_sets(
            name=name, query_sets=query_sets, contexts=contexts
        )
        summaries.append(summary)
        query_rows.extend(rows)

    group, semantic, fine = _semantic_tables(
        selected_keys,
        annotations,
        (
            exclusive_refinement_dir / "annotations.jsonl",
            overlap_refinement_dir / "annotations.jsonl",
        ),
    )
    membership = _membership_rows(selected_query_sets, references, contexts)
    pool_sizes = [int(row["intersection_pool_count"]) for row in group_rows]
    diagnostics = {
        "group_count": len(group_rows),
        "event_count": len(seen_events),
        "requested_occurrence_budget": sum(int(row["requested_occurrence_budget"]) for row in group_rows),
        "selected_occurrence_count": sum(int(row["selected_occurrence_count"]) for row in group_rows),
        "budget_shortfall": sum(int(row["budget_shortfall"]) for row in group_rows),
        "groups_below_budget": sum(int(row["budget_shortfall"]) > 0 for row in group_rows),
        "empty_group_count": sum(int(row["intersection_pool_count"]) == 0 for row in group_rows),
        "min_intersection_pool_size": min(pool_sizes) if pool_sizes else 0,
        "median_intersection_pool_size": statistics.median(pool_sizes) if pool_sizes else 0.0,
        "max_intersection_pool_size": max(pool_sizes) if pool_sizes else 0,
    }

    rerank._atomic_write_jsonl(output_dir / "subquery_intersection_pools.jsonl", group_rows)
    rerank._atomic_write_jsonl(output_dir / "event_ret_selections.jsonl", event_rows)
    rerank._atomic_write_jsonl(output_dir / "query_set_results.jsonl", query_rows)
    rerank._atomic_write_csv(output_dir / "retrieval_summary.csv", summaries)
    rerank._atomic_write_jsonl(output_dir / "retrieval_summary.jsonl", summaries)
    rerank._atomic_write_csv(output_dir / "reference_ret_membership.csv", membership)
    rerank._atomic_write_csv(output_dir / "relatedness.csv", [group])
    rerank._atomic_write_csv(output_dir / "semantic_roles.csv", semantic)
    rerank._atomic_write_csv(output_dir / "fine_relations.csv", fine)

    report = _report(
        formula=formula,
        summaries=summaries,
        diagnostics=diagnostics,
        membership=membership,
        group=group,
        semantic=semantic,
        fine=fine,
    )
    report_path = output_dir / "report_zh.md"
    temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)

    summary = {
        "complete": True,
        "implementation_version": IMPLEMENTATION_VERSION,
        "method": METHOD,
        "formula": formula,
        "feature_policy": "best-scoring Graph occurrence per paper within stable subquery",
        "pool_definition": "union(Graph event pools in stable subquery) intersect Deep merged pool",
        "k_definition": "len(baseline.selector_decision.candidate_rows) per retrieval_event_id",
        "selector_run": False,
        "diagnostics": diagnostics,
        "retrieval_summary": summaries,
        "reference_ret_membership": membership,
        "relatedness": group,
        "semantic_roles": semantic,
        "fine_relations": fine,
        "output_dir": str(output_dir),
    }
    rerank._atomic_write_json(output_dir / "summary.json", summary)
    rerank._atomic_write_json(
        output_dir / "run_config.json",
        {
            "implementation_version": IMPLEMENTATION_VERSION,
            "run_dir": str(run_dir),
            "reference_ret_dir": str(reference_ret_dir),
            "reference_formula": args.reference_formula,
            "annotation_work_dir": str(annotation_work_dir),
            "exclusive_refinement_analysis_dir": str(exclusive_refinement_dir),
            "overlap_refinement_analysis_dir": str(overlap_refinement_dir),
            "output_dir": str(output_dir),
            "query_weight": query_weight,
            "subquery_weight": subquery_weight,
            "intent_weight": intent_weight,
            "path_weight": path_weight,
            "formula": formula,
            "feature_policy": "best-scoring Graph occurrence per paper within stable subquery",
            "selector_run": False,
            "retriever_run": False,
            "embedding_run": False,
            "annotation_run": False,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--reference-ret-dir", required=True)
    parser.add_argument("--reference-formula", default="hybrid_30_40_15_15")
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
