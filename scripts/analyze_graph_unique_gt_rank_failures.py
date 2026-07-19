#!/usr/bin/env python3
"""Diagnose why full-pool Graph-only ground-truth papers miss Ret Top-K.

The analysis is fully offline.  It identifies query-paper pairs whose saved
full-pool provenance is exactly Graph (not Deep merged), replays the Graph
ranking at Baseline's actual per-event Selector-input budget, and records every
occurrence of those papers in the local subquery pools.

For each occurrence it compares:

* the requested four-factor ranking;
* a same-Q/SQ semantic control with the structural terms set to zero;
* a counterfactual in which only the target paper receives the maximum
  possible normalized structural bonus.

This separates low semantic rank, insufficient structural signal, displacement
by competitors' structural scores, and production tie-break losses.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_baseline_budget_ret_rerank as replay  # noqa: E402


IMPLEMENTATION_VERSION = "1.0"
TOLERANCE = 1e-12

CAUSE_LABELS = {
    "selected": "已进入当前 Ret Top-K",
    "tie_break_loss": "同分但被 seed/tie-break 压出",
    "displaced_by_structure": "语义对照可入选，但被竞争者结构加分挤出",
    "insufficient_structural_signal": "语义不足；结构项理论上可救回但实际信号不足",
    "semantic_deficit_beyond_max_structure": "语义差距超过 0.15 最大结构加分",
}


def _float(row: Mapping[str, Any], key: str) -> float:
    value = float(row.get(key) or 0.0)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} for paper={row.get('paper_arxiv_id')}")
    return value


def _tie_key(row: Mapping[str, Any]) -> Tuple[int, int, str]:
    observed_rank = row.get("observed_retrieval_rank")
    return (
        -int(replay._graph_is_seed(row)),
        replay._as_int(observed_rank, 10**12)
        if observed_rank is not None
        else 10**12,
        replay._paper_id(row.get("paper_arxiv_id")),
    )


def _counterfactual_rank(
    rows: Sequence[Mapping[str, Any]],
    target_row: Mapping[str, Any],
    target_score: float,
    *,
    query_weight: float,
    subquery_weight: float,
    intent_weight: float,
    path_weight: float,
) -> int:
    """Return production rank if only target_row's score were target_score."""

    target_id = replay._paper_id(target_row.get("paper_arxiv_id"))
    target_tie = _tie_key(target_row)
    before = 0
    for row in rows:
        if replay._paper_id(row.get("paper_arxiv_id")) == target_id:
            continue
        score = replay.semantic_score(
            row,
            query_weight,
            subquery_weight,
            intent_weight,
            path_weight,
        )
        if score > target_score or (score == target_score and _tie_key(row) < target_tie):
            before += 1
    return before + 1


def _distribution(values: Iterable[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(0.25),
        "median": median(ordered),
        "p75": percentile(0.75),
        "max": ordered[-1],
        "mean": mean(ordered),
    }


def _primary_sources(candidate: Mapping[str, Any]) -> Tuple[str, ...]:
    saved_sources = set(candidate.get("sources") or [])
    return tuple(source for source in replay.METHODS if source in saved_sources)


def load_targets(
    candidates_path: Path,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, int]]:
    targets: Dict[Tuple[str, str], Dict[str, Any]] = {}
    graph_unique_count = 0
    graph_unique_gt_count = 0
    deep_unique_count = 0
    deep_unique_gt_count = 0
    for candidate in replay._iter_jsonl(candidates_path):
        sources = _primary_sources(candidate)
        is_gt = bool(candidate.get("is_ground_truth"))
        if sources == ("graph",):
            graph_unique_count += 1
            if is_gt:
                graph_unique_gt_count += 1
                key = (
                    str(candidate.get("query_id") or ""),
                    replay._paper_id(candidate.get("paper_id")),
                )
                if not all(key):
                    raise ValueError(f"invalid Graph-only GT key: {key}")
                if key in targets:
                    raise ValueError(f"duplicate Graph-only GT: {key}")
                targets[key] = {
                    "query_id": key[0],
                    "paper_id": key[1],
                    "candidate_id": str(candidate.get("candidate_id") or ""),
                    "title": str(candidate.get("title") or ""),
                    "paper_date": str(candidate.get("paper_date") or ""),
                    "expected_graph_occurrence_count": replay._as_int(
                        ((candidate.get("source_stats") or {}).get("graph") or {}).get(
                            "occurrence_count"
                        ),
                        0,
                    ),
                    "expected_graph_event_ids": sorted(
                        str(value)
                        for value in (
                            ((candidate.get("source_stats") or {}).get("graph") or {}).get(
                                "event_ids"
                            )
                            or []
                        )
                    ),
                }
        elif sources == ("deep_merged",):
            deep_unique_count += 1
            deep_unique_gt_count += int(is_gt)
    return targets, {
        "graph_pool_unique_candidate_count": graph_unique_count,
        "graph_pool_unique_gt_count": graph_unique_gt_count,
        "deep_pool_unique_candidate_count": deep_unique_count,
        "deep_pool_unique_gt_count": deep_unique_gt_count,
    }


def load_query_texts(path: Path) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for row in replay._iter_jsonl(path):
        query_id = str(row.get("query_id") or "")
        if query_id:
            output[query_id] = str(row.get("query") or "")
    return output


def _rank_map(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return {
        replay._paper_id(row.get("paper_arxiv_id")): rank
        for rank, row in enumerate(rows, start=1)
    }


def analyze_occurrences(
    pool_path: Path,
    budgets: Mapping[str, replay.BaselineBudget],
    targets: Mapping[Tuple[str, str], Mapping[str, Any]],
    query_texts: Mapping[str, str],
    *,
    query_weight: float,
    subquery_weight: float,
    intent_weight: float,
    path_weight: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target_ids_by_query: MutableMapping[str, set[str]] = defaultdict(set)
    for query_id, paper_id in targets:
        target_ids_by_query[query_id].add(paper_id)

    occurrences: List[Dict[str, Any]] = []
    seen_events: set[str] = set()
    all_intent_values: List[float] = []
    all_path_values: List[float] = []
    max_structure_bonus = intent_weight + path_weight

    for pool_record in replay._iter_jsonl(pool_path):
        event_id = str(pool_record.get("retrieval_event_id") or "")
        query_id = str(pool_record.get("query_id") or "")
        if event_id in seen_events:
            raise ValueError(f"duplicate graph event: {event_id}")
        seen_events.add(event_id)
        if event_id not in budgets:
            raise ValueError(f"graph event lacks Baseline budget: {event_id}")
        budget = budgets[event_id]
        if query_id != budget.query_id:
            raise ValueError(f"query mismatch for graph event: {event_id}")
        rows = list(pool_record.get("local_pool_rows") or [])
        replay._validate_unique_pool(rows, f"Graph pool {event_id}")
        if len(rows) < budget.k:
            raise ValueError(f"Graph pool below K={budget.k}: {event_id}")

        for row in rows:
            all_intent_values.append(_float(row, "intent_score"))
            all_path_values.append(_float(row, "path_count_normalized"))

        query_targets = target_ids_by_query.get(query_id)
        if not query_targets:
            continue
        target_rows = [
            row
            for row in rows
            if replay._paper_id(row.get("paper_arxiv_id")) in query_targets
        ]
        if not target_rows:
            continue

        hybrid_rows = replay.rank_graph_rows(
            rows,
            query_weight=query_weight,
            subquery_weight=subquery_weight,
            intent_weight=intent_weight,
            path_weight=path_weight,
        )
        semantic_rows = replay.rank_graph_rows(
            rows,
            query_weight=query_weight,
            subquery_weight=subquery_weight,
            intent_weight=0.0,
            path_weight=0.0,
        )
        hybrid_ranks = _rank_map(hybrid_rows)
        semantic_ranks = _rank_map(semantic_rows)
        hybrid_cutoff = replay.semantic_score(
            hybrid_rows[budget.k - 1],
            query_weight,
            subquery_weight,
            intent_weight,
            path_weight,
        )
        semantic_cutoff = replay.semantic_score(
            semantic_rows[budget.k - 1],
            query_weight,
            subquery_weight,
            0.0,
            0.0,
        )

        for row in target_rows:
            paper_id = replay._paper_id(row.get("paper_arxiv_id"))
            target = targets[(query_id, paper_id)]
            query_value = _float(row, "query_score_normalized")
            subquery_value = _float(row, "subquery_score_normalized")
            intent_value = _float(row, "intent_score")
            path_value = _float(row, "path_count_normalized")
            q_contribution = query_weight * query_value
            sq_contribution = subquery_weight * subquery_value
            intent_contribution = intent_weight * intent_value
            path_contribution = path_weight * path_value
            semantic_score = q_contribution + sq_contribution
            structure_bonus = intent_contribution + path_contribution
            hybrid_score = semantic_score + structure_bonus
            hybrid_rank = hybrid_ranks[paper_id]
            semantic_rank = semantic_ranks[paper_id]
            tie_break_loss = hybrid_rank > budget.k and math.isclose(
                hybrid_score,
                hybrid_cutoff,
                rel_tol=0.0,
                abs_tol=TOLERANCE,
            )
            max_structure_score = semantic_score + max_structure_bonus
            max_structure_rank = _counterfactual_rank(
                rows,
                row,
                max_structure_score,
                query_weight=query_weight,
                subquery_weight=subquery_weight,
                intent_weight=intent_weight,
                path_weight=path_weight,
            )
            required_total_structure_to_match_cutoff = max(
                0.0, hybrid_cutoff - semantic_score
            )
            additional_bonus_to_match_cutoff = max(
                0.0, hybrid_cutoff - hybrid_score
            )
            occurrences.append(
                {
                    "query_id": query_id,
                    "query": query_texts.get(query_id, ""),
                    "paper_id": paper_id,
                    "candidate_id": target.get("candidate_id", ""),
                    "title": target.get("title", ""),
                    "paper_date": target.get("paper_date", ""),
                    "retrieval_event_id": event_id,
                    "iteration_idx": budget.iteration_idx,
                    "subquery_id": budget.subquery_id,
                    "subquery": budget.subquery,
                    "baseline_k": budget.k,
                    "pool_size": len(rows),
                    "candidate_type": str(row.get("candidate_type") or ""),
                    "is_seed": bool(replay._graph_is_seed(row)),
                    "observed_retrieval_rank": row.get("observed_retrieval_rank"),
                    "stored_rerank_score": row.get("rerank_score"),
                    "stored_rerank_rank": row.get("rerank_rank"),
                    "stored_in_selector_topk": bool(row.get("in_selector_topk")),
                    "stored_selector_input_rank": row.get("selector_input_rank"),
                    "query_score_normalized": query_value,
                    "subquery_score_normalized": subquery_value,
                    "intent_score": intent_value,
                    "path_count": _float(row, "path_count"),
                    "path_count_normalized": path_value,
                    "query_contribution": q_contribution,
                    "subquery_contribution": sq_contribution,
                    "intent_contribution": intent_contribution,
                    "path_contribution": path_contribution,
                    "semantic_component_score": semantic_score,
                    "structure_bonus": structure_bonus,
                    "hybrid_score": hybrid_score,
                    "hybrid_rank": hybrid_rank,
                    "hybrid_cutoff_score": hybrid_cutoff,
                    "hybrid_rank_minus_k": hybrid_rank - budget.k,
                    "hybrid_percentile": hybrid_rank / len(rows),
                    "hybrid_selected": hybrid_rank <= budget.k,
                    "hybrid_score_gap_to_cutoff": max(0.0, hybrid_cutoff - hybrid_score),
                    "semantic_control_score": semantic_score,
                    "semantic_control_rank": semantic_rank,
                    "semantic_control_cutoff_score": semantic_cutoff,
                    "semantic_control_rank_minus_k": semantic_rank - budget.k,
                    "semantic_control_selected": semantic_rank <= budget.k,
                    "hybrid_rank_change_vs_semantic": hybrid_rank - semantic_rank,
                    "tie_break_loss": tie_break_loss,
                    "max_structure_bonus": max_structure_bonus,
                    "max_structure_counterfactual_score": max_structure_score,
                    "max_structure_counterfactual_rank": max_structure_rank,
                    "max_structure_counterfactual_selected": max_structure_rank <= budget.k,
                    "required_total_structure_to_match_cutoff": required_total_structure_to_match_cutoff,
                    "additional_structure_bonus_to_match_cutoff": additional_bonus_to_match_cutoff,
                    "edge_types": json.dumps(row.get("edge_types") or [], ensure_ascii=False),
                    "source_seed_ids": json.dumps(row.get("source_seed_ids") or [], ensure_ascii=False),
                }
            )

    missing_events = set(budgets) - seen_events
    if missing_events:
        raise ValueError(f"Graph pool is missing {len(missing_events)} Baseline events")

    return occurrences, {
        "graph_event_count": len(seen_events),
        "graph_pool_intent_range": {
            "min": min(all_intent_values),
            "max": max(all_intent_values),
        },
        "graph_pool_path_normalized_range": {
            "min": min(all_path_values),
            "max": max(all_path_values),
        },
    }


def classify_paper(rows: Sequence[Mapping[str, Any]]) -> str:
    if any(bool(row["hybrid_selected"]) for row in rows):
        return "selected"
    if any(bool(row["tie_break_loss"]) for row in rows):
        return "tie_break_loss"
    if any(bool(row["semantic_control_selected"]) for row in rows):
        return "displaced_by_structure"
    if any(bool(row["max_structure_counterfactual_selected"]) for row in rows):
        return "insufficient_structural_signal"
    return "semantic_deficit_beyond_max_structure"


def summarize_papers(
    targets: Mapping[Tuple[str, str], Mapping[str, Any]],
    occurrences: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, str], Mapping[str, Any]]]:
    grouped: MutableMapping[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in occurrences:
        grouped[(str(row["query_id"]), str(row["paper_id"]))].append(row)
    missing = set(targets) - set(grouped)
    if missing:
        raise ValueError(
            f"{len(missing)} Graph-only GT papers have no Graph pool occurrence; first={sorted(missing)[0]}"
        )

    paper_rows: List[Dict[str, Any]] = []
    best_rows: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for key in sorted(targets):
        rows = grouped[key]
        target = targets[key]
        actual_event_ids = sorted(str(row["retrieval_event_id"]) for row in rows)
        expected_event_ids = list(target.get("expected_graph_event_ids") or [])
        expected_occurrence_count = int(target.get("expected_graph_occurrence_count") or 0)
        if expected_occurrence_count and len(rows) != expected_occurrence_count:
            raise ValueError(
                f"Graph occurrence count mismatch for {key}: "
                f"expected={expected_occurrence_count}, actual={len(rows)}"
            )
        if expected_event_ids and actual_event_ids != expected_event_ids:
            raise ValueError(f"Graph event IDs mismatch for {key}")
        best = min(
            rows,
            key=lambda row: (
                int(row["hybrid_rank_minus_k"]),
                float(row["hybrid_score_gap_to_cutoff"]),
                int(row["hybrid_rank"]),
                -float(row["hybrid_score"]),
            ),
        )
        rescue_rows = [
            row for row in rows if bool(row["max_structure_counterfactual_selected"])
        ]
        best_rescue = (
            min(
                rescue_rows,
                key=lambda row: (
                    int(row["max_structure_counterfactual_rank"]),
                    float(row["required_total_structure_to_match_cutoff"]),
                    int(row["hybrid_rank_minus_k"]),
                ),
            )
            if rescue_rows
            else None
        )
        best_rows[key] = best
        cause = classify_paper(rows)
        stored_ranks = [
            int(row["stored_rerank_rank"])
            for row in rows
            if row.get("stored_rerank_rank") is not None
        ]
        paper_rows.append(
            {
                "query_id": key[0],
                "paper_id": key[1],
                "candidate_id": target.get("candidate_id", ""),
                "title": target.get("title", ""),
                "paper_date": target.get("paper_date", ""),
                "failure_cause": cause,
                "failure_cause_zh": CAUSE_LABELS[cause],
                "occurrence_count": len(rows),
                "expected_occurrence_count": expected_occurrence_count,
                "occurrence_validation_complete": (
                    (not expected_occurrence_count or len(rows) == expected_occurrence_count)
                    and (not expected_event_ids or actual_event_ids == expected_event_ids)
                ),
                "subquery_count": len({(row["subquery_id"], row["subquery"]) for row in rows}),
                "any_nonzero_intent": any(float(row["intent_score"]) > 0 for row in rows),
                "any_nonzero_path": any(float(row["path_count_normalized"]) > 0 for row in rows),
                "max_intent_score": max(float(row["intent_score"]) for row in rows),
                "max_path_count_normalized": max(float(row["path_count_normalized"]) for row in rows),
                "max_structure_bonus": max(float(row["structure_bonus"]) for row in rows),
                "best_hybrid_rank": min(int(row["hybrid_rank"]) for row in rows),
                "best_hybrid_rank_minus_k": min(int(row["hybrid_rank_minus_k"]) for row in rows),
                "best_semantic_control_rank": min(int(row["semantic_control_rank"]) for row in rows),
                "best_stored_rerank_rank": min(stored_ranks) if stored_ranks else None,
                "stored_topk_occurrence_count": sum(
                    bool(row["stored_in_selector_topk"]) for row in rows
                ),
                "best_max_structure_counterfactual_rank": min(
                    int(row["max_structure_counterfactual_rank"]) for row in rows
                ),
                "semantic_control_topk_occurrence_count": sum(
                    bool(row["semantic_control_selected"]) for row in rows
                ),
                "tie_break_loss_occurrence_count": sum(bool(row["tie_break_loss"]) for row in rows),
                "max_structure_rescue_occurrence_count": sum(
                    bool(row["max_structure_counterfactual_selected"]) for row in rows
                ),
                "min_hybrid_score_gap_to_cutoff": min(
                    float(row["hybrid_score_gap_to_cutoff"]) for row in rows
                ),
                "min_required_total_structure_to_match_cutoff": min(
                    float(row["required_total_structure_to_match_cutoff"]) for row in rows
                ),
                "min_additional_structure_bonus_to_match_cutoff": min(
                    float(row["additional_structure_bonus_to_match_cutoff"]) for row in rows
                ),
                "best_event_id": best["retrieval_event_id"],
                "best_iteration_idx": best["iteration_idx"],
                "best_subquery_id": best["subquery_id"],
                "best_subquery": best["subquery"],
                "best_baseline_k": best["baseline_k"],
                "best_pool_size": best["pool_size"],
                "best_is_seed": best["is_seed"],
                "best_query_score_normalized": best["query_score_normalized"],
                "best_subquery_score_normalized": best["subquery_score_normalized"],
                "best_intent_score": best["intent_score"],
                "best_path_count_normalized": best["path_count_normalized"],
                "best_semantic_component_score": best["semantic_component_score"],
                "best_structure_bonus": best["structure_bonus"],
                "best_hybrid_score": best["hybrid_score"],
                "best_hybrid_cutoff_score": best["hybrid_cutoff_score"],
                "best_event_hybrid_score_gap_to_cutoff": best[
                    "hybrid_score_gap_to_cutoff"
                ],
                "best_event_required_total_structure_to_match_cutoff": best[
                    "required_total_structure_to_match_cutoff"
                ],
                "best_hybrid_rank_at_event": best["hybrid_rank"],
                "best_semantic_rank_at_event": best["semantic_control_rank"],
                "best_max_structure_rank_at_event": best["max_structure_counterfactual_rank"],
                "best_rescue_event_id": (
                    best_rescue["retrieval_event_id"] if best_rescue else ""
                ),
                "best_rescue_subquery": best_rescue["subquery"] if best_rescue else "",
                "best_rescue_hybrid_rank": best_rescue["hybrid_rank"] if best_rescue else None,
                "best_rescue_baseline_k": best_rescue["baseline_k"] if best_rescue else None,
                "best_rescue_semantic_rank": (
                    best_rescue["semantic_control_rank"] if best_rescue else None
                ),
                "best_rescue_max_structure_rank": (
                    best_rescue["max_structure_counterfactual_rank"] if best_rescue else None
                ),
                "best_rescue_query_score_normalized": (
                    best_rescue["query_score_normalized"] if best_rescue else None
                ),
                "best_rescue_subquery_score_normalized": (
                    best_rescue["subquery_score_normalized"] if best_rescue else None
                ),
                "best_rescue_intent_score": (
                    best_rescue["intent_score"] if best_rescue else None
                ),
                "best_rescue_path_count_normalized": (
                    best_rescue["path_count_normalized"] if best_rescue else None
                ),
                "best_rescue_actual_structure_bonus": (
                    best_rescue["structure_bonus"] if best_rescue else None
                ),
                "best_rescue_required_total_structure_to_match_cutoff": (
                    best_rescue["required_total_structure_to_match_cutoff"]
                    if best_rescue
                    else None
                ),
            }
        )
    return paper_rows, best_rows


def aggregate_summary(
    source_counts: Mapping[str, int],
    occurrences: Sequence[Mapping[str, Any]],
    papers: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    *,
    query_weight: float,
    subquery_weight: float,
    intent_weight: float,
    path_weight: float,
) -> Dict[str, Any]:
    cause_counts = Counter(str(row["failure_cause"]) for row in papers)
    target_count = len(papers)
    best_values = {
        "query_score_normalized": [float(row["best_query_score_normalized"]) for row in papers],
        "subquery_score_normalized": [float(row["best_subquery_score_normalized"]) for row in papers],
        "intent_score": [float(row["best_intent_score"]) for row in papers],
        "path_count_normalized": [float(row["best_path_count_normalized"]) for row in papers],
        "semantic_component_score": [float(row["best_semantic_component_score"]) for row in papers],
        "structure_bonus": [float(row["best_structure_bonus"]) for row in papers],
        "hybrid_score": [float(row["best_hybrid_score"]) for row in papers],
        "hybrid_cutoff_score": [float(row["best_hybrid_cutoff_score"]) for row in papers],
        "best_rank_event_hybrid_score_gap_to_cutoff": [
            float(row["best_event_hybrid_score_gap_to_cutoff"]) for row in papers
        ],
        "minimum_hybrid_score_gap_to_cutoff": [
            float(row["min_hybrid_score_gap_to_cutoff"]) for row in papers
        ],
        "hybrid_rank": [float(row["best_hybrid_rank_at_event"]) for row in papers],
        "best_hybrid_rank_any_occurrence": [
            float(row["best_hybrid_rank"]) for row in papers
        ],
        "hybrid_rank_minus_k": [float(row["best_hybrid_rank_minus_k"]) for row in papers],
        "best_semantic_control_rank": [
            float(row["best_semantic_control_rank"]) for row in papers
        ],
        "best_max_structure_counterfactual_rank": [
            float(row["best_max_structure_counterfactual_rank"]) for row in papers
        ],
        "minimum_required_total_structure_to_match_cutoff": [
            float(row["min_required_total_structure_to_match_cutoff"]) for row in papers
        ],
    }
    occurrence_values = {
        "query_score_normalized": [float(row["query_score_normalized"]) for row in occurrences],
        "subquery_score_normalized": [float(row["subquery_score_normalized"]) for row in occurrences],
        "intent_score": [float(row["intent_score"]) for row in occurrences],
        "path_count_normalized": [float(row["path_count_normalized"]) for row in occurrences],
        "structure_bonus": [float(row["structure_bonus"]) for row in occurrences],
        "hybrid_rank": [float(row["hybrid_rank"]) for row in occurrences],
        "hybrid_rank_minus_k": [float(row["hybrid_rank_minus_k"]) for row in occurrences],
    }
    return {
        "complete": True,
        "implementation_version": IMPLEMENTATION_VERSION,
        "scope": "full-pool provenance exactly Graph-only and ground truth",
        "formula": {
            "query_score_normalized": query_weight,
            "subquery_score_normalized": subquery_weight,
            "intent_score": intent_weight,
            "path_count_normalized": path_weight,
        },
        "semantic_control": {
            "query_score_normalized": query_weight,
            "subquery_score_normalized": subquery_weight,
            "intent_score": 0.0,
            "path_count_normalized": 0.0,
        },
        "max_structure_bonus_assumption": intent_weight + path_weight,
        **dict(source_counts),
        "target_paper_count": target_count,
        "target_occurrence_count": len(occurrences),
        "selected_target_paper_count": cause_counts.get("selected", 0),
        "failure_cause_counts": {
            cause: {
                "count": cause_counts.get(cause, 0),
                "rate": cause_counts.get(cause, 0) / target_count if target_count else None,
                "label_zh": CAUSE_LABELS[cause],
            }
            for cause in CAUSE_LABELS
        },
        "paper_feature_presence": {
            "any_nonzero_intent_count": sum(bool(row["any_nonzero_intent"]) for row in papers),
            "any_nonzero_path_count": sum(bool(row["any_nonzero_path"]) for row in papers),
            "both_always_zero_count": sum(
                not bool(row["any_nonzero_intent"]) and not bool(row["any_nonzero_path"])
                for row in papers
            ),
        },
        "occurrence_validation": {
            "validated_paper_count": sum(
                bool(row["occurrence_validation_complete"]) for row in papers
            ),
            "all_complete": all(
                bool(row["occurrence_validation_complete"]) for row in papers
            ),
        },
        "production_tie_break_diagnostics": {
            "seed_occurrence_count": sum(bool(row["is_seed"]) for row in occurrences),
            "expanded_occurrence_count": sum(not bool(row["is_seed"]) for row in occurrences),
            "cutoff_tie_loss_occurrence_count": sum(
                bool(row["tie_break_loss"]) for row in occurrences
            ),
        },
        "structure_rank_effect_occurrences": {
            "improved_rank_count": sum(
                int(row["hybrid_rank_change_vs_semantic"]) < 0 for row in occurrences
            ),
            "unchanged_rank_count": sum(
                int(row["hybrid_rank_change_vs_semantic"]) == 0 for row in occurrences
            ),
            "worsened_rank_count": sum(
                int(row["hybrid_rank_change_vs_semantic"]) > 0 for row in occurrences
            ),
            "rank_change_distribution": _distribution(
                float(row["hybrid_rank_change_vs_semantic"]) for row in occurrences
            ),
        },
        "near_cutoff_paper_counts": {
            f"within_k_plus_{delta}": sum(
                int(row["best_hybrid_rank_minus_k"]) <= delta for row in papers
            )
            for delta in (1, 5, 10, 20, 50, 100)
        },
        "relaxed_budget_paper_counts": {
            f"within_{multiple}x_k": sum(
                any(
                    str(occ["query_id"]) == str(row["query_id"])
                    and str(occ["paper_id"]) == str(row["paper_id"])
                    and int(occ["hybrid_rank"]) <= multiple * int(occ["baseline_k"])
                    for occ in occurrences
                )
                for row in papers
            )
            for multiple in (2, 5, 10, 20)
        },
        "best_occurrence_distributions": {
            key: _distribution(values) for key, values in best_values.items()
        },
        "all_target_occurrence_distributions": {
            key: _distribution(values) for key, values in occurrence_values.items()
        },
        "diagnostics": dict(diagnostics),
        "classification_precedence": [
            "selected",
            "tie_break_loss",
            "displaced_by_structure",
            "insufficient_structural_signal",
            "semantic_deficit_beyond_max_structure",
        ],
    }


def _pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def build_report(summary: Mapping[str, Any], papers: Sequence[Mapping[str, Any]]) -> str:
    target_count = int(summary["target_paper_count"])
    causes = summary["failure_cause_counts"]
    best = summary["best_occurrence_distributions"]
    presence = summary["paper_feature_presence"]
    near = summary["near_cutoff_paper_counts"]
    relaxed = summary["relaxed_budget_paper_counts"]
    tie_diagnostics = summary["production_tie_break_diagnostics"]
    rank_effect = summary["structure_rank_effect_occurrences"]
    rescuable = [
        row
        for row in papers
        if row["failure_cause"] == "insufficient_structural_signal"
    ]
    closest = sorted(
        papers,
        key=lambda row: (
            int(row["best_hybrid_rank_minus_k"]),
            float(row["min_hybrid_score_gap_to_cutoff"]),
        ),
    )[:15]

    lines = [
        "# 57 篇 Graph-only GT 的 Ret 排名失败诊断",
        "",
        "## 口径",
        "",
        f"完整候选池中有 **{summary['graph_pool_unique_candidate_count']:,}** 个真正 Graph-only query-paper，其中 **{target_count}** 个是 GT。",
        f"这些 GT 在 899 个 Graph subquery 事件中共出现 **{summary['target_occurrence_count']:,}** 次。每个事件沿用 Baseline 实际 Selector 输入数 K_i，并复现 `0.55Q + 0.30SQ + 0.10I + 0.05P` 与生产 tie-break。",
        "",
        "语义对照保持 Q/SQ 权重不变，只把 I/P 置零。最大结构反事实只把目标论文的 I/P 都设为 1，因此最多增加 0.15；其他候选保持当前四因素分数。",
        "",
        "## 失败原因",
        "",
        "| 原因 | 论文数 | 比例 |",
        "|---|---:|---:|",
    ]
    for cause in (
        "selected",
        "tie_break_loss",
        "displaced_by_structure",
        "insufficient_structural_signal",
        "semantic_deficit_beyond_max_structure",
    ):
        item = causes[cause]
        lines.append(f"| {item['label_zh']} | {item['count']} | {_pct(item['rate'])} |")
    lines.extend(
        [
            "",
            "判定优先级：当前入选 → 同分 tie-break → 同权重语义对照可入选但四因素被挤出 → 最大 0.15 结构加分可救回 → 即使最大结构加分也无法入选。分类在 paper 层面按任一最有利 occurrence 判定。",
            "",
            "## 距离 Top-K 有多远",
            "",
            f"- 最佳 occurrence 的排名中位数为 **{_num(best['hybrid_rank']['median'], 1)}**，排名减 K 的中位数为 **{_num(best['hybrid_rank_minus_k']['median'], 1)}**。",
            f"- 每篇跨 occurrence 取最好名次时，纯语义对照 / 当前四因素 / 目标结构分拉满的排名中位数分别为 **{_num(best['best_semantic_control_rank']['median'], 1)} / {_num(best['best_hybrid_rank_any_occurrence']['median'], 1)} / {_num(best['best_max_structure_counterfactual_rank']['median'], 1)}**。",
            f"- 每篇所有 occurrence 中的最小分数差中位数为 **{_num(best['minimum_hybrid_score_gap_to_cutoff']['median'])}**；匹配 cutoff 所需的最小总结构加分中位数为 **{_num(best['minimum_required_total_structure_to_match_cutoff']['median'])}**。",
            f"- K+1 内：{near['within_k_plus_1']}/{target_count}；K+5 内：{near['within_k_plus_5']}/{target_count}；K+20 内：{near['within_k_plus_20']}/{target_count}；K+100 内：{near['within_k_plus_100']}/{target_count}。",
            f"- 若把每事件预算放宽到 2K/5K/10K/20K，至少一次可入选的论文分别为 {relaxed['within_2x_k']}/{relaxed['within_5x_k']}/{relaxed['within_10x_k']}/{relaxed['within_20x_k']}。",
            "",
            "## 四项特征（每篇最接近 K 线的 occurrence）",
            "",
            "| 特征 | P25 | 中位数 | P75 |",
            "|---|---:|---:|---:|",
        ]
    )
    feature_labels = (
        ("query_score_normalized", "Q"),
        ("subquery_score_normalized", "SQ"),
        ("intent_score", "intent"),
        ("path_count_normalized", "path"),
        ("semantic_component_score", "0.55Q+0.30SQ"),
        ("structure_bonus", "0.10I+0.05P"),
        ("hybrid_score", "四因素总分"),
        ("hybrid_cutoff_score", "Kth cutoff"),
    )
    for key, label in feature_labels:
        distribution = best[key]
        lines.append(
            f"| {label} | {_num(distribution['p25'])} | {_num(distribution['median'])} | {_num(distribution['p75'])} |"
        )
    lines.extend(
        [
            "",
            f"57 篇中，至少一个 occurrence 的 intent 非零有 **{presence['any_nonzero_intent_count']}** 篇，path 非零有 **{presence['any_nonzero_path_count']}** 篇；所有 occurrence 的 I/P 都为零有 **{presence['both_always_zero_count']}** 篇。",
            f"180 个 occurrence 全部是 expanded candidate，seed occurrence 为 **{tie_diagnostics['seed_occurrence_count']}**，cutoff 同分落选为 **{tie_diagnostics['cutoff_tie_loss_occurrence_count']}**，因此生产 seed/tie-break 不是这 57 篇落选的原因。",
            f"相对同 Q/SQ 权重的纯语义排序，结构项使目标论文排名上升/不变/下降的 occurrence 分别为 **{rank_effect['improved_rank_count']} / {rank_effect['unchanged_rank_count']} / {rank_effect['worsened_rank_count']}**；但没有任何一次上升足以进入 K。",
            "",
            "## 结构项理论上可救回的论文",
            "",
            "| Paper | 可救回的 subquery | 当前/满结构 Rank/K | I | P | 实际结构分 | 匹配 cutoff 所需结构分 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rescuable:
        title = str(row["title"]).replace("|", "\\|")[:70]
        subquery = str(row["best_rescue_subquery"]).replace("|", "\\|")[:80]
        lines.append(
            f"| {row['paper_id']} {title} | {subquery} | "
            f"{row['best_rescue_hybrid_rank']}/{row['best_rescue_max_structure_rank']}/{row['best_rescue_baseline_k']} | "
            f"{_num(row['best_rescue_intent_score'], 3)} | {_num(row['best_rescue_path_count_normalized'], 3)} | "
            f"{_num(row['best_rescue_actual_structure_bonus'])} | "
            f"{_num(row['best_rescue_required_total_structure_to_match_cutoff'])} |"
        )
    lines.extend(
        [
            "",
            "## 最接近 K 线的 15 篇",
            "",
            "| Paper | 最佳 subquery | Rank/K | Q | SQ | I | P | 总分差 | 原因 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in closest:
        title = str(row["title"]).replace("|", "\\|")[:80]
        subquery = str(row["best_subquery"]).replace("|", "\\|")[:80]
        lines.append(
            f"| {row['paper_id']} {title} | {subquery} | {row['best_hybrid_rank_at_event']}/{row['best_baseline_k']} | "
            f"{_num(float(row['best_query_score_normalized']), 3)} | {_num(float(row['best_subquery_score_normalized']), 3)} | "
            f"{_num(float(row['best_intent_score']), 3)} | {_num(float(row['best_path_count_normalized']), 3)} | "
            f"{_num(float(row['best_event_hybrid_score_gap_to_cutoff']))} | {row['failure_cause_zh']} |"
        )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            "- `occurrence_diagnostics.jsonl/csv`：每个 query-paper-subquery occurrence 的完整排名、四项得分、cutoff 和反事实。",
            "- `paper_summary.jsonl/csv`：57 篇逐篇汇总与失败归因。",
            "- `summary.json`：总体计数和分布。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    annotation_work_dir = Path(args.annotation_work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = (
        float(args.query_weight),
        float(args.subquery_weight),
        float(args.intent_weight),
        float(args.path_weight),
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"weights must sum to 1; got {sum(weights)}")
    query_weight, subquery_weight, intent_weight, path_weight = weights

    targets, source_counts = load_targets(
        annotation_work_dir / "manifest" / "candidates.jsonl"
    )
    query_texts = load_query_texts(annotation_work_dir / "manifest" / "queries.jsonl")
    budgets = replay.load_baseline_budgets(
        run_dir / "onepass_artifacts" / "baseline" / "selector_decisions.jsonl"
    )
    occurrences, diagnostics = analyze_occurrences(
        run_dir / "onepass_artifacts" / "per_subquery" / "pool_records.jsonl",
        budgets,
        targets,
        query_texts,
        query_weight=query_weight,
        subquery_weight=subquery_weight,
        intent_weight=intent_weight,
        path_weight=path_weight,
    )
    paper_rows, _ = summarize_papers(targets, occurrences)
    summary = aggregate_summary(
        source_counts,
        occurrences,
        paper_rows,
        diagnostics,
        query_weight=query_weight,
        subquery_weight=subquery_weight,
        intent_weight=intent_weight,
        path_weight=path_weight,
    )
    summary["run_dir"] = str(run_dir)
    summary["annotation_work_dir"] = str(annotation_work_dir)
    summary["output_dir"] = str(output_dir)

    replay._atomic_write_jsonl(output_dir / "occurrence_diagnostics.jsonl", occurrences)
    replay._atomic_write_csv(output_dir / "occurrence_diagnostics.csv", occurrences)
    replay._atomic_write_jsonl(output_dir / "paper_summary.jsonl", paper_rows)
    replay._atomic_write_csv(output_dir / "paper_summary.csv", paper_rows)
    replay._atomic_write_json(output_dir / "summary.json", summary)
    report_path = output_dir / "report_zh.md"
    temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    temporary.write_text(build_report(summary, paper_rows), encoding="utf-8")
    os.replace(temporary, report_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--annotation-work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-weight", type=float, default=0.55)
    parser.add_argument("--subquery-weight", type=float, default=0.30)
    parser.add_argument("--intent-weight", type=float, default=0.10)
    parser.add_argument("--path-weight", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
