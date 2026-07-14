#!/usr/bin/env python3
"""Replay two BM25 deep-retrieval shadows on a completed legacy OnePass run.

Equivalent controls are now integrated into ``code/eval.py``; this standalone
script remains useful for replaying already completed full-artifact runs.
It freezes the committed baseline trajectory, uses the per-retrieval-event graph
pool sizes only as retrieval budgets, and never calls Planner or Semantic
Scholar.  The two independently selectable arms are:

* event-offset matched: one deep pool and one rerank per baseline retrieval
  event, using that event's exclusion snapshot and saved offset;
* merged-subquery sum budget: one deep pool and one rerank per stable subquery
  id, followed by chronological, disjoint top-k Selector slices.

Paper text is used in memory for BM25 and Selector calls but is never emitted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import config  # noqa: E402
import api as llm_api  # noqa: E402
from agent.selector import Selector  # noqa: E402
from graph_methods import CandidateIndex, normalize_arxiv_id  # noqa: E402
from structures import Paper, SubQuery  # noqa: E402


EVENT_METHOD = "event_offset_matched_text_deep_retrieval"
MERGED_METHOD = "merged_subquery_sum_budget_text_deep_retrieval"
METHODS = (EVENT_METHOD, MERGED_METHOD)
SCHEMA_VERSION = "1.0-experimental"
IMPLEMENTATION_VERSION = "1.2"
TEXT_ONLY_WEIGHTS = {
    "query_score_normalized": 0.30,
    "subquery_score_normalized": 0.40,
    "intent_score": 0.15,
    "path_count_normalized": 0.15,
}


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(recall: float, precision: float) -> float:
    return 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ordered_unique(values: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: Set[str] = set()
    for value in values:
        paper_id = normalize_arxiv_id(value)
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            output.append(paper_id)
    return output


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_detailed_results(path: Path) -> Dict[int, Dict[str, Any]]:
    """Use the last valid detailed-result record for each benchmark index."""
    results: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid detailed JSONL at line {line_number}: {exc}") from exc
            idx = row.get("idx")
            if isinstance(idx, int) and idx >= 0:
                results[idx] = row
    return results


def iter_jsonl_groups(path: Path, allowed_indices: Set[int]) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
    """Yield contiguous benchmark-index groups without loading a full artifact."""
    current_idx: Optional[int] = None
    current_rows: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid artifact {path} at line {line_number}: {exc}") from exc
            idx = row.get("benchmark_idx")
            if not isinstance(idx, int) or idx not in allowed_indices:
                continue
            if current_idx is None:
                current_idx = idx
            if idx != current_idx:
                if idx in seen:
                    raise ValueError(f"artifact {path} contains non-contiguous duplicate idx={idx}")
                seen.add(current_idx)
                yield current_idx, current_rows
                current_idx = idx
                current_rows = []
            current_rows.append(row)
    if current_idx is not None:
        if current_idx in seen:
            raise ValueError(f"artifact {path} contains non-contiguous duplicate idx={current_idx}")
        yield current_idx, current_rows


class GroupCursor:
    def __init__(self, path: Path, allowed_indices: Set[int], order: Mapping[int, int]) -> None:
        self.path = path
        self._order = order
        self._iterator = iter_jsonl_groups(path, allowed_indices)
        self._current: Optional[Tuple[int, List[Dict[str, Any]]]] = None

    def _ensure_current(self) -> None:
        if self._current is None:
            self._current = next(self._iterator, None)

    def take(self, idx: int) -> List[Dict[str, Any]]:
        self._ensure_current()
        target_position = self._order[idx]
        while self._current is not None and self._order.get(self._current[0], 10**12) < target_position:
            self._current = next(self._iterator, None)
        if self._current is None or self._current[0] != idx:
            return []
        rows = self._current[1]
        self._current = None
        return rows


def _planner_trajectories(rows: Sequence[Mapping[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split retries whenever Planner iteration numbering restarts or repeats."""
    trajectories: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    previous_iteration = -1
    for row in rows:
        iteration = _as_int(row.get("iteration_idx"), -1)
        if iteration < 0:
            continue
        if current and iteration <= previous_iteration:
            trajectories.append(current)
            current = []
        current.append(dict(row))
        previous_iteration = iteration
    if current:
        trajectories.append(current)
    return trajectories


def canonical_planner_events(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select the last contiguous Planner trajectory without mixing retries."""
    trajectories = _planner_trajectories(rows)
    canonical = trajectories[-1] if trajectories else []
    iteration_counts: Dict[int, int] = defaultdict(int)
    for row in rows:
        iteration = _as_int(row.get("iteration_idx"), -1)
        if iteration < 0:
            continue
        iteration_counts[iteration] += 1
    return canonical, {
        "raw_planner_row_count": len(rows),
        "canonical_planner_row_count": len(canonical),
        "duplicate_planner_row_count": max(0, len(rows) - len(canonical)),
        "planner_trajectory_count": len(trajectories),
        "selected_planner_trajectory_index": len(trajectories) if trajectories else None,
        "planner_rows_per_iteration": {str(key): value for key, value in sorted(iteration_counts.items())},
        "selection_rule": "last contiguous Planner trajectory",
    }


def planner_expectations(
    planner_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[Tuple[int, str], Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    expected: Dict[Tuple[int, str], Dict[str, Any]] = {}
    planner_by_iteration: Dict[int, Dict[str, Any]] = {}
    for planner in planner_rows:
        iteration = _as_int(planner.get("iteration_idx"), -1)
        if iteration < 0:
            continue
        planner_by_iteration[iteration] = dict(planner)
        for subquery_order, subquery in enumerate(planner.get("subqueries") or [], start=1):
            subquery_id = str(subquery.get("subquery_id"))
            expected[(iteration, subquery_id)] = {
                **dict(subquery),
                "planner_checklist": str(planner.get("planner_checklist") or ""),
                "planner_subquery_order": subquery_order,
            }
    return expected, planner_by_iteration


def _matches_final_planner_row(
    row: Mapping[str, Any],
    expected: Mapping[Tuple[int, str], Mapping[str, Any]],
) -> bool:
    key = (_as_int(row.get("iteration_idx"), -1), str(row.get("subquery_id")))
    target = expected.get(key)
    if target is None:
        return False
    if str(row.get("planner_checklist") or "") != str(target.get("planner_checklist") or ""):
        return False
    if str(row.get("subquery") or "") != str(target.get("subquery") or ""):
        return False
    row_target = row.get("subquery_target_k")
    target_k = target.get("target_k")
    if row_target is not None and target_k is not None and _as_int(row_target, -1) != _as_int(target_k, -2):
        return False
    return True


def _dedupe_paper_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_paper: Dict[str, Dict[str, Any]] = {}
    last_position: Dict[str, int] = {}
    for position, row in enumerate(rows):
        paper_id = normalize_arxiv_id(row.get("paper_arxiv_id"))
        if not paper_id:
            continue
        value = dict(row)
        value["paper_arxiv_id"] = paper_id
        by_paper[paper_id] = value
        last_position[paper_id] = position
    return [by_paper[paper_id] for paper_id in sorted(by_paper, key=lambda value: last_position[value])]


def canonical_event_blocks(
    rows: Sequence[Mapping[str, Any]],
    expected: Mapping[Tuple[int, str], Mapping[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, int], Dict[str, Any]]:
    """Keep the last matching contiguous block for each retrieval event."""
    blocks: Dict[str, List[Dict[str, Any]]] = {}
    block_order: Dict[str, int] = {}
    matching_count = 0
    replaced = 0
    current_event = ""
    current_rows: List[Mapping[str, Any]] = []
    block_index = 0

    def flush() -> None:
        nonlocal replaced, block_index, current_event, current_rows
        if not current_event or not current_rows:
            return
        if current_event in blocks:
            replaced += 1
        blocks[current_event] = _dedupe_paper_rows(current_rows)
        block_order[current_event] = block_index
        block_index += 1

    for row in rows:
        if not _matches_final_planner_row(row, expected):
            flush()
            current_event = ""
            current_rows = []
            continue
        event_id = str(row.get("retrieval_event_id") or "")
        if not event_id:
            flush()
            current_event = ""
            current_rows = []
            continue
        matching_count += 1
        if current_event and event_id != current_event:
            flush()
            current_rows = []
        current_event = event_id
        current_rows.append(row)
    flush()
    return blocks, block_order, {
        "raw_row_count": len(rows),
        "matching_final_planner_row_count": matching_count,
        "canonical_event_block_count": len(blocks),
        "replaced_earlier_event_block_count": replaced,
        "canonical_unique_paper_row_count": sum(len(value) for value in blocks.values()),
        "selection_rule": "final Planner metadata, then last contiguous block per event, then last row per arXiv ID",
    }


def _ranked_paper_ids(rows: Sequence[Mapping[str, Any]], rank_keys: Sequence[str]) -> List[str]:
    def key(row: Mapping[str, Any]) -> Tuple[int, str]:
        rank = 10**12
        for name in rank_keys:
            value = row.get(name)
            if value is not None:
                rank = _as_int(value, 10**12)
                break
        return rank, normalize_arxiv_id(row.get("paper_arxiv_id"))

    return _ordered_unique(normalize_arxiv_id(row.get("paper_arxiv_id")) for row in sorted(rows, key=key))


def build_source_context(
    detail: Mapping[str, Any],
    planner_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    local_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    trajectories = _planner_trajectories(planner_rows)
    if not trajectories:
        raise ValueError(f"idx={detail.get('idx')} has no committed Planner events")
    baseline_summary = (detail.get("postprocess_results") or {}).get("baseline") or {}
    has_expected_candidates = "candidate_arxiv_ids" in baseline_summary
    has_expected_selected = "selected_arxiv_ids" in baseline_summary
    expected_candidate_ids = {
        normalize_arxiv_id(value)
        for value in baseline_summary.get("candidate_arxiv_ids") or []
        if normalize_arxiv_id(value)
    }
    expected_selected_ids = {
        normalize_arxiv_id(value)
        for value in baseline_summary.get("selected_arxiv_ids") or []
        if normalize_arxiv_id(value)
    }
    per_subquery_summary = (detail.get("postprocess_results") or {}).get("per_subquery") or {}
    has_expected_local_candidates = "candidate_arxiv_ids" in per_subquery_summary
    has_expected_local_selected = "selected_arxiv_ids" in per_subquery_summary
    expected_local_candidate_ids = {
        normalize_arxiv_id(value)
        for value in per_subquery_summary.get("candidate_arxiv_ids") or []
        if normalize_arxiv_id(value)
    }
    expected_local_selected_ids = {
        normalize_arxiv_id(value)
        for value in per_subquery_summary.get("selected_arxiv_ids") or []
        if normalize_arxiv_id(value)
    }
    chosen: Optional[Tuple[Any, ...]] = None
    trajectory_attempts: List[Dict[str, Any]] = []
    for trajectory_index in range(len(trajectories) - 1, -1, -1):
        candidate_planners = trajectories[trajectory_index]
        candidate_expected, candidate_planner_by_iteration = planner_expectations(candidate_planners)
        candidate_baseline_blocks, candidate_baseline_order, candidate_baseline_stats = canonical_event_blocks(
            baseline_rows, candidate_expected
        )
        candidate_local_blocks, _, candidate_local_stats = canonical_event_blocks(local_rows, candidate_expected)
        retrieved_ids = {
            normalize_arxiv_id(row.get("paper_arxiv_id"))
            for rows in candidate_baseline_blocks.values()
            for row in rows
            if normalize_arxiv_id(row.get("paper_arxiv_id"))
        }
        selected_ids = {
            normalize_arxiv_id(row.get("paper_arxiv_id"))
            for rows in candidate_baseline_blocks.values()
            for row in rows
            if row.get("selector_selected") and normalize_arxiv_id(row.get("paper_arxiv_id"))
        }
        candidate_match = not has_expected_candidates or retrieved_ids == expected_candidate_ids
        selected_match = not has_expected_selected or selected_ids == expected_selected_ids
        local_candidate_ids = {
            normalize_arxiv_id(row.get("paper_arxiv_id"))
            for rows in candidate_local_blocks.values()
            for row in rows
            if row.get("in_selector_topk") and normalize_arxiv_id(row.get("paper_arxiv_id"))
        }
        local_selected_ids = {
            normalize_arxiv_id(row.get("paper_arxiv_id"))
            for rows in candidate_local_blocks.values()
            for row in rows
            if row.get("selector_selected") and normalize_arxiv_id(row.get("paper_arxiv_id"))
        }
        local_candidate_match = (
            not has_expected_local_candidates or local_candidate_ids == expected_local_candidate_ids
        )
        local_selected_match = (
            not has_expected_local_selected or local_selected_ids == expected_local_selected_ids
        )
        local_complete = bool(candidate_baseline_blocks) and all(
            event_id in candidate_local_blocks and bool(candidate_local_blocks[event_id])
            for event_id in candidate_baseline_blocks
        )
        trajectory_attempts.append(
            {
                "trajectory_index": trajectory_index + 1,
                "planner_row_count": len(candidate_planners),
                "baseline_event_count": len(candidate_baseline_blocks),
                "retrieved_unique_count": len(retrieved_ids),
                "selected_unique_count": len(selected_ids),
                "matches_detailed_baseline_candidates": candidate_match,
                "matches_detailed_baseline_selected": selected_match,
                "matches_detailed_per_subquery_candidates": local_candidate_match,
                "matches_detailed_per_subquery_selected": local_selected_match,
                "has_local_pool_for_every_event": local_complete,
            }
        )
        if (
            candidate_match
            and selected_match
            and local_candidate_match
            and local_selected_match
            and local_complete
        ):
            chosen = (
                trajectory_index,
                candidate_planners,
                candidate_expected,
                candidate_planner_by_iteration,
                candidate_baseline_blocks,
                candidate_baseline_order,
                candidate_baseline_stats,
                candidate_local_blocks,
                candidate_local_stats,
            )
            break
    if chosen is None:
        raise ValueError(
            f"idx={detail.get('idx')} has no artifact trajectory matching detailed baseline commit: "
            f"{trajectory_attempts}"
        )
    (
        chosen_trajectory_index,
        planners,
        expected,
        planner_by_iteration,
        baseline_blocks,
        baseline_order,
        baseline_stats,
        local_blocks,
        local_stats,
    ) = chosen
    _, planner_stats = canonical_planner_events(planner_rows)
    planner_stats.update(
        {
            "canonical_planner_row_count": len(planners),
            "duplicate_planner_row_count": max(0, len(planner_rows) - len(planners)),
            "selected_planner_trajectory_index": chosen_trajectory_index + 1,
            "selection_rule": "newest contiguous trajectory matching detailed baseline and per-subquery candidate/selected IDs with complete local pools",
            "trajectory_match_attempts_newest_first": trajectory_attempts,
        }
    )
    if not baseline_blocks:
        raise ValueError(f"idx={detail.get('idx')} has no canonical baseline retrieval events")

    events: List[Dict[str, Any]] = []
    for event_id in sorted(baseline_blocks, key=lambda value: baseline_order[value]):
        rows = baseline_blocks[event_id]
        metadata = rows[-1]
        iteration = _as_int(metadata.get("iteration_idx"), -1)
        subquery_id = str(metadata.get("subquery_id"))
        target = expected.get((iteration, subquery_id)) or {}
        planner = planner_by_iteration.get(iteration) or {}
        local_event_rows = [
            row
            for row in (local_blocks.get(event_id) or [])
            if row.get("passed_date_cutoff") is not False
        ]
        if not local_event_rows:
            raise ValueError(f"idx={detail.get('idx')} event={event_id} has no canonical per-subquery pool rows")
        local_ids = _ranked_paper_ids(local_event_rows, ("rerank_rank", "paper_arxiv_id"))
        if not local_ids:
            raise ValueError(f"idx={detail.get('idx')} event={event_id} has an empty local graph pool")
        exclusion = _ordered_unique(
            (planner.get("planner_input_state") or {}).get("retrieval_exclusion_arxiv_ids") or []
        )
        events.append(
            {
                "event_order": len(events) + 1,
                "source_artifact_block_order": baseline_order[event_id] + 1,
                "planner_subquery_order": _as_int(target.get("planner_subquery_order"), 10**9),
                "retrieval_event_id": event_id,
                "query_id": str(metadata.get("query_id") or ""),
                "benchmark_idx": _as_int(detail.get("idx"), -1),
                "iteration_idx": iteration,
                "retrieval_page_idx": _as_int(metadata.get("retrieval_page_idx"), 1),
                "retrieval_offset": _as_int(metadata.get("retrieval_offset"), 0),
                "subquery_id": subquery_id,
                "subquery": str(target.get("subquery") or metadata.get("subquery") or ""),
                "subquery_target_k": _as_int(target.get("target_k") or metadata.get("subquery_target_k"), 0),
                "subquery_link_type": target.get("link_type") or metadata.get("subquery_link_type"),
                "parent_subquery_id": target.get("parent_subquery_id") or metadata.get("parent_subquery_id"),
                "subquery_before_date": target.get("subquery_before_date") or metadata.get("subquery_before_date") or metadata.get("query_date"),
                "planner_checklist": str(planner.get("planner_checklist") or metadata.get("planner_checklist") or ""),
                "selector_top_k": _as_int(metadata.get("selector_top_k"), len(rows)),
                "baseline_seed_arxiv_ids": _ranked_paper_ids(
                    rows,
                    ("observed_retrieval_rank", "observed_retrieval_absolute_rank", "paper_arxiv_id"),
                ),
                "retrieval_exclusion_arxiv_ids": exclusion,
                "source_local_graph_pool_size": len(local_ids),
                "source_local_graph_pool_arxiv_ids": local_ids,
            }
        )

    events.sort(
        key=lambda event: (
            event["iteration_idx"],
            event["planner_subquery_order"],
            event["retrieval_page_idx"],
            event["source_artifact_block_order"],
        )
    )
    for event_order, event in enumerate(events, start=1):
        event["event_order"] = event_order

    groups_by_subquery: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups_by_subquery[event["subquery_id"]].append(event)
    groups: List[Dict[str, Any]] = []
    for subquery_id, group_events in groups_by_subquery.items():
        group_events.sort(
            key=lambda event: (
                event["iteration_idx"],
                event["retrieval_page_idx"],
                event["event_order"],
            )
        )
        texts = {event["subquery"] for event in group_events}
        cutoffs = {str(event.get("subquery_before_date") or "")[:7] for event in group_events}
        if len(texts) != 1 or len(cutoffs) != 1:
            raise ValueError(
                f"idx={detail.get('idx')} subquery_id={subquery_id} changed text/date across continue events"
            )
        occurrence_budget = sum(event["source_local_graph_pool_size"] for event in group_events)
        local_union = _ordered_unique(
            paper_id
            for event in group_events
            for paper_id in event["source_local_graph_pool_arxiv_ids"]
        )
        baseline_seed_union = _ordered_unique(
            paper_id
            for event in group_events
            for paper_id in event["baseline_seed_arxiv_ids"]
        )
        groups.append(
            {
                "group_order": len(groups) + 1,
                "subquery_id": subquery_id,
                "subquery": group_events[0]["subquery"],
                "subquery_before_date": group_events[0]["subquery_before_date"],
                "events": group_events,
                "source_event_ids": [event["retrieval_event_id"] for event in group_events],
                "source_event_count": len(group_events),
                "source_event_budgets": [
                    {
                        "retrieval_event_id": event["retrieval_event_id"],
                        "iteration_idx": event["iteration_idx"],
                        "retrieval_page_idx": event["retrieval_page_idx"],
                        "retrieval_offset": event["retrieval_offset"],
                        "selector_top_k": event["selector_top_k"],
                        "source_local_graph_pool_size": event["source_local_graph_pool_size"],
                        "source_local_graph_pool_arxiv_ids": event["source_local_graph_pool_arxiv_ids"],
                        "baseline_seed_arxiv_ids": event["baseline_seed_arxiv_ids"],
                        "retrieval_exclusion_count": len(event["retrieval_exclusion_arxiv_ids"]),
                        "retrieval_exclusion_arxiv_ids": event["retrieval_exclusion_arxiv_ids"],
                        "planner_checklist": event["planner_checklist"],
                    }
                    for event in group_events
                ],
                "source_local_graph_pool_occurrence_budget": occurrence_budget,
                "source_local_graph_pool_union_count": len(local_union),
                "source_local_graph_pool_overlap_occurrence_count": occurrence_budget - len(local_union),
                "source_local_graph_pool_union_over_sum": _safe_div(len(local_union), occurrence_budget),
                "source_local_graph_pool_union_arxiv_ids": local_union,
                "source_baseline_seed_occurrence_count": sum(
                    len(event["baseline_seed_arxiv_ids"]) for event in group_events
                ),
                "source_baseline_seed_union_count": len(baseline_seed_union),
                "source_baseline_seed_union_arxiv_ids": baseline_seed_union,
                "frozen_first_event_exclusion_arxiv_ids": list(group_events[0]["retrieval_exclusion_arxiv_ids"]),
            }
        )

    query_id = events[0]["query_id"] or str(
        (detail.get("postprocess_results") or {}).get("query_id") or f"idx-{detail.get('idx')}"
    )
    return {
        "benchmark_idx": _as_int(detail.get("idx"), -1),
        "query_id": query_id,
        "query": str(detail.get("query") or ""),
        "gt_ids": {
            normalize_arxiv_id(value)
            for value in detail.get("ground_truth_arxiv_ids") or []
            if normalize_arxiv_id(value)
        },
        "events": events,
        "groups": groups,
        "canonicalization": {
            "planner": planner_stats,
            "baseline_paper_rows": baseline_stats,
            "per_subquery_paper_rows": local_stats,
        },
    }


def retrieve_bm25_requests(
    rag_system: Any,
    *,
    subquery: str,
    before_date: Optional[str],
    requests: Mapping[Any, Mapping[str, Any]],
) -> Tuple[Dict[Any, List[Dict[str, Any]]], Dict[Any, Dict[str, Any]]]:
    """Score one subquery once and materialize several exclusion/offset slices."""
    outputs: Dict[Any, List[Dict[str, Any]]] = {key: [] for key in requests}
    states: Dict[Any, Dict[str, Any]] = {}
    for key, request in requests.items():
        states[key] = {
            "offset": max(0, _as_int(request.get("offset"), 0)),
            "count": max(0, _as_int(request.get("count"), 0)),
            "eligible_rank": 0,
            "completion_global_rank": None,
            "exclude": {
                normalize_arxiv_id(value)
                for value in request.get("exclude_arxiv_ids") or []
                if normalize_arxiv_id(value)
            },
        }
    def request_diagnostics(date_valid_count: int) -> Dict[Any, Dict[str, Any]]:
        return {
            key: {
                "requested_count": state["count"],
                "retrieval_offset": state["offset"],
                "retrieval_exclusion_count": len(state["exclude"]),
                "actual_count": len(outputs[key]),
                "budget_fulfillment_rate": _safe_div(len(outputs[key]), state["count"]),
                "eligible_rank_scanned_for_request": state["eligible_rank"],
                "completion_global_date_valid_unique_rank": state["completion_global_rank"],
                "date_valid_positive_document_count_before_arxiv_dedup": date_valid_count,
                "canonical_arxiv_deduplication_before_exclusion_offset": True,
            }
            for key, state in states.items()
        }

    if not states or all(state["count"] == 0 for state in states.values()):
        return outputs, request_diagnostics(0)

    tokens = rag_system._preprocess_text_for_bm25(subquery)
    if not tokens:
        return outputs, request_diagnostics(0)
    scores = rag_system.bm25_index.get_scores(tokens)
    positive_indices = np.where(scores > 0)[0]
    date_valid_indices: List[int] = []
    cutoff_month = str(before_date or "")[:7]
    for index in positive_indices:
        raw_paper_id = rag_system.bm25_index_to_id.get(int(index))
        if not raw_paper_id:
            continue
        metadata = rag_system.paper_metadata.get(raw_paper_id, {})
        paper_date = str(metadata.get("date") or "")
        if cutoff_month and (not paper_date or paper_date[:7] > cutoff_month):
            continue
        date_valid_indices.append(int(index))
    sorted_indices = sorted(date_valid_indices, key=lambda index: scores[index], reverse=True)

    seen: Set[str] = set()
    global_rank = 0
    for index in sorted_indices:
        raw_paper_id = rag_system.bm25_index_to_id.get(index)
        metadata = dict(rag_system.paper_metadata.get(raw_paper_id, {}) or {})
        paper_id = normalize_arxiv_id(metadata.get("arxiv_id") or raw_paper_id)
        if not paper_id:
            continue
        if paper_id in seen:
            continue
        seen.add(paper_id)
        global_rank += 1
        for key, state in states.items():
            if (
                state["count"] <= 0
                or state["eligible_rank"] >= state["offset"] + state["count"]
                or paper_id in state["exclude"]
            ):
                continue
            state["eligible_rank"] += 1
            eligible_rank = state["eligible_rank"]
            if state["offset"] < eligible_rank <= state["offset"] + state["count"]:
                metadata["arxiv_id"] = paper_id
                outputs[key].append(
                    {
                        "paper_arxiv_id": paper_id,
                        "deep_retrieval_score_raw": float(scores[index]),
                        "deep_retrieval_rank_global_date_valid": global_rank,
                        "deep_retrieval_rank_after_exclusion": eligible_rank,
                        "_metadata": metadata,
                    }
                )
            if eligible_rank >= state["offset"] + state["count"]:
                state["completion_global_rank"] = global_rank
        if all(
            state["count"] <= 0
            or state["eligible_rank"] >= state["offset"] + state["count"]
            for state in states.values()
        ):
            break
    return outputs, request_diagnostics(len(sorted_indices))


def text_only_formula_score(query_normalized: float, subquery_normalized: float) -> float:
    return (
        TEXT_ONLY_WEIGHTS["query_score_normalized"] * float(query_normalized)
        + TEXT_ONLY_WEIGHTS["subquery_score_normalized"] * float(subquery_normalized)
    )


def rerank_text_pool(
    pool: Sequence[Mapping[str, Any]],
    *,
    query: str,
    subquery: str,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], List[str]]:
    metadata = {
        normalize_arxiv_id(hit.get("paper_arxiv_id")): dict(hit.get("_metadata") or {})
        for hit in pool
        if normalize_arxiv_id(hit.get("paper_arxiv_id"))
    }
    retrieval = {
        normalize_arxiv_id(hit.get("paper_arxiv_id")): hit
        for hit in pool
        if normalize_arxiv_id(hit.get("paper_arxiv_id"))
    }
    candidate_ids = _ordered_unique(hit.get("paper_arxiv_id") for hit in pool)
    index = CandidateIndex(candidate_ids, metadata, "bm25")
    query_raw, query_normalized, query_rank = index.score(query)
    subquery_raw, subquery_normalized, subquery_rank = index.score(subquery)
    scorable_ids = [paper_id for paper_id in candidate_ids if paper_id in query_raw and paper_id in subquery_raw]
    scorable_set = set(scorable_ids)
    unscorable_ids = [paper_id for paper_id in candidate_ids if paper_id not in scorable_set]
    scores = {
        paper_id: text_only_formula_score(
            query_normalized.get(paper_id, 0.0),
            subquery_normalized.get(paper_id, 0.0),
        )
        for paper_id in scorable_ids
    }
    ordered = sorted(
        scorable_ids,
        key=lambda paper_id: (
            -scores[paper_id],
            _as_int(retrieval[paper_id].get("deep_retrieval_rank_after_exclusion"), 10**12),
            paper_id,
        ),
    )
    rerank_rank = {paper_id: rank for rank, paper_id in enumerate(ordered, start=1)}
    features = {
        paper_id: {
            "paper_arxiv_id": paper_id,
            "deep_retrieval_score_raw": float(retrieval[paper_id].get("deep_retrieval_score_raw") or 0.0),
            "deep_retrieval_rank_global_date_valid": retrieval[paper_id].get("deep_retrieval_rank_global_date_valid"),
            "deep_retrieval_rank_after_exclusion": retrieval[paper_id].get("deep_retrieval_rank_after_exclusion"),
            "query_score_raw": float(query_raw.get(paper_id, 0.0)),
            "query_score_normalized": float(query_normalized.get(paper_id, 0.0)),
            "query_component_rank": query_rank.get(paper_id),
            "subquery_score_raw": float(subquery_raw.get(paper_id, 0.0)),
            "subquery_score_normalized": float(subquery_normalized.get(paper_id, 0.0)),
            "subquery_component_rank": subquery_rank.get(paper_id),
            "intent_labels": [],
            "intent_score": 0.0,
            "path_count": 0,
            "path_count_normalized": 0.0,
            "feature_weights": dict(TEXT_ONLY_WEIGHTS),
            "rerank_score": float(scores[paper_id]),
            "rerank_rank": rerank_rank[paper_id],
        }
        for paper_id in ordered
    }
    return ordered, features, unscorable_ids


def deep_paper_scoring_fields(
    paper_id: str,
    *,
    features: Mapping[str, Mapping[str, Any]],
    hit: Mapping[str, Any],
) -> Dict[str, Any]:
    if paper_id in features:
        return {
            **dict(features[paper_id]),
            "rerankable": True,
            "rerank_drop_reason": None,
        }
    return {
        "paper_arxiv_id": paper_id,
        "deep_retrieval_score_raw": float(hit.get("deep_retrieval_score_raw") or 0.0),
        "deep_retrieval_rank_global_date_valid": hit.get("deep_retrieval_rank_global_date_valid"),
        "deep_retrieval_rank_after_exclusion": hit.get("deep_retrieval_rank_after_exclusion"),
        "query_score_raw": None,
        "query_score_normalized": None,
        "query_component_rank": None,
        "subquery_score_raw": None,
        "subquery_score_normalized": None,
        "subquery_component_rank": None,
        "intent_labels": [],
        "intent_score": 0.0,
        "path_count": 0,
        "path_count_normalized": 0.0,
        "feature_weights": dict(TEXT_ONLY_WEIGHTS),
        "rerank_score": None,
        "rerank_rank": None,
        "rerankable": False,
        "rerank_drop_reason": "missing non-empty title/abstract in loaded BM25 metadata",
    }


def contiguous_selector_slices(
    ordered_ids: Sequence[str],
    events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Assign chronological, mutually exclusive top-k slices."""
    slices: List[Dict[str, Any]] = []
    cursor = 0
    for slice_index, event in enumerate(events, start=1):
        requested = max(0, _as_int(event.get("selector_top_k"), 0))
        input_ids = list(ordered_ids[cursor : cursor + requested])
        slices.append(
            {
                "selector_slice_idx": slice_index,
                "rerank_start_rank": cursor + 1 if input_ids else None,
                "rerank_end_rank": cursor + len(input_ids) if input_ids else None,
                "selector_requested_top_k": requested,
                "selector_input_arxiv_ids": input_ids,
                "event": event,
            }
        )
        cursor += requested
    return slices


async def run_selector_for_event(
    selector: Optional[Selector],
    *,
    context: Mapping[str, Any],
    event: Mapping[str, Any],
    input_ids: Sequence[str],
    features: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    if selector is None or not input_ids:
        return {
            "selector_executed": False,
            "selector_input_arxiv_ids": list(input_ids),
            "selector_selected_arxiv_ids": [],
            "selector_reasons": {},
            "selector_overview": "",
        }
    papers: List[Paper] = []
    for paper_id in input_ids:
        item = metadata.get(paper_id) or {}
        papers.append(
            Paper(
                id=paper_id,
                arxiv_id=paper_id,
                title=str(item.get("title") or "N/A"),
                abstract=str(item.get("abstract") or "N/A"),
                date=item.get("date") or "",
                score=float((features.get(paper_id) or {}).get("rerank_score") or 0.0),
            )
        )
    subquery_id: Any = event.get("subquery_id")
    try:
        subquery_id = int(subquery_id)
    except (TypeError, ValueError):
        pass
    subquery = SubQuery(
        id=subquery_id,
        text=str(event.get("subquery") or ""),
        before_date=event.get("subquery_before_date"),
        target_k=_as_int(event.get("subquery_target_k"), len(input_ids)),
        link_type=event.get("subquery_link_type"),
        source_subquery_id=event.get("parent_subquery_id"),
        iter_index=_as_int(event.get("iteration_idx"), 1),
    )
    kept, overview, _, details = await selector.decide_for_subquery(
        user_query=str(context.get("query") or ""),
        sub_query=subquery,
        planner_checklist=str(event.get("planner_checklist") or ""),
        papers=papers,
        iteration_index=_as_int(event.get("iteration_idx"), 1),
        idx=_as_int(context.get("benchmark_idx"), -1),
        old_overview="",
        is_after_browsing=False,
        return_details=True,
    )
    selected_set = {
        normalize_arxiv_id(paper.arxiv_id or paper.id)
        for paper in kept
        if normalize_arxiv_id(paper.arxiv_id or paper.id)
    }
    selected_ids = [paper_id for paper_id in input_ids if paper_id in selected_set]
    raw_reasons = dict((details or {}).get("reasons") or {})
    reasons = {
        normalize_arxiv_id(key): str(value)
        for key, value in raw_reasons.items()
        if normalize_arxiv_id(key) in set(input_ids)
    }
    return {
        "selector_executed": True,
        "selector_input_arxiv_ids": list(input_ids),
        "selector_selected_arxiv_ids": selected_ids,
        "selector_reasons": reasons,
        "selector_overview": overview or "",
    }


def _metadata_for_pool(pool: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        normalize_arxiv_id(hit.get("paper_arxiv_id")): dict(hit.get("_metadata") or {})
        for hit in pool
        if normalize_arxiv_id(hit.get("paper_arxiv_id"))
    }


def stage_metrics(gt_ids: Set[str], candidate_ids: Sequence[str]) -> Dict[str, Any]:
    ordered = _ordered_unique(candidate_ids)
    hits = [paper_id for paper_id in ordered if paper_id in gt_ids]
    recall = _safe_div(len(hits), len(gt_ids))
    precision = _safe_div(len(hits), len(ordered))
    return {
        "count": len(ordered),
        "arxiv_ids": ordered,
        "gt_count": len(hits),
        "gt_arxiv_ids": hits,
        "recall": recall,
        "precision": precision,
        "f1": _f1(recall, precision),
    }


def build_query_summary(
    *,
    context: Mapping[str, Any],
    method: str,
    deep_pool_ids: Sequence[str],
    selector_input_ids: Sequence[str],
    selected_ids: Sequence[str],
    selector_enabled: bool,
    selector_call_count: int,
    deep_requested_occurrences: int,
    deep_actual_occurrences: int,
    rerankable_occurrences: int,
    source_graph_pool_ids: Sequence[str],
) -> Dict[str, Any]:
    gt_ids = set(context.get("gt_ids") or set())
    deep = stage_metrics(gt_ids, deep_pool_ids)
    rerank_input = stage_metrics(gt_ids, selector_input_ids)
    selected = stage_metrics(gt_ids, selected_ids)
    result: Dict[str, Any] = {
        "method": method,
        "query_id": context["query_id"],
        "benchmark_idx": context["benchmark_idx"],
        "gt_count": len(gt_ids),
        "deep_pool_count": deep["count"],
        "deep_pool_arxiv_ids": deep["arxiv_ids"],
        "deep_pool_gt_count": deep["gt_count"],
        "deep_pool_gt_arxiv_ids": deep["gt_arxiv_ids"],
        "deep_pool_recall": deep["recall"],
        "deep_pool_precision": deep["precision"],
        "deep_pool_f1": deep["f1"],
        "rerank_input_count": rerank_input["count"],
        "rerank_input_arxiv_ids": rerank_input["arxiv_ids"],
        "rerank_input_gt_count": rerank_input["gt_count"],
        "rerank_input_gt_arxiv_ids": rerank_input["gt_arxiv_ids"],
        "rerank_input_recall": rerank_input["recall"],
        "rerank_input_precision": rerank_input["precision"],
        "rerank_input_f1": rerank_input["f1"],
        "selector_enabled": selector_enabled,
        "selector_call_count": selector_call_count,
        "event_count": len(context["events"]),
        "subquery_group_count": len(context["groups"]),
        "deep_retrieval_requested_occurrence_count": deep_requested_occurrences,
        "deep_retrieval_actual_occurrence_count": deep_actual_occurrences,
        "deep_retrieval_budget_fulfillment_rate": _safe_div(
            deep_actual_occurrences, deep_requested_occurrences
        ),
        "rerankable_occurrence_count": rerankable_occurrences,
        "unscorable_occurrence_count": max(0, deep_actual_occurrences - rerankable_occurrences),
        "source_graph_pool_occurrence_count": deep_requested_occurrences,
        "source_graph_pool_unique_count": len(_ordered_unique(source_graph_pool_ids)),
        "source_graph_pool_overlap_occurrence_count": max(
            0, deep_requested_occurrences - len(_ordered_unique(source_graph_pool_ids))
        ),
        # Compatibility aliases: candidate means the deduplicated Selector input.
        "candidate_count": rerank_input["count"],
        "candidate_arxiv_ids": rerank_input["arxiv_ids"],
        "candidate_gt_ids": rerank_input["gt_arxiv_ids"],
        "candidate_recall": rerank_input["recall"],
        "candidate_precision": rerank_input["precision"],
    }
    if selector_enabled:
        result.update(
            {
                "selected_count": selected["count"],
                "selected_arxiv_ids": selected["arxiv_ids"],
                "selected_gt_count": selected["gt_count"],
                "selected_gt_ids": selected["gt_arxiv_ids"],
                "selection_recall": selected["recall"],
                "selection_precision": selected["precision"],
                "selection_f1": selected["f1"],
                "retrieved_to_selected_gt_gap": rerank_input["gt_count"] - selected["gt_count"],
                "gt_conversion_rate": _safe_div(selected["gt_count"], rerank_input["gt_count"]),
            }
        )
    else:
        result.update(
            {
                "selected_count": None,
                "selected_arxiv_ids": [],
                "selected_gt_count": None,
                "selected_gt_ids": [],
                "selection_recall": None,
                "selection_precision": None,
                "selection_f1": None,
                "retrieved_to_selected_gt_gap": None,
                "gt_conversion_rate": None,
            }
        )
    return result


async def build_event_method_output(
    *,
    context: Mapping[str, Any],
    pools: Mapping[Any, Sequence[Mapping[str, Any]]],
    retrieval_diagnostics: Mapping[Any, Mapping[str, Any]],
    selector: Optional[Selector],
    save_level: str,
    run_signature: str,
) -> Dict[str, Any]:
    event_outputs: List[Dict[str, Any]] = []
    paper_rows: List[Dict[str, Any]] = []
    all_deep: List[str] = []
    all_inputs: List[str] = []
    all_selected: List[str] = []
    all_source_graph: List[str] = []
    deep_requested_occurrences = 0
    deep_actual_occurrences = 0
    rerankable_occurrences = 0
    selector_calls = 0
    for event in context["events"]:
        event_id = event["retrieval_event_id"]
        pool = list(pools[(EVENT_METHOD, event_id)])
        ordered, features, unscorable = rerank_text_pool(
            pool,
            query=context["query"],
            subquery=event["subquery"],
        )
        input_ids = ordered[: event["selector_top_k"]]
        decision = await run_selector_for_event(
            selector,
            context=context,
            event=event,
            input_ids=input_ids,
            features=features,
            metadata=_metadata_for_pool(pool),
        )
        if decision["selector_executed"]:
            selector_calls += 1
        deep_ids = [normalize_arxiv_id(hit.get("paper_arxiv_id")) for hit in pool]
        deep_ids = [paper_id for paper_id in deep_ids if paper_id]
        all_deep.extend(deep_ids)
        all_inputs.extend(input_ids)
        all_selected.extend(decision["selector_selected_arxiv_ids"])
        all_source_graph.extend(event["source_local_graph_pool_arxiv_ids"])
        deep_requested_occurrences += event["source_local_graph_pool_size"]
        deep_actual_occurrences += len(deep_ids)
        rerankable_occurrences += len(ordered)
        selected_set = set(decision["selector_selected_arxiv_ids"])
        input_rank = {paper_id: rank for rank, paper_id in enumerate(input_ids, start=1)}
        event_output = {
            **dict(event),
            "method": EVENT_METHOD,
            "deep_retrieval_pagination": "exclude event snapshot, then apply saved baseline offset",
            "deep_retrieval_requested_count": event["source_local_graph_pool_size"],
            "deep_retrieval_actual_count": len(pool),
            "rerankable_count": len(ordered),
            "unscorable_arxiv_ids": unscorable,
            "deep_retrieval_arxiv_ids": [hit["paper_arxiv_id"] for hit in pool],
            "reranked_arxiv_ids": ordered,
            "selector_input_arxiv_ids": input_ids,
            "selector_selected_arxiv_ids": decision["selector_selected_arxiv_ids"],
            "selector_reasons": decision["selector_reasons"],
            "selector_overview": decision["selector_overview"],
            "selector_executed": decision["selector_executed"],
            "selector_old_overview": "",
            "retrieval_diagnostics": dict(
                retrieval_diagnostics.get((EVENT_METHOD, event_id)) or {}
            ),
        }
        event_outputs.append(event_output)
        if save_level == "full":
            source_seed_set = set(event["baseline_seed_arxiv_ids"])
            source_local_set = set(event["source_local_graph_pool_arxiv_ids"])
            hit_by_id = {
                normalize_arxiv_id(hit.get("paper_arxiv_id")): hit
                for hit in pool
                if normalize_arxiv_id(hit.get("paper_arxiv_id"))
            }
            for paper_id in deep_ids:
                feature = deep_paper_scoring_fields(
                    paper_id,
                    features=features,
                    hit=hit_by_id[paper_id],
                )
                paper_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "method": EVENT_METHOD,
                        "query_id": context["query_id"],
                        "benchmark_idx": context["benchmark_idx"],
                        "query": context["query"],
                        "subquery_id": event["subquery_id"],
                        "subquery": event["subquery"],
                        "subquery_before_date": event["subquery_before_date"],
                        "retrieval_event_id": event_id,
                        "iteration_idx": event["iteration_idx"],
                        "retrieval_page_idx": event["retrieval_page_idx"],
                        "retrieval_offset": event["retrieval_offset"],
                        "source_local_graph_pool_size": event["source_local_graph_pool_size"],
                        "paper_arxiv_id": paper_id,
                        "candidate_type": "deep_retrieved",
                        "retrieval_backend": "bm25",
                        "passed_date_cutoff": True,
                        "date_cutoff_month": str(event.get("subquery_before_date") or "")[:7],
                        "is_ground_truth": paper_id in set(context.get("gt_ids") or set()),
                        "deep_retrieval_rank_scope": "full_corpus_positive_date_valid_canonical_arxiv",
                        "component_rank_scope": "closed_event_deep_pool",
                        "normalization_scope": event_id,
                        "in_source_baseline_seed_page": paper_id in source_seed_set,
                        "in_source_local_graph_pool": paper_id in source_local_set,
                        **feature,
                        "in_selector_input": paper_id in input_rank,
                        "selector_input_rank": input_rank.get(paper_id),
                        "selector_selected": paper_id in selected_set,
                        "selector_reason": decision["selector_reasons"].get(paper_id, ""),
                    }
                )
    summary = build_query_summary(
        context=context,
        method=EVENT_METHOD,
        deep_pool_ids=all_deep,
        selector_input_ids=all_inputs,
        selected_ids=all_selected,
        selector_enabled=selector is not None,
        selector_call_count=selector_calls,
        deep_requested_occurrences=deep_requested_occurrences,
        deep_actual_occurrences=deep_actual_occurrences,
        rerankable_occurrences=rerankable_occurrences,
        source_graph_pool_ids=all_source_graph,
    )
    output: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_signature": run_signature,
        "method": EVENT_METHOD,
        "llm_model": config.LLM_MODEL_NAME if selector is not None else None,
        "selector_enabled": selector is not None,
        "query_id": context["query_id"],
        "benchmark_idx": context["benchmark_idx"],
        "query": context["query"],
        "source_artifact_canonicalization": context["canonicalization"],
        "summary": summary,
        "events": event_outputs,
    }
    if save_level == "full":
        output["paper_rows"] = paper_rows
    return output


async def build_merged_method_output(
    *,
    context: Mapping[str, Any],
    pools: Mapping[Any, Sequence[Mapping[str, Any]]],
    retrieval_diagnostics: Mapping[Any, Mapping[str, Any]],
    selector: Optional[Selector],
    save_level: str,
    run_signature: str,
) -> Dict[str, Any]:
    group_outputs: List[Dict[str, Any]] = []
    paper_rows: List[Dict[str, Any]] = []
    all_deep: List[str] = []
    all_inputs: List[str] = []
    all_selected: List[str] = []
    all_source_graph: List[str] = []
    deep_requested_occurrences = 0
    deep_actual_occurrences = 0
    rerankable_occurrences = 0
    selector_calls = 0
    for group in context["groups"]:
        subquery_id = group["subquery_id"]
        pool = list(pools[(MERGED_METHOD, subquery_id)])
        ordered, features, unscorable = rerank_text_pool(
            pool,
            query=context["query"],
            subquery=group["subquery"],
        )
        slices = contiguous_selector_slices(ordered, group["events"])
        group_selected: List[str] = []
        slice_outputs: List[Dict[str, Any]] = []
        assignment: Dict[str, Dict[str, Any]] = {}
        metadata = _metadata_for_pool(pool)
        for slice_value in slices:
            event = slice_value["event"]
            input_ids = slice_value["selector_input_arxiv_ids"]
            decision = await run_selector_for_event(
                selector,
                context=context,
                event=event,
                input_ids=input_ids,
                features=features,
                metadata=metadata,
            )
            if decision["selector_executed"]:
                selector_calls += 1
            group_selected.extend(decision["selector_selected_arxiv_ids"])
            for within_slice_rank, paper_id in enumerate(input_ids, start=1):
                assignment[paper_id] = {
                    "retrieval_event_id": event["retrieval_event_id"],
                    "selector_slice_idx": slice_value["selector_slice_idx"],
                    "selector_input_rank": within_slice_rank,
                    "selector_selected": paper_id in set(decision["selector_selected_arxiv_ids"]),
                    "selector_reason": decision["selector_reasons"].get(paper_id, ""),
                }
            slice_outputs.append(
                {
                    "selector_slice_idx": slice_value["selector_slice_idx"],
                    "retrieval_event_id": event["retrieval_event_id"],
                    "iteration_idx": event["iteration_idx"],
                    "retrieval_page_idx": event["retrieval_page_idx"],
                    "source_retrieval_offset": event["retrieval_offset"],
                    "source_local_graph_pool_size": event["source_local_graph_pool_size"],
                    "source_retrieval_exclusion_arxiv_ids": event["retrieval_exclusion_arxiv_ids"],
                    "subquery_id": event["subquery_id"],
                    "subquery": event["subquery"],
                    "planner_checklist": event["planner_checklist"],
                    "selector_requested_top_k": slice_value["selector_requested_top_k"],
                    "rerank_start_rank": slice_value["rerank_start_rank"],
                    "rerank_end_rank": slice_value["rerank_end_rank"],
                    "selector_input_arxiv_ids": input_ids,
                    "selector_selected_arxiv_ids": decision["selector_selected_arxiv_ids"],
                    "selector_reasons": decision["selector_reasons"],
                    "selector_overview": decision["selector_overview"],
                    "selector_executed": decision["selector_executed"],
                    "selector_old_overview": "",
                }
            )
        group_inputs = [paper_id for slice_value in slices for paper_id in slice_value["selector_input_arxiv_ids"]]
        deep_ids = [normalize_arxiv_id(hit.get("paper_arxiv_id")) for hit in pool]
        deep_ids = [paper_id for paper_id in deep_ids if paper_id]
        all_deep.extend(deep_ids)
        all_inputs.extend(group_inputs)
        all_selected.extend(group_selected)
        all_source_graph.extend(
            paper_id
            for event in group["events"]
            for paper_id in event["source_local_graph_pool_arxiv_ids"]
        )
        deep_requested_occurrences += group["source_local_graph_pool_occurrence_budget"]
        deep_actual_occurrences += len(deep_ids)
        rerankable_occurrences += len(ordered)
        group_outputs.append(
            {
                **{key: value for key, value in group.items() if key != "events"},
                "method": MERGED_METHOD,
                "deep_retrieval_pagination": "freeze first-event exclusion, offset=0, one sum-budget retrieval",
                "deep_retrieval_requested_count": group["source_local_graph_pool_occurrence_budget"],
                "deep_retrieval_actual_count": len(pool),
                "rerankable_count": len(ordered),
                "unscorable_arxiv_ids": unscorable,
                "deep_retrieval_arxiv_ids": [hit["paper_arxiv_id"] for hit in pool],
                "reranked_arxiv_ids": ordered,
                "selector_slicing": "chronological contiguous disjoint slices using each event's actual selector_top_k",
                "selector_input_arxiv_ids": group_inputs,
                "selector_selected_arxiv_ids": _ordered_unique(group_selected),
                "selector_slices": slice_outputs,
                "retrieval_diagnostics": dict(
                    retrieval_diagnostics.get((MERGED_METHOD, subquery_id)) or {}
                ),
            }
        )
        if save_level == "full":
            source_seed_set = set(group["source_baseline_seed_union_arxiv_ids"])
            source_local_set = set(group["source_local_graph_pool_union_arxiv_ids"])
            hit_by_id = {
                normalize_arxiv_id(hit.get("paper_arxiv_id")): hit
                for hit in pool
                if normalize_arxiv_id(hit.get("paper_arxiv_id"))
            }
            for paper_id in deep_ids:
                assigned = assignment.get(paper_id) or {}
                paper_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "method": MERGED_METHOD,
                        "query_id": context["query_id"],
                        "benchmark_idx": context["benchmark_idx"],
                        "query": context["query"],
                        "subquery_id": subquery_id,
                        "subquery": group["subquery"],
                        "subquery_before_date": group["subquery_before_date"],
                        "source_event_ids": group["source_event_ids"],
                        "source_local_graph_pool_occurrence_budget": group["source_local_graph_pool_occurrence_budget"],
                        "source_local_graph_pool_union_count": group["source_local_graph_pool_union_count"],
                        "paper_arxiv_id": paper_id,
                        "candidate_type": "deep_retrieved",
                        "retrieval_backend": "bm25",
                        "passed_date_cutoff": True,
                        "date_cutoff_month": str(group.get("subquery_before_date") or "")[:7],
                        "is_ground_truth": paper_id in set(context.get("gt_ids") or set()),
                        "deep_retrieval_rank_scope": "full_corpus_positive_date_valid_canonical_arxiv",
                        "component_rank_scope": "closed_merged_subquery_deep_pool",
                        "normalization_scope": f"{context['query_id']}:subquery:{subquery_id}",
                        "in_any_source_baseline_seed_page": paper_id in source_seed_set,
                        "in_source_local_graph_union": paper_id in source_local_set,
                        **deep_paper_scoring_fields(
                            paper_id,
                            features=features,
                            hit=hit_by_id[paper_id],
                        ),
                        "assigned_retrieval_event_id": assigned.get("retrieval_event_id"),
                        "selector_slice_idx": assigned.get("selector_slice_idx"),
                        "in_selector_input": paper_id in assignment,
                        "selector_input_rank": assigned.get("selector_input_rank"),
                        "selector_selected": bool(assigned.get("selector_selected", False)),
                        "selector_reason": assigned.get("selector_reason", ""),
                    }
                )
    summary = build_query_summary(
        context=context,
        method=MERGED_METHOD,
        deep_pool_ids=all_deep,
        selector_input_ids=all_inputs,
        selected_ids=all_selected,
        selector_enabled=selector is not None,
        selector_call_count=selector_calls,
        deep_requested_occurrences=deep_requested_occurrences,
        deep_actual_occurrences=deep_actual_occurrences,
        rerankable_occurrences=rerankable_occurrences,
        source_graph_pool_ids=all_source_graph,
    )
    output: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_signature": run_signature,
        "method": MERGED_METHOD,
        "llm_model": config.LLM_MODEL_NAME if selector is not None else None,
        "selector_enabled": selector is not None,
        "query_id": context["query_id"],
        "benchmark_idx": context["benchmark_idx"],
        "query": context["query"],
        "source_artifact_canonicalization": context["canonicalization"],
        "summary": summary,
        "subquery_groups": group_outputs,
    }
    if save_level == "full":
        output["paper_rows"] = paper_rows
    return output


def prepare_deep_pools(
    rag_system: Any,
    context: Mapping[str, Any],
    pending_methods: Set[str],
) -> Tuple[Dict[Any, List[Dict[str, Any]]], Dict[Any, Dict[str, Any]]]:
    pools: Dict[Any, List[Dict[str, Any]]] = {}
    diagnostics: Dict[Any, Dict[str, Any]] = {}
    for group in context["groups"]:
        requests: Dict[Any, Dict[str, Any]] = {}
        if EVENT_METHOD in pending_methods:
            for event in group["events"]:
                requests[(EVENT_METHOD, event["retrieval_event_id"])] = {
                    "offset": event["retrieval_offset"],
                    "count": event["source_local_graph_pool_size"],
                    "exclude_arxiv_ids": event["retrieval_exclusion_arxiv_ids"],
                }
        if MERGED_METHOD in pending_methods:
            requests[(MERGED_METHOD, group["subquery_id"])] = {
                "offset": 0,
                "count": group["source_local_graph_pool_occurrence_budget"],
                "exclude_arxiv_ids": group["frozen_first_event_exclusion_arxiv_ids"],
            }
        group_pools, group_diagnostics = retrieve_bm25_requests(
            rag_system,
            subquery=group["subquery"],
            before_date=group["subquery_before_date"],
            requests=requests,
        )
        pools.update(group_pools)
        diagnostics.update(group_diagnostics)
    return pools, diagnostics


def aggregate_query_outputs(outputs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summaries = [dict(output.get("summary") or {}) for output in outputs]
    count = len(summaries)
    result: Dict[str, Any] = {
        "evaluated_query_count": count,
        "selector_enabled": bool(summaries and summaries[0].get("selector_enabled")),
    }
    for stage in ("deep_pool", "rerank_input"):
        for metric in ("count", "gt_count", "recall", "precision", "f1"):
            result[f"avg_{stage}_{metric}"] = (
                sum(float(summary.get(f"{stage}_{metric}") or 0.0) for summary in summaries) / count
                if count
                else 0.0
            )
        result[f"mean_query_{stage}_f1"] = result[f"avg_{stage}_f1"]
        result[f"macro_{stage}_f1_from_avg_recall_precision"] = _f1(
            result[f"avg_{stage}_recall"], result[f"avg_{stage}_precision"]
        )
        # Keep the project's main-table convention: harmonic mean of macro R/P.
        result[f"avg_{stage}_f1"] = result[f"macro_{stage}_f1_from_avg_recall_precision"]
        total_candidates = sum(_as_int(summary.get(f"{stage}_count"), 0) for summary in summaries)
        total_hits = sum(_as_int(summary.get(f"{stage}_gt_count"), 0) for summary in summaries)
        total_gt = sum(_as_int(summary.get("gt_count"), 0) for summary in summaries)
        micro_recall = _safe_div(total_hits, total_gt)
        micro_precision = _safe_div(total_hits, total_candidates)
        result.update(
            {
                f"total_{stage}_count": total_candidates,
                f"total_{stage}_gt_count": total_hits,
                f"micro_{stage}_recall": micro_recall,
                f"micro_{stage}_precision": micro_precision,
                f"micro_{stage}_f1": _f1(micro_recall, micro_precision),
            }
        )
    selector_enabled = result["selector_enabled"]
    if selector_enabled:
        for metric in (
            "selected_count",
            "selected_gt_count",
            "selection_recall",
            "selection_precision",
            "selection_f1",
            "retrieved_to_selected_gt_gap",
            "gt_conversion_rate",
        ):
            result[f"avg_{metric}"] = (
                sum(float(summary.get(metric) or 0.0) for summary in summaries) / count if count else 0.0
            )
        total_selected = sum(_as_int(summary.get("selected_count"), 0) for summary in summaries)
        total_selected_hits = sum(_as_int(summary.get("selected_gt_count"), 0) for summary in summaries)
        total_gt = sum(_as_int(summary.get("gt_count"), 0) for summary in summaries)
        total_retrieved_hits = sum(_as_int(summary.get("rerank_input_gt_count"), 0) for summary in summaries)
        micro_selection_recall = _safe_div(total_selected_hits, total_gt)
        micro_selection_precision = _safe_div(total_selected_hits, total_selected)
        result.update(
            {
                "total_selected_count": total_selected,
                "total_selected_gt_count": total_selected_hits,
                "micro_selection_recall": micro_selection_recall,
                "micro_selection_precision": micro_selection_precision,
                "micro_selection_f1": _f1(micro_selection_recall, micro_selection_precision),
                "micro_gt_conversion_rate": _safe_div(total_selected_hits, total_retrieved_hits),
                "total_retrieved_to_selected_gt_gap": total_retrieved_hits - total_selected_hits,
            }
        )
        result["mean_query_selection_f1"] = result["avg_selection_f1"]
        result["macro_selection_f1_from_avg_recall_precision"] = _f1(
            result["avg_selection_recall"], result["avg_selection_precision"]
        )
        result["avg_selection_f1"] = result["macro_selection_f1_from_avg_recall_precision"]
    else:
        for key in (
            "avg_selected_count",
            "avg_selected_gt_count",
            "avg_selection_recall",
            "avg_selection_precision",
            "avg_selection_f1",
            "mean_query_selection_f1",
            "macro_selection_f1_from_avg_recall_precision",
            "avg_retrieved_to_selected_gt_gap",
            "avg_gt_conversion_rate",
            "total_selected_count",
            "total_selected_gt_count",
            "micro_selection_recall",
            "micro_selection_precision",
            "micro_selection_f1",
            "micro_gt_conversion_rate",
            "total_retrieved_to_selected_gt_gap",
        ):
            result[key] = None
    budget_fields = (
        "deep_retrieval_requested_occurrence_count",
        "deep_retrieval_actual_occurrence_count",
        "rerankable_occurrence_count",
        "unscorable_occurrence_count",
        "source_graph_pool_occurrence_count",
        "source_graph_pool_unique_count",
        "source_graph_pool_overlap_occurrence_count",
    )
    for field in budget_fields:
        total = sum(_as_int(summary.get(field), 0) for summary in summaries)
        result[f"total_{field}"] = total
        result[f"avg_{field}"] = _safe_div(total, count)
    result["overall_deep_retrieval_budget_fulfillment_rate"] = _safe_div(
        result["total_deep_retrieval_actual_occurrence_count"],
        result["total_deep_retrieval_requested_occurrence_count"],
    )
    result["avg_deep_retrieval_budget_fulfillment_rate"] = (
        sum(float(summary.get("deep_retrieval_budget_fulfillment_rate") or 0.0) for summary in summaries)
        / count
        if count
        else 0.0
    )
    result["total_selector_call_count"] = sum(
        _as_int(summary.get("selector_call_count"), 0) for summary in summaries
    )
    # Main-table aliases: Ret is the union of reranked slices actually shown to Selector.
    result["avg_candidate_recall"] = result.get("avg_rerank_input_recall", 0.0)
    result["avg_candidate_precision"] = result.get("avg_rerank_input_precision", 0.0)
    result["avg_candidate_f1"] = result.get("avg_rerank_input_f1", 0.0)
    result["avg_retrieved_gt_count"] = result.get("avg_rerank_input_gt_count", 0.0)
    result["total_candidate_count"] = result.get("total_rerank_input_count", 0)
    result["total_candidate_gt_count"] = result.get("total_rerank_input_gt_count", 0)
    result["micro_candidate_recall"] = result.get("micro_rerank_input_recall", 0.0)
    result["micro_candidate_precision"] = result.get("micro_rerank_input_precision", 0.0)
    result["micro_candidate_f1"] = result.get("micro_rerank_input_f1", 0.0)
    return result


def load_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("deep_retrieval_replay_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_selector(config_path: Path, llm_model: Optional[str]) -> Tuple[Selector, Dict[str, Any]]:
    cfg = load_config(config_path)
    config.LLM_MODEL_NAME = llm_model or cfg.LLM_MODEL_NAME
    config.IS_LOCAL_LLM = cfg.IS_LOCAL_LLM
    config.LLM_GEN_PARAMS = cfg.LLM_GEN_PARAMS
    config.ENABLE_REASONING = cfg.ENABLE_REASONING
    config.ENABLE_STRUCTURED_OUTPUT = cfg.ENABLE_STRUCTURED_OUTPUT
    config.ENABLE_LLM_FILTERING = getattr(cfg, "ENABLE_LLM_FILTERING", True)
    if not config.ENABLE_LLM_FILTERING:
        raise ValueError(
            "Selector replay requires ENABLE_LLM_FILTERING=True; use --skip_selector for retrieval/rerank-only mode"
        )
    config.BROWSER_MODE = "NONE"
    config.DEBUG = False
    config.SAVE_AGENT_TRACES = False
    selector = Selector(config.LLM_MODEL_NAME, config.LLM_GEN_PARAMS, config.IS_LOCAL_LLM)
    provider_route = "ollama" if config.IS_LOCAL_LLM else next(
        (
            prefix
            for prefix in llm_api.PROVIDER_CONFIG
            if prefix != "ollama" and config.LLM_MODEL_NAME.lower().startswith(prefix)
        ),
        "gpt",
    )
    provider = llm_api._resolve_provider(config.LLM_MODEL_NAME, config.IS_LOCAL_LLM)
    return selector, {
        "llm_model": config.LLM_MODEL_NAME,
        "is_local_llm": config.IS_LOCAL_LLM,
        "provider_route": provider_route,
        "provider_base_url_sha256": hashlib.sha256(
            str(provider.get("base_url") or "").encode("utf-8")
        ).hexdigest(),
        "llm_gen_params": config.LLM_GEN_PARAMS,
        "enable_reasoning": config.ENABLE_REASONING,
        "enable_structured_output": config.ENABLE_STRUCTURED_OUTPUT,
        "enable_llm_filtering": config.ENABLE_LLM_FILTERING,
        "browser_mode": config.BROWSER_MODE,
    }


def _sha256_json(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def ensure_compatible_output_dir(output_dir: Path, run_signature: str) -> None:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing output manifest is unreadable: {manifest_path}") from exc
    existing_signature = existing.get("run_signature")
    has_query_files = any(output_dir.glob("*/queries/*.json"))
    if existing_signature and existing_signature != run_signature and has_query_files:
        raise ValueError(
            "output_dir already contains query files from a different run signature; "
            "use a new --output_dir instead of mixing experiments"
        )


def completed_outputs(method_dir: Path, run_signature: str) -> Dict[int, Dict[str, Any]]:
    outputs: Dict[int, Dict[str, Any]] = {}
    query_dir = method_dir / "queries"
    for path in sorted(query_dir.glob("*.json")) if query_dir.exists() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        idx = value.get("benchmark_idx")
        if (
            isinstance(idx, int)
            and value.get("method") == method_dir.name
            and value.get("run_signature") == run_signature
            and isinstance(value.get("summary"), Mapping)
        ):
            outputs[idx] = value
    return outputs


def write_method_summary(method_dir: Path, outputs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = [outputs[idx] for idx in sorted(outputs)]
    summary = aggregate_query_outputs(ordered)
    summary["method"] = method_dir.name
    query_results_path = method_dir / "query_results.jsonl"
    tmp = query_results_path.with_suffix(query_results_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as handle:
        for output in ordered:
            handle.write(json.dumps(output.get("summary") or {}, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, query_results_path)
    atomic_write_json(method_dir / "summary.json", summary)
    return summary


def resolve_bm25_path(explicit: Optional[str], run_dir: Path) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"BM25 index does not exist: {path}")
        return path
    manifest_path = run_dir / "onepass_artifacts" / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("--bm25_path is required because the source run has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = manifest.get("bm25_path")
    if not raw:
        raise ValueError("--bm25_path is required because the source manifest has no bm25_path")
    candidates = [Path(raw), PACKAGE_ROOT / str(raw), run_dir / str(raw), manifest_path.parent / str(raw)]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"could not resolve source-manifest bm25_path={raw!r}; pass --bm25_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_run_dir", required=True, help="Completed full-mode OnePass run")
    parser.add_argument("--bm25_path", default=None, help="Baseline BM25 pickle; inferred from source manifest when possible")
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs/config_qwen30b_api.py"))
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_level", choices=("minimal", "full"), default="full")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--skip_selector", action="store_true", help="Run retrieval/rerank only; makes no paid LLM calls")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute compatible method/query files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_methods = list(dict.fromkeys(args.methods))
    run_dir = Path(args.baseline_run_dir).expanduser().resolve()
    artifacts = run_dir / "onepass_artifacts"
    detailed_path = run_dir / "detailed_results.jsonl"
    required = {
        "detailed results": detailed_path,
        "baseline Planner events": artifacts / "baseline/planner_events.jsonl",
        "baseline paper rows": artifacts / "baseline/paper_rows.jsonl",
        "per-subquery paper rows": artifacts / "per_subquery/paper_rows.jsonl",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required full-mode source artifacts:\n" + "\n".join(missing))
    bm25_path = resolve_bm25_path(args.bm25_path, run_dir)
    config_path = Path(args.config).expanduser().resolve()

    selector: Optional[Selector] = None
    selector_config: Dict[str, Any]
    if args.skip_selector:
        selector_config = {
            "llm_model": None,
            "enable_reasoning": False,
            "browser_mode": "NONE",
            "selector_enabled": False,
        }
    else:
        selector, selector_config = configure_selector(config_path, args.llm_model)
        selector_config["selector_enabled"] = True

    source_fingerprint_paths = dict(required)
    source_run_manifest = artifacts / "run_manifest.json"
    if source_run_manifest.exists():
        source_fingerprint_paths["source run manifest"] = source_run_manifest
    code_fingerprint_paths = {
        "deep replay script": Path(__file__).resolve(),
        "selector": CODE_DIR / "agent" / "selector.py",
        "selector utilities": CODE_DIR / "utils.py",
        "selector prompts": CODE_DIR / "prompt.py",
        "LLM API adapter": CODE_DIR / "api.py",
        "BM25 loader/preprocessor": CODE_DIR / "rag.py",
        "closed-pool scorer": CODE_DIR / "graph_methods.py",
        "paper/subquery structures": CODE_DIR / "structures.py",
        "base config": CODE_DIR / "config.py",
    }
    if not args.skip_selector:
        code_fingerprint_paths["selector config"] = config_path

    signature_payload = {
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "baseline_run_dir": str(run_dir),
        "bm25_path": str(bm25_path),
        "bm25_fingerprint": _stat_fingerprint(bm25_path),
        "source_file_fingerprints": {
            name: _stat_fingerprint(path) for name, path in source_fingerprint_paths.items()
        },
        "code_sha256": {
            name: _sha256_file(path) for name, path in code_fingerprint_paths.items()
        },
        "save_level": args.save_level,
        "selector": selector_config,
        "formula_weights": TEXT_ONLY_WEIGHTS,
        "event_pagination": "event exclusion snapshot -> saved event offset -> N_i",
        "merged_pagination": "first-event exclusion snapshot -> offset 0 -> sum_i N_i",
        "merged_selector_slicing": "one rerank, chronological disjoint slices by actual event selector_top_k",
        "source_canonicalization": "newest contiguous Planner trajectory matching committed baseline and per-subquery candidate/selected IDs; last matching contiguous event block",
        "deep_retrieval_identity_policy": "normalize and deduplicate canonical arXiv IDs before exclusion/offset",
    }
    run_signature = _sha256_json(signature_payload)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_compatible_output_dir(output_dir, run_signature)
    requested_methods_history = list(selected_methods)
    existing_manifest_path = output_dir / "run_manifest.json"
    if existing_manifest_path.exists():
        try:
            existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_manifest = {}
        if existing_manifest.get("run_signature") == run_signature:
            requested_methods_history = list(
                dict.fromkeys((existing_manifest.get("requested_methods") or []) + selected_methods)
            )
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            **signature_payload,
            "run_signature": run_signature,
            "requested_methods": requested_methods_history,
            "detailed_results_path": str(detailed_path),
            "source_artifacts": {key: str(value) for key, value in required.items()},
            "rendered_selector_prompt_saved": False,
            "raw_paper_title_abstract_saved": False,
            "planner_checklist_saved": True,
            "selector_reason_and_overview_saved": True,
            "semantic_scholar_called": False,
            "planner_called": False,
            "paper_db_loaded_separately": False,
            "deep_pool_budget_definition": "distinct date-valid source graph pool arXiv IDs per event",
            "merged_budget_counts_cross_event_graph_paper_repeats_as_occurrences": True,
            "intent_and_path_features_for_deep_papers": 0,
        },
    )

    detailed = load_detailed_results(detailed_path)
    indices = list(detailed)
    if args.limit is not None:
        allowed_limit = set(sorted(detailed)[: max(0, args.limit)])
        indices = [idx for idx in indices if idx in allowed_limit]
    allowed = set(indices)
    order = {idx: position for position, idx in enumerate(indices)}
    cursors = {
        "planner": GroupCursor(required["baseline Planner events"], allowed, order),
        "baseline": GroupCursor(required["baseline paper rows"], allowed, order),
        "local": GroupCursor(required["per-subquery paper rows"], allowed, order),
    }
    method_outputs = {
        method: {
            idx: value
            for idx, value in completed_outputs(output_dir / method, run_signature).items()
            if idx in allowed
        }
        for method in selected_methods
    }
    rag_system: Any = None
    for position, idx in enumerate(indices, start=1):
        planner_rows = cursors["planner"].take(idx)
        baseline_rows = cursors["baseline"].take(idx)
        local_rows = cursors["local"].take(idx)
        pending = {
            method
            for method in selected_methods
            if args.force or idx not in method_outputs[method]
        }
        if not pending:
            continue
        context = build_source_context(detailed[idx], planner_rows, baseline_rows, local_rows)
        if rag_system is None:
            from rag import CitationRAGSystem  # Imported lazily so offline unit tests stay lightweight.

            rag_system = CitationRAGSystem(search_method="bm25", device="cpu")
            rag_system.load_bm25_index(str(bm25_path))
        pools, retrieval_diagnostics = prepare_deep_pools(rag_system, context, pending)
        for method in selected_methods:
            if method not in pending:
                continue
            print(f"[{position}/{len(indices)}] idx={idx} method={method}", flush=True)
            if method == EVENT_METHOD:
                output = asyncio.run(
                    build_event_method_output(
                        context=context,
                        pools=pools,
                        retrieval_diagnostics=retrieval_diagnostics,
                        selector=selector,
                        save_level=args.save_level,
                        run_signature=run_signature,
                    )
                )
            elif method == MERGED_METHOD:
                output = asyncio.run(
                    build_merged_method_output(
                        context=context,
                        pools=pools,
                        retrieval_diagnostics=retrieval_diagnostics,
                        selector=selector,
                        save_level=args.save_level,
                        run_signature=run_signature,
                    )
                )
            else:
                raise ValueError(f"unsupported method: {method}")
            query_path = output_dir / method / "queries" / f"{idx:06d}.json"
            atomic_write_json(query_path, output)
            method_outputs[method][idx] = output

    all_method_outputs: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for method in METHODS:
        outputs = method_outputs.get(method)
        if outputs is None:
            outputs = {
                idx: value
                for idx, value in completed_outputs(output_dir / method, run_signature).items()
                if idx in allowed
            }
        if outputs:
            all_method_outputs[method] = outputs
    overall = {
        "schema_version": SCHEMA_VERSION,
        "run_signature": run_signature,
        "source_committed_query_count": len(indices),
        "selector_enabled": selector is not None,
        "methods": {
            method: write_method_summary(output_dir / method, outputs)
            for method, outputs in all_method_outputs.items()
        },
    }
    atomic_write_json(output_dir / "evaluation_summary.json", overall)
    print(f"Saved deep-retrieval replay outputs to {output_dir}")


if __name__ == "__main__":
    main()
