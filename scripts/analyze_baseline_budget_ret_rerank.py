#!/usr/bin/env python3
"""Replay Graph/Deep-merged Ret Top-K at Baseline's actual per-event budget.

The script is intentionally offline.  It reads the completed OnePass artifacts,
derives ``K_i`` from the number of papers in each Baseline Selector input, and
then applies the same event budgets to Graph and Deep merged.  It never calls
the retriever, embedding backend, Planner, or Selector.

Two selections are compared:

* ``stored_pipeline``: the Selector inputs saved by the completed run;
* a configurable replay formula over the complete saved local pool.

Deep merged is ranked once per merged subquery pool.  Its ranked list is split
chronologically into source-event slices whose lengths are the corresponding
Baseline ``K_i`` values.  Query-level metrics use the union of all event slices
with query-local paper deduplication.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import analyze_graph_vs_deep_merged as semantic_analysis  # noqa: E402


IMPLEMENTATION_VERSION = "1.0"
STORED_FORMULA = "stored_pipeline"
NEW_FORMULA = "q030_sq040_intent015_path015_closed_pool_minmax_v1"
FORMULAS = (STORED_FORMULA, NEW_FORMULA)
METHODS = ("graph", "deep_merged")
AXIS_LABELS: Mapping[str, Sequence[str]] = {
    "historical_relation": (
        "direct_predecessor",
        "enabling_foundation",
        "historical_background",
        "retrospective_or_survey",
        "insufficient_evidence",
    ),
    "mechanism_relation": (
        "explicit_target_mechanism",
        "implicit_explanatory_mechanism",
        "generic_theory",
        "insufficient_evidence",
    ),
    "domain_relation": (
        "same_domain",
        "adjacent_domain",
        "cross_domain_transfer",
        "unrelated_domain_drift",
        "insufficient_evidence",
    ),
}
AXIS_TRIGGERS = {
    "historical_relation": "historical_context",
    "mechanism_relation": "mechanism_or_theory",
    "domain_relation": "application_domain",
}
INFORMATION_GRADES = {"direct", "partial", "contextual"}


@dataclass(frozen=True)
class BaselineBudget:
    query_id: str
    benchmark_idx: int
    retrieval_event_id: str
    iteration_idx: int
    subquery_id: int
    subquery: str
    k: int
    selector_input_ids: Tuple[str, ...]


@dataclass(frozen=True)
class QueryContext:
    query_id: str
    benchmark_idx: int
    query: str
    ground_truth_ids: frozenset[str]


def _paper_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("arxiv:"):
        text = text.split(":", 1)[1].strip()
    return text


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(recall: float, precision: float) -> float:
    return 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
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


def _ordered_unique(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen: Set[str] = set()
    for value in values:
        paper_id = _paper_id(value)
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            output.append(paper_id)
    return output


def load_baseline_budgets(path: Path) -> Dict[str, BaselineBudget]:
    """Load K_i from the actual Baseline candidate rows sent to Selector."""

    budgets: Dict[str, BaselineBudget] = {}
    for row in _iter_jsonl(path):
        candidates = list(row.get("candidate_rows") or [])
        if not candidates:
            raise ValueError(
                f"empty Baseline Selector input for query={row.get('query_id')} "
                f"iteration={row.get('iteration_idx')} subquery={row.get('subquery_id')}"
            )
        event_ids = {str(candidate.get("retrieval_event_id") or "") for candidate in candidates}
        if "" in event_ids or len(event_ids) != 1:
            raise ValueError(
                "each Baseline Selector decision must contain exactly one retrieval_event_id; "
                f"got {sorted(event_ids)}"
            )
        event_id = next(iter(event_ids))
        paper_ids = [_paper_id(candidate.get("paper_arxiv_id")) for candidate in candidates]
        if any(not paper_id for paper_id in paper_ids):
            raise ValueError(f"missing paper ID in Baseline Selector input {event_id}")
        if len(set(paper_ids)) != len(paper_ids):
            raise ValueError(f"duplicate paper IDs in Baseline Selector input {event_id}")
        if event_id in budgets:
            raise ValueError(f"duplicate Baseline retrieval event {event_id}")
        query_id = str(row.get("query_id") or "")
        candidate_query_ids = {str(candidate.get("query_id") or "") for candidate in candidates}
        if candidate_query_ids != {query_id}:
            raise ValueError(f"query mismatch inside Baseline Selector input {event_id}")
        budgets[event_id] = BaselineBudget(
            query_id=query_id,
            benchmark_idx=_as_int(row.get("benchmark_idx"), -1),
            retrieval_event_id=event_id,
            iteration_idx=_as_int(row.get("iteration_idx"), -1),
            subquery_id=_as_int(row.get("subquery_id"), -1),
            subquery=str(row.get("subquery") or ""),
            k=len(candidates),
            selector_input_ids=tuple(paper_ids),
        )
    if not budgets:
        raise ValueError(f"no Baseline Selector decisions found in {path}")
    return budgets


def load_query_contexts(path: Path) -> Dict[str, QueryContext]:
    contexts: Dict[str, QueryContext] = {}
    for row in _iter_jsonl(path):
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in contexts:
            raise ValueError(f"missing or duplicate query_id in {path}: {query_id!r}")
        contexts[query_id] = QueryContext(
            query_id=query_id,
            benchmark_idx=_as_int(row.get("benchmark_idx"), -1),
            query=str(row.get("query") or ""),
            ground_truth_ids=frozenset(
                _ordered_unique(row.get("ground_truth_ids") or [])
            ),
        )
    if not contexts:
        raise ValueError(f"no query contexts found in {path}")
    return contexts


def load_expanded_endpoint_path_overrides(
    path: Path,
) -> Tuple[Dict[str, Dict[str, Tuple[int, float]]], Dict[str, Any]]:
    """Count each graph edge only at its expanded endpoint, then event-minmax.

    The saved path feature is an undirected degree: each retained edge adds one
    neighbor to both the seed and expanded endpoints.  This alternative uses
    ``expansion_path_count`` from the full materialized rows, which counts the
    incoming provenance edges for the expanded role.  Pure seeds therefore
    receive zero; a seed that was also reached as an expansion keeps only its
    incoming expanded-role count.
    """

    raw_by_event: Dict[str, Dict[str, int]] = {}
    candidate_type_counts: Counter[str] = Counter()
    raw_path_count_distribution: Counter[int] = Counter()
    is_seed_count = 0
    is_expanded_count = 0
    for row in _iter_jsonl(path):
        event_id = str(row.get("retrieval_event_id") or "")
        paper_id = _paper_id(row.get("paper_arxiv_id"))
        if not event_id or not paper_id:
            raise ValueError(f"missing event/paper ID in Graph paper row: {path}")
        event_values = raw_by_event.setdefault(event_id, {})
        if paper_id in event_values:
            raise ValueError(f"duplicate Graph paper row for {event_id}/{paper_id}")
        is_seed = bool(row.get("is_seed"))
        is_expanded = bool(row.get("is_expanded"))
        raw_value = _as_int(row.get("expansion_path_count"), 0) if is_expanded else 0
        if raw_value < 0:
            raise ValueError(f"negative expansion_path_count for {event_id}/{paper_id}")
        event_values[paper_id] = raw_value
        candidate_type_counts[str(row.get("candidate_type") or "unknown")] += 1
        raw_path_count_distribution[raw_value] += 1
        is_seed_count += int(is_seed)
        is_expanded_count += int(is_expanded)

    overrides: Dict[str, Dict[str, Tuple[int, float]]] = {}
    nonzero_raw_count = 0
    nonzero_normalized_count = 0
    all_zero_event_count = 0
    max_raw_path_count = 0
    for event_id, raw_values in raw_by_event.items():
        values = list(raw_values.values())
        low = min(values)
        high = max(values)
        max_raw_path_count = max(max_raw_path_count, high)
        nonzero_raw_count += sum(value != 0 for value in values)
        if high - low <= 1e-12:
            normalized = {paper_id: 0.0 for paper_id in raw_values}
            all_zero_event_count += 1
        else:
            normalized = {
                paper_id: (value - low) / (high - low)
                for paper_id, value in raw_values.items()
            }
        nonzero_normalized_count += sum(value != 0.0 for value in normalized.values())
        overrides[event_id] = {
            paper_id: (raw_values[paper_id], normalized[paper_id])
            for paper_id in raw_values
        }

    return overrides, {
        "paper_row_count": sum(len(rows) for rows in raw_by_event.values()),
        "event_count": len(raw_by_event),
        "candidate_type_counts": dict(sorted(candidate_type_counts.items())),
        "raw_path_count_distribution": {
            str(value): count
            for value, count in sorted(raw_path_count_distribution.items())
        },
        "is_seed_row_count": is_seed_count,
        "is_expanded_row_count": is_expanded_count,
        "nonzero_raw_path_count": nonzero_raw_count,
        "nonzero_normalized_path_count": nonzero_normalized_count,
        "all_zero_event_count": all_zero_event_count,
        "max_raw_path_count": max_raw_path_count,
        "normalization_scope": "per retrieval event minmax",
        "edge_contribution": "seed endpoint +0; expanded endpoint +1",
    }


def semantic_score(
    row: Mapping[str, Any],
    query_weight: float = 0.30,
    subquery_weight: float = 0.40,
    intent_weight: float = 0.15,
    path_weight: float = 0.15,
) -> float:
    score = (
        float(query_weight) * float(row.get("query_score_normalized") or 0.0)
        + float(subquery_weight) * float(row.get("subquery_score_normalized") or 0.0)
        + float(intent_weight) * float(row.get("intent_score") or 0.0)
        + float(path_weight) * float(row.get("path_count_normalized") or 0.0)
    )
    if not math.isfinite(score):
        raise ValueError(f"non-finite semantic score for paper={row.get('paper_arxiv_id')}")
    return score


def _graph_is_seed(row: Mapping[str, Any]) -> bool:
    if "is_seed" in row:
        return bool(row.get("is_seed"))
    return str(row.get("candidate_type") or "") in {"seed", "seed_and_expanded"}


def _graph_is_expanded(row: Mapping[str, Any]) -> bool:
    if "is_expanded" in row:
        return bool(row.get("is_expanded"))
    return str(row.get("candidate_type") or "") in {"expanded", "seed_and_expanded"}


def rank_graph_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    query_weight: float = 0.30,
    subquery_weight: float = 0.40,
    intent_weight: float = 0.15,
    path_weight: float = 0.15,
) -> List[Mapping[str, Any]]:
    """Rank Graph rows with the configured formula and production tie breaks."""

    return sorted(
        rows,
        key=lambda row: (
            -semantic_score(
                row,
                query_weight,
                subquery_weight,
                intent_weight,
                path_weight,
            ),
            -int(_graph_is_seed(row)),
            _as_int(row.get("observed_retrieval_rank"), 10**12)
            if row.get("observed_retrieval_rank") is not None
            else 10**12,
            _paper_id(row.get("paper_arxiv_id")),
        ),
    )


def rank_deep_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    query_weight: float = 0.30,
    subquery_weight: float = 0.40,
    intent_weight: float = 0.15,
    path_weight: float = 0.15,
) -> List[Mapping[str, Any]]:
    """Rank Deep rows with the configured formula and production tie breaks."""

    scorable = [row for row in rows if bool(row.get("rerankable", True))]
    return sorted(
        scorable,
        key=lambda row: (
            -semantic_score(
                row,
                query_weight,
                subquery_weight,
                intent_weight,
                path_weight,
            ),
            _as_int(row.get("deep_retrieval_rank_in_local_pool"), 10**12),
            _paper_id(row.get("paper_arxiv_id")),
        ),
    )


def _validate_unique_pool(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    paper_ids = [_paper_id(row.get("paper_arxiv_id")) for row in rows]
    if any(not paper_id for paper_id in paper_ids):
        raise ValueError(f"missing paper ID in {label}")
    if len(set(paper_ids)) != len(paper_ids):
        raise ValueError(f"duplicate paper ID in {label}")


def _selection_container() -> Dict[str, Dict[str, MutableMapping[str, Set[str]]]]:
    return {
        formula: {method: defaultdict(set) for method in METHODS}
        for formula in FORMULAS
    }


def _event_record(
    *,
    formula: str,
    method: str,
    budget: BaselineBudget,
    pool_size: int,
    selected_rows: Sequence[Mapping[str, Any]],
    start_rank: int,
    query_weight: float,
    subquery_weight: float,
    intent_weight: float = 0.0,
    path_weight: float = 0.0,
) -> Dict[str, Any]:
    output_rows = []
    for local_index, row in enumerate(selected_rows, start=1):
        output_rows.append(
            {
                "paper_id": _paper_id(row.get("paper_arxiv_id")),
                "slice_rank": local_index,
                "merged_rank": start_rank + local_index - 1,
                "rerank_score": (
                    semantic_score(
                        row,
                        query_weight,
                        subquery_weight,
                        intent_weight,
                        path_weight,
                    )
                    if formula == NEW_FORMULA
                    else row.get("rerank_score")
                ),
                "query_score_normalized": row.get("query_score_normalized"),
                "subquery_score_normalized": row.get("subquery_score_normalized"),
                "intent_score": row.get("intent_score"),
                "path_count": row.get("path_count"),
                "path_count_normalized": row.get("path_count_normalized"),
                "candidate_type": row.get("candidate_type"),
                "is_seed": _graph_is_seed(row) if method == "graph" else None,
                "is_expanded": (
                    _graph_is_expanded(row) if method == "graph" else None
                ),
            }
        )
    return {
        "formula": formula,
        "method": method,
        "query_id": budget.query_id,
        "benchmark_idx": budget.benchmark_idx,
        "retrieval_event_id": budget.retrieval_event_id,
        "iteration_idx": budget.iteration_idx,
        "subquery_id": budget.subquery_id,
        "subquery": budget.subquery,
        "baseline_selector_input_k": budget.k,
        "pool_size": pool_size,
        "selection_start_rank": start_rank,
        "selection_end_rank": start_rank + len(selected_rows) - 1,
        "selected": output_rows,
    }


def replay_graph(
    path: Path,
    budgets: Mapping[str, BaselineBudget],
    selections: MutableMapping[str, MutableMapping[str, MutableMapping[str, Set[str]]]],
    event_rows: List[Dict[str, Any]],
    occurrence_counts: MutableMapping[Tuple[str, str], int],
    *,
    query_weight: float,
    subquery_weight: float,
    intent_weight: float = 0.0,
    path_weight: float = 0.0,
    feature_diagnostics: Optional[MutableMapping[str, Any]] = None,
    path_overrides: Optional[Mapping[str, Mapping[str, Tuple[int, float]]]] = None,
) -> Set[str]:
    seen_events: Set[str] = set()
    for pool_record in _iter_jsonl(path):
        event_id = str(pool_record.get("retrieval_event_id") or "")
        if event_id in seen_events:
            raise ValueError(f"duplicate Graph pool event {event_id}")
        seen_events.add(event_id)
        if event_id not in budgets:
            raise ValueError(f"Graph event has no Baseline budget: {event_id}")
        budget = budgets[event_id]
        if str(pool_record.get("query_id") or "") != budget.query_id:
            raise ValueError(f"Graph/Baseline query mismatch for {event_id}")
        rows = list(pool_record.get("local_pool_rows") or [])
        effective_rows: List[Mapping[str, Any]] = rows
        if path_overrides is not None:
            event_overrides = path_overrides.get(event_id)
            if event_overrides is None:
                raise ValueError(f"Graph event lacks path overrides: {event_id}")
            row_ids = {_paper_id(row.get("paper_arxiv_id")) for row in rows}
            if row_ids != set(event_overrides):
                raise ValueError(
                    f"Graph path override pool mismatch for {event_id}: "
                    f"rows={len(row_ids)}, overrides={len(event_overrides)}"
                )
            effective_rows = []
            for row in rows:
                paper_id = _paper_id(row.get("paper_arxiv_id"))
                raw_path_count, normalized_path_count = event_overrides[paper_id]
                effective_row = dict(row)
                effective_row["path_count"] = raw_path_count
                effective_row["path_count_normalized"] = normalized_path_count
                effective_rows.append(effective_row)
        if feature_diagnostics is not None:
            feature_diagnostics["graph_pool_row_occurrences"] = int(
                feature_diagnostics.get("graph_pool_row_occurrences", 0)
            ) + len(rows)
            feature_diagnostics["graph_nonzero_intent_occurrences"] = int(
                feature_diagnostics.get("graph_nonzero_intent_occurrences", 0)
            ) + sum(float(row.get("intent_score") or 0.0) != 0.0 for row in rows)
            feature_diagnostics["graph_saved_nonzero_path_occurrences"] = int(
                feature_diagnostics.get("graph_saved_nonzero_path_occurrences", 0)
            ) + sum(
                float(row.get("path_count_normalized") or 0.0) != 0.0
                for row in rows
            )
            feature_diagnostics["graph_nonzero_path_occurrences"] = int(
                feature_diagnostics.get("graph_nonzero_path_occurrences", 0)
            ) + sum(
                float(row.get("path_count_normalized") or 0.0) != 0.0
                for row in effective_rows
            )
        _validate_unique_pool(rows, f"Graph pool {event_id}")
        if len(rows) < budget.k:
            raise ValueError(
                f"Graph pool {event_id} has {len(rows)} rows, below Baseline K_i={budget.k}"
            )

        stored_rows = sorted(
            [row for row in rows if bool(row.get("in_selector_topk"))],
            key=lambda row: (
                _as_int(row.get("selector_input_rank"), 10**12),
                _as_int(row.get("rerank_rank"), 10**12),
                _paper_id(row.get("paper_arxiv_id")),
            ),
        )
        if len(stored_rows) != budget.k:
            raise ValueError(
                f"stored Graph Ret count {len(stored_rows)} != Baseline K_i={budget.k} "
                f"for {event_id}"
            )
        new_rows = rank_graph_rows(
            effective_rows,
            query_weight=query_weight,
            subquery_weight=subquery_weight,
            intent_weight=intent_weight,
            path_weight=path_weight,
        )[: budget.k]

        for formula, selected_rows in (
            (STORED_FORMULA, stored_rows),
            (NEW_FORMULA, new_rows),
        ):
            selected_ids = [_paper_id(row.get("paper_arxiv_id")) for row in selected_rows]
            selections[formula]["graph"][budget.query_id].update(selected_ids)
            occurrence_counts[(formula, "graph")] += len(selected_ids)
            event_rows.append(
                _event_record(
                    formula=formula,
                    method="graph",
                    budget=budget,
                    pool_size=len(rows),
                    selected_rows=selected_rows,
                    start_rank=1,
                    query_weight=query_weight,
                    subquery_weight=subquery_weight,
                    intent_weight=intent_weight,
                    path_weight=path_weight,
                )
            )
    return seen_events


def replay_deep_merged(
    path: Path,
    budgets: Mapping[str, BaselineBudget],
    selections: MutableMapping[str, MutableMapping[str, MutableMapping[str, Set[str]]]],
    event_rows: List[Dict[str, Any]],
    occurrence_counts: MutableMapping[Tuple[str, str], int],
    *,
    query_weight: float,
    subquery_weight: float,
    intent_weight: float = 0.0,
    path_weight: float = 0.0,
    feature_diagnostics: Optional[MutableMapping[str, Any]] = None,
) -> Set[str]:
    seen_events: Set[str] = set()
    for pool_record in _iter_jsonl(path):
        query_id = str(pool_record.get("query_id") or "")
        rows = list(pool_record.get("deep_pool_rows") or [])
        if feature_diagnostics is not None:
            feature_diagnostics["deep_pool_row_occurrences"] = int(
                feature_diagnostics.get("deep_pool_row_occurrences", 0)
            ) + len(rows)
            feature_diagnostics["deep_nonzero_intent_occurrences"] = int(
                feature_diagnostics.get("deep_nonzero_intent_occurrences", 0)
            ) + sum(float(row.get("intent_score") or 0.0) != 0.0 for row in rows)
            feature_diagnostics["deep_nonzero_path_occurrences"] = int(
                feature_diagnostics.get("deep_nonzero_path_occurrences", 0)
            ) + sum(
                float(row.get("path_count_normalized") or 0.0) != 0.0
                for row in rows
            )
        _validate_unique_pool(rows, f"Deep merged pool query={query_id}")
        slices = sorted(
            list(pool_record.get("selector_slices") or []),
            key=lambda row: _as_int(row.get("selector_slice_idx"), 10**12),
        )
        if not slices:
            raise ValueError(f"Deep merged pool has no source slices for query={query_id}")
        slice_budgets: List[BaselineBudget] = []
        for slice_row in slices:
            event_id = str(slice_row.get("retrieval_event_id") or "")
            if event_id in seen_events:
                raise ValueError(f"duplicate Deep merged source event {event_id}")
            seen_events.add(event_id)
            if event_id not in budgets:
                raise ValueError(f"Deep merged slice has no Baseline budget: {event_id}")
            budget = budgets[event_id]
            if budget.query_id != query_id:
                raise ValueError(f"Deep merged/Baseline query mismatch for {event_id}")
            stored_ids = _ordered_unique(slice_row.get("selector_input_arxiv_ids") or [])
            if len(stored_ids) != budget.k:
                raise ValueError(
                    f"stored Deep merged Ret count {len(stored_ids)} != Baseline K_i={budget.k} "
                    f"for {event_id}"
                )
            slice_budgets.append(budget)

        row_by_id = {_paper_id(row.get("paper_arxiv_id")): row for row in rows}
        ranked = rank_deep_rows(
            rows,
            query_weight=query_weight,
            subquery_weight=subquery_weight,
            intent_weight=intent_weight,
            path_weight=path_weight,
        )
        required = sum(budget.k for budget in slice_budgets)
        if len(ranked) < required:
            raise ValueError(
                f"Deep merged rerankable pool for query={query_id} has {len(ranked)} rows, "
                f"below summed Baseline budget {required}"
            )

        offset = 0
        for slice_row, budget in zip(slices, slice_budgets):
            stored_ids = _ordered_unique(slice_row.get("selector_input_arxiv_ids") or [])
            missing_stored = set(stored_ids) - set(row_by_id)
            if missing_stored:
                raise ValueError(
                    f"stored Deep merged slice {budget.retrieval_event_id} references missing rows: "
                    f"{sorted(missing_stored)[:5]}"
                )
            stored_rows = [row_by_id[paper_id] for paper_id in stored_ids]
            new_rows = ranked[offset : offset + budget.k]
            for formula, selected_rows in (
                (STORED_FORMULA, stored_rows),
                (NEW_FORMULA, new_rows),
            ):
                selected_ids = [_paper_id(row.get("paper_arxiv_id")) for row in selected_rows]
                selections[formula]["deep_merged"][query_id].update(selected_ids)
                occurrence_counts[(formula, "deep_merged")] += len(selected_ids)
                event_rows.append(
                    _event_record(
                        formula=formula,
                        method="deep_merged",
                        budget=budget,
                        pool_size=len(rows),
                        selected_rows=selected_rows,
                        start_rank=offset + 1,
                        query_weight=query_weight,
                        subquery_weight=subquery_weight,
                        intent_weight=intent_weight,
                        path_weight=path_weight,
                    )
                )
            offset += budget.k
    return seen_events


def _load_saved_query_candidates(path: Path) -> Dict[str, Set[str]]:
    output: Dict[str, Set[str]] = {}
    for row in _iter_jsonl(path):
        query_id = str(row.get("query_id") or "")
        if not query_id or query_id in output:
            raise ValueError(f"missing or duplicate query result in {path}: {query_id!r}")
        output[query_id] = set(_ordered_unique(row.get("candidate_arxiv_ids") or []))
    return output


def validate_stored_query_results(
    artifact_dir: Path,
    selections: Mapping[str, Mapping[str, Mapping[str, Set[str]]]],
) -> None:
    paths = {
        "graph": artifact_dir / "per_subquery" / "query_results.jsonl",
        "deep_merged": artifact_dir / "deep_merged" / "query_results.jsonl",
    }
    for method, path in paths.items():
        expected = _load_saved_query_candidates(path)
        actual = selections[STORED_FORMULA][method]
        if set(expected) != set(actual):
            raise ValueError(
                f"stored {method} query IDs do not reproduce saved query_results.jsonl"
            )
        mismatches = [
            query_id
            for query_id in expected
            if expected[query_id] != set(actual.get(query_id, set()))
        ]
        if mismatches:
            raise ValueError(
                f"stored {method} candidates fail artifact reproduction for "
                f"{len(mismatches)} queries; first={mismatches[0]}"
            )


def query_metric_rows(
    contexts: Mapping[str, QueryContext],
    selections: Mapping[str, Mapping[str, Mapping[str, Set[str]]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for formula in FORMULAS:
        for method in METHODS:
            for context in sorted(contexts.values(), key=lambda value: value.benchmark_idx):
                candidate_ids = set(selections[formula][method].get(context.query_id, set()))
                gt_ids = candidate_ids & set(context.ground_truth_ids)
                recall = _safe_rate(len(gt_ids), len(context.ground_truth_ids))
                precision = _safe_rate(len(gt_ids), len(candidate_ids))
                rows.append(
                    {
                        "formula": formula,
                        "method": method,
                        "benchmark_idx": context.benchmark_idx,
                        "query_id": context.query_id,
                        "gt_count": len(context.ground_truth_ids),
                        "candidate_count": len(candidate_ids),
                        "candidate_gt_count": len(gt_ids),
                        "candidate_recall": recall,
                        "candidate_precision": precision,
                        "candidate_f1": _f1(recall, precision),
                        "candidate_ids": sorted(candidate_ids),
                        "candidate_gt_ids": sorted(gt_ids),
                    }
                )
    return rows


def retrieval_summary_rows(
    query_rows: Sequence[Mapping[str, Any]],
    occurrence_counts: Mapping[Tuple[str, str], int],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in query_rows:
        grouped[(str(row["formula"]), str(row["method"]))].append(row)
    output = []
    for formula in FORMULAS:
        for method in METHODS:
            rows = grouped[(formula, method)]
            total_gt = sum(_as_int(row["gt_count"]) for row in rows)
            total_candidates = sum(_as_int(row["candidate_count"]) for row in rows)
            total_candidate_gt = sum(_as_int(row["candidate_gt_count"]) for row in rows)
            output.append(
                {
                    "formula": formula,
                    "method": method,
                    "query_count": len(rows),
                    "baseline_budget_occurrence_count": occurrence_counts[(formula, method)],
                    "candidate_count": total_candidates,
                    "candidate_gt_count": total_candidate_gt,
                    "micro_recall": _safe_rate(total_candidate_gt, total_gt),
                    "micro_precision": _safe_rate(total_candidate_gt, total_candidates),
                    "micro_f1": _f1(
                        _safe_rate(total_candidate_gt, total_gt),
                        _safe_rate(total_candidate_gt, total_candidates),
                    ),
                    "macro_recall": sum(float(row["candidate_recall"]) for row in rows)
                    / len(rows),
                    "macro_precision": sum(
                        float(row["candidate_precision"]) for row in rows
                    )
                    / len(rows),
                    "macro_f1": sum(float(row["candidate_f1"]) for row in rows)
                    / len(rows),
                }
            )
    return output


def partition_rows(
    contexts: Mapping[str, QueryContext],
    selections: Mapping[str, Mapping[str, Mapping[str, Set[str]]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    query_rows = []
    summary_rows = []
    for formula in FORMULAS:
        aggregate = Counter()
        jaccards = []
        for context in sorted(contexts.values(), key=lambda value: value.benchmark_idx):
            graph = set(selections[formula]["graph"].get(context.query_id, set()))
            deep = set(selections[formula]["deep_merged"].get(context.query_id, set()))
            graph_only = graph - deep
            deep_only = deep - graph
            overlap = graph & deep
            union = graph | deep
            gt = set(context.ground_truth_ids)
            jaccard = _safe_rate(len(overlap), len(union))
            jaccards.append(jaccard)
            row = {
                "formula": formula,
                "benchmark_idx": context.benchmark_idx,
                "query_id": context.query_id,
                "graph_count": len(graph),
                "deep_merged_count": len(deep),
                "graph_only_count": len(graph_only),
                "deep_merged_only_count": len(deep_only),
                "overlap_count": len(overlap),
                "union_count": len(union),
                "jaccard": jaccard,
                "graph_gt_count": len(graph & gt),
                "deep_merged_gt_count": len(deep & gt),
                "graph_only_gt_count": len(graph_only & gt),
                "deep_merged_only_gt_count": len(deep_only & gt),
                "overlap_gt_count": len(overlap & gt),
                "union_gt_count": len(union & gt),
                "graph_only_ids": sorted(graph_only),
                "deep_merged_only_ids": sorted(deep_only),
                "overlap_ids": sorted(overlap),
            }
            query_rows.append(row)
            for key, value in row.items():
                if key.endswith("_count") and key not in {"benchmark_idx"}:
                    aggregate[key] += int(value)
        summary_rows.append(
            {
                "formula": formula,
                **dict(aggregate),
                "micro_jaccard": _safe_rate(
                    aggregate["overlap_count"], aggregate["union_count"]
                ),
                "macro_query_jaccard": sum(jaccards) / len(jaccards),
            }
        )
    return query_rows, summary_rows


def churn_rows(
    contexts: Mapping[str, QueryContext],
    selections: Mapping[str, Mapping[str, Mapping[str, Set[str]]]],
) -> List[Dict[str, Any]]:
    output = []
    for method in METHODS:
        aggregate = Counter()
        per_query_jaccards = []
        for context in contexts.values():
            old = set(selections[STORED_FORMULA][method].get(context.query_id, set()))
            new = set(selections[NEW_FORMULA][method].get(context.query_id, set()))
            retained = old & new
            added = new - old
            removed = old - new
            union = old | new
            gt = set(context.ground_truth_ids)
            aggregate["stored_candidate_count"] += len(old)
            aggregate["new_candidate_count"] += len(new)
            aggregate["retained_count"] += len(retained)
            aggregate["added_count"] += len(added)
            aggregate["removed_count"] += len(removed)
            aggregate["union_count"] += len(union)
            aggregate["stored_gt_count"] += len(old & gt)
            aggregate["new_gt_count"] += len(new & gt)
            aggregate["added_gt_count"] += len(added & gt)
            aggregate["removed_gt_count"] += len(removed & gt)
            per_query_jaccards.append(_safe_rate(len(retained), len(union)))
        output.append(
            {
                "method": method,
                **dict(aggregate),
                "candidate_count_delta": aggregate["new_candidate_count"]
                - aggregate["stored_candidate_count"],
                "gt_count_delta": aggregate["new_gt_count"] - aggregate["stored_gt_count"],
                "micro_jaccard": _safe_rate(
                    aggregate["retained_count"], aggregate["union_count"]
                ),
                "macro_query_jaccard": sum(per_query_jaccards)
                / len(per_query_jaccards),
            }
        )
    return output


def _primary_partition(sources: Iterable[str]) -> str:
    source_set = set(sources)
    if source_set == {"graph"}:
        return "graph_only"
    if source_set == {"deep_merged"}:
        return "deep_merged_only"
    if source_set == {"graph", "deep_merged"}:
        return "graph_and_deep_merged"
    raise ValueError(f"unexpected primary sources: {sorted(source_set)}")


def _membership_maps(
    selections: Mapping[str, Mapping[str, Mapping[str, Set[str]]]],
) -> Dict[str, Dict[Tuple[str, str], Tuple[str, ...]]]:
    output: Dict[str, Dict[Tuple[str, str], Tuple[str, ...]]] = {}
    for formula in FORMULAS:
        membership: Dict[Tuple[str, str], Tuple[str, ...]] = {}
        query_ids = set(selections[formula]["graph"]) | set(
            selections[formula]["deep_merged"]
        )
        for query_id in query_ids:
            graph = set(selections[formula]["graph"].get(query_id, set()))
            deep = set(selections[formula]["deep_merged"].get(query_id, set()))
            for paper_id in graph | deep:
                sources = tuple(
                    source
                    for source in METHODS
                    if paper_id in selections[formula][source].get(query_id, set())
                )
                membership[(query_id, paper_id)] = sources
        output[formula] = membership
    return output


def build_annotation_workspaces(
    annotation_work_dir: Path,
    output_dir: Path,
    memberships: Mapping[str, Mapping[Tuple[str, str], Tuple[str, ...]]],
) -> Tuple[
    Dict[str, Path],
    Dict[Tuple[str, str], Mapping[str, Any]],
    Dict[Tuple[str, str], Tuple[str, ...]],
]:
    """Create small, auditable Top-K annotation views without relabeling papers."""

    workspaces: Dict[str, Path] = {}
    candidate_temp: Dict[str, Path] = {}
    annotation_temp: Dict[str, Path] = {}
    candidate_handles = {}
    annotation_handles = {}
    seen: Dict[str, Set[Tuple[str, str]]] = {formula: set() for formula in FORMULAS}
    selected_annotations: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    original_primary_sources: Dict[Tuple[str, str], Tuple[str, ...]] = {}

    for formula in FORMULAS:
        workspace = output_dir / "annotation_views" / formula
        workspaces[formula] = workspace
        (workspace / "manifest").mkdir(parents=True, exist_ok=True)
        (workspace / "analysis").mkdir(parents=True, exist_ok=True)
        (workspace / "outputs" / "rubrics").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            annotation_work_dir / "manifest" / "queries.jsonl",
            workspace / "manifest" / "queries.jsonl",
        )
        for rubric_path in (annotation_work_dir / "outputs" / "rubrics").glob("*.json"):
            shutil.copy2(rubric_path, workspace / "outputs" / "rubrics" / rubric_path.name)
        candidate_path = workspace / "manifest" / "candidates.jsonl"
        annotation_path = workspace / "analysis" / "annotations.jsonl"
        candidate_temp[formula] = candidate_path.with_name(
            f".{candidate_path.name}.tmp-{os.getpid()}"
        )
        annotation_temp[formula] = annotation_path.with_name(
            f".{annotation_path.name}.tmp-{os.getpid()}"
        )
        candidate_handles[formula] = candidate_temp[formula].open("w", encoding="utf-8")
        annotation_handles[formula] = annotation_temp[formula].open("w", encoding="utf-8")

    candidate_iter = _iter_jsonl(annotation_work_dir / "manifest" / "candidates.jsonl")
    annotation_iter = _iter_jsonl(annotation_work_dir / "analysis" / "annotations.jsonl")
    try:
        for candidate, annotation_row in itertools.zip_longest(candidate_iter, annotation_iter):
            if candidate is None or annotation_row is None:
                raise ValueError("first-stage candidate and annotation files differ in length")
            if candidate.get("candidate_id") != annotation_row.get("candidate_id"):
                raise ValueError("first-stage candidate and annotation order/IDs do not match")
            key = (str(candidate.get("query_id") or ""), _paper_id(candidate.get("paper_id")))
            for formula in FORMULAS:
                sources = memberships[formula].get(key)
                if not sources:
                    continue
                if key in seen[formula]:
                    raise ValueError(f"duplicate selected candidate in annotation manifest: {key}")
                seen[formula].add(key)
                partition = _primary_partition(sources)
                candidate_output = dict(candidate)
                candidate_output["sources"] = list(sources)
                candidate_output["source_partition"] = partition
                original_stats = candidate.get("source_stats") or {}
                candidate_output["source_stats"] = {
                    source: original_stats[source]
                    for source in sources
                    if source in original_stats
                }
                annotation_output = dict(annotation_row)
                annotation_output["sources"] = list(sources)
                annotation_output["source_partition"] = partition
                candidate_handles[formula].write(
                    json.dumps(candidate_output, ensure_ascii=False, sort_keys=True) + "\n"
                )
                annotation_handles[formula].write(
                    json.dumps(annotation_output, ensure_ascii=False, sort_keys=True) + "\n"
                )
                selected_annotations[key] = annotation_row.get("annotation") or {}
                original_sources = tuple(
                    source
                    for source in METHODS
                    if source in set(candidate.get("sources") or [])
                )
                previous_sources = original_primary_sources.get(key)
                if previous_sources is not None and previous_sources != original_sources:
                    raise ValueError(f"inconsistent original source provenance for {key}")
                original_primary_sources[key] = original_sources
    finally:
        for handle in candidate_handles.values():
            handle.close()
        for handle in annotation_handles.values():
            handle.close()

    for formula in FORMULAS:
        missing = set(memberships[formula]) - seen[formula]
        if missing:
            raise ValueError(
                f"{len(missing)} selected {formula} query-paper pairs lack first-stage annotations; "
                f"first={sorted(missing)[0]}"
            )
        candidate_path = workspaces[formula] / "manifest" / "candidates.jsonl"
        annotation_path = workspaces[formula] / "analysis" / "annotations.jsonl"
        os.replace(candidate_temp[formula], candidate_path)
        os.replace(annotation_temp[formula], annotation_path)
        _atomic_write_json(
            workspaces[formula] / "analysis" / "summary.json",
            {
                "complete": True,
                "candidate_count": len(seen[formula]),
                "derived_view": True,
                "formula": formula,
            },
        )
    return workspaces, selected_annotations, original_primary_sources


def run_semantic_analyses(
    workspaces: Mapping[str, Path],
    output_dir: Path,
    *,
    bootstrap_samples: int,
    formula_description: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    summaries: Dict[str, Any] = {}
    all_group_rows: List[Dict[str, Any]] = []
    all_semantic_rows: List[Dict[str, Any]] = []
    for formula in FORMULAS:
        scope = {
            "report_title": (
                "Graph expansion vs Deep merged — stored pipeline Ret budget"
                if formula == STORED_FORMULA
                else f"Graph expansion vs Deep merged — Baseline-budget Ret Top-K ({formula_description})"
            ),
            "candidate_scope": (
                "query-local union after per-event Ret truncation at Baseline Selector-input K_i; "
                "no Selector decisions applied"
            ),
            "unit_description": (
                "The unit is a query-local deduplicated paper after per-event Ret truncation at "
                "the actual Baseline Selector-input budget K_i. Selector output is not used."
            ),
            "interpretation_boundary": (
                "These are descriptive, model-annotated Ret-stage results using saved scores. "
                "They isolate rerank-and-truncation effects, not retrieval, embedding, Selector, "
                "or causal edge-type effects."
            ),
        }
        analysis_dir = output_dir / "semantic_analysis" / formula
        summaries[formula] = semantic_analysis.analyze(
            workspaces[formula],
            output_dir=analysis_dir,
            bootstrap_samples=bootstrap_samples,
            rarefaction_samples=0,
            analysis_scope=scope,
        )
        for row in _iter_jsonl(analysis_dir / "group_summary.jsonl"):
            all_group_rows.append({"formula": formula, **row})
        for row in _iter_jsonl(analysis_dir / "semantic_complement.jsonl"):
            all_semantic_rows.append({"formula": formula, **row})
    return summaries, all_group_rows, all_semantic_rows


def fine_relation_rows(
    refinement_path: Path,
    memberships: Mapping[str, Mapping[Tuple[str, str], Tuple[str, ...]]],
    first_stage_annotations: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    counts: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    missing_records: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    def groups_for(sources: Sequence[str]) -> List[str]:
        return ["union", _primary_partition(sources), *sources]

    # The expected denominator comes from complete first-stage labels.  The
    # original second stage intentionally covered only candidates exclusive in
    # the *full* pools, whereas a Top-K replay can create new exclusive rows.
    expected: Dict[Tuple[str, str, str], Set[Tuple[str, str]]] = defaultdict(set)
    for formula in FORMULAS:
        for key, sources in memberships[formula].items():
            annotation = first_stage_annotations.get(key)
            if annotation is None:
                raise ValueError(f"missing first-stage annotation for fine relation denominator: {key}")
            information = set(
                annotation.get(
                    "information_added", annotation.get("contribution_types", [])
                )
                or []
            )
            for axis, trigger in AXIS_TRIGGERS.items():
                if trigger not in information:
                    continue
                for group in groups_for(sources):
                    expected[(formula, group, axis)].add(key)

    observed: Dict[Tuple[str, str, str], Set[Tuple[str, str]]] = defaultdict(set)
    for row in _iter_jsonl(refinement_path):
        key = (str(row.get("query_id") or ""), _paper_id(row.get("paper_id")))
        annotation = row.get("annotation") or {}
        applicable_axes = set(row.get("applicable_axes") or [])
        for formula in FORMULAS:
            sources = memberships[formula].get(key)
            if not sources:
                continue
            partition = _primary_partition(sources)
            groups = ["union", partition, *sources]
            for group in groups:
                for axis in AXIS_LABELS:
                    axis_value = annotation.get(axis) or {}
                    label = str(axis_value.get("label") or "")
                    applicable = axis in applicable_axes or label not in {"", "not_applicable"}
                    if not applicable:
                        continue
                    if key not in expected[(formula, group, axis)]:
                        raise ValueError(
                            f"second-stage axis {axis} is not triggered by the selected first-stage "
                            f"annotation for {formula}/{group}/{key}"
                        )
                    counter = counts[(formula, group, axis)]
                    counter[f"label:{label}"] += 1
                    observed[(formula, group, axis)].add(key)
                    if label != "insufficient_evidence":
                        counter["conclusive_count"] += 1

    output: List[Dict[str, Any]] = []
    groups = (
        "union",
        "graph",
        "deep_merged",
        "graph_only",
        "deep_merged_only",
        "graph_and_deep_merged",
    )
    for formula in FORMULAS:
        for group in groups:
            for axis, labels in AXIS_LABELS.items():
                counter = counts[(formula, group, axis)]
                triggered_keys = expected[(formula, group, axis)]
                refined_keys = observed[(formula, group, axis)]
                unexpected = refined_keys - triggered_keys
                if unexpected:
                    raise ValueError(
                        f"unexpected refined candidates for {formula}/{group}/{axis}: "
                        f"{sorted(unexpected)[0]}"
                    )
                missing_keys = triggered_keys - refined_keys
                applicable = len(triggered_keys)
                refined = len(refined_keys)
                conclusive = counter["conclusive_count"]
                manifest_missing_keys = (
                    missing_keys
                    if group in {"graph_only", "deep_merged_only"}
                    else set()
                )
                for query_id, paper_id in manifest_missing_keys:
                    missing_key = (formula, group, query_id, paper_id)
                    record = missing_records.setdefault(
                        missing_key,
                        {
                            "formula": formula,
                            "group": group,
                            "query_id": query_id,
                            "paper_id": paper_id,
                            "missing_axes": [],
                        },
                    )
                    record["missing_axes"].append(axis)
                for label in labels:
                    paper_count = counter[f"label:{label}"]
                    output.append(
                        {
                            "formula": formula,
                            "group": group,
                            "axis": axis,
                            "label": label,
                            "applicable_count": applicable,
                            "triggered_count": applicable,
                            "refined_count": refined,
                            "missing_refinement_count": len(missing_keys),
                            "refinement_coverage_rate": (
                                _safe_rate(refined, applicable) if applicable else None
                            ),
                            "conclusive_count": conclusive,
                            "paper_count": paper_count,
                            "rate_among_applicable": (
                                _safe_rate(paper_count, applicable) if applicable else None
                            ),
                            "rate_among_refined": (
                                _safe_rate(paper_count, refined) if refined else None
                            ),
                            "rate_among_conclusive": (
                                _safe_rate(paper_count, conclusive)
                                if conclusive and label != "insufficient_evidence"
                                else None
                            ),
                        }
                    )
    missing_rows = sorted(
        (
            {**row, "missing_axes": sorted(set(row["missing_axes"]))}
            for row in missing_records.values()
        ),
        key=lambda row: (
            row["formula"],
            row["group"],
            row["query_id"],
            row["paper_id"],
        ),
    )
    return output, missing_rows


def _strict_provenance_key_sets(
    selections: Mapping[str, Mapping[str, Mapping[str, Set[str]]]],
    original_sources: Mapping[Tuple[str, str], Tuple[str, ...]],
) -> Dict[Tuple[str, str], Set[Tuple[str, str]]]:
    """Selected candidates that are genuinely unique in the saved full pools."""

    output: Dict[Tuple[str, str], Set[Tuple[str, str]]] = {}
    specs = (
        ("graph_pool_unique_selected", "graph", ("graph",)),
        ("deep_pool_unique_selected", "deep_merged", ("deep_merged",)),
    )
    for formula in FORMULAS:
        for group, method, required_sources in specs:
            selected = {
                (query_id, paper_id)
                for query_id, paper_ids in selections[formula][method].items()
                for paper_id in paper_ids
            }
            output[(formula, group)] = {
                key for key in selected if original_sources.get(key) == required_sources
            }
    return output


def provenance_semantic_rows(
    contexts: Mapping[str, QueryContext],
    selections: Mapping[str, Mapping[str, Mapping[str, Set[str]]]],
    annotations: Mapping[Tuple[str, str], Mapping[str, Any]],
    original_sources: Mapping[Tuple[str, str], Tuple[str, ...]],
) -> List[Dict[str, Any]]:
    key_sets = _strict_provenance_key_sets(selections, original_sources)
    output = []
    relevance_scores = {
        "direct": 1.0,
        "partial": 2.0 / 3.0,
        "contextual": 1.0 / 3.0,
        "unrelated": 0.0,
    }
    for formula in FORMULAS:
        for group in ("graph_pool_unique_selected", "deep_pool_unique_selected"):
            keys = key_sets[(formula, group)]
            missing = keys - set(annotations)
            if missing:
                raise ValueError(
                    f"strict provenance candidates lack first-stage annotations: {sorted(missing)[0]}"
                )
            grades = Counter()
            information = Counter()
            roles = Counter()
            gt_count = 0
            relevance_total = 0.0
            for query_id, paper_id in keys:
                annotation = annotations[(query_id, paper_id)]
                grade = _annotation_grade(annotation)
                grades[grade] += 1
                relevance_total += relevance_scores.get(grade, 0.0)
                information.update(
                    str(value)
                    for value in (
                        annotation.get(
                            "information_added",
                            annotation.get("contribution_types", []),
                        )
                        or []
                    )
                )
                roles.update(
                    str(value)
                    for value in (
                        annotation.get(
                            "scholarly_roles", annotation.get("relation_roles", [])
                        )
                        or []
                    )
                )
                gt_count += int(paper_id in contexts[query_id].ground_truth_ids)
            count = len(keys)
            information_count = sum(grades[grade] for grade in INFORMATION_GRADES)
            output.append(
                {
                    "formula": formula,
                    "group": group,
                    "candidate_count": count,
                    "ground_truth_count": gt_count,
                    "ground_truth_rate": _safe_rate(gt_count, count),
                    "direct_count": grades["direct"],
                    "direct_rate": _safe_rate(grades["direct"], count),
                    "direct_or_partial_count": grades["direct"] + grades["partial"],
                    "direct_or_partial_rate": _safe_rate(
                        grades["direct"] + grades["partial"], count
                    ),
                    "information_bearing_count": information_count,
                    "information_bearing_rate": _safe_rate(information_count, count),
                    "mean_relevance_score": _safe_rate(relevance_total, count),
                    "historical_context_count": information["historical_context"],
                    "historical_context_pool_rate": _safe_rate(
                        information["historical_context"], count
                    ),
                    "mechanism_or_theory_count": information["mechanism_or_theory"],
                    "mechanism_or_theory_pool_rate": _safe_rate(
                        information["mechanism_or_theory"], count
                    ),
                    "application_domain_count": information["application_domain"],
                    "application_domain_pool_rate": _safe_rate(
                        information["application_domain"], count
                    ),
                    "method_component_count": roles["method_component"],
                    "method_component_pool_rate": _safe_rate(
                        roles["method_component"], count
                    ),
                    "task_or_application_count": roles["task_or_application"],
                    "task_or_application_pool_rate": _safe_rate(
                        roles["task_or_application"], count
                    ),
                    "background_or_foundation_count": roles["background_or_foundation"],
                    "background_or_foundation_pool_rate": _safe_rate(
                        roles["background_or_foundation"], count
                    ),
                }
            )
    return output


def provenance_fine_relation_rows(
    refinement_path: Path,
    selections: Mapping[str, Mapping[str, Mapping[str, Set[str]]]],
    first_stage_annotations: Mapping[Tuple[str, str], Mapping[str, Any]],
    original_sources: Mapping[Tuple[str, str], Tuple[str, ...]],
) -> List[Dict[str, Any]]:
    """Fine labels on full-pool-provenance exclusive selected candidates.

    This is the strict source-complement analysis.  Unlike Top-K selection
    exclusivity, its triggered candidates are exactly inside the completed
    second-stage annotation universe.
    """

    key_sets = _strict_provenance_key_sets(selections, original_sources)
    expected: Dict[Tuple[str, str, str], Set[Tuple[str, str]]] = defaultdict(set)
    for (formula, group), keys in key_sets.items():
        for key in keys:
            annotation = first_stage_annotations[key]
            information = set(
                annotation.get(
                    "information_added", annotation.get("contribution_types", [])
                )
                or []
            )
            for axis, trigger in AXIS_TRIGGERS.items():
                if trigger in information:
                    expected[(formula, group, axis)].add(key)

    counts: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    observed: Dict[Tuple[str, str, str], Set[Tuple[str, str]]] = defaultdict(set)
    for row in _iter_jsonl(refinement_path):
        key = (str(row.get("query_id") or ""), _paper_id(row.get("paper_id")))
        annotation = row.get("annotation") or {}
        applicable_axes = set(row.get("applicable_axes") or [])
        for formula in FORMULAS:
            for group in ("graph_pool_unique_selected", "deep_pool_unique_selected"):
                if key not in key_sets[(formula, group)]:
                    continue
                for axis in AXIS_LABELS:
                    axis_value = annotation.get(axis) or {}
                    label = str(axis_value.get("label") or "")
                    applicable = axis in applicable_axes or label not in {"", "not_applicable"}
                    if not applicable:
                        continue
                    if key not in expected[(formula, group, axis)]:
                        raise ValueError(
                            f"strict provenance refinement axis mismatch for {formula}/{group}/{axis}/{key}"
                        )
                    counts[(formula, group, axis)][f"label:{label}"] += 1
                    if label != "insufficient_evidence":
                        counts[(formula, group, axis)]["conclusive_count"] += 1
                    observed[(formula, group, axis)].add(key)

    output = []
    for formula in FORMULAS:
        for group in ("graph_pool_unique_selected", "deep_pool_unique_selected"):
            for axis, labels in AXIS_LABELS.items():
                expected_keys = expected[(formula, group, axis)]
                observed_keys = observed[(formula, group, axis)]
                missing = expected_keys - observed_keys
                if missing:
                    raise ValueError(
                        f"strict provenance fine-label coverage is incomplete for "
                        f"{formula}/{group}/{axis}: first={sorted(missing)[0]}"
                    )
                counter = counts[(formula, group, axis)]
                applicable = len(expected_keys)
                conclusive = counter["conclusive_count"]
                for label in labels:
                    paper_count = counter[f"label:{label}"]
                    output.append(
                        {
                            "formula": formula,
                            "group": group,
                            "axis": axis,
                            "label": label,
                            "applicable_count": applicable,
                            "refined_count": len(observed_keys),
                            "refinement_coverage_rate": (
                                _safe_rate(len(observed_keys), applicable)
                                if applicable
                                else None
                            ),
                            "conclusive_count": conclusive,
                            "paper_count": paper_count,
                            "rate_among_applicable": (
                                _safe_rate(paper_count, applicable) if applicable else None
                            ),
                            "rate_among_conclusive": (
                                _safe_rate(paper_count, conclusive)
                                if conclusive and label != "insufficient_evidence"
                                else None
                            ),
                        }
                    )
    return output


def _annotation_grade(annotation: Mapping[str, Any]) -> str:
    return str(annotation.get("relevance_grade", annotation.get("relationship", "")))


def annotated_churn_rows(
    contexts: Mapping[str, QueryContext],
    selections: Mapping[str, Mapping[str, Mapping[str, Set[str]]]],
    annotations: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    def profile(keys: Set[Tuple[str, str]]) -> Dict[str, Any]:
        grades = Counter()
        information_types = Counter()
        gt_count = 0
        for key in keys:
            query_id, paper_id = key
            annotation = annotations[key]
            grade = _annotation_grade(annotation)
            grades[grade] += 1
            gt_count += int(paper_id in contexts[query_id].ground_truth_ids)
            values = annotation.get("information_added", annotation.get("contribution_types", [])) or []
            information_types.update(str(value) for value in values)
        count = len(keys)
        return {
            "candidate_count": count,
            "gt_count": gt_count,
            "direct_count": grades["direct"],
            "direct_or_partial_count": grades["direct"] + grades["partial"],
            "information_bearing_count": sum(grades[grade] for grade in INFORMATION_GRADES),
            "direct_rate": _safe_rate(grades["direct"], count),
            "direct_or_partial_rate": _safe_rate(
                grades["direct"] + grades["partial"], count
            ),
            "information_bearing_rate": _safe_rate(
                sum(grades[grade] for grade in INFORMATION_GRADES), count
            ),
            "historical_context_count": information_types["historical_context"],
            "mechanism_or_theory_count": information_types["mechanism_or_theory"],
            "application_domain_count": information_types["application_domain"],
        }

    output = []
    for method in METHODS:
        old = {
            (query_id, paper_id)
            for query_id, paper_ids in selections[STORED_FORMULA][method].items()
            for paper_id in paper_ids
        }
        new = {
            (query_id, paper_id)
            for query_id, paper_ids in selections[NEW_FORMULA][method].items()
            for paper_id in paper_ids
        }
        for change, keys in (("added", new - old), ("removed", old - new)):
            missing = keys - set(annotations)
            if missing:
                raise ValueError(f"missing first-stage annotations for churn set: {sorted(missing)[0]}")
            output.append({"method": method, "change": change, **profile(keys)})
    return output


def _pct(value: Any) -> str:
    return "NA" if value is None else f"{100.0 * float(value):.2f}%"


def build_report(
    *,
    budgets: Mapping[str, BaselineBudget],
    retrieval_rows: Sequence[Mapping[str, Any]],
    partition_summary: Sequence[Mapping[str, Any]],
    churn: Sequence[Mapping[str, Any]],
    annotated_churn: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    fine_rows: Sequence[Mapping[str, Any]],
    provenance_semantic: Sequence[Mapping[str, Any]],
    provenance_fine: Sequence[Mapping[str, Any]],
    formula_description: str,
) -> str:
    retrieval = {(row["formula"], row["method"]): row for row in retrieval_rows}
    partitions = {row["formula"]: row for row in partition_summary}
    group = {(row["formula"], row["group"]): row for row in group_rows}
    k_distribution = Counter(budget.k for budget in budgets.values())
    lines = [
        "# Baseline-budget Ret rerank: Graph vs Deep merged",
        "",
        f"Replay formula: `{formula_description}`.",
        "Each event uses `K_i = len(Baseline selector_decision.candidate_rows)`. No Selector is run.",
        f"Validated events: **{len(budgets):,}**; occurrence budget: "
        f"**{sum(budget.k for budget in budgets.values()):,}**; K distribution: "
        + ", ".join(f"K={key}: {value}" for key, value in sorted(k_distribution.items()))
        + ".",
        "",
        "## Retrieval outcome",
        "",
        "| Formula | Method | Query-paper candidates | GT | Macro recall | Macro precision | Macro F1 | Micro recall |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for formula in FORMULAS:
        for method in METHODS:
            row = retrieval[(formula, method)]
            lines.append(
                f"| {formula} | {method} | {row['candidate_count']:,} | "
                f"{row['candidate_gt_count']:,} | {_pct(row['macro_recall'])} | "
                f"{_pct(row['macro_precision'])} | {_pct(row['macro_f1'])} | "
                f"{_pct(row['micro_recall'])} |"
            )
    lines.extend(
        [
            "",
            "## Graph–Deep partition after query-local union",
            "",
            "| Formula | Graph only | Deep only | Overlap | Union | Jaccard | Graph-only GT | Deep-only GT | Overlap GT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for formula in FORMULAS:
        row = partitions[formula]
        lines.append(
            f"| {formula} | {row['graph_only_count']:,} | "
            f"{row['deep_merged_only_count']:,} | {row['overlap_count']:,} | "
            f"{row['union_count']:,} | {_pct(row['micro_jaccard'])} | "
            f"{row['graph_only_gt_count']:,} | {row['deep_merged_only_gt_count']:,} | "
            f"{row['overlap_gt_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Formula churn",
            "",
            "| Method | Added | Removed | Retained | Jaccard | GT delta | Added GT | Removed GT |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in churn:
        lines.append(
            f"| {row['method']} | {row['added_count']:,} | {row['removed_count']:,} | "
            f"{row['retained_count']:,} | {_pct(row['micro_jaccard'])} | "
            f"{row['gt_count_delta']:+,} | {row['added_gt_count']:,} | "
            f"{row['removed_gt_count']:,} |"
        )
    lines.extend(
        [
            "",
            "## Semantic properties of exclusive sets",
            "",
            "| Formula | Partition | Candidates | Direct | Direct+Partial | Information-bearing | Mean relevance |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for formula in FORMULAS:
        for partition in ("graph_only", "deep_merged_only", "graph_and_deep_merged"):
            row = group[(formula, partition)]
            lines.append(
                f"| {formula} | {partition} | {row['candidate_count']:,} | "
                f"{_pct(row['direct_rate'])} | {_pct(row['direct_or_partial_rate'])} | "
                f"{_pct(row['information_bearing_rate'])} | "
                f"{float(row['mean_relevance_score']):.4f} |"
            )
    provenance_semantic_index = {
        (row["formula"], row["group"]): row for row in provenance_semantic
    }
    lines.extend(
        [
            "",
            "## Strict retrieval-source complement among Ret-selected papers",
            "",
            "Here `unique` means absent from the other method's complete saved candidate pool, "
            "not merely absent after Top-K truncation.",
            "",
            "| Formula | Full-pool provenance | Selected candidates | GT | Direct | Direct+Partial | Information-bearing | Historical | Mechanism/theory | Application domain |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for formula in FORMULAS:
        for provenance_group in (
            "graph_pool_unique_selected",
            "deep_pool_unique_selected",
        ):
            row = provenance_semantic_index[(formula, provenance_group)]
            lines.append(
                f"| {formula} | {provenance_group} | {row['candidate_count']:,} | "
                f"{row['ground_truth_count']:,} | {_pct(row['direct_rate'])} | "
                f"{_pct(row['direct_or_partial_rate'])} | "
                f"{_pct(row['information_bearing_rate'])} | "
                f"{_pct(row['historical_context_pool_rate'])} | "
                f"{_pct(row['mechanism_or_theory_pool_rate'])} | "
                f"{_pct(row['application_domain_pool_rate'])} |"
            )
    lines.extend(
        [
            "",
            f"## Papers entering/leaving under {NEW_FORMULA}",
            "",
            "| Method | Change | Candidates | GT | Direct | Direct+Partial | Information-bearing |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in annotated_churn:
        lines.append(
            f"| {row['method']} | {row['change']} | {row['candidate_count']:,} | "
            f"{row['gt_count']:,} | {_pct(row['direct_rate'])} | "
            f"{_pct(row['direct_or_partial_rate'])} | "
            f"{_pct(row['information_bearing_rate'])} |"
        )

    fine_index = {
        (row["formula"], row["group"], row["axis"], row["label"]): row
        for row in fine_rows
    }
    key_labels = (
        ("historical_relation", "direct_predecessor"),
        ("historical_relation", "enabling_foundation"),
        ("mechanism_relation", "implicit_explanatory_mechanism"),
        ("domain_relation", "cross_domain_transfer"),
        ("domain_relation", "unrelated_domain_drift"),
    )
    lines.extend(
        [
            "",
            "## Fine semantic relations for Top-K selection exclusivity (coverage diagnostic)",
            "",
            "| Formula | Partition | Axis | Label | Triggered | Refined | Coverage | Count | Rate among refined |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for formula in FORMULAS:
        for partition in ("graph_only", "deep_merged_only"):
            for axis, label in key_labels:
                row = fine_index[(formula, partition, axis, label)]
                lines.append(
                    f"| {formula} | {partition} | {axis} | {label} | "
                    f"{row['triggered_count']:,} | {row['refined_count']:,} | "
                    f"{_pct(row['refinement_coverage_rate'])} | {row['paper_count']:,} | "
                    f"{_pct(row['rate_among_refined'])} |"
                )
    provenance_fine_index = {
        (row["formula"], row["group"], row["axis"], row["label"]): row
        for row in provenance_fine
    }
    lines.extend(
        [
            "",
            "The table above is not the strict source-complement result: many candidates were "
            "present in both complete pools and became exclusive only after truncation. The "
            "following full-pool-provenance table has complete second-stage coverage.",
            "",
            "## Fine relations for strict retrieval-source complement",
            "",
            "| Formula | Full-pool provenance | Axis | Label | Applicable | Count | Rate | Coverage |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for formula in FORMULAS:
        for provenance_group in (
            "graph_pool_unique_selected",
            "deep_pool_unique_selected",
        ):
            for axis, label in key_labels:
                row = provenance_fine_index[
                    (formula, provenance_group, axis, label)
                ]
                lines.append(
                    f"| {formula} | {provenance_group} | {axis} | {label} | "
                    f"{row['applicable_count']:,} | {row['paper_count']:,} | "
                    f"{_pct(row['rate_among_applicable'])} | "
                    f"{_pct(row['refinement_coverage_rate'])} |"
                )
    old_gap = retrieval[(STORED_FORMULA, "graph")]["macro_recall"] - retrieval[
        (STORED_FORMULA, "deep_merged")
    ]["macro_recall"]
    new_gap = retrieval[(NEW_FORMULA, "graph")]["macro_recall"] - retrieval[
        (NEW_FORMULA, "deep_merged")
    ]["macro_recall"]
    lines.extend(
        [
            "",
            "## Direct comparison",
            "",
            f"The macro-recall gap Graph−Deep merged changes from **{_pct(old_gap)}** "
            f"to **{_pct(new_gap)}** (shift **{_pct(new_gap - old_gap)}**).",
            "Candidate counts are query-paper pairs after union/deduplication; event occurrence "
            "budgets remain identical across both formulas and both methods.",
            "All semantic labels are reused from the completed blinded annotations; no paper was relabeled.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    global NEW_FORMULA, FORMULAS

    run_dir = Path(args.run_dir).resolve()
    annotation_work_dir = Path(args.annotation_work_dir).resolve()
    refinement_work_dir = Path(args.refinement_work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_dir / "onepass_artifacts"

    query_weight = float(args.query_weight)
    subquery_weight = float(args.subquery_weight)
    intent_weight = float(args.intent_weight)
    path_weight = float(args.path_weight)
    graph_path_definition = str(
        getattr(args, "graph_path_definition", "saved_undirected_degree")
    )
    weights = (query_weight, subquery_weight, intent_weight, path_weight)
    if any(weight < 0 for weight in weights):
        raise ValueError("rerank weights must be non-negative")
    if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"rerank weights must sum to 1.0; got {sum(weights):.12g}"
        )
    formula_id = str(args.formula_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", formula_id):
        raise ValueError("--formula-id may contain only letters, digits, dot, underscore, and hyphen")
    if formula_id == STORED_FORMULA:
        raise ValueError(f"--formula-id may not be {STORED_FORMULA!r}")
    NEW_FORMULA = formula_id
    FORMULAS = (STORED_FORMULA, NEW_FORMULA)
    path_feature_name = (
        "path_count_normalized"
        if graph_path_definition == "saved_undirected_degree"
        else "expanded_endpoint_path_count_normalized"
    )
    formula_description = (
        f"{query_weight:.2f} * query_score_normalized + "
        f"{subquery_weight:.2f} * subquery_score_normalized + "
        f"{intent_weight:.2f} * intent_score + "
        f"{path_weight:.2f} * {path_feature_name}"
    )

    budgets = load_baseline_budgets(artifact_dir / "baseline" / "selector_decisions.jsonl")
    contexts = load_query_contexts(annotation_work_dir / "manifest" / "queries.jsonl")
    selections = _selection_container()
    event_rows: List[Dict[str, Any]] = []
    occurrence_counts: Counter[Tuple[str, str]] = Counter()
    feature_diagnostics: Dict[str, Any] = {}
    path_overrides: Optional[Dict[str, Dict[str, Tuple[int, float]]]] = None
    path_definition_diagnostics: Dict[str, Any] = {
        "definition": graph_path_definition
    }
    if graph_path_definition == "expanded_endpoint_edge_count":
        path_overrides, loaded_diagnostics = load_expanded_endpoint_path_overrides(
            artifact_dir / "per_subquery" / "paper_rows.jsonl"
        )
        path_definition_diagnostics.update(loaded_diagnostics)
    graph_events = replay_graph(
        artifact_dir / "per_subquery" / "pool_records.jsonl",
        budgets,
        selections,
        event_rows,
        occurrence_counts,
        query_weight=query_weight,
        subquery_weight=subquery_weight,
        intent_weight=intent_weight,
        path_weight=path_weight,
        feature_diagnostics=feature_diagnostics,
        path_overrides=path_overrides,
    )
    if path_overrides is not None and set(path_overrides) != graph_events:
        raise ValueError(
            f"path override events do not match replayed Graph events: "
            f"overrides={len(path_overrides)}, graph={len(graph_events)}"
        )
    deep_events = replay_deep_merged(
        artifact_dir / "deep_merged" / "pool_records.jsonl",
        budgets,
        selections,
        event_rows,
        occurrence_counts,
        query_weight=query_weight,
        subquery_weight=subquery_weight,
        intent_weight=intent_weight,
        path_weight=path_weight,
        feature_diagnostics=feature_diagnostics,
    )
    budget_events = set(budgets)
    if graph_events != budget_events:
        raise ValueError(
            f"Graph event coverage mismatch: missing={len(budget_events - graph_events)} "
            f"extra={len(graph_events - budget_events)}"
        )
    if deep_events != budget_events:
        raise ValueError(
            f"Deep merged event coverage mismatch: missing={len(budget_events - deep_events)} "
            f"extra={len(deep_events - budget_events)}"
        )
    validate_stored_query_results(artifact_dir, selections)

    query_rows = query_metric_rows(contexts, selections)
    retrieval_rows = retrieval_summary_rows(query_rows, occurrence_counts)
    per_query_partitions, partition_summary = partition_rows(contexts, selections)
    churn = churn_rows(contexts, selections)
    memberships = _membership_maps(selections)

    workspaces, selected_annotations, original_primary_sources = build_annotation_workspaces(
        annotation_work_dir, output_dir, memberships
    )
    semantic_summaries, group_rows, semantic_rows = run_semantic_analyses(
        workspaces,
        output_dir,
        bootstrap_samples=int(args.bootstrap_samples),
        formula_description=formula_description,
    )
    fine_rows, missing_fine_rows = fine_relation_rows(
        refinement_work_dir / "analysis" / "annotations.jsonl",
        memberships,
        selected_annotations,
    )
    provenance_semantic = provenance_semantic_rows(
        contexts,
        selections,
        selected_annotations,
        original_primary_sources,
    )
    provenance_fine = provenance_fine_relation_rows(
        refinement_work_dir / "analysis" / "annotations.jsonl",
        selections,
        selected_annotations,
        original_primary_sources,
    )
    annotated_churn = annotated_churn_rows(
        contexts, selections, selected_annotations
    )

    _atomic_write_jsonl(output_dir / "event_ret_selections.jsonl", event_rows)
    _atomic_write_jsonl(output_dir / "query_ret_results.jsonl", query_rows)
    _atomic_write_jsonl(output_dir / "query_graph_deep_partitions.jsonl", per_query_partitions)
    _atomic_write_jsonl(output_dir / "retrieval_summary.jsonl", retrieval_rows)
    _atomic_write_csv(output_dir / "retrieval_summary.csv", retrieval_rows)
    _atomic_write_jsonl(output_dir / "partition_summary.jsonl", partition_summary)
    _atomic_write_csv(output_dir / "partition_summary.csv", partition_summary)
    _atomic_write_jsonl(output_dir / "formula_churn.jsonl", churn)
    _atomic_write_csv(output_dir / "formula_churn.csv", churn)
    _atomic_write_jsonl(output_dir / "annotated_formula_churn.jsonl", annotated_churn)
    _atomic_write_csv(output_dir / "annotated_formula_churn.csv", annotated_churn)
    _atomic_write_jsonl(output_dir / "semantic_group_summary.jsonl", group_rows)
    _atomic_write_csv(output_dir / "semantic_group_summary.csv", group_rows)
    _atomic_write_jsonl(output_dir / "semantic_complement.jsonl", semantic_rows)
    _atomic_write_csv(output_dir / "semantic_complement.csv", semantic_rows)
    _atomic_write_jsonl(output_dir / "fine_relation_summary.jsonl", fine_rows)
    _atomic_write_csv(output_dir / "fine_relation_summary.csv", fine_rows)
    _atomic_write_jsonl(
        output_dir / "provenance_semantic_summary.jsonl", provenance_semantic
    )
    _atomic_write_csv(
        output_dir / "provenance_semantic_summary.csv", provenance_semantic
    )
    _atomic_write_jsonl(
        output_dir / "provenance_fine_relation_summary.jsonl", provenance_fine
    )
    _atomic_write_csv(
        output_dir / "provenance_fine_relation_summary.csv", provenance_fine
    )
    _atomic_write_jsonl(
        output_dir / "fine_relation_backfill_manifest.jsonl", missing_fine_rows
    )

    report = build_report(
        budgets=budgets,
        retrieval_rows=retrieval_rows,
        partition_summary=partition_summary,
        churn=churn,
        annotated_churn=annotated_churn,
        group_rows=group_rows,
        fine_rows=fine_rows,
        provenance_semantic=provenance_semantic,
        provenance_fine=provenance_fine,
        formula_description=formula_description,
    )
    report_path = output_dir / "report.md"
    temporary_report = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    temporary_report.write_text(report, encoding="utf-8")
    os.replace(temporary_report, report_path)

    k_distribution = Counter(budget.k for budget in budgets.values())
    summary = {
        "complete": True,
        "implementation_version": IMPLEMENTATION_VERSION,
        "event_count": len(budgets),
        "baseline_occurrence_budget": sum(budget.k for budget in budgets.values()),
        "baseline_k_distribution": {str(key): value for key, value in sorted(k_distribution.items())},
        "query_count": len(contexts),
        "formula": {
            "id": NEW_FORMULA,
            "query_score_normalized": query_weight,
            "subquery_score_normalized": subquery_weight,
            "intent_score": intent_weight,
            "path_count_normalized": path_weight,
            "description": formula_description,
        },
        "k_definition": "len(baseline.selector_decision.candidate_rows) per retrieval_event_id",
        "deep_merged_policy": "rank merged subquery pool once, then split by chronological Baseline K_i slices",
        "selector_run": False,
        "annotations_reused": True,
        "fine_relation_backfill_row_count": len(missing_fine_rows),
        "strict_provenance_fine_relation_complete": True,
        "feature_diagnostics": feature_diagnostics,
        "graph_path_definition": graph_path_definition,
        "path_definition_diagnostics": path_definition_diagnostics,
        "retrieval_summary": retrieval_rows,
        "partition_summary": partition_summary,
        "formula_churn": churn,
        "semantic_analysis": semantic_summaries,
        "output_dir": str(output_dir),
    }
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_json(
        output_dir / "run_config.json",
        {
            "implementation_version": IMPLEMENTATION_VERSION,
            "run_dir": str(run_dir),
            "annotation_work_dir": str(annotation_work_dir),
            "refinement_work_dir": str(refinement_work_dir),
            "output_dir": str(output_dir),
            "query_weight": query_weight,
            "subquery_weight": subquery_weight,
            "intent_weight": intent_weight,
            "path_weight": path_weight,
            "graph_path_definition": graph_path_definition,
            "path_definition_diagnostics": path_definition_diagnostics,
            "formula_id": formula_id,
            "formula_description": formula_description,
            "feature_diagnostics": feature_diagnostics,
            "bootstrap_samples": int(args.bootstrap_samples),
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
    parser.add_argument("--annotation-work-dir", required=True)
    parser.add_argument("--refinement-work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-weight", type=float, default=0.30)
    parser.add_argument("--subquery-weight", type=float, default=0.40)
    parser.add_argument("--intent-weight", type=float, default=0.15)
    parser.add_argument("--path-weight", type=float, default=0.15)
    parser.add_argument(
        "--graph-path-definition",
        choices=("saved_undirected_degree", "expanded_endpoint_edge_count"),
        default="saved_undirected_degree",
        help=(
            "saved undirected degree, or count each edge only for its expanded "
            "endpoint and re-normalize within the event"
        ),
    )
    parser.add_argument(
        "--formula-id",
        default="q030_sq040_intent015_path015_closed_pool_minmax_v1",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
