#!/usr/bin/env python3
"""Offline replay of graph-pool reranking formulas on a completed OnePass run.

The replay reads only committed ``full`` artifacts.  It does not call the
retriever, embedding service, Semantic Scholar, Planner, or Selector.  Every
strategy receives the original event pool and the original ``selector_top_k``;
the output can therefore be used to screen rerankers before paying for a
Selector replay.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


IMPLEMENTATION_VERSION = "1.0"
DEFAULT_STRATEGIES = (
    "stored_current",
    "semantic_only",
    "semantic_40_60",
    "dense_reduced_intent",
    "dense_low_structure",
    "dense_no_intent",
    "corrected_support",
    "structure_heavy",
    "rank_fusion",
    "expanded_quota_30",
    "graph_novel_quota_30",
    "current_graph_novel_quota_10",
    "current_graph_novel_quota_20",
    "semantic_graph_novel_quota_10",
)


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(recall: float, precision: float) -> float:
    return 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _paper_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("arxiv:"):
        text = text.split(":", 1)[1].strip()
    return text


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


@dataclass(frozen=True)
class QueryContext:
    benchmark_idx: int
    query_id: str
    gt_ids: frozenset[str]
    baseline_ids: frozenset[str]
    graph_pool_ids: frozenset[str]
    deep_pool_ids: frozenset[str]
    stored_candidate_ids: frozenset[str]


@dataclass(frozen=True)
class Candidate:
    paper_id: str
    is_seed: bool
    is_expanded: bool
    observed_rank: Optional[int]
    query_score: float
    subquery_score: float
    intent_score: float
    original_path_score: float
    source_seed_count: int
    stored_top: bool

    @property
    def expanded_nonseed(self) -> bool:
        return self.is_expanded and not self.is_seed

    @property
    def source_support(self) -> float:
        """Saturating support for expanded papers; seeds receive no graph bonus.

        One source edge receives 0.5, two receive about 0.79, and three or more
        saturate at 1.0.  This avoids normalizing high-degree seeds together
        with low-degree expanded papers.
        """

        if not self.expanded_nonseed or self.source_seed_count <= 0:
            return 0.0
        return min(1.0, math.log1p(self.source_seed_count) / math.log(4.0))


def load_query_contexts(path: Path) -> Dict[str, QueryContext]:
    latest_by_idx: Dict[int, Dict[str, Any]] = {}
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
                latest_by_idx[idx] = row

    contexts: Dict[str, QueryContext] = {}
    for idx, row in sorted(latest_by_idx.items()):
        postprocess = row.get("postprocess_results") or {}
        baseline = postprocess.get("baseline") or {}
        graph = postprocess.get("per_subquery") or {}
        deep = postprocess.get("deep_event") or {}
        required = {
            "baseline candidate IDs": baseline.get("candidate_arxiv_ids"),
            "graph candidate IDs": graph.get("candidate_arxiv_ids"),
            "graph pool IDs": deep.get("source_graph_pool_arxiv_ids"),
            "deep pool IDs": deep.get("deep_pool_arxiv_ids"),
        }
        missing = [name for name, value in required.items() if not isinstance(value, list)]
        if missing:
            raise ValueError(f"idx={idx} is not a complete paired run; missing {', '.join(missing)}")
        query_id = str(deep.get("query_id") or postprocess.get("query_id") or f"idx-{idx}")
        context = QueryContext(
            benchmark_idx=idx,
            query_id=query_id,
            gt_ids=frozenset(_paper_id(value) for value in row.get("ground_truth_arxiv_ids") or [] if _paper_id(value)),
            baseline_ids=frozenset(_paper_id(value) for value in baseline["candidate_arxiv_ids"] if _paper_id(value)),
            graph_pool_ids=frozenset(_paper_id(value) for value in deep["source_graph_pool_arxiv_ids"] if _paper_id(value)),
            deep_pool_ids=frozenset(_paper_id(value) for value in deep["deep_pool_arxiv_ids"] if _paper_id(value)),
            stored_candidate_ids=frozenset(_paper_id(value) for value in graph["candidate_arxiv_ids"] if _paper_id(value)),
        )
        if query_id in contexts:
            raise ValueError(f"duplicate committed query_id={query_id}")
        contexts[query_id] = context
    if not contexts:
        raise ValueError(f"no committed query results found in {path}")
    return contexts


def _candidate_from_row(row: Mapping[str, Any]) -> Candidate:
    observed_rank = row.get("observed_retrieval_rank")
    try:
        observed_rank = int(observed_rank) if observed_rank is not None else None
    except (TypeError, ValueError):
        observed_rank = None
    source_seed_ids = {
        _paper_id(value)
        for value in row.get("source_seed_arxiv_ids") or []
        if _paper_id(value)
    }
    return Candidate(
        paper_id=_paper_id(row.get("paper_arxiv_id")),
        is_seed=bool(row.get("is_seed")),
        is_expanded=bool(row.get("is_expanded")),
        observed_rank=observed_rank,
        query_score=float(row.get("query_score_normalized") or 0.0),
        subquery_score=float(row.get("subquery_score_normalized") or 0.0),
        intent_score=float(row.get("intent_score") or 0.0),
        original_path_score=float(row.get("path_count_normalized") or 0.0),
        source_seed_count=len(source_seed_ids),
        stored_top=bool(row.get("in_selector_topk")),
    )


def iter_event_groups(
    path: Path,
    allowed_queries: Set[str],
) -> Iterator[Tuple[Dict[str, Any], List[Candidate]]]:
    current_key: Optional[Tuple[str, str]] = None
    current_meta: Dict[str, Any] = {}
    current_rows: List[Candidate] = []
    seen: Set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid graph paper rows at line {line_number}: {exc}") from exc
            query_id = str(row.get("query_id") or "")
            if query_id not in allowed_queries:
                continue
            event_id = str(row.get("retrieval_event_id") or "")
            if not event_id:
                raise ValueError(f"missing retrieval_event_id at line {line_number}")
            key = (query_id, event_id)
            if current_key is None:
                current_key = key
                current_meta = {
                    "benchmark_idx": row.get("benchmark_idx"),
                    "query_id": query_id,
                    "retrieval_event_id": event_id,
                    "iteration_idx": row.get("iteration_idx"),
                    "subquery_id": row.get("subquery_id"),
                    "subquery": row.get("subquery"),
                    "selector_top_k": int(row.get("selector_top_k") or 0),
                }
            if key != current_key:
                if key in seen:
                    raise ValueError(f"non-contiguous duplicate graph event {key}")
                seen.add(current_key)
                yield current_meta, current_rows
                current_key = key
                current_meta = {
                    "benchmark_idx": row.get("benchmark_idx"),
                    "query_id": query_id,
                    "retrieval_event_id": event_id,
                    "iteration_idx": row.get("iteration_idx"),
                    "subquery_id": row.get("subquery_id"),
                    "subquery": row.get("subquery"),
                    "selector_top_k": int(row.get("selector_top_k") or 0),
                }
                current_rows = []
            candidate = _candidate_from_row(row)
            if not candidate.paper_id:
                raise ValueError(f"missing paper_arxiv_id at line {line_number}")
            current_rows.append(candidate)
    if current_key is not None:
        if current_key in seen:
            raise ValueError(f"non-contiguous duplicate graph event {current_key}")
        yield current_meta, current_rows


def _descending_percentiles(values: Sequence[float], *, zero_is_zero: bool = False) -> List[float]:
    if not values:
        return []
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    output = [0.0] * len(values)
    denominator = max(1, len(values) - 1)
    start = 0
    while start < len(order):
        end = start + 1
        value = values[order[start]]
        while end < len(order) and values[order[end]] == value:
            end += 1
        percentile = 1.0 - (((start + end - 1) / 2.0) / denominator)
        if zero_is_zero and value <= 0.0:
            percentile = 0.0
        for position in range(start, end):
            output[order[position]] = percentile
        start = end
    return output


def _score_maps(candidates: Sequence[Candidate]) -> Dict[str, Dict[str, float]]:
    supports = [candidate.source_support for candidate in candidates]
    q_percentiles = _descending_percentiles([candidate.query_score for candidate in candidates])
    sq_percentiles = _descending_percentiles([candidate.subquery_score for candidate in candidates])
    intent_percentiles = _descending_percentiles(
        [candidate.intent_score for candidate in candidates], zero_is_zero=True
    )
    support_percentiles = _descending_percentiles(supports, zero_is_zero=True)
    output: Dict[str, Dict[str, float]] = {
        "current_recomputed": {},
        "semantic_only": {},
        "semantic_40_60": {},
        "dense_reduced_intent": {},
        "dense_low_structure": {},
        "dense_no_intent": {},
        "corrected_support": {},
        "structure_heavy": {},
        "rank_fusion": {},
    }
    for index, candidate in enumerate(candidates):
        output["current_recomputed"][candidate.paper_id] = (
            0.30 * candidate.query_score
            + 0.40 * candidate.subquery_score
            + 0.15 * candidate.intent_score
            + 0.15 * candidate.original_path_score
        )
        output["semantic_only"][candidate.paper_id] = (
            (3.0 / 7.0) * candidate.query_score
            + (4.0 / 7.0) * candidate.subquery_score
        )
        output["semantic_40_60"][candidate.paper_id] = (
            0.40 * candidate.query_score
            + 0.60 * candidate.subquery_score
        )
        output["dense_reduced_intent"][candidate.paper_id] = (
            0.35 * candidate.query_score
            + 0.45 * candidate.subquery_score
            + 0.05 * candidate.intent_score
            + 0.15 * candidate.original_path_score
        )
        output["dense_low_structure"][candidate.paper_id] = (
            0.35 * candidate.query_score
            + 0.50 * candidate.subquery_score
            + 0.05 * candidate.intent_score
            + 0.10 * candidate.original_path_score
        )
        output["dense_no_intent"][candidate.paper_id] = (
            0.35 * candidate.query_score
            + 0.50 * candidate.subquery_score
            + 0.15 * candidate.original_path_score
        )
        output["corrected_support"][candidate.paper_id] = (
            0.30 * candidate.query_score
            + 0.40 * candidate.subquery_score
            + 0.15 * candidate.intent_score
            + 0.15 * supports[index]
        )
        output["structure_heavy"][candidate.paper_id] = (
            0.20 * candidate.query_score
            + 0.30 * candidate.subquery_score
            + 0.20 * candidate.intent_score
            + 0.30 * supports[index]
        )
        output["rank_fusion"][candidate.paper_id] = (
            0.30 * q_percentiles[index]
            + 0.40 * sq_percentiles[index]
            + 0.15 * intent_percentiles[index]
            + 0.15 * support_percentiles[index]
        )
    return output


def _ranking(
    candidates: Sequence[Candidate],
    scores: Mapping[str, float],
) -> List[Candidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(scores[candidate.paper_id]),
            -int(candidate.is_seed),
            candidate.observed_rank if candidate.observed_rank is not None else 10**12,
            candidate.paper_id,
        ),
    )


def select_candidates(
    strategy: str,
    candidates: Sequence[Candidate],
    top_k: int,
    deep_pool_ids: Set[str] | frozenset[str],
    quota_ratio: float = 0.30,
) -> List[Tuple[Candidate, Optional[float]]]:
    top_k = max(0, int(top_k))
    if not candidates or top_k <= 0:
        return []
    if strategy == "stored_current":
        selected = [candidate for candidate in candidates if candidate.stored_top]
        if len(selected) != min(top_k, len(candidates)):
            raise ValueError(
                f"stored top-k count {len(selected)} does not match requested {top_k} "
                f"for an event with {len(candidates)} candidates"
            )
        return [(candidate, None) for candidate in selected]

    score_maps = _score_maps(candidates)
    base_name = strategy
    quota_predicate = None
    if strategy == "expanded_quota_30":
        base_name = "corrected_support"
        quota_predicate = lambda candidate: candidate.expanded_nonseed
    elif strategy == "graph_novel_quota_30":
        base_name = "corrected_support"
        quota_predicate = lambda candidate: (
            candidate.expanded_nonseed and candidate.paper_id not in deep_pool_ids
        )
    elif strategy == "current_graph_novel_quota_10":
        base_name = "current_recomputed"
        quota_ratio = 0.10
        quota_predicate = lambda candidate: (
            candidate.expanded_nonseed and candidate.paper_id not in deep_pool_ids
        )
    elif strategy == "current_graph_novel_quota_20":
        base_name = "current_recomputed"
        quota_ratio = 0.20
        quota_predicate = lambda candidate: (
            candidate.expanded_nonseed and candidate.paper_id not in deep_pool_ids
        )
    elif strategy == "semantic_graph_novel_quota_10":
        base_name = "semantic_only"
        quota_ratio = 0.10
        quota_predicate = lambda candidate: (
            candidate.expanded_nonseed and candidate.paper_id not in deep_pool_ids
        )
    if base_name not in score_maps:
        raise ValueError(f"unknown rerank strategy: {strategy}")

    scores = score_maps[base_name]
    ranked = _ranking(candidates, scores)
    if quota_predicate is None:
        return [(candidate, scores[candidate.paper_id]) for candidate in ranked[:top_k]]

    quota_k = min(top_k, max(1, int(top_k * quota_ratio)))
    quota_rows = [candidate for candidate in ranked if quota_predicate(candidate)][:quota_k]
    selected_ids = {candidate.paper_id for candidate in quota_rows}
    fill_rows = [candidate for candidate in ranked if candidate.paper_id not in selected_ids]
    selected = quota_rows + fill_rows[: max(0, top_k - len(quota_rows))]
    return [(candidate, scores[candidate.paper_id]) for candidate in selected]


def _query_metrics(
    context: QueryContext,
    candidate_ids: Set[str],
) -> Dict[str, Any]:
    candidate_gt_ids = candidate_ids & set(context.gt_ids)
    graph_only_ids = set(context.graph_pool_ids) - set(context.deep_pool_ids)
    graph_only_candidate_ids = candidate_ids & graph_only_ids
    graph_only_gt_ids = graph_only_candidate_ids & set(context.gt_ids)
    baseline_candidate_ids = candidate_ids & set(context.baseline_ids)
    common_new_ids = candidate_ids - set(context.baseline_ids) - graph_only_ids
    recall = _safe_div(len(candidate_gt_ids), len(context.gt_ids))
    precision = _safe_div(len(candidate_gt_ids), len(candidate_ids))
    return {
        "benchmark_idx": context.benchmark_idx,
        "query_id": context.query_id,
        "gt_count": len(context.gt_ids),
        "candidate_count": len(candidate_ids),
        "candidate_gt_count": len(candidate_gt_ids),
        "candidate_recall": recall,
        "candidate_precision": precision,
        "candidate_f1": _f1(recall, precision),
        "candidate_arxiv_ids": sorted(candidate_ids),
        "candidate_gt_arxiv_ids": sorted(candidate_gt_ids),
        "baseline_candidate_count": len(baseline_candidate_ids),
        "common_new_candidate_count": len(common_new_ids),
        "graph_only_candidate_count": len(graph_only_candidate_ids),
        "graph_only_candidate_gt_count": len(graph_only_gt_ids),
        "graph_only_candidate_arxiv_ids": sorted(graph_only_candidate_ids),
        "graph_only_candidate_gt_arxiv_ids": sorted(graph_only_gt_ids),
    }


def _aggregate_strategy(
    strategy: str,
    query_rows: Sequence[Mapping[str, Any]],
    occurrence_count: int,
) -> Dict[str, Any]:
    total_candidates = sum(int(row["candidate_count"]) for row in query_rows)
    total_candidate_gt = sum(int(row["candidate_gt_count"]) for row in query_rows)
    total_gt = sum(int(row["gt_count"]) for row in query_rows)
    total_graph_only = sum(int(row["graph_only_candidate_count"]) for row in query_rows)
    total_graph_only_gt = sum(int(row["graph_only_candidate_gt_count"]) for row in query_rows)
    macro_recall = _mean([float(row["candidate_recall"]) for row in query_rows])
    macro_precision = _mean([float(row["candidate_precision"]) for row in query_rows])
    macro_f1 = _mean([float(row["candidate_f1"]) for row in query_rows])
    return {
        "strategy": strategy,
        "evaluated_query_count": len(query_rows),
        "total_gt_count": total_gt,
        "total_candidate_occurrence_count": occurrence_count,
        "total_candidate_count": total_candidates,
        "total_candidate_gt_count": total_candidate_gt,
        "avg_candidate_count": _safe_div(total_candidates, len(query_rows)),
        "avg_candidate_recall": macro_recall,
        "avg_candidate_precision": macro_precision,
        "avg_candidate_f1": macro_f1,
        "micro_candidate_recall": _safe_div(total_candidate_gt, total_gt),
        "micro_candidate_precision": _safe_div(total_candidate_gt, total_candidates),
        "micro_candidate_f1": _f1(
            _safe_div(total_candidate_gt, total_gt),
            _safe_div(total_candidate_gt, total_candidates),
        ),
        "deduplication_factor": _safe_div(occurrence_count, total_candidates),
        "graph_only_candidate_count": total_graph_only,
        "graph_only_candidate_gt_count": total_graph_only_gt,
        "graph_only_candidate_share": _safe_div(total_graph_only, total_candidates),
        "graph_only_candidate_precision": _safe_div(total_graph_only_gt, total_graph_only),
    }


def replay(
    run_dir: Path,
    output_dir: Path,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    quota_ratio: float = 0.30,
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    detailed_path = run_dir / "detailed_results.jsonl"
    graph_rows_path = run_dir / "onepass_artifacts" / "per_subquery" / "paper_rows.jsonl"
    if not detailed_path.exists():
        raise FileNotFoundError(detailed_path)
    if not graph_rows_path.exists():
        raise FileNotFoundError(graph_rows_path)
    if not 0.0 <= quota_ratio <= 1.0:
        raise ValueError("quota_ratio must be between 0 and 1")
    strategies = tuple(dict.fromkeys(strategies))
    unknown = set(strategies) - set(DEFAULT_STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies: {sorted(unknown)}")

    contexts = load_query_contexts(detailed_path)
    candidate_ids: Dict[str, MutableMapping[str, Set[str]]] = {
        strategy: defaultdict(set) for strategy in strategies
    }
    occurrence_counts = {strategy: 0 for strategy in strategies}
    event_rows: List[Dict[str, Any]] = []
    event_count = 0

    for meta, candidates in iter_event_groups(graph_rows_path, set(contexts)):
        event_count += 1
        query_id = str(meta["query_id"])
        context = contexts[query_id]
        top_k = int(meta["selector_top_k"])
        if len({candidate.paper_id for candidate in candidates}) != len(candidates):
            raise ValueError(f"duplicate paper in graph event {meta['retrieval_event_id']}")
        strategy_outputs: Dict[str, Any] = {}
        for strategy in strategies:
            selected = select_candidates(
                strategy,
                candidates,
                top_k,
                context.deep_pool_ids,
                quota_ratio,
            )
            selected_ids = [candidate.paper_id for candidate, _ in selected]
            occurrence_counts[strategy] += len(selected_ids)
            candidate_ids[strategy][query_id].update(selected_ids)
            strategy_outputs[strategy] = {
                "candidate_arxiv_ids": selected_ids,
                "candidate_scores": [score for _, score in selected],
                "candidate_gt_arxiv_ids": [paper_id for paper_id in selected_ids if paper_id in context.gt_ids],
                "graph_only_arxiv_ids": [
                    paper_id
                    for paper_id in selected_ids
                    if paper_id in context.graph_pool_ids and paper_id not in context.deep_pool_ids
                ],
            }
        event_rows.append({**meta, "strategies": strategy_outputs})

    query_results: List[Dict[str, Any]] = []
    per_strategy_query_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for context in sorted(contexts.values(), key=lambda value: value.benchmark_idx):
        methods: Dict[str, Any] = {}
        for strategy in strategies:
            metrics = _query_metrics(context, set(candidate_ids[strategy].get(context.query_id, set())))
            methods[strategy] = metrics
            per_strategy_query_rows[strategy].append(metrics)
        query_results.append(
            {
                "benchmark_idx": context.benchmark_idx,
                "query_id": context.query_id,
                "gt_count": len(context.gt_ids),
                "strategies": methods,
            }
        )

    if "stored_current" in strategies:
        mismatches = []
        for context in contexts.values():
            replayed = set(candidate_ids["stored_current"].get(context.query_id, set()))
            if replayed != set(context.stored_candidate_ids):
                mismatches.append(
                    {
                        "query_id": context.query_id,
                        "missing": sorted(set(context.stored_candidate_ids) - replayed),
                        "extra": sorted(replayed - set(context.stored_candidate_ids)),
                    }
                )
        if mismatches:
            raise ValueError(f"stored-current reconstruction mismatch for {len(mismatches)} query(s)")

    summaries = {
        strategy: _aggregate_strategy(
            strategy,
            per_strategy_query_rows[strategy],
            occurrence_counts[strategy],
        )
        for strategy in strategies
    }
    baseline = summaries.get("stored_current")
    if baseline:
        for strategy, summary in summaries.items():
            summary["delta_avg_candidate_recall_vs_stored"] = (
                summary["avg_candidate_recall"] - baseline["avg_candidate_recall"]
            )
            summary["delta_avg_candidate_precision_vs_stored"] = (
                summary["avg_candidate_precision"] - baseline["avg_candidate_precision"]
            )
            summary["delta_total_candidate_gt_vs_stored"] = (
                summary["total_candidate_gt_count"] - baseline["total_candidate_gt_count"]
            )

    result = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "run_dir": str(run_dir),
        "detailed_results": str(detailed_path),
        "graph_paper_rows": str(graph_rows_path),
        "evaluated_query_count": len(contexts),
        "event_count": event_count,
        "quota_ratio": quota_ratio,
        "candidate_scope": "deduplicated per-query union of each event's original selector_top_k",
        "strategies": summaries,
    }
    atomic_write_json(output_dir / "summary.json", result)
    atomic_write_jsonl(output_dir / "query_results.jsonl", query_results)
    atomic_write_jsonl(output_dir / "event_results.jsonl", event_rows)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=DEFAULT_STRATEGIES,
        default=list(DEFAULT_STRATEGIES),
    )
    parser.add_argument("--quota-ratio", type=float, default=0.30)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = replay(
        args.run_dir,
        args.output_dir,
        strategies=args.strategies,
        quota_ratio=args.quota_ratio,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
