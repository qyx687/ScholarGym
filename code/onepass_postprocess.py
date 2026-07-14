#!/usr/bin/env python3
"""Per-query graph rerank and matched text-only deep-retrieval shadows."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from agent.selector import Selector
from deep_retrieval import (
    DEEP_EVENT_METHOD,
    DEEP_MERGED_METHOD,
    DeepRetrievalProcessor,
    compact_deep_rows,
    id_set_comparison,
    ordered_unique,
)
from graph_methods import (
    ArtifactWriter,
    PerSubqueryProcessor,
    S2GraphClient,
    normalize_arxiv_id,
    selector_decision_record,
)
from structures import Paper, SubQuery


POSTPROCESS_METHODS = ("per_subquery", "deep_event", "deep_merged")
EVENT_FIELDS = (
    "schema_version",
    "run_id",
    "retrieval_event_id",
    "query_id",
    "benchmark_idx",
    "query",
    "query_source",
    "query_date",
    "iteration_idx",
    "subquery_id",
    "subquery",
    "subquery_target_k",
    "subquery_link_type",
    "parent_subquery_id",
    "subquery_before_date",
    "retrieval_page_idx",
    "retrieval_offset",
    "raw_retrieval_page_count",
    "results_per_query",
    "selector_top_k",
    "planner_checklist",
    "retrieval_exclusion_arxiv_ids",
)
GRAPH_COMPACT_FIELDS = (
    "paper_arxiv_id",
    "candidate_type",
    "observed_retrieval_score",
    "observed_retrieval_rank",
    "observed_retrieval_rank_scope",
    "observed_retrieval_rank_after_exclusion",
    "retrieval_score_raw",
    "retrieval_score_normalized",
    "retrieval_rank",
    "query_score_raw",
    "query_score_normalized",
    "query_component_rank",
    "subquery_score_raw",
    "subquery_score_normalized",
    "subquery_component_rank",
    "intent_score",
    "path_count",
    "path_count_normalized",
    "rerank_score",
    "rerank_rank",
    "in_selector_topk",
    "selector_input_rank",
    "selector_selected",
)


def _stable_thread_map(function, items: Sequence[Any], max_workers: int) -> List[Any]:
    """Run independent synchronous jobs concurrently and retain input order."""
    values = list(items)
    if len(values) <= 1 or int(max_workers) <= 1:
        return [function(value) for value in values]
    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(values))) as executor:
        return list(executor.map(function, values))


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(recall: float, precision: float) -> float:
    return 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0


def _stage_metrics(gt_ids: Set[str], candidate_ids: Iterable[str]) -> Dict[str, Any]:
    candidates = ordered_unique(candidate_ids)
    hits = [paper_id for paper_id in candidates if paper_id in gt_ids]
    recall = _safe_div(len(hits), len(gt_ids))
    precision = _safe_div(len(hits), len(candidates))
    return {
        "count": len(candidates),
        "arxiv_ids": candidates,
        "gt_count": len(hits),
        "gt_arxiv_ids": hits,
        "recall": recall,
        "precision": precision,
        "f1": _f1(recall, precision),
    }


def _metrics(gt_ids: Set[str], candidate_ids: Iterable[str], selected_ids: Iterable[str]) -> Dict[str, Any]:
    candidates = {normalize_arxiv_id(value) for value in candidate_ids if normalize_arxiv_id(value)}
    selected = {normalize_arxiv_id(value) for value in selected_ids if normalize_arxiv_id(value)}
    candidate_hits = sorted(gt_ids & candidates)
    selected_hits = sorted(gt_ids & selected)
    candidate_recall = _safe_div(len(candidate_hits), len(gt_ids))
    candidate_precision = _safe_div(len(candidate_hits), len(candidates))
    selection_recall = _safe_div(len(selected_hits), len(gt_ids))
    selection_precision = _safe_div(len(selected_hits), len(selected))
    return {
        "gt_count": len(gt_ids),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidate_arxiv_ids": sorted(candidates),
        "selected_arxiv_ids": sorted(selected),
        "candidate_gt_ids": candidate_hits,
        "selected_gt_ids": selected_hits,
        "candidate_gt_count": len(candidate_hits),
        "selected_gt_count": len(selected_hits),
        "candidate_recall": candidate_recall,
        "candidate_precision": candidate_precision,
        "candidate_f1": _f1(candidate_recall, candidate_precision),
        "selection_recall": selection_recall,
        "selection_precision": selection_precision,
        "selection_f1": _f1(selection_recall, selection_precision),
        "retrieved_to_selected_gt_gap": len(candidate_hits) - len(selected_hits),
        "gt_conversion_rate": _safe_div(len(selected_hits), len(candidate_hits)),
    }


def _prefix_stage(summary: Dict[str, Any], prefix: str, stage: Mapping[str, Any]) -> None:
    for key, value in stage.items():
        summary[f"{prefix}_{key}"] = value


def _compact_graph_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {key: row.get(key) for key in GRAPH_COMPACT_FIELDS if key in row}
        for row in rows
    ]


def aggregate_postprocess_metrics(detailed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate the graph/deep shadows over authoritative benchmark records."""
    canonical_results: Dict[Tuple[str, Any], Mapping[str, Any]] = {}
    for position, query_result in enumerate(detailed_results or []):
        if not isinstance(query_result, Mapping):
            continue
        benchmark_idx = query_result.get("idx")
        identity = ("idx", benchmark_idx) if benchmark_idx is not None else ("position", position)
        canonical_results[identity] = query_result

    methods: Dict[str, Dict[str, Any]] = {}
    metric_names = (
        "candidate_count",
        "selected_count",
        "candidate_recall",
        "candidate_precision",
        "candidate_f1",
        "selection_recall",
        "selection_precision",
        "selection_f1",
        "retrieved_to_selected_gt_gap",
        "gt_conversion_rate",
    )
    for method_name in POSTPROCESS_METHODS:
        method_results: List[Mapping[str, Any]] = []
        failed_query_count = 0
        missing_query_count = 0
        for query_result in canonical_results.values():
            postprocess_results = query_result.get("postprocess_results") or {}
            method_result = (
                postprocess_results.get(method_name)
                if isinstance(postprocess_results, Mapping)
                else None
            )
            if not isinstance(method_result, Mapping):
                missing_query_count += 1
                continue
            if method_result.get("error") or "gt_count" not in method_result:
                failed_query_count += 1
                continue
            method_results.append(method_result)

        count = len(method_results)
        totals = {
            "gt_count": sum(int(result.get("gt_count") or 0) for result in method_results),
            "candidate_count": sum(int(result.get("candidate_count") or 0) for result in method_results),
            "selected_count": sum(int(result.get("selected_count") or 0) for result in method_results),
            "candidate_gt_count": sum(len(result.get("candidate_gt_ids") or []) for result in method_results),
            "selected_gt_count": sum(len(result.get("selected_gt_ids") or []) for result in method_results),
        }
        aggregate: Dict[str, Any] = {
            "evaluated_query_count": count,
            "failed_query_count": failed_query_count,
            "missing_query_count": missing_query_count,
            **{f"total_{name}": value for name, value in totals.items()},
        }
        for metric_name in metric_names:
            aggregate[f"avg_{metric_name}"] = (
                sum(float(result.get(metric_name) or 0.0) for result in method_results) / count
                if count
                else 0.0
            )
        aggregate.update(
            {
                "micro_candidate_recall": _safe_div(totals["candidate_gt_count"], totals["gt_count"]),
                "micro_candidate_precision": _safe_div(totals["candidate_gt_count"], totals["candidate_count"]),
                "micro_selection_recall": _safe_div(totals["selected_gt_count"], totals["gt_count"]),
                "micro_selection_precision": _safe_div(totals["selected_gt_count"], totals["selected_count"]),
            }
        )
        aggregate["micro_candidate_f1"] = _f1(
            aggregate["micro_candidate_recall"], aggregate["micro_candidate_precision"]
        )
        aggregate["micro_selection_f1"] = _f1(
            aggregate["micro_selection_recall"], aggregate["micro_selection_precision"]
        )
        methods[method_name] = aggregate

    return {
        "source_query_count": len(canonical_results),
        "macro_average_scope": "successful unique benchmark queries for each method",
        "candidate_scope": "deduplicated rerank top-k papers passed to each shadow Selector",
        **methods,
    }


class OnePassPostprocessor:
    def __init__(
        self,
        *,
        selector: Optional[Selector],
        paper_db: Mapping[str, Mapping[str, Any]],
        writer: ArtifactWriter,
        s2_client: S2GraphClient,
        per_subquery_processor: PerSubqueryProcessor,
        deep_retrieval_processor: Optional[DeepRetrievalProcessor],
        scoring_backend: str,
        embedding_provider: Any,
        run_per_subquery: bool,
        run_deep_event: bool = False,
        run_deep_merged: bool = False,
        run_id: str = "",
        event_workers: int = 1,
        selector_concurrency: int = 1,
    ) -> None:
        self.selector = selector
        self.paper_db = dict(paper_db)
        self.writer = writer
        self.s2 = s2_client
        self.per_subquery = per_subquery_processor
        self.deep = deep_retrieval_processor
        self.backend = scoring_backend
        self.embedding_provider = embedding_provider
        self.run_per_subquery = run_per_subquery
        self.run_deep_event = run_deep_event
        self.run_deep_merged = run_deep_merged
        self.run_id = run_id
        self.event_workers = int(event_workers)
        self.selector_concurrency = int(selector_concurrency)
        if self.event_workers != 1 or self.selector_concurrency != 1:
            raise ValueError(
                "the sequential package requires event_workers=1 and "
                "selector_concurrency=1"
            )
        if (run_deep_event or run_deep_merged) and deep_retrieval_processor is None:
            raise ValueError("deep_retrieval_processor is required when a deep shadow is enabled")

    def process_query(
        self,
        query: Mapping[str, Any],
        planner_events: Sequence[Mapping[str, Any]],
        retrieval_events: Sequence[Mapping[str, Any]],
        gt_ids: Set[str],
    ) -> Dict[str, Any]:
        query_id = (
            retrieval_events[0].get("query_id")
            if retrieval_events
            else (query.get("qid") or query.get("query_id"))
        )
        benchmark_idx = retrieval_events[0].get("benchmark_idx") if retrieval_events else None
        self.writer.begin_query(query_id, benchmark_idx)
        try:
            result = self._process_query(query, planner_events, retrieval_events, gt_ids)
            self.writer.commit_query()
            return result
        except BaseException:
            self.writer.abort_query()
            raise

    def _process_query(
        self,
        query: Mapping[str, Any],
        planner_events: Sequence[Mapping[str, Any]],
        retrieval_events: Sequence[Mapping[str, Any]],
        gt_ids: Set[str],
    ) -> Dict[str, Any]:
        s2_before = self.s2.snapshot_stats()
        for event in list(planner_events) + list(retrieval_events):
            if isinstance(event, dict):
                event.setdefault("run_id", self.run_id)
        for event in planner_events:
            self.writer.append("baseline/planner_events.jsonl", event, full_only=True)
        self._write_baseline(retrieval_events, gt_ids)
        result: Dict[str, Any] = {
            "query_id": retrieval_events[0].get("query_id") if retrieval_events else (query.get("qid") or query.get("query_id")),
            "benchmark_idx": retrieval_events[0].get("benchmark_idx") if retrieval_events else None,
            "baseline": self._baseline_summary(retrieval_events, gt_ids),
        }

        graph_events: List[Dict[str, Any]] = []
        if self.run_per_subquery:
            try:
                result["per_subquery"], graph_events = self._process_per_subquery(
                    retrieval_events, gt_ids
                )
            except Exception as exc:
                result["per_subquery"] = {"error": str(exc)}
                self.writer.append(
                    "per_subquery/errors.jsonl",
                    {"query_id": result["query_id"], "error": str(exc)},
                )

        if self.run_deep_event or self.run_deep_merged:
            try:
                per_subquery_result = result.get("per_subquery") or {}
                graph_failed_events = int(
                    per_subquery_result.get("graph_failed_events") or 0
                ) if isinstance(per_subquery_result, Mapping) else 0
                if graph_failed_events:
                    raise ValueError(
                        f"deep shadows require all graph pools; {graph_failed_events} graph event(s) failed"
                    )
                if not graph_events:
                    raise ValueError(
                        "deep shadows require successful per-subquery graph pools for this query"
                    )
                prepared = self.deep.prepare_pools(
                    graph_events,
                    include_event=self.run_deep_event,
                    include_merged=self.run_deep_merged,
                )
            except Exception as exc:
                for name, enabled in (
                    ("deep_event", self.run_deep_event),
                    ("deep_merged", self.run_deep_merged),
                ):
                    if enabled:
                        result[name] = {"error": str(exc)}
                        self.writer.append(
                            f"{name}/errors.jsonl",
                            {"query_id": result["query_id"], "error": str(exc)},
                        )
            else:
                if self.run_deep_event:
                    try:
                        result["deep_event"] = self._process_deep_event(prepared, gt_ids)
                    except Exception as exc:
                        result["deep_event"] = {"error": str(exc)}
                        self.writer.append(
                            "deep_event/errors.jsonl",
                            {"query_id": result["query_id"], "error": str(exc)},
                        )
                if self.run_deep_merged:
                    try:
                        result["deep_merged"] = self._process_deep_merged(prepared, gt_ids)
                    except Exception as exc:
                        result["deep_merged"] = {"error": str(exc)}
                        self.writer.append(
                            "deep_merged/errors.jsonl",
                            {"query_id": result["query_id"], "error": str(exc)},
                        )

        s2_after = self.s2.snapshot_stats()
        result["s2_stats_delta"] = {
            key: int(s2_after.get(key, 0)) - int(s2_before.get(key, 0))
            for key in set(s2_before) | set(s2_after)
        }
        result["s2_stats_cumulative"] = s2_after
        self.writer.append("query_results.jsonl", result)
        return result

    def reconcile_artifacts(
        self,
        committed_indices: Iterable[Any],
        committed_query_ids: Iterable[Any],
    ) -> Dict[str, Any]:
        stats = self.writer.reconcile_with_checkpoint(committed_indices, committed_query_ids)
        stats["artifact_write_mode"] = "query_staging_then_flat_jsonl_commit"
        self.writer.write_json("resume_reconciliation.json", stats)
        return stats

    def _write_baseline(self, events: Sequence[Mapping[str, Any]], gt_ids: Set[str]) -> None:
        for event in events:
            selected = set(event.get("baseline_selected_arxiv_ids") or [])
            candidate_rows = []
            for seed in event.get("seed_papers") or []:
                paper_id = normalize_arxiv_id(seed.get("paper_arxiv_id"))
                if not paper_id:
                    continue
                observed_rank = seed.get("observed_retrieval_rank")
                rank_after_exclusion = (
                    int(event.get("retrieval_offset") or 0) + int(observed_rank)
                    if observed_rank is not None
                    else None
                )
                row = {
                    **{key: event.get(key) for key in EVENT_FIELDS},
                    "method": "baseline",
                    "paper_arxiv_id": paper_id,
                    "candidate_type": "seed",
                    "is_seed": True,
                    "is_expanded": False,
                    "passed_date_cutoff": True,
                    "date_cutoff_month": str(event.get("subquery_before_date") or event.get("query_date") or "")[:7],
                    "retrieval_backend": event.get("retrieval_backend"),
                    "observed_retrieval_score": seed.get("observed_retrieval_score"),
                    "observed_retrieval_rank": observed_rank,
                    "observed_retrieval_rank_scope": "one_based_rank_in_returned_baseline_page",
                    "observed_retrieval_rank_after_exclusion": rank_after_exclusion,
                    # Backward-compatible alias; it is not a pre-exclusion global rank.
                    "observed_retrieval_absolute_rank": rank_after_exclusion,
                    "observed_retrieval_absolute_rank_scope": "one_based_rank_after_frozen_exclusion",
                    "in_selector_topk": True,
                    "selector_selected": paper_id in selected,
                    "is_ground_truth": paper_id in gt_ids,
                }
                candidate_rows.append(row)
                self.writer.append("baseline/paper_rows.jsonl", row, full_only=True)
            self.writer.append(
                "baseline/selector_decisions.jsonl",
                {
                    "query_id": event.get("query_id"),
                    "benchmark_idx": event.get("benchmark_idx"),
                    "iteration_idx": event.get("iteration_idx"),
                    "subquery_id": event.get("subquery_id"),
                    "subquery": event.get("subquery"),
                    "planner_checklist": event.get("planner_checklist"),
                    "candidate_rows": candidate_rows,
                    "selected_arxiv_ids": sorted(selected),
                    "selector_overview": event.get("baseline_selector_overview") or "",
                },
                full_only=True,
            )

    @staticmethod
    def _baseline_summary(events: Sequence[Mapping[str, Any]], gt_ids: Set[str]) -> Dict[str, Any]:
        candidates = []
        selected = []
        for event in events:
            candidates.extend(row.get("paper_arxiv_id") for row in event.get("seed_papers") or [])
            selected.extend(event.get("baseline_selected_arxiv_ids") or [])
        return _metrics(gt_ids, candidates, selected)

    @staticmethod
    def _selector_subquery(
        event: Mapping[str, Any],
        papers: Sequence[Paper],
    ) -> SubQuery:
        return SubQuery(
            id=int(event.get("subquery_id") or 0),
            text=str(event.get("subquery") or event.get("query") or ""),
            before_date=event.get("subquery_before_date") or event.get("query_date"),
            target_k=int(event.get("subquery_target_k") or event.get("selector_top_k") or len(papers)),
            link_type=event.get("subquery_link_type"),
            source_subquery_id=event.get("parent_subquery_id"),
            iter_index=int(event.get("iteration_idx") or 0),
        )

    async def _call_selector_async(
        self,
        event: Mapping[str, Any],
        papers: Sequence[Paper],
        checklist: str,
    ) -> Tuple[List[str], str, Dict[str, str]]:
        if not papers:
            return [], "", {}
        if self.selector is None:
            raise ValueError("Selector is not attached to the one-pass postprocessor")
        selected, overview, _, details = await self.selector.decide_for_subquery(
            user_query=str(event.get("query") or ""),
            sub_query=self._selector_subquery(event, papers),
            planner_checklist=checklist,
            papers=list(papers),
            iteration_index=int(event.get("iteration_idx") or 1),
            idx=int(event.get("benchmark_idx") or 0),
            old_overview="",
            is_after_browsing=False,
            return_details=True,
        )
        selected_ids = ordered_unique(paper.arxiv_id or paper.id for paper in selected)
        raw_reasons = dict((details or {}).get("reasons") or {})
        reasons = {
            normalize_arxiv_id(key): str(value)
            for key, value in raw_reasons.items()
            if normalize_arxiv_id(key)
        }
        return selected_ids, overview or "", reasons

    def _call_selectors(
        self,
        requests: Sequence[Tuple[Mapping[str, Any], Sequence[Paper], str]],
    ) -> List[Tuple[Optional[Tuple[List[str], str, Dict[str, str]]], Optional[Exception]]]:
        """Call shadow Selectors with bounded concurrency and stable results."""

        async def run_all():
            semaphore = asyncio.Semaphore(self.selector_concurrency)

            async def run_one(event, papers, checklist):
                if not papers:
                    return (([], "", {}), None)
                try:
                    async with semaphore:
                        result = await self._call_selector_async(event, papers, checklist)
                    return result, None
                except Exception as exc:
                    return None, exc

            tasks = [run_one(event, papers, checklist) for event, papers, checklist in requests]
            return await asyncio.gather(*tasks)

        return asyncio.run(run_all()) if requests else []

    def _call_selector(
        self,
        event: Mapping[str, Any],
        papers: Sequence[Paper],
        checklist: str,
    ) -> Tuple[List[str], str, Dict[str, str]]:
        result, error = self._call_selectors([(event, papers, checklist)])[0]
        if error is not None:
            raise error
        return result

    def _process_per_subquery(
        self,
        events: Sequence[Mapping[str, Any]],
        gt_ids: Set[str],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        all_pool: List[str] = []
        all_candidates: List[str] = []
        all_selected: List[str] = []
        graph_events: List[Dict[str, Any]] = []
        graph_failures = 0
        selector_failures = 0

        def process_graph_event(baseline_event):
            event = dict(baseline_event)
            event["method"] = "per_subquery_new_formula_shadow"
            try:
                processed = self.per_subquery.process(
                    event,
                    gt_ids,
                    exclude_arxiv_ids=set(
                        event.get("retrieval_exclusion_arxiv_ids") or []
                    ),
                )
            except Exception as exc:
                return event, None, exc
            return event, processed, None

        graph_outcomes = _stable_thread_map(
            process_graph_event, list(events), self.event_workers
        )
        successful_positions = [
            index
            for index, (_, processed, error) in enumerate(graph_outcomes)
            if processed is not None and error is None
        ]
        selector_requests = [
            (
                graph_outcomes[index][0],
                graph_outcomes[index][1]["papers"],
                str(graph_outcomes[index][0].get("planner_checklist") or ""),
            )
            for index in successful_positions
        ]
        selector_outcomes = self._call_selectors(selector_requests)
        selector_by_position = dict(zip(successful_positions, selector_outcomes))

        for index, (event, processed, graph_error) in enumerate(graph_outcomes):
            if graph_error is not None:
                graph_failures += 1
                self.writer.append(
                    "per_subquery/errors.jsonl",
                    {
                        "query_id": event.get("query_id"),
                        "retrieval_event_id": event.get("retrieval_event_id"),
                        "subquery_id": event.get("subquery_id"),
                        "stage": "graph_expand_rerank",
                        "error": str(graph_error),
                    },
                )
                continue

            selector_result, selector_error = selector_by_position[index]
            if selector_error is not None:
                selector_failures += 1
                selected, overview, reasons = [], "", {}
                self.writer.append(
                    "per_subquery/errors.jsonl",
                    {
                        "query_id": event.get("query_id"),
                        "retrieval_event_id": event.get("retrieval_event_id"),
                        "subquery_id": event.get("subquery_id"),
                        "stage": "selector",
                        "error": str(selector_error),
                    },
                )
            else:
                selected, overview, reasons = selector_result

            selected_set = set(selected)
            top_ids = ordered_unique(row["paper_arxiv_id"] for row in processed["top_rows"])
            top_set = set(top_ids)
            for row in processed["rows"]:
                paper_id = row["paper_arxiv_id"]
                row["in_selector_topk"] = paper_id in top_set
                row["selector_input_rank"] = row["rerank_rank"] if paper_id in top_set else None
                row["selector_selected"] = paper_id in selected_set
                row["selector_reason"] = reasons.get(paper_id, "")
                row["affects_next_iteration"] = False
                self.writer.append("per_subquery/paper_rows.jsonl", row, full_only=True)
            for edge in processed["edges"]:
                self.writer.append("per_subquery/expansion_edges.jsonl", edge, full_only=True)
            self.writer.append(
                "per_subquery/filter_stats.jsonl",
                {
                    "query_id": event.get("query_id"),
                    "benchmark_idx": event.get("benchmark_idx"),
                    "retrieval_event_id": event.get("retrieval_event_id"),
                    "iteration_idx": event.get("iteration_idx"),
                    "subquery_id": event.get("subquery_id"),
                    **processed.get("filter_stats", {}),
                },
            )
            self.writer.append(
                "per_subquery/pool_records.jsonl",
                {
                    **{key: event.get(key) for key in EVENT_FIELDS},
                    "method": "per_subquery_new_formula_shadow",
                    "local_pool_size": len(processed["rows"]),
                    "local_pool_arxiv_ids": ordered_unique(
                        row["paper_arxiv_id"] for row in processed["rows"]
                    ),
                    "selector_topk_arxiv_ids": top_ids,
                    "selected_arxiv_ids": selected,
                    "local_pool_rows": _compact_graph_rows(processed["rows"]),
                },
            )
            self.writer.append(
                "per_subquery/selector_decisions.jsonl",
                selector_decision_record(event, processed["top_rows"], selected, overview, reasons),
                full_only=True,
            )
            graph_events.append(
                {
                    "event": event,
                    "rows": processed["rows"],
                    "top_rows": processed["top_rows"],
                    "selected_arxiv_ids": selected,
                }
            )
            all_pool.extend(row["paper_arxiv_id"] for row in processed["rows"])
            all_candidates.extend(top_ids)
            all_selected.extend(selected)

        summary = _metrics(gt_ids, all_candidates, all_selected)
        _prefix_stage(summary, "local_pool", _stage_metrics(gt_ids, all_pool))
        summary.update(
            {
                "query_id": events[0].get("query_id") if events else None,
                "benchmark_idx": events[0].get("benchmark_idx") if events else None,
                "method": "per_subquery_new_formula_shadow",
                "event_count": len(graph_events),
                "failed_events": graph_failures + selector_failures,
                "graph_failed_events": graph_failures,
                "selector_failed_events": selector_failures,
                "candidate_metrics_valid": graph_failures == 0,
                "selection_metrics_valid": graph_failures == 0 and selector_failures == 0,
            }
        )
        summary["metrics_valid"] = graph_failures == 0 and selector_failures == 0
        if graph_failures or selector_failures:
            summary["error"] = (
                f"per-subquery shadow had {graph_failures} graph failure(s) and "
                f"{selector_failures} selector failure(s); "
                "partial metrics are excluded from aggregate evaluation"
            )
        self.writer.append("per_subquery/query_results.jsonl", summary)
        return summary, graph_events

    @staticmethod
    def _deep_summary(
        *,
        method: str,
        first_event: Optional[Mapping[str, Any]],
        gt_ids: Set[str],
        deep_pool_ids: Sequence[str],
        selector_input_ids: Sequence[str],
        selected_ids: Sequence[str],
        source_graph_ids: Sequence[str],
        requested_occurrences: int,
        actual_occurrences: int,
        rerankable_occurrences: int,
        event_count: int,
        group_count: int,
        selector_call_count: int,
    ) -> Dict[str, Any]:
        summary = _metrics(gt_ids, selector_input_ids, selected_ids)
        _prefix_stage(summary, "deep_pool", _stage_metrics(gt_ids, deep_pool_ids))
        _prefix_stage(summary, "source_graph_pool", _stage_metrics(gt_ids, source_graph_ids))
        summary.update(
            {
                "method": method,
                "query_id": first_event.get("query_id") if first_event else None,
                "benchmark_idx": first_event.get("benchmark_idx") if first_event else None,
                "event_count": event_count,
                "subquery_group_count": group_count,
                "selector_call_count": selector_call_count,
                "deep_retrieval_requested_occurrence_count": requested_occurrences,
                "deep_retrieval_actual_occurrence_count": actual_occurrences,
                "deep_retrieval_budget_fulfillment_rate": _safe_div(actual_occurrences, requested_occurrences),
                "rerankable_occurrence_count": rerankable_occurrences,
                "unscorable_occurrence_count": max(0, actual_occurrences - rerankable_occurrences),
            }
        )
        return summary

    def _process_deep_event(self, prepared: Mapping[str, Any], gt_ids: Set[str]) -> Dict[str, Any]:
        all_deep_pool: List[str] = []
        all_selector_input: List[str] = []
        all_selected: List[str] = []
        all_source_graph: List[str] = []
        requested_occurrences = actual_occurrences = rerankable_occurrences = selector_calls = 0
        sources = [source for group in prepared["groups"] for source in group["events"]]
        first_event: Optional[Mapping[str, Any]] = sources[0]["event"] if sources else None

        def rerank_source(source):
            event = source["event"]
            key = (DEEP_EVENT_METHOD, source["retrieval_event_id"])
            pool = prepared["pools"].get(key) or []
            diagnostics = prepared["diagnostics"].get(key) or {}
            reranked = self.deep.rerank_pool(
                pool,
                query=str(event.get("query") or ""),
                subquery=str(event.get("subquery") or ""),
                cutoff=event.get("subquery_before_date") or event.get("query_date"),
            )
            target_k = max(0, int(event.get("selector_top_k") or 0))
            input_ids = reranked["rerank_order_arxiv_ids"][:target_k]
            papers = self.deep.papers_for_ids(
                input_ids, reranked["rows"], reranked["metadata"]
            )
            return {
                "source": source,
                "event": event,
                "diagnostics": diagnostics,
                "reranked": reranked,
                "input_ids": input_ids,
                "papers": papers,
            }

        contexts = _stable_thread_map(rerank_source, sources, self.event_workers)
        selector_outcomes = self._call_selectors(
            [
                (context["event"], context["papers"], str(context["event"].get("planner_checklist") or ""))
                for context in contexts
            ]
        )
        context_outcomes = list(zip(contexts, selector_outcomes))
        for context, (_, selector_error) in context_outcomes:
            event = context["event"]
            if selector_error is not None:
                raise RuntimeError(
                    f"deep-event Selector failed for {event.get('retrieval_event_id')}: {selector_error}"
                ) from selector_error

        for context, (selector_result, _) in context_outcomes:
            event = context["event"]
            selected, overview, reasons = selector_result
            source = context["source"]
            diagnostics = context["diagnostics"]
            reranked = context["reranked"]
            input_ids = context["input_ids"]
            selector_calls += int(bool(input_ids))
            selected_set = set(selected)
            input_rank = {paper_id: rank for rank, paper_id in enumerate(input_ids, start=1)}
            graph_pool = source["graph_local_pool_arxiv_ids"]
            graph_topk = source["graph_topk_arxiv_ids"]
            graph_pool_set, graph_topk_set = set(graph_pool), set(graph_topk)
            graph_row_by_id = {
                row["paper_arxiv_id"]: row for row in source.get("rows") or []
            }
            enriched_rows: List[Dict[str, Any]] = []
            for row in reranked["rows"]:
                paper_id = row["paper_arxiv_id"]
                graph_row = graph_row_by_id.get(paper_id) or {}
                enriched = {
                    **{field: event.get(field) for field in EVENT_FIELDS},
                    **dict(row),
                    "method": DEEP_EVENT_METHOD,
                    "source_graph_local_pool_size": len(graph_pool),
                    "source_graph_topk_size": len(graph_topk),
                    "in_source_graph_local_pool": paper_id in graph_pool_set,
                    "in_source_graph_topk": paper_id in graph_topk_set,
                    "source_graph_query_score_normalized": graph_row.get("query_score_normalized"),
                    "source_graph_subquery_score_normalized": graph_row.get("subquery_score_normalized"),
                    "source_graph_intent_score": graph_row.get("intent_score"),
                    "source_graph_path_count_normalized": graph_row.get("path_count_normalized"),
                    "source_graph_rerank_score": graph_row.get("rerank_score"),
                    "source_graph_rerank_rank": graph_row.get("rerank_rank"),
                    "in_selector_topk": paper_id in input_rank,
                    "selector_input_rank": input_rank.get(paper_id),
                    "selector_selected": paper_id in selected_set,
                    "selector_reason": reasons.get(paper_id, ""),
                    "is_ground_truth": paper_id in gt_ids,
                    "affects_next_iteration": False,
                }
                enriched_rows.append(enriched)
                self.writer.append("deep_event/paper_rows.jsonl", enriched, full_only=True)

            self.writer.append(
                "deep_event/comparisons.jsonl",
                {
                    **{field: event.get(field) for field in EVENT_FIELDS},
                    "method": DEEP_EVENT_METHOD,
                    "pool_comparison": id_set_comparison(
                        graph_pool, reranked["retrieval_order_arxiv_ids"]
                    ),
                    "topk_comparison": id_set_comparison(graph_topk, input_ids),
                },
            )
            self.writer.append(
                "deep_event/pool_records.jsonl",
                {
                    **{field: event.get(field) for field in EVENT_FIELDS},
                    "method": DEEP_EVENT_METHOD,
                    "source_graph_local_pool_arxiv_ids": graph_pool,
                    "source_graph_topk_arxiv_ids": graph_topk,
                    "deep_retrieval_order_arxiv_ids": reranked["retrieval_order_arxiv_ids"],
                    "deep_rerank_order_arxiv_ids": reranked["rerank_order_arxiv_ids"],
                    "deep_selector_topk_arxiv_ids": input_ids,
                    "deep_selected_arxiv_ids": selected,
                    "retrieval_diagnostics": diagnostics,
                    "deep_pool_rows": compact_deep_rows(enriched_rows),
                },
            )
            top_rows = [row for row in enriched_rows if row["paper_arxiv_id"] in input_rank]
            top_rows.sort(key=lambda row: row["selector_input_rank"])
            self.writer.append(
                "deep_event/selector_decisions.jsonl",
                selector_decision_record(event, top_rows, selected, overview, reasons),
                full_only=True,
            )

            requested_occurrences += int(diagnostics.get("requested_count") or 0)
            actual_occurrences += len(reranked["retrieval_order_arxiv_ids"])
            rerankable_occurrences += len(reranked["rerank_order_arxiv_ids"])
            all_deep_pool.extend(reranked["retrieval_order_arxiv_ids"])
            all_selector_input.extend(input_ids)
            all_selected.extend(selected)
            all_source_graph.extend(graph_pool)

        summary = self._deep_summary(
            method=DEEP_EVENT_METHOD,
            first_event=first_event,
            gt_ids=gt_ids,
            deep_pool_ids=all_deep_pool,
            selector_input_ids=all_selector_input,
            selected_ids=all_selected,
            source_graph_ids=all_source_graph,
            requested_occurrences=requested_occurrences,
            actual_occurrences=actual_occurrences,
            rerankable_occurrences=rerankable_occurrences,
            event_count=len(sources),
            group_count=len(prepared["groups"]),
            selector_call_count=selector_calls,
        )
        self.writer.append("deep_event/query_results.jsonl", summary)
        return summary

    def _process_deep_merged(self, prepared: Mapping[str, Any], gt_ids: Set[str]) -> Dict[str, Any]:
        all_deep_pool: List[str] = []
        all_selector_input: List[str] = []
        all_selected: List[str] = []
        all_source_graph: List[str] = []
        requested_occurrences = actual_occurrences = rerankable_occurrences = selector_calls = 0
        indexed_groups = list(enumerate(prepared["groups"], start=1))

        def prepare_group(indexed_group):
            group_index, group = indexed_group
            source_events = group["events"]
            event0 = source_events[0]["event"]
            key = (DEEP_MERGED_METHOD, group["subquery_key"])
            pool = prepared["pools"].get(key) or []
            diagnostics = prepared["diagnostics"].get(key) or {}
            reranked = self.deep.rerank_pool(
                pool,
                query=str(event0.get("query") or ""),
                subquery=group["subquery"],
                cutoff=group["subquery_before_date"],
            )
            row_by_id = {row["paper_arxiv_id"]: dict(row) for row in reranked["rows"]}
            graph_event_ids_by_paper: Dict[str, List[Any]] = {}
            graph_top_event_ids_by_paper: Dict[str, List[Any]] = {}
            graph_event_features_by_paper: Dict[str, List[Dict[str, Any]]] = {}
            for source in source_events:
                event_id = source["retrieval_event_id"]
                graph_top_set = set(source["graph_topk_arxiv_ids"])
                for graph_row in source.get("rows") or []:
                    paper_id = graph_row["paper_arxiv_id"]
                    graph_event_ids_by_paper.setdefault(paper_id, []).append(event_id)
                    graph_event_features_by_paper.setdefault(paper_id, []).append(
                        {
                            "retrieval_event_id": event_id,
                            "iteration_idx": source["event"].get("iteration_idx"),
                            "retrieval_page_idx": source["event"].get("retrieval_page_idx"),
                            "query_score_normalized": graph_row.get("query_score_normalized"),
                            "subquery_score_normalized": graph_row.get("subquery_score_normalized"),
                            "intent_score": graph_row.get("intent_score"),
                            "path_count_normalized": graph_row.get("path_count_normalized"),
                            "rerank_score": graph_row.get("rerank_score"),
                            "rerank_rank": graph_row.get("rerank_rank"),
                            "in_graph_selector_topk": paper_id in graph_top_set,
                        }
                    )
                for paper_id in source["graph_topk_arxiv_ids"]:
                    graph_top_event_ids_by_paper.setdefault(paper_id, []).append(event_id)

            graph_union = group["source_graph_pool_union_arxiv_ids"]
            graph_top_union = ordered_unique(
                paper_id
                for source in source_events
                for paper_id in source["graph_topk_arxiv_ids"]
            )
            cursor = 0
            slices: List[Dict[str, Any]] = []
            for slice_index, source in enumerate(source_events, start=1):
                event = source["event"]
                target_k = max(0, int(event.get("selector_top_k") or 0))
                input_ids = reranked["rerank_order_arxiv_ids"][cursor : cursor + target_k]
                rerank_start_rank = cursor + 1 if input_ids else None
                cursor += target_k
                papers = self.deep.papers_for_ids(
                    input_ids, reranked["rows"], reranked["metadata"]
                )
                slices.append(
                    {
                        "slice_index": slice_index,
                        "source": source,
                        "event": event,
                        "target_k": target_k,
                        "input_ids": input_ids,
                        "papers": papers,
                        "rerank_start_rank": rerank_start_rank,
                        "rerank_end_rank": (
                            rerank_start_rank + len(input_ids) - 1 if input_ids else None
                        ),
                    }
                )
            return {
                "group_index": group_index,
                "group": group,
                "source_events": source_events,
                "event0": event0,
                "diagnostics": diagnostics,
                "reranked": reranked,
                "row_by_id": row_by_id,
                "graph_event_ids_by_paper": graph_event_ids_by_paper,
                "graph_top_event_ids_by_paper": graph_top_event_ids_by_paper,
                "graph_event_features_by_paper": graph_event_features_by_paper,
                "graph_union": graph_union,
                "graph_top_union": graph_top_union,
                "slices": slices,
            }

        contexts = _stable_thread_map(prepare_group, indexed_groups, self.event_workers)
        selector_requests = [
            (slice_["event"], slice_["papers"], str(slice_["event"].get("planner_checklist") or ""))
            for context in contexts
            for slice_ in context["slices"]
        ]
        selector_outcomes = iter(self._call_selectors(selector_requests))
        for context in contexts:
            for slice_ in context["slices"]:
                selector_result, selector_error = next(selector_outcomes)
                if selector_error is not None:
                    raise RuntimeError(
                        "deep-merged Selector failed for "
                        f"{slice_['source'].get('retrieval_event_id')}: {selector_error}"
                    ) from selector_error
                slice_["selector_result"] = selector_result

        first_event: Optional[Mapping[str, Any]] = contexts[0]["event0"] if contexts else None
        event_count = sum(len(context["source_events"]) for context in contexts)

        for context in contexts:
            group_index = context["group_index"]
            group = context["group"]
            source_events = context["source_events"]
            event0 = context["event0"]
            diagnostics = context["diagnostics"]
            reranked = context["reranked"]
            row_by_id = context["row_by_id"]
            graph_event_ids_by_paper = context["graph_event_ids_by_paper"]
            graph_top_event_ids_by_paper = context["graph_top_event_ids_by_paper"]
            graph_event_features_by_paper = context["graph_event_features_by_paper"]
            graph_union = context["graph_union"]
            graph_top_union = context["graph_top_union"]
            slice_records: List[Dict[str, Any]] = []

            for slice_ in context["slices"]:
                slice_index = slice_["slice_index"]
                source = slice_["source"]
                event = slice_["event"]
                target_k = slice_["target_k"]
                input_ids = slice_["input_ids"]
                selected, overview, reasons = slice_["selector_result"]
                selector_calls += int(bool(input_ids))
                selected_set = set(selected)
                matching_pool = set(source["graph_local_pool_arxiv_ids"])
                matching_topk = set(source["graph_topk_arxiv_ids"])
                top_rows: List[Dict[str, Any]] = []
                for input_position, paper_id in enumerate(input_ids, start=1):
                    row = row_by_id[paper_id]
                    row.update(
                        {
                            "in_selector_topk": True,
                            "selector_slice_idx": slice_index,
                            "selector_event_id": source["retrieval_event_id"],
                            "selector_input_rank": input_position,
                            "selector_selected": paper_id in selected_set,
                            "selector_reason": reasons.get(paper_id, ""),
                            "in_matching_event_graph_local_pool": paper_id in matching_pool,
                            "in_matching_event_graph_topk": paper_id in matching_topk,
                        }
                    )
                    top_rows.append(row)
                slice_comparison = id_set_comparison(
                    source["graph_topk_arxiv_ids"], input_ids
                )
                slice_record = {
                    "selector_slice_idx": slice_index,
                    "retrieval_event_id": source["retrieval_event_id"],
                    "iteration_idx": event.get("iteration_idx"),
                    "retrieval_page_idx": event.get("retrieval_page_idx"),
                    "planner_checklist": event.get("planner_checklist"),
                    "selector_requested_top_k": target_k,
                    "selector_input_arxiv_ids": input_ids,
                    "selector_selected_arxiv_ids": selected,
                    "rerank_start_rank": slice_["rerank_start_rank"],
                    "rerank_end_rank": slice_["rerank_end_rank"],
                    "topk_comparison": slice_comparison,
                }
                slice_records.append(slice_record)
                decision_event = dict(event)
                decision_event["retrieval_event_id"] = source["retrieval_event_id"]
                self.writer.append(
                    "deep_merged/selector_decisions.jsonl",
                    selector_decision_record(
                        decision_event, top_rows, selected, overview, reasons
                    ),
                    full_only=True,
                )
                all_selector_input.extend(input_ids)
                all_selected.extend(selected)

            graph_union_set, graph_top_union_set = set(graph_union), set(graph_top_union)
            enriched_rows: List[Dict[str, Any]] = []
            for paper_id in reranked["rerank_order_arxiv_ids"] + reranked["unscorable_arxiv_ids"]:
                row = row_by_id[paper_id]
                enriched = {
                    **{field: event0.get(field) for field in EVENT_FIELDS},
                    **row,
                    "method": DEEP_MERGED_METHOD,
                    "merged_subquery_group_index": group_index,
                    "source_event_count": len(source_events),
                    "source_event_ids": [source["retrieval_event_id"] for source in source_events],
                    "source_graph_pool_occurrence_budget": group["source_graph_pool_occurrence_budget"],
                    "source_graph_pool_union_count": group["source_graph_pool_union_count"],
                    "source_graph_event_ids": graph_event_ids_by_paper.get(paper_id, []),
                    "source_graph_topk_event_ids": graph_top_event_ids_by_paper.get(paper_id, []),
                    "source_graph_event_features": graph_event_features_by_paper.get(paper_id, []),
                    "in_source_graph_local_pool": paper_id in graph_union_set,
                    "in_source_graph_topk": paper_id in graph_top_union_set,
                    "in_selector_topk": bool(row.get("in_selector_topk", False)),
                    "selector_selected": bool(row.get("selector_selected", False)),
                    "is_ground_truth": paper_id in gt_ids,
                    "affects_next_iteration": False,
                }
                enriched_rows.append(enriched)
                self.writer.append("deep_merged/paper_rows.jsonl", enriched, full_only=True)

            pool_comparison = id_set_comparison(
                graph_union, reranked["retrieval_order_arxiv_ids"]
            )
            all_slices_comparison = id_set_comparison(graph_top_union, [
                paper_id
                for record in slice_records
                for paper_id in record["selector_input_arxiv_ids"]
            ])
            comparison_record = {
                **{field: event0.get(field) for field in EVENT_FIELDS},
                "method": DEEP_MERGED_METHOD,
                "subquery_id": group["subquery_id"],
                "source_graph_pool_occurrence_budget": group["source_graph_pool_occurrence_budget"],
                "source_graph_pool_union_count": group["source_graph_pool_union_count"],
                "source_graph_pool_overlap_occurrence_count": group["source_graph_pool_overlap_occurrence_count"],
                "pool_comparison": pool_comparison,
                "all_slices_topk_comparison": all_slices_comparison,
                "slice_comparisons": slice_records,
            }
            self.writer.append("deep_merged/comparisons.jsonl", comparison_record)
            self.writer.append(
                "deep_merged/pool_records.jsonl",
                {
                    **{field: event0.get(field) for field in EVENT_FIELDS},
                    "method": DEEP_MERGED_METHOD,
                    "subquery_id": group["subquery_id"],
                    "source_event_budgets": [
                        {
                            "retrieval_event_id": source["retrieval_event_id"],
                            "iteration_idx": source["event"].get("iteration_idx"),
                            "retrieval_page_idx": source["event"].get("retrieval_page_idx"),
                            "retrieval_offset": source["event"].get("retrieval_offset"),
                            "selector_top_k": source["event"].get("selector_top_k"),
                            "source_graph_local_pool_size": source["source_local_graph_pool_size"],
                            "source_graph_local_pool_arxiv_ids": source["graph_local_pool_arxiv_ids"],
                            "source_graph_topk_arxiv_ids": source["graph_topk_arxiv_ids"],
                            "retrieval_exclusion_arxiv_ids": source["event"].get("retrieval_exclusion_arxiv_ids") or [],
                        }
                        for source in source_events
                    ],
                    "source_graph_pool_occurrence_budget": group["source_graph_pool_occurrence_budget"],
                    "source_graph_pool_union_arxiv_ids": graph_union,
                    "deep_retrieval_order_arxiv_ids": reranked["retrieval_order_arxiv_ids"],
                    "deep_rerank_order_arxiv_ids": reranked["rerank_order_arxiv_ids"],
                    "selector_slices": slice_records,
                    "retrieval_diagnostics": diagnostics,
                    "deep_pool_rows": compact_deep_rows(enriched_rows),
                },
            )

            requested_occurrences += int(diagnostics.get("requested_count") or 0)
            actual_occurrences += len(reranked["retrieval_order_arxiv_ids"])
            rerankable_occurrences += len(reranked["rerank_order_arxiv_ids"])
            all_deep_pool.extend(reranked["retrieval_order_arxiv_ids"])
            all_source_graph.extend(graph_union)

        summary = self._deep_summary(
            method=DEEP_MERGED_METHOD,
            first_event=first_event,
            gt_ids=gt_ids,
            deep_pool_ids=all_deep_pool,
            selector_input_ids=all_selector_input,
            selected_ids=all_selected,
            source_graph_ids=all_source_graph,
            requested_occurrences=requested_occurrences,
            actual_occurrences=actual_occurrences,
            rerankable_occurrences=rerankable_occurrences,
            event_count=event_count,
            group_count=len(prepared["groups"]),
            selector_call_count=selector_calls,
        )
        self.writer.append("deep_merged/query_results.jsonl", summary)
        return summary
