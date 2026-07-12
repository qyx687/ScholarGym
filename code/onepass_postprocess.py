#!/usr/bin/env python3
"""Per-query replay of per-subquery and global graph postprocessors."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from agent.selector import Selector
from graph_methods import (
    ArtifactWriter,
    CandidateIndex,
    PerSubqueryProcessor,
    S2GraphClient,
    month_key,
    normalize_arxiv_id,
    selector_decision_record,
)
from structures import Paper, SubQuery


DEFAULT_GLOBAL_CHECKLIST = (
    "Select papers that directly answer the original query; prefer concrete "
    "method papers, seminal works, and papers matching the requested entity/task."
)


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _phase3_max_normalize(raw_scores: Mapping[str, float]) -> Dict[str, float]:
    """Normalization used by phase3's named global rerank method."""
    if not raw_scores:
        return {}
    maximum = max(raw_scores.values())
    if maximum <= 0:
        return {paper_id: 0.0 for paper_id in raw_scores}
    return {paper_id: float(score) / (float(maximum) + 1e-12) for paper_id, score in raw_scores.items()}


def _metrics(gt_ids: Set[str], candidate_ids: Iterable[str], selected_ids: Iterable[str]) -> Dict[str, Any]:
    candidates = {normalize_arxiv_id(value) for value in candidate_ids if normalize_arxiv_id(value)}
    selected = {normalize_arxiv_id(value) for value in selected_ids if normalize_arxiv_id(value)}
    candidate_hits = sorted(gt_ids & candidates)
    selected_hits = sorted(gt_ids & selected)
    return {
        "gt_count": len(gt_ids),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidate_arxiv_ids": sorted(candidates),
        "selected_arxiv_ids": sorted(selected),
        "candidate_gt_ids": candidate_hits,
        "selected_gt_ids": selected_hits,
        "candidate_recall": _safe_div(len(candidate_hits), len(gt_ids)),
        "candidate_precision": _safe_div(len(candidate_hits), len(candidates)),
        "selection_recall": _safe_div(len(selected_hits), len(gt_ids)),
        "selection_precision": _safe_div(len(selected_hits), len(selected)),
    }


def aggregate_postprocess_metrics(detailed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate the two shadow postprocessors over unique benchmark queries.

    Checkpoint files are append-only and may contain more than one record for an
    ``idx`` after retries.  The last record for each benchmark index is therefore
    authoritative, matching the resume semantics used by evaluation.
    """
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
        "selection_recall",
        "selection_precision",
    )
    for method_name in ("per_subquery", "global"):
        method_results: List[Mapping[str, Any]] = []
        failed_query_count = 0
        missing_query_count = 0
        for query_result in canonical_results.values():
            postprocess_results = query_result.get("postprocess_results") or {}
            method_result = postprocess_results.get(method_name) if isinstance(postprocess_results, Mapping) else None
            if not isinstance(method_result, Mapping):
                missing_query_count += 1
                continue
            if method_result.get("error") or "gt_count" not in method_result:
                failed_query_count += 1
                continue
            method_results.append(method_result)

        evaluated_query_count = len(method_results)
        totals = {
            "gt_count": sum(int(result.get("gt_count") or 0) for result in method_results),
            "candidate_count": sum(int(result.get("candidate_count") or 0) for result in method_results),
            "selected_count": sum(int(result.get("selected_count") or 0) for result in method_results),
            "candidate_gt_count": sum(len(result.get("candidate_gt_ids") or []) for result in method_results),
            "selected_gt_count": sum(len(result.get("selected_gt_ids") or []) for result in method_results),
        }
        aggregate = {
            "evaluated_query_count": evaluated_query_count,
            "failed_query_count": failed_query_count,
            "missing_query_count": missing_query_count,
            **{f"total_{name}": value for name, value in totals.items()},
        }
        for metric_name in metric_names:
            aggregate[f"avg_{metric_name}"] = (
                sum(float(result.get(metric_name) or 0.0) for result in method_results) / evaluated_query_count
                if evaluated_query_count else 0.0
            )
        aggregate.update(
            {
                "micro_candidate_recall": _safe_div(totals["candidate_gt_count"], totals["gt_count"]),
                "micro_candidate_precision": _safe_div(totals["candidate_gt_count"], totals["candidate_count"]),
                "micro_selection_recall": _safe_div(totals["selected_gt_count"], totals["gt_count"]),
                "micro_selection_precision": _safe_div(totals["selected_gt_count"], totals["selected_count"]),
            }
        )
        methods[method_name] = aggregate

    return {
        "source_query_count": len(canonical_results),
        "macro_average_scope": "successful unique benchmark queries for each method",
        "candidate_scope": "deduplicated rerank top-k papers passed to the shadow Selector",
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
        scoring_backend: str,
        embedding_provider: Any,
        run_per_subquery: bool,
        run_global: bool,
        run_id: str = "",
        global_alpha: float = 0.5,
        global_checklist: str = DEFAULT_GLOBAL_CHECKLIST,
    ) -> None:
        self.selector = selector
        self.paper_db = dict(paper_db)
        self.writer = writer
        self.s2 = s2_client
        self.per_subquery = per_subquery_processor
        self.backend = scoring_backend
        self.embedding_provider = embedding_provider
        self.run_per_subquery = run_per_subquery
        self.run_global = run_global
        self.run_id = run_id
        self.global_alpha = float(global_alpha)
        self.global_checklist = global_checklist

    def process_query(
        self,
        query: Mapping[str, Any],
        planner_events: Sequence[Mapping[str, Any]],
        retrieval_events: Sequence[Mapping[str, Any]],
        gt_ids: Set[str],
    ) -> Dict[str, Any]:
        query_id = (
            retrieval_events[0].get("query_id")
            if retrieval_events else (query.get("qid") or query.get("query_id"))
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
        for event in planner_events:
            if isinstance(event, dict):
                event.setdefault("run_id", self.run_id)
        for event in retrieval_events:
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
        if self.run_per_subquery:
            try:
                result["per_subquery"] = self._process_per_subquery(retrieval_events, gt_ids)
            except Exception as exc:
                result["per_subquery"] = {"error": str(exc)}
                self.writer.append("per_subquery/errors.jsonl", {"query_id": result["query_id"], "error": str(exc)})
        if self.run_global:
            try:
                result["global"] = self._process_global(query, retrieval_events, gt_ids)
            except Exception as exc:
                result["global"] = {"error": str(exc)}
                self.writer.append("global/errors.jsonl", {"query_id": result["query_id"], "error": str(exc)})
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
                row = {
                    **{key: event.get(key) for key in (
                        "schema_version", "run_id", "retrieval_event_id", "query_id", "benchmark_idx", "query", "query_source", "query_date",
                        "iteration_idx", "subquery_id", "subquery", "subquery_target_k", "subquery_link_type",
                        "parent_subquery_id", "subquery_before_date",
                        "retrieval_page_idx", "retrieval_offset", "raw_retrieval_page_count", "results_per_query", "selector_top_k", "planner_checklist",
                    )},
                    "method": "baseline",
                    "paper_arxiv_id": paper_id,
                    "candidate_type": "seed",
                    "is_seed": True,
                    "is_expanded": False,
                    "passed_date_cutoff": True,
                    "date_cutoff_month": month_key(event.get("subquery_before_date") or event.get("query_date")),
                    "retrieval_backend": event.get("retrieval_backend"),
                    "observed_retrieval_score": seed.get("observed_retrieval_score"),
                    "observed_retrieval_rank": seed.get("observed_retrieval_rank"),
                    "observed_retrieval_absolute_rank": (
                        int(event.get("retrieval_offset") or 0) + int(seed["observed_retrieval_rank"])
                        if seed.get("observed_retrieval_rank") is not None else None
                    ),
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

    def _call_selector(self, event: Mapping[str, Any], papers: Sequence[Paper], checklist: str) -> Tuple[List[str], str, Dict[str, str]]:
        subquery = SubQuery(
            id=int(event.get("subquery_id") or 0),
            text=str(event.get("subquery") or event.get("query") or ""),
            before_date=event.get("subquery_before_date") or event.get("query_date"),
            target_k=int(event.get("subquery_target_k") or event.get("selector_top_k") or len(papers)),
            link_type=event.get("subquery_link_type"),
            source_subquery_id=event.get("parent_subquery_id"),
            iter_index=int(event.get("iteration_idx") or 0),
        )
        result = asyncio.run(
            self.selector.decide_for_subquery(
                user_query=str(event.get("query") or ""),
                sub_query=subquery,
                planner_checklist=checklist,
                papers=list(papers),
                iteration_index=int(event.get("iteration_idx") or 1),
                idx=int(event.get("benchmark_idx") or 0),
                old_overview="",
                is_after_browsing=False,
                return_details=True,
            )
        )
        selected, overview, _, details = result
        return [normalize_arxiv_id(paper.arxiv_id or paper.id) for paper in selected], overview, details.get("reasons") or {}

    def _process_per_subquery(self, events: Sequence[Mapping[str, Any]], gt_ids: Set[str]) -> Dict[str, Any]:
        all_candidates: List[str] = []
        all_selected: List[str] = []
        failures = 0
        for baseline_event in events:
            event = dict(baseline_event)
            event["method"] = "per_subquery_shadow"
            try:
                processed = self.per_subquery.process(event, gt_ids)
                selected, overview, reasons = self._call_selector(
                    event,
                    processed["papers"],
                    str(event.get("planner_checklist") or ""),
                )
                selected_set = set(selected)
                top_ids = {row["paper_arxiv_id"] for row in processed["top_rows"]}
                for row in processed["rows"]:
                    row["in_selector_topk"] = row["paper_arxiv_id"] in top_ids
                    row["selector_input_rank"] = row["rerank_rank"] if row["in_selector_topk"] else None
                    row["selector_selected"] = row["paper_arxiv_id"] in selected_set
                    row["selector_reason"] = reasons.get(row["paper_arxiv_id"], "")
                    row["affects_next_iteration"] = False
                    self.writer.append("per_subquery/paper_rows.jsonl", row, full_only=True)
                for edge in processed["edges"]:
                    self.writer.append("per_subquery/expansion_edges.jsonl", edge, full_only=True)
                self.writer.append(
                    "per_subquery/filter_stats.jsonl",
                    {
                        "query_id": event.get("query_id"),
                        "iteration_idx": event.get("iteration_idx"),
                        "subquery_id": event.get("subquery_id"),
                        **processed.get("filter_stats", {}),
                    },
                )
                decision = selector_decision_record(event, processed["top_rows"], selected, overview, reasons)
                self.writer.append("per_subquery/selector_decisions.jsonl", decision, full_only=True)
                all_candidates.extend(row["paper_arxiv_id"] for row in processed["top_rows"])
                all_selected.extend(selected)
            except Exception as exc:
                failures += 1
                self.writer.append(
                    "per_subquery/errors.jsonl",
                    {"query_id": event.get("query_id"), "subquery_id": event.get("subquery_id"), "error": str(exc)},
                )
        summary = _metrics(gt_ids, all_candidates, all_selected)
        summary["query_id"] = events[0].get("query_id") if events else None
        summary["benchmark_idx"] = events[0].get("benchmark_idx") if events else None
        summary["failed_events"] = failures
        self.writer.append("per_subquery/query_results.jsonl", summary)
        return summary

    def _process_global(self, query: Mapping[str, Any], events: Sequence[Mapping[str, Any]], gt_ids: Set[str]) -> Dict[str, Any]:
        if not events:
            summary = _metrics(gt_ids, [], [])
            self.writer.append("global/query_results.jsonl", summary)
            return summary
        first = events[0]
        seed_origins: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        observed_by_seed: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for event in events:
            for seed in event.get("seed_papers") or []:
                paper_id = normalize_arxiv_id(seed.get("paper_arxiv_id"))
                if paper_id:
                    seed_origins[paper_id].append(event)
                    observed_by_seed[paper_id].append(seed)
        seed_ids = list(seed_origins)
        # Count actual baseline retrieval records, including a paper retrieved
        # by more than one subquery/page. Expansion itself remains deduplicated.
        baseline_query_retrieval_count = sum(len(rows) for rows in observed_by_seed.values())
        raw_edges = self.s2.expand(seed_ids, self.per_subquery.method, self.per_subquery.limit)
        cutoff = month_key(query.get("date") or first.get("query_date"))
        kept_edges = []
        filter_stats = {
            "query_id": first.get("query_id"),
            "raw_edge_count": len(raw_edges),
            "missing_from_paper_db_count": 0,
            "missing_date_count": 0,
            "after_cutoff_count": 0,
            "kept_edge_count": 0,
        }
        for edge in raw_edges:
            expanded = normalize_arxiv_id(edge.get("expanded_arxiv_id"))
            metadata = self.paper_db.get(expanded)
            paper_month = month_key((metadata or {}).get("date"))
            if not metadata:
                filter_stats["missing_from_paper_db_count"] += 1
            elif not paper_month:
                filter_stats["missing_date_count"] += 1
            elif cutoff and paper_month > cutoff:
                filter_stats["after_cutoff_count"] += 1
            else:
                kept_edges.append(edge)
        filter_stats["kept_edge_count"] = len(kept_edges)
        self.writer.append("global/filter_stats.jsonl", filter_stats)
        expanded_ids = list(dict.fromkeys(edge["expanded_arxiv_id"] for edge in kept_edges))
        index = CandidateIndex(seed_ids + expanded_ids, self.paper_db, self.backend, self.embedding_provider)
        candidate_ids = index.ids
        original_query = str(query.get("query") or first.get("query") or "")
        query_raw, _, query_rank = index.score(original_query)
        query_norm = _phase3_max_normalize(query_raw)

        scoring_subqueries: List[Tuple[Any, str]] = []
        seen_subqueries = set()
        for event in events:
            key = (event.get("subquery_id"), str(event.get("subquery") or ""))
            if key not in seen_subqueries:
                seen_subqueries.add(key)
                scoring_subqueries.append(key)
        component_scores: Dict[Any, Dict[str, float]] = {}
        component_raw: Dict[Any, Dict[str, float]] = {}
        component_rank: Dict[Any, Dict[str, int]] = {}
        for subquery_id, subquery_text in scoring_subqueries:
            raw, _, ranks = index.score(subquery_text)
            component_raw[subquery_id] = raw
            component_scores[subquery_id] = _phase3_max_normalize(raw)
            component_rank[subquery_id] = ranks
        max_subquery_id: Dict[str, Any] = {}
        max_subquery_score: Dict[str, float] = {}
        for paper_id in candidate_ids:
            best_id, best_score = None, 0.0
            for subquery_id, _ in scoring_subqueries:
                score = component_scores[subquery_id].get(paper_id, 0.0)
                if score > best_score:
                    best_id, best_score = subquery_id, score
            max_subquery_id[paper_id] = best_id
            max_subquery_score[paper_id] = best_score
        if scoring_subqueries:
            final_scores = {
                paper_id: self.global_alpha * query_norm.get(paper_id, 0.0)
                + (1.0 - self.global_alpha) * max_subquery_score.get(paper_id, 0.0)
                for paper_id in candidate_ids
            }
        else:
            # Match phase3's no-subquery fallback to original-query-only.
            final_scores = {paper_id: query_norm.get(paper_id, 0.0) for paper_id in candidate_ids}
        ordered = sorted(candidate_ids, key=lambda paper_id: (-final_scores[paper_id], -int(paper_id in seed_origins), paper_id))
        final_rank = {paper_id: rank for rank, paper_id in enumerate(ordered, start=1)}
        edges_by_expanded: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for edge in kept_edges:
            edges_by_expanded[edge["expanded_arxiv_id"]].append(edge)

        for subquery_id, subquery_text in scoring_subqueries:
            for paper_id in ordered:
                source_edges = edges_by_expanded.get(paper_id, [])
                origin_subquery_ids = sorted(
                    {origin.get("subquery_id") for origin in seed_origins.get(paper_id, []) if origin.get("subquery_id") is not None},
                    key=str,
                )
                source_seed_ids = sorted({edge["seed_arxiv_id"] for edge in source_edges})
                source_subquery_ids = sorted(
                    {
                        origin.get("subquery_id")
                        for seed_id in source_seed_ids
                        for origin in seed_origins.get(seed_id, [])
                        if origin.get("subquery_id") is not None
                    },
                    key=str,
                )
                row = {
                    "run_id": first.get("run_id"),
                    "method": "global_original_plus_all_subqueries_max",
                    "query_id": first.get("query_id"),
                    "benchmark_idx": first.get("benchmark_idx"),
                    "query": original_query,
                    "subquery_id": subquery_id,
                    "subquery": subquery_text,
                    "paper_arxiv_id": paper_id,
                    "candidate_type": "seed_and_expanded" if paper_id in seed_origins and source_edges else ("seed" if paper_id in seed_origins else "expanded"),
                    "origin_retrieval_subquery_ids": origin_subquery_ids,
                    "source_seed_arxiv_ids": source_seed_ids,
                    "source_seed_subquery_ids": source_subquery_ids,
                    "retrieval_backend": self.backend,
                    "retrieval_score_raw": component_raw[subquery_id].get(paper_id, 0.0),
                    "retrieval_score_normalized": component_scores[subquery_id].get(paper_id, 0.0),
                    "retrieval_rank": component_rank[subquery_id].get(paper_id),
                    "retrieval_rank_scope": "global_closed_seed_expanded_pool",
                    "query_score_raw": query_raw.get(paper_id, 0.0),
                    "query_score_normalized": query_norm.get(paper_id, 0.0),
                    "query_component_rank": query_rank.get(paper_id),
                    "subquery_score_raw": component_raw[subquery_id].get(paper_id, 0.0),
                    "subquery_score_normalized": component_scores[subquery_id].get(paper_id, 0.0),
                    "subquery_component_rank": component_rank[subquery_id].get(paper_id),
                    "is_max_subquery": subquery_id == max_subquery_id.get(paper_id),
                    "final_max_subquery_id": max_subquery_id.get(paper_id),
                    "final_max_subquery_score": max_subquery_score.get(paper_id, 0.0),
                    "global_alpha": self.global_alpha,
                    "global_final_score": final_scores.get(paper_id, 0.0),
                    "global_final_rank": final_rank.get(paper_id),
                    "date_cutoff_month": cutoff,
                    "passed_date_cutoff": True,
                    "is_ground_truth": paper_id in gt_ids,
                }
                self.writer.append("global/subquery_paper_scores.jsonl", row, full_only=True)

        # Match the global Selector input budget to the total number of actual
        # baseline retrieval records across every subquery/page. This is not
        # the per-request --results_per_query value.
        top_k = baseline_query_retrieval_count
        top_ids = ordered[:top_k]
        papers = []
        final_rows = []
        for paper_id in ordered:
            metadata = self.paper_db.get(paper_id, {})
            row = {
                "run_id": first.get("run_id"),
                "method": "global_original_plus_all_subqueries_max",
                "query_id": first.get("query_id"),
                "benchmark_idx": first.get("benchmark_idx"),
                "paper_arxiv_id": paper_id,
                "query_score_raw": query_raw.get(paper_id, 0.0),
                "query_score_normalized": query_norm.get(paper_id, 0.0),
                "max_subquery_id": max_subquery_id.get(paper_id),
                "max_subquery_score": max_subquery_score.get(paper_id, 0.0),
                "global_alpha": self.global_alpha,
                "global_final_score": final_scores.get(paper_id, 0.0),
                "global_final_rank": final_rank.get(paper_id),
                "baseline_query_retrieval_count": baseline_query_retrieval_count,
                "baseline_unique_retrieved_paper_count": len(seed_ids),
                "global_selector_top_k": top_k,
                "in_selector_topk": paper_id in set(top_ids),
                "date_cutoff_month": cutoff,
                "passed_date_cutoff": True,
                "is_ground_truth": paper_id in gt_ids,
            }
            final_rows.append(row)
            if paper_id in top_ids:
                papers.append(Paper(id=paper_id, arxiv_id=paper_id, title=str(metadata.get("title") or "N/A"), abstract=str(metadata.get("abstract") or "N/A"), date=metadata.get("date") or "", score=final_scores[paper_id]))
        global_event = {
            **dict(first),
            "method": "global_original_plus_all_subqueries_max",
            "subquery_id": None,
            "subquery": original_query,
            "subquery_target_k": top_k,
            "subquery_link_type": None,
            "parent_subquery_id": None,
            "planner_checklist": self.global_checklist,
            "results_per_query": top_k,
            "selector_top_k": top_k,
        }
        selected, overview, reasons = self._call_selector(global_event, papers, self.global_checklist)
        selected_set = set(selected)
        for row in final_rows:
            row["selector_selected"] = row["paper_arxiv_id"] in selected_set
            row["selector_reason"] = reasons.get(row["paper_arxiv_id"], "")
            self.writer.append("global/final_paper_rows.jsonl", row, full_only=True)
        for edge in kept_edges:
            for origin in seed_origins.get(edge["seed_arxiv_id"], []):
                edge_row = {
                    "run_id": first.get("run_id"),
                    "method": "global_original_plus_all_subqueries_max",
                    "query_id": first.get("query_id"),
                    "benchmark_idx": first.get("benchmark_idx"),
                    "query": original_query,
                    "iteration_idx": origin.get("iteration_idx"),
                    "subquery_id": origin.get("subquery_id"),
                    "subquery": origin.get("subquery"),
                    "retrieval_page_idx": origin.get("retrieval_page_idx"),
                    **edge,
                    "date_cutoff_month": cutoff,
                    "passed_date_cutoff": True,
                }
                self.writer.append("global/expansion_edges.jsonl", edge_row, full_only=True)
        decision_rows = [row for row in final_rows if row["in_selector_topk"]]
        self.writer.append(
            "global/selector_decisions.jsonl",
            selector_decision_record(global_event, decision_rows, selected, overview, reasons),
            full_only=True,
        )
        summary = _metrics(gt_ids, top_ids, selected)
        summary["query_id"] = first.get("query_id")
        summary["benchmark_idx"] = first.get("benchmark_idx")
        summary["candidate_pool_count"] = len(candidate_ids)
        summary["baseline_query_retrieval_count"] = baseline_query_retrieval_count
        summary["baseline_unique_retrieved_paper_count"] = len(seed_ids)
        summary["global_selector_top_k"] = top_k
        self.writer.append("global/query_results.jsonl", summary)
        return summary
