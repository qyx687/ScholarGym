#!/usr/bin/env python3
"""Per-query graph rerank and the merged text-only deep-retrieval shadow."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from agent.selector import Selector
from deep_retrieval import (
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


POSTPROCESS_METHODS = ("per_subquery", "deep_merged")
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
    "is_seed",
    "is_expanded",
    "source_seed_arxiv_ids",
    "source_subquery_ids",
    "edge_types",
    "expansion_path_count",
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
    "component_rank_scope",
    "normalization_scope",
    "intent_labels",
    "intent_score",
    "path_count",
    "path_count_normalized",
    "materialization_order_rank",
    "materialization_order_scope",
    "rerank_formula_id",
    "rerank_policy_id",
    "dynamic_rerank_enabled",
    "rerank_used_fallback",
    "feature_weights",
    "compiled_feature_weights",
    "intent_background",
    "intent_method",
    "intent_result",
    "paper_type_probs",
    "paper_type_backend",
    "paper_type_namespace",
    "paper_type_classifier_confidence",
    "paper_type_evidence_source",
    "paper_type_publication_types",
    "paper_type_supported_types",
    "paper_type_negative_evidence_types",
    "paper_type_alignment",
    "paper_type_filter_action",
    "paper_type_filter_reason",
    "hard_filtered",
    "component_contributions",
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


def _aggregate_materialization_metrics(
    canonical_results: Mapping[Tuple[str, Any], Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate pool coverage without inventing rerank/Selector metrics."""
    methods: Dict[str, Dict[str, Any]] = {}
    for method_name in POSTPROCESS_METHODS:
        materialized: List[Mapping[str, Any]] = []
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
            if method_result.get("error") or not method_result.get(
                "materialization_complete", False
            ):
                failed_query_count += 1
                continue
            materialized.append(method_result)

        pool_prefix = "local_pool" if method_name == "per_subquery" else "deep_pool"
        count = len(materialized)
        pool_evaluable = [
            result
            for result in materialized
            if int(result.get("ground_truth_count") or 0) > 0
        ]
        pool_evaluated_count = len(pool_evaluable)
        total_materialized_pool = sum(
            int(result.get(f"{pool_prefix}_count") or 0)
            for result in materialized
        )
        total_ground_truth = sum(
            int(result.get("ground_truth_count") or 0)
            for result in pool_evaluable
        )
        total_pool = sum(
            int(result.get(f"{pool_prefix}_count") or 0)
            for result in pool_evaluable
        )
        total_pool_gt = sum(
            int(result.get(f"{pool_prefix}_gt_count") or 0)
            for result in pool_evaluable
        )
        aggregate: Dict[str, Any] = {
            "materialized_query_count": count,
            "pool_evaluated_query_count": pool_evaluated_count,
            "queries_without_ground_truth_count": count - pool_evaluated_count,
            "failed_query_count": failed_query_count,
            "missing_query_count": missing_query_count,
            "total_materialized_pool_count": total_materialized_pool,
            "total_ground_truth_count": total_ground_truth,
            "total_pool_count": total_pool,
            "total_pool_gt_count": total_pool_gt,
            "avg_pool_recall": (
                sum(
                    float(result.get(f"{pool_prefix}_recall") or 0.0)
                    for result in pool_evaluable
                )
                / pool_evaluated_count
                if pool_evaluated_count
                else None
            ),
            "avg_pool_precision": (
                sum(
                    float(result.get(f"{pool_prefix}_precision") or 0.0)
                    for result in pool_evaluable
                )
                / pool_evaluated_count
                if pool_evaluated_count
                else None
            ),
            "micro_pool_recall": (
                _safe_div(total_pool_gt, total_ground_truth)
                if total_ground_truth
                else None
            ),
            "micro_pool_precision": (
                _safe_div(total_pool_gt, total_pool) if total_pool else None
            ),
            "candidate_metrics_available": False,
            "selection_metrics_available": False,
            "candidate_evaluated_query_count": 0,
            "selection_evaluated_query_count": 0,
            "avg_candidate_recall": None,
            "avg_candidate_precision": None,
            "avg_selection_recall": None,
            "avg_selection_precision": None,
            "micro_candidate_recall": None,
            "micro_candidate_precision": None,
            "micro_selection_recall": None,
            "micro_selection_precision": None,
        }
        if method_name != "per_subquery":
            total_materialized_source_pool = sum(
                int(result.get("source_graph_pool_count") or 0)
                for result in materialized
            )
            total_source_pool = sum(
                int(result.get("source_graph_pool_count") or 0)
                for result in pool_evaluable
            )
            total_source_gt = sum(
                int(result.get("source_graph_pool_gt_count") or 0)
                for result in pool_evaluable
            )
            aggregate.update(
                {
                    "total_materialized_source_graph_pool_count": total_materialized_source_pool,
                    "total_source_graph_pool_count": total_source_pool,
                    "total_source_graph_pool_gt_count": total_source_gt,
                    "avg_source_graph_pool_recall": (
                        sum(
                            float(result.get("source_graph_pool_recall") or 0.0)
                            for result in pool_evaluable
                        )
                        / pool_evaluated_count
                        if pool_evaluated_count
                        else None
                    ),
                    "micro_source_graph_pool_recall": (
                        _safe_div(total_source_gt, total_ground_truth)
                        if total_ground_truth
                        else None
                    ),
                }
            )
        methods[method_name] = aggregate

    return {
        "source_query_count": len(canonical_results),
        "postprocess_stage": "materialize",
        "macro_average_scope": "successful unique benchmark queries for each materialized pool",
        "candidate_scope": None,
        "candidate_scope_status": "not_computed_in_stage_a",
        "selection_scope_status": "not_computed_in_stage_a",
        **methods,
    }


def aggregate_postprocess_metrics(detailed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate the graph/deep shadows over authoritative benchmark records."""
    canonical_results: Dict[Tuple[str, Any], Mapping[str, Any]] = {}
    for position, query_result in enumerate(detailed_results or []):
        if not isinstance(query_result, Mapping):
            continue
        benchmark_idx = query_result.get("idx")
        identity = ("idx", benchmark_idx) if benchmark_idx is not None else ("position", position)
        canonical_results[identity] = query_result

    if any(
        isinstance(query_result.get("postprocess_results"), Mapping)
        and query_result["postprocess_results"].get("postprocess_stage")
        == "materialize"
        for query_result in canonical_results.values()
    ):
        return _aggregate_materialization_metrics(canonical_results)

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
        run_deep_merged: bool = False,
        postprocess_stage: str = "full",
        run_id: str = "",
        event_workers: int = 4,
        selector_concurrency: int = 4,
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
        self.run_deep_merged = run_deep_merged
        if postprocess_stage not in {"full", "materialize"}:
            raise ValueError("postprocess_stage must be full or materialize")
        self.postprocess_stage = postprocess_stage
        self.materialize_only = postprocess_stage == "materialize"
        self.run_id = run_id
        self.event_workers = int(event_workers)
        self.selector_concurrency = int(selector_concurrency)
        if self.event_workers < 1:
            raise ValueError("event_workers must be >= 1")
        if self.selector_concurrency < 1:
            raise ValueError("selector_concurrency must be >= 1")
        if run_deep_merged and deep_retrieval_processor is None:
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
        begin_cache_scope = getattr(
            self.embedding_provider, "begin_query_scope", None
        )
        end_cache_scope = getattr(self.embedding_provider, "end_query_scope", None)
        cache_scope_started = False
        if callable(begin_cache_scope):
            begin_cache_scope(
                f"benchmark_idx={benchmark_idx};query_id={query_id}"
            )
            cache_scope_started = True
        try:
            self.writer.begin_query(query_id, benchmark_idx)
            try:
                result = self._process_query(
                    query, planner_events, retrieval_events, gt_ids
                )
                self.writer.commit_query()
                return result
            except BaseException:
                self.writer.abort_query()
                raise
        finally:
            if cache_scope_started and callable(end_cache_scope):
                end_cache_scope()

    def _process_query(
        self,
        query: Mapping[str, Any],
        planner_events: Sequence[Mapping[str, Any]],
        retrieval_events: Sequence[Mapping[str, Any]],
        gt_ids: Set[str],
    ) -> Dict[str, Any]:
        s2_before = self.s2.snapshot_stats()
        resolver = self.per_subquery.paper_type_resolver
        paper_type_before = resolver.snapshot_stats() if resolver else {}
        for event in list(planner_events) + list(retrieval_events):
            if isinstance(event, dict):
                event.setdefault("run_id", self.run_id)
        for event in planner_events:
            self.writer.append("baseline/planner_events.jsonl", event, full_only=True)
        self._write_baseline(retrieval_events, gt_ids)
        result: Dict[str, Any] = {
            "query_id": retrieval_events[0].get("query_id") if retrieval_events else (query.get("qid") or query.get("query_id")),
            "benchmark_idx": retrieval_events[0].get("benchmark_idx") if retrieval_events else None,
            "postprocess_stage": self.postprocess_stage,
            "baseline": self._baseline_summary(retrieval_events, gt_ids),
        }

        original_query = str(
            query.get("query")
            or (retrieval_events[0].get("query") if retrieval_events else "")
            or ""
        )
        query_id = str(result.get("query_id") or "")
        benchmark_idx = result.get("benchmark_idx")
        policy = compiled_policy = None
        if not self.materialize_only:
            policy, compiled_policy = self.per_subquery.configure_query(original_query)
        paper_type_provenance = {
            "paper_type_backend": getattr(resolver, "backend", None),
            "paper_type_namespace": getattr(
                self.per_subquery.rerank_skill,
                "paper_type_namespace",
                None,
            ),
            "paper_type_evidence_source": getattr(
                resolver, "evidence_source", None
            ),
            "paper_type_classifier_version": getattr(
                resolver, "classifier_version", None
            ),
            "paper_type_model": getattr(resolver, "model", None),
            "paper_type_supported_types": list(
                getattr(resolver, "supported_types", ()) or ()
            ),
        }
        if policy is not None and compiled_policy is not None:
            assert self.per_subquery.rerank_skill is not None
            policy_record = self.per_subquery.rerank_skill.artifact_record(
                query_id=query_id,
                original_query=original_query,
                policy=policy,
                compiled=compiled_policy,
            )
            policy_record.update(
                {
                    "schema_version": "1.0",
                    "run_id": self.run_id,
                    "benchmark_idx": benchmark_idx,
                    "formula_scope": "one_policy_per_original_query_all_events",
                    "affects_next_iteration": False,
                    "dynamic_rerank_enabled": not compiled_policy.used_fallback,
                    "rerank_formula_id": (
                        "q030_sq040_intent015_path015_closed_pool_minmax_v1"
                        if compiled_policy.used_fallback
                        else "dynamic_rerank_v1"
                    ),
                    **paper_type_provenance,
                }
            )
        else:
            policy_record = {
                "schema_version": "1.0",
                "run_id": self.run_id,
                "query_id": query_id,
                "benchmark_idx": benchmark_idx,
                "original_query": original_query,
                "rerank_policy_id": "legacy-static",
                "rerank_formula_id": "q030_sq040_intent015_path015_closed_pool_minmax_v1",
                "compiled_weights": dict(self.per_subquery.weights),
                "dynamic_rerank_enabled": False,
                "formula_scope": "one_static_formula_all_queries",
                "affects_next_iteration": False,
                **paper_type_provenance,
            }
        result["rerank_policy"] = policy_record
        self.writer.append("query_rerank_policies.jsonl", policy_record)

        graph_events: List[Dict[str, Any]] = []
        if self.run_per_subquery:
            try:
                result["per_subquery"], graph_events = self._process_per_subquery(
                    retrieval_events, gt_ids
                )
            except Exception as exc:
                result["per_subquery"] = {
                    "postprocess_stage": self.postprocess_stage,
                    "materialization_complete": False,
                    "error": str(exc),
                }
                self.writer.append(
                    "per_subquery/errors.jsonl",
                    {"query_id": result["query_id"], "error": str(exc)},
                )

        if self.run_deep_merged:
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
                prepared = self.deep.prepare_pools(graph_events)
            except Exception as exc:
                result["deep_merged"] = {
                    "postprocess_stage": self.postprocess_stage,
                    "materialization_complete": False,
                    "error": str(exc),
                }
                self.writer.append(
                    "deep_merged/errors.jsonl",
                    {"query_id": result["query_id"], "error": str(exc)},
                )
            else:
                try:
                    result["deep_merged"] = self._process_deep_merged(prepared, gt_ids)
                except Exception as exc:
                    result["deep_merged"] = {
                        "postprocess_stage": self.postprocess_stage,
                        "materialization_complete": False,
                        "error": str(exc),
                    }
                    self.writer.append(
                        "deep_merged/errors.jsonl",
                        {"query_id": result["query_id"], "error": str(exc)},
                    )

        if self.materialize_only:
            enabled_methods = [
                name
                for name, enabled in (
                    ("per_subquery", self.run_per_subquery),
                    ("deep_merged", self.run_deep_merged),
                )
                if enabled
            ]
            incomplete_methods = [
                name
                for name in enabled_methods
                if not isinstance(result.get(name), Mapping)
                or not result[name].get("materialization_complete", False)
            ]
            if incomplete_methods:
                raise RuntimeError(
                    "Stage A query materialization is incomplete for: "
                    + ", ".join(incomplete_methods)
                    + "; the query transaction was not committed and can be retried"
                )

        s2_after = self.s2.snapshot_stats()
        result["s2_stats_delta"] = {
            key: int(s2_after.get(key, 0)) - int(s2_before.get(key, 0))
            for key in set(s2_before) | set(s2_after)
        }
        result["s2_stats_cumulative"] = s2_after
        paper_type_after = resolver.snapshot_stats() if resolver else {}
        result["paper_type_stats_delta"] = {
            key: int(paper_type_after.get(key, 0))
            - int(paper_type_before.get(key, 0))
            for key in set(paper_type_before) | set(paper_type_after)
        }
        result["paper_type_stats_cumulative"] = paper_type_after
        result.update(paper_type_provenance)
        snapshot_embedding_cache = getattr(
            self.embedding_provider, "snapshot_query_stats", None
        )
        if callable(snapshot_embedding_cache):
            result["postprocess_embedding_cache_stats"] = (
                snapshot_embedding_cache()
            )
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
        method_name = (
            "per_subquery_pool_feature_materialization"
            if self.materialize_only
            else (
                "per_subquery_dynamic_rerank_shadow"
                if self.per_subquery.dynamic_rerank_enabled
                else "per_subquery_static_rerank_shadow"
            )
        )
        all_pool: List[str] = []
        all_candidates: List[str] = []
        all_selected: List[str] = []
        graph_events: List[Dict[str, Any]] = []
        graph_failures = 0
        selector_failures = 0

        def process_graph_event(baseline_event):
            event = dict(baseline_event)
            event["method"] = method_name
            try:
                processor = (
                    self.per_subquery.materialize
                    if self.materialize_only
                    else self.per_subquery.process
                )
                processed = processor(
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
        selector_by_position: Dict[
            int,
            Tuple[
                Optional[Tuple[List[str], str, Dict[str, str]]],
                Optional[Exception],
            ],
        ] = {}
        if not self.materialize_only:
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
                        "stage": (
                            "graph_expand_feature_materialization"
                            if self.materialize_only
                            else "graph_expand_rerank"
                        ),
                        "error": str(graph_error),
                    },
                )
                continue

            if self.materialize_only:
                selected, overview, reasons = [], "", {}
            else:
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
            top_rows = list(processed.get("top_rows") or [])
            top_ids = ordered_unique(row["paper_arxiv_id"] for row in top_rows)
            top_set = set(top_ids)
            for row in processed["rows"]:
                paper_id = row["paper_arxiv_id"]
                row["postprocess_stage"] = self.postprocess_stage
                row["features_materialized"] = True
                row["legacy_rerank_applied"] = bool(
                    processed.get("legacy_rerank_applied")
                )
                row["shadow_selector_applied"] = not self.materialize_only
                if not self.materialize_only:
                    row["in_selector_topk"] = paper_id in top_set
                    row["selector_input_rank"] = (
                        row["rerank_rank"] if paper_id in top_set else None
                    )
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
            pool_record = {
                **{key: event.get(key) for key in EVENT_FIELDS},
                "method": method_name,
                "postprocess_stage": self.postprocess_stage,
                "features_materialized": True,
                "legacy_rerank_applied": bool(
                    processed.get("legacy_rerank_applied")
                ),
                "dynamic_rerank_enabled": bool(
                    (processed.get("compiled_rerank_policy") or {}).get(
                        "used_fallback"
                    )
                    is False
                ),
                "rerank_policy_id": processed.get("rerank_policy_id"),
                "compiled_rerank_policy": processed.get(
                    "compiled_rerank_policy"
                ),
                "shadow_selector_applied": not self.materialize_only,
                "local_pool_size": len(processed["rows"]),
                "local_pool_arxiv_ids": ordered_unique(
                    row["paper_arxiv_id"] for row in processed["rows"]
                ),
                "local_pool_rows": _compact_graph_rows(processed["rows"]),
            }
            if not self.materialize_only:
                pool_record.update(
                    {
                        "selector_topk_arxiv_ids": top_ids,
                        "selected_arxiv_ids": selected,
                    }
                )
            self.writer.append("per_subquery/pool_records.jsonl", pool_record)
            if not self.materialize_only:
                self.writer.append(
                    "per_subquery/selector_decisions.jsonl",
                    selector_decision_record(
                        event, top_rows, selected, overview, reasons
                    ),
                    full_only=True,
                )
            graph_events.append(
                {
                    "event": event,
                    "rows": processed["rows"],
                    "top_rows": top_rows,
                    "selected_arxiv_ids": selected,
                }
            )
            all_pool.extend(row["paper_arxiv_id"] for row in processed["rows"])
            all_candidates.extend(top_ids)
            all_selected.extend(selected)

        summary = (
            {} if self.materialize_only else _metrics(gt_ids, all_candidates, all_selected)
        )
        _prefix_stage(summary, "local_pool", _stage_metrics(gt_ids, all_pool))
        summary.update(
            {
                "query_id": events[0].get("query_id") if events else None,
                "benchmark_idx": events[0].get("benchmark_idx") if events else None,
                "method": method_name,
                "postprocess_stage": self.postprocess_stage,
                "ground_truth_count": len(gt_ids),
                "event_count": len(graph_events),
                "failed_events": graph_failures + selector_failures,
                "graph_failed_events": graph_failures,
                "selector_failed_events": selector_failures,
                "selector_call_count": (
                    0
                    if self.materialize_only
                    else sum(
                        bool(graph_outcomes[index][1].get("papers"))
                        for index in successful_positions
                    )
                ),
                "features_materialized": graph_failures == 0,
                "legacy_rerank_applied": bool(
                    not self.materialize_only
                    and (
                        self.per_subquery.active_compiled_policy is None
                        or self.per_subquery.active_compiled_policy.used_fallback
                    )
                ),
                "dynamic_rerank_enabled": bool(
                    not self.materialize_only
                    and self.per_subquery.active_compiled_policy is not None
                    and not self.per_subquery.active_compiled_policy.used_fallback
                ),
                "rerank_policy_id": (
                    self.per_subquery.active_compiled_policy.policy_id
                    if self.per_subquery.active_compiled_policy is not None
                    else None
                ),
                "compiled_rerank_policy": (
                    self.per_subquery.active_compiled_policy.to_dict()
                    if self.per_subquery.active_compiled_policy is not None
                    else None
                ),
                "shadow_selector_applied": not self.materialize_only,
                "materialization_complete": graph_failures == 0,
                "candidate_metrics_available": not self.materialize_only,
                "selection_metrics_available": not self.materialize_only,
                "candidate_metrics_valid": (
                    False if self.materialize_only else graph_failures == 0
                ),
                "selection_metrics_valid": (
                    False
                    if self.materialize_only
                    else graph_failures == 0 and selector_failures == 0
                ),
            }
        )
        summary["metrics_valid"] = (
            graph_failures == 0
            if self.materialize_only
            else graph_failures == 0 and selector_failures == 0
        )
        if graph_failures or selector_failures:
            if self.materialize_only:
                summary["error"] = (
                    f"per-subquery materialization had {graph_failures} graph/feature "
                    "failure(s); partial pools are excluded from aggregate evaluation"
                )
            else:
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

    @staticmethod
    def _deep_materialization_summary(
        *,
        method: str,
        first_event: Optional[Mapping[str, Any]],
        gt_ids: Set[str],
        deep_pool_ids: Sequence[str],
        source_graph_ids: Sequence[str],
        requested_occurrences: int,
        actual_occurrences: int,
        feature_scorable_occurrences: int,
        event_count: int,
        group_count: int,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        _prefix_stage(summary, "deep_pool", _stage_metrics(gt_ids, deep_pool_ids))
        _prefix_stage(
            summary, "source_graph_pool", _stage_metrics(gt_ids, source_graph_ids)
        )
        summary.update(
            {
                "method": method,
                "postprocess_stage": "materialize",
                "query_id": first_event.get("query_id") if first_event else None,
                "benchmark_idx": (
                    first_event.get("benchmark_idx") if first_event else None
                ),
                "ground_truth_count": len(gt_ids),
                "event_count": event_count,
                "subquery_group_count": group_count,
                "selector_call_count": 0,
                "deep_retrieval_requested_occurrence_count": requested_occurrences,
                "deep_retrieval_actual_occurrence_count": actual_occurrences,
                "deep_retrieval_budget_fulfillment_rate": _safe_div(
                    actual_occurrences, requested_occurrences
                ),
                "feature_scorable_occurrence_count": feature_scorable_occurrences,
                "feature_unscorable_occurrence_count": max(
                    0, actual_occurrences - feature_scorable_occurrences
                ),
                "features_materialized": True,
                "legacy_rerank_applied": False,
                "shadow_selector_applied": False,
                "materialization_complete": True,
                "candidate_metrics_available": False,
                "selection_metrics_available": False,
                "candidate_metrics_valid": False,
                "selection_metrics_valid": False,
            }
        )
        return summary

    def _materialize_deep_merged(
        self,
        prepared: Mapping[str, Any],
        gt_ids: Set[str],
    ) -> Dict[str, Any]:
        """Persist merged stable-subquery pools without formula/slices/Selector."""
        all_deep_pool: List[str] = []
        all_source_graph: List[str] = []
        requested_occurrences = actual_occurrences = scorable_occurrences = 0
        groups = list(prepared["groups"])
        first_event: Optional[Mapping[str, Any]] = (
            groups[0]["events"][0]["event"] if groups else None
        )
        event_count = sum(len(group["events"]) for group in groups)

        def materialize_group(indexed_group):
            group_index, group = indexed_group
            source_events = group["events"]
            event0 = source_events[0]["event"]
            key = (DEEP_MERGED_METHOD, group["subquery_key"])
            pool = prepared["pools"].get(key) or []
            return {
                "group_index": group_index,
                "group": group,
                "source_events": source_events,
                "event0": event0,
                "diagnostics": prepared["diagnostics"].get(key) or {},
                "materialized": self.deep.materialize_pool_features(
                    pool,
                    query=str(event0.get("query") or ""),
                    subquery=group["subquery"],
                    cutoff=group["subquery_before_date"],
                ),
            }

        contexts = _stable_thread_map(
            materialize_group,
            list(enumerate(groups, start=1)),
            self.event_workers,
        )
        for context in contexts:
            group_index = context["group_index"]
            group = context["group"]
            source_events = context["source_events"]
            event0 = context["event0"]
            diagnostics = context["diagnostics"]
            materialized = context["materialized"]
            graph_union = group["source_graph_pool_union_arxiv_ids"]
            graph_union_set = set(graph_union)

            graph_event_ids_by_paper: Dict[str, List[Any]] = {}
            graph_event_features_by_paper: Dict[str, List[Dict[str, Any]]] = {}
            for source in source_events:
                event_id = source["retrieval_event_id"]
                for graph_row in source.get("rows") or []:
                    paper_id = graph_row["paper_arxiv_id"]
                    graph_event_ids_by_paper.setdefault(paper_id, []).append(event_id)
                    graph_event_features_by_paper.setdefault(paper_id, []).append(
                        {
                            "retrieval_event_id": event_id,
                            "iteration_idx": source["event"].get("iteration_idx"),
                            "retrieval_page_idx": source["event"].get(
                                "retrieval_page_idx"
                            ),
                            "query_score_raw": graph_row.get("query_score_raw"),
                            "query_score_normalized": graph_row.get(
                                "query_score_normalized"
                            ),
                            "query_component_rank": graph_row.get(
                                "query_component_rank"
                            ),
                            "subquery_score_raw": graph_row.get("subquery_score_raw"),
                            "subquery_score_normalized": graph_row.get(
                                "subquery_score_normalized"
                            ),
                            "subquery_component_rank": graph_row.get(
                                "subquery_component_rank"
                            ),
                            "intent_score": graph_row.get("intent_score"),
                            "path_count": graph_row.get("path_count"),
                            "path_count_normalized": graph_row.get(
                                "path_count_normalized"
                            ),
                        }
                    )

            enriched_rows: List[Dict[str, Any]] = []
            for row in materialized["rows"]:
                paper_id = row["paper_arxiv_id"]
                enriched = {
                    **{field: event0.get(field) for field in EVENT_FIELDS},
                    **dict(row),
                    "method": DEEP_MERGED_METHOD,
                    "postprocess_stage": "materialize",
                    "features_materialized": True,
                    "legacy_rerank_applied": False,
                    "shadow_selector_applied": False,
                    "merged_subquery_group_index": group_index,
                    "source_event_count": len(source_events),
                    "source_event_ids": [
                        source["retrieval_event_id"] for source in source_events
                    ],
                    "source_graph_pool_occurrence_budget": group[
                        "source_graph_pool_occurrence_budget"
                    ],
                    "source_graph_pool_union_count": group[
                        "source_graph_pool_union_count"
                    ],
                    "source_graph_event_ids": graph_event_ids_by_paper.get(
                        paper_id, []
                    ),
                    "source_graph_event_features": graph_event_features_by_paper.get(
                        paper_id, []
                    ),
                    "in_source_graph_local_pool": paper_id in graph_union_set,
                    "is_ground_truth": paper_id in gt_ids,
                    "affects_next_iteration": False,
                }
                enriched_rows.append(enriched)
                self.writer.append(
                    "deep_merged/paper_rows.jsonl", enriched, full_only=True
                )

            self.writer.append(
                "deep_merged/comparisons.jsonl",
                {
                    **{field: event0.get(field) for field in EVENT_FIELDS},
                    "method": DEEP_MERGED_METHOD,
                    "postprocess_stage": "materialize",
                    "comparison_scope": "complete_candidate_pools_only",
                    "subquery_id": group["subquery_id"],
                    "source_graph_pool_occurrence_budget": group[
                        "source_graph_pool_occurrence_budget"
                    ],
                    "source_graph_pool_union_count": group[
                        "source_graph_pool_union_count"
                    ],
                    "source_graph_pool_overlap_occurrence_count": group[
                        "source_graph_pool_overlap_occurrence_count"
                    ],
                    "pool_comparison": id_set_comparison(
                        graph_union, materialized["retrieval_order_arxiv_ids"]
                    ),
                },
            )
            self.writer.append(
                "deep_merged/pool_records.jsonl",
                {
                    **{field: event0.get(field) for field in EVENT_FIELDS},
                    "method": DEEP_MERGED_METHOD,
                    "postprocess_stage": "materialize",
                    "features_materialized": True,
                    "legacy_rerank_applied": False,
                    "shadow_selector_applied": False,
                    "subquery_id": group["subquery_id"],
                    "source_event_budgets": [
                        {
                            **{
                                field: source["event"].get(field)
                                for field in EVENT_FIELDS
                            },
                            "source_graph_local_pool_size": source[
                                "source_local_graph_pool_size"
                            ],
                            "source_graph_local_pool_arxiv_ids": source[
                                "graph_local_pool_arxiv_ids"
                            ],
                        }
                        for source in source_events
                    ],
                    "source_graph_pool_occurrence_budget": group[
                        "source_graph_pool_occurrence_budget"
                    ],
                    "source_graph_pool_union_arxiv_ids": graph_union,
                    "deep_retrieval_order_arxiv_ids": materialized[
                        "retrieval_order_arxiv_ids"
                    ],
                    "deep_pool_size": len(
                        materialized["retrieval_order_arxiv_ids"]
                    ),
                    "deep_feature_scorable_arxiv_ids": materialized[
                        "scorable_arxiv_ids"
                    ],
                    "deep_feature_scorable_count": len(
                        materialized["scorable_arxiv_ids"]
                    ),
                    "deep_feature_unscorable_arxiv_ids": materialized[
                        "unscorable_arxiv_ids"
                    ],
                    "retrieval_diagnostics": diagnostics,
                    "deep_pool_rows": compact_deep_rows(enriched_rows),
                },
            )

            requested_occurrences += int(diagnostics.get("requested_count") or 0)
            actual_occurrences += len(materialized["retrieval_order_arxiv_ids"])
            scorable_occurrences += len(materialized["scorable_arxiv_ids"])
            all_deep_pool.extend(materialized["retrieval_order_arxiv_ids"])
            all_source_graph.extend(graph_union)

        summary = self._deep_materialization_summary(
            method=DEEP_MERGED_METHOD,
            first_event=first_event,
            gt_ids=gt_ids,
            deep_pool_ids=all_deep_pool,
            source_graph_ids=all_source_graph,
            requested_occurrences=requested_occurrences,
            actual_occurrences=actual_occurrences,
            feature_scorable_occurrences=scorable_occurrences,
            event_count=event_count,
            group_count=len(groups),
        )
        self.writer.append("deep_merged/query_results.jsonl", summary)
        return summary

    def _process_deep_merged(self, prepared: Mapping[str, Any], gt_ids: Set[str]) -> Dict[str, Any]:
        if self.materialize_only:
            return self._materialize_deep_merged(prepared, gt_ids)
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
