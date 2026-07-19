#!/usr/bin/env python3
"""Artifact and state hook for the online per-subquery graph method."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from graph_methods import ArtifactWriter, PerSubqueryProcessor, normalize_arxiv_id, selector_decision_record


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


class OnlinePerSubqueryManager:
    def __init__(self, processor: PerSubqueryProcessor, writer: ArtifactWriter, run_id: str = "") -> None:
        self.processor = processor
        self.writer = writer
        self.run_id = run_id
        self._query_candidates: List[str] = []
        self._query_selected: List[str] = []
        self._s2_before: Dict[str, int] = {}
        self._paper_type_before: Dict[str, int] = {}
        self._query_policy_record: Dict[str, Any] = {}

    @property
    def query_policy_id(self) -> Optional[str]:
        value = self._query_policy_record.get("rerank_policy_id")
        return str(value) if value else None

    @property
    def dynamic_rerank_enabled(self) -> bool:
        return bool(
            self.processor.active_compiled_policy
            and not self.processor.active_compiled_policy.used_fallback
        )

    def start_query(
        self,
        original_query: str = "",
        *,
        query_id: str = "",
        benchmark_idx: Optional[int] = None,
    ) -> None:
        self._query_candidates = []
        self._query_selected = []
        self._s2_before = self.processor.s2.snapshot_stats()
        resolver = self.processor.paper_type_resolver
        self._paper_type_before = resolver.snapshot_stats() if resolver else {}
        policy, compiled = self.processor.configure_query(original_query)
        paper_type_provenance = {
            "paper_type_backend": getattr(resolver, "backend", None),
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
        if policy is not None and compiled is not None:
            assert self.processor.rerank_skill is not None
            record = self.processor.rerank_skill.artifact_record(
                query_id=str(query_id),
                original_query=original_query,
                policy=policy,
                compiled=compiled,
            )
            record.update(
                {
                    "schema_version": "1.0",
                    "run_id": self.run_id,
                    "benchmark_idx": benchmark_idx,
                    "formula_scope": "one_policy_per_original_query_all_iterations",
                    "affects_selector_and_next_iteration": True,
                    **paper_type_provenance,
                }
            )
        else:
            record = {
                "schema_version": "1.0",
                "run_id": self.run_id,
                "query_id": str(query_id),
                "benchmark_idx": benchmark_idx,
                "original_query": original_query,
                "rerank_policy_id": "legacy-static",
                "rerank_formula_id": "q030_sq040_intent015_path015_closed_pool_minmax_v1",
                "compiled_weights": dict(self.processor.weights),
                "dynamic_rerank_enabled": False,
                "formula_scope": "one_static_formula_all_queries",
                "affects_selector_and_next_iteration": True,
                **paper_type_provenance,
            }
        self._query_policy_record = record
        self.writer.append("query_rerank_policies.jsonl", record)

    def process_event(
        self,
        event: Mapping[str, Any],
        gt_ids: Set[str],
        exclude_arxiv_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        if isinstance(event, dict):
            event.setdefault("run_id", self.run_id)
            event.setdefault(
                "rerank_policy_id",
                self._query_policy_record.get("rerank_policy_id"),
            )
            event.setdefault(
                "dynamic_rerank_enabled",
                bool(
                    self.processor.active_compiled_policy
                    and not self.processor.active_compiled_policy.used_fallback
                ),
            )
        for seed in event.get("seed_papers") or []:
            paper_id = normalize_arxiv_id(seed.get("paper_arxiv_id"))
            if not paper_id:
                continue
            self.writer.append(
                "raw_retrieval_rows.jsonl",
                {
                    **{key: event.get(key) for key in (
                        "schema_version", "run_id", "retrieval_event_id", "query_id", "benchmark_idx", "query", "query_source", "query_date",
                        "iteration_idx", "subquery_id", "subquery", "subquery_target_k", "subquery_link_type",
                        "parent_subquery_id", "subquery_before_date",
                        "retrieval_page_idx", "retrieval_offset", "raw_retrieval_page_count", "results_per_query", "selector_top_k", "planner_checklist",
                    )},
                    "method": "online_per_subquery_raw_retrieval",
                    "paper_arxiv_id": paper_id,
                    "retrieval_backend": event.get("retrieval_backend"),
                    "observed_retrieval_score": seed.get("observed_retrieval_score"),
                    "observed_retrieval_rank": seed.get("observed_retrieval_rank"),
                    "observed_retrieval_absolute_rank": (
                        int(event.get("retrieval_offset") or 0) + int(seed["observed_retrieval_rank"])
                        if seed.get("observed_retrieval_rank") is not None else None
                    ),
                    "passed_date_cutoff": True,
                    "is_ground_truth": paper_id in gt_ids,
                },
                full_only=True,
            )
        processed = self.processor.process(event, gt_ids, exclude_arxiv_ids=exclude_arxiv_ids)
        for edge in processed["edges"]:
            self.writer.append("expansion_edges.jsonl", edge, full_only=True)
        self.writer.append(
            "filter_stats.jsonl",
            {
                "schema_version": event.get("schema_version", "1.0"),
                "retrieval_event_id": event.get("retrieval_event_id"),
                "query_id": event.get("query_id"),
                "iteration_idx": event.get("iteration_idx"),
                "subquery_id": event.get("subquery_id"),
                **processed.get("filter_stats", {}),
            },
        )
        return processed

    def record_selector_pass(
        self,
        event: Mapping[str, Any],
        input_papers: Iterable[Any],
        result: Any,
        *,
        pass_id: int,
        is_after_browsing: bool,
    ) -> None:
        input_papers = list(input_papers)
        kept, overview, to_browse = result[:3] if result else ([], "", {})
        details = result[3] if result and len(result) > 3 else {}
        input_ids = [
            normalize_arxiv_id(getattr(paper, "arxiv_id", None) or getattr(paper, "id", None))
            for paper in input_papers
        ]
        selected_ids = [
            normalize_arxiv_id(getattr(paper, "arxiv_id", None) or getattr(paper, "id", None))
            for paper in kept
        ]
        browse_goals = {}
        for paper_id, value in (to_browse or {}).items():
            normalized = normalize_arxiv_id(paper_id)
            if normalized:
                browse_goals[normalized] = value.get("goal") if isinstance(value, Mapping) else str(value)
        self.writer.append(
            "selector_passes.jsonl",
            {
                "schema_version": event.get("schema_version", "1.0"),
                "retrieval_event_id": event.get("retrieval_event_id"),
                "selector_call_id": f'{event.get("retrieval_event_id")}:selector:p{pass_id}',
                "query_id": event.get("query_id"),
                "benchmark_idx": event.get("benchmark_idx"),
                "iteration_idx": event.get("iteration_idx"),
                "subquery_id": event.get("subquery_id"),
                "selector_pass_id": pass_id,
                "is_after_browsing": is_after_browsing,
                "input_arxiv_ids": [paper_id for paper_id in input_ids if paper_id],
                "selected_arxiv_ids": sorted(paper_id for paper_id in selected_ids if paper_id),
                "to_browse_goals": browse_goals,
                "selector_reasons": (details or {}).get("reasons") or {},
                "selector_overview": overview or "",
                "rerank_policy_id": self._query_policy_record.get(
                    "rerank_policy_id"
                ),
                "dynamic_rerank_enabled": bool(
                    self.processor.active_compiled_policy
                    and not self.processor.active_compiled_policy.used_fallback
                ),
                "input_rerank_scores": {
                    paper_id: float(getattr(paper, "score", 0.0) or 0.0)
                    for paper_id, paper in zip(input_ids, input_papers)
                    if paper_id
                },
            },
            full_only=True,
        )

    def finish_event(
        self,
        event: Mapping[str, Any],
        processed: Mapping[str, Any],
        selected_ids: Iterable[str],
        overview: str,
        reasons: Optional[Mapping[str, str]] = None,
    ) -> None:
        selected = {normalize_arxiv_id(value) for value in selected_ids if normalize_arxiv_id(value)}
        top_ids = {row["paper_arxiv_id"] for row in processed.get("top_rows") or []}
        for row_value in processed.get("rows") or []:
            row = dict(row_value)
            paper_id = row["paper_arxiv_id"]
            row.update(
                {
                    "in_selector_topk": paper_id in top_ids,
                    "selector_input_rank": row.get("rerank_rank") if paper_id in top_ids else None,
                    "selector_selected": paper_id in selected,
                    "selector_reason": (reasons or {}).get(paper_id, ""),
                    "written_to_retrieved_memory": paper_id in top_ids,
                    "written_to_selected_memory": paper_id in selected,
                    "affects_next_iteration": True,
                }
            )
            self.writer.append("paper_rows.jsonl", row, full_only=True)
        self.writer.append(
            "selector_decisions.jsonl",
            selector_decision_record(event, processed.get("top_rows") or [], selected, overview, reasons),
            full_only=True,
        )
        self.writer.append(
            "memory_transitions.jsonl",
            {
                "schema_version": event.get("schema_version", "1.0"),
                "retrieval_event_id": event.get("retrieval_event_id"),
                "query_id": event.get("query_id"),
                "benchmark_idx": event.get("benchmark_idx"),
                "iteration_idx": event.get("iteration_idx"),
                "subquery_id": event.get("subquery_id"),
                "retrieved_memory_arxiv_ids": sorted(top_ids),
                "selected_memory_arxiv_ids": sorted(selected),
                "retrieved_memory_rerank_scores": {
                    row["paper_arxiv_id"]: row.get("rerank_score")
                    for row in processed.get("top_rows") or []
                },
                "rerank_policy_id": processed.get("rerank_policy_id"),
                "dynamic_rerank_enabled": bool(
                    (processed.get("compiled_rerank_policy") or {}).get(
                        "used_fallback"
                    )
                    is False
                ),
                "affects_next_iteration": True,
            },
            full_only=True,
        )
        self._query_candidates.extend(top_ids)
        self._query_selected.extend(selected)

    def finish_query(self, query_id: str, benchmark_idx: int, gt_ids: Set[str]) -> Dict[str, Any]:
        candidates = set(self._query_candidates)
        selected = set(self._query_selected)
        candidate_hits = gt_ids & candidates
        selected_hits = gt_ids & selected
        s2_after = self.processor.s2.snapshot_stats()
        resolver = self.processor.paper_type_resolver
        paper_type_after = resolver.snapshot_stats() if resolver else {}
        summary = {
            "query_id": query_id,
            "benchmark_idx": benchmark_idx,
            "gt_count": len(gt_ids),
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "candidate_arxiv_ids": sorted(candidates),
            "selected_arxiv_ids": sorted(selected),
            "candidate_gt_ids": sorted(candidate_hits),
            "selected_gt_ids": sorted(selected_hits),
            "candidate_recall": _safe_div(len(candidate_hits), len(gt_ids)),
            "candidate_precision": _safe_div(len(candidate_hits), len(candidates)),
            "selection_recall": _safe_div(len(selected_hits), len(gt_ids)),
            "selection_precision": _safe_div(len(selected_hits), len(selected)),
            "rerank_policy_id": self._query_policy_record.get("rerank_policy_id"),
            "compiled_rerank_policy": self._query_policy_record.get(
                "compiled_policy"
            ),
            "dynamic_rerank_enabled": bool(
                self.processor.active_compiled_policy
                and not self.processor.active_compiled_policy.used_fallback
            ),
            "s2_stats_delta": {
                key: int(s2_after.get(key, 0)) - int(self._s2_before.get(key, 0))
                for key in set(self._s2_before) | set(s2_after)
            },
            "s2_stats_cumulative": s2_after,
            "paper_type_stats_delta": {
                key: int(paper_type_after.get(key, 0))
                - int(self._paper_type_before.get(key, 0))
                for key in set(self._paper_type_before) | set(paper_type_after)
            },
            "paper_type_stats_cumulative": paper_type_after,
            "paper_type_backend": getattr(resolver, "backend", None),
            "paper_type_evidence_source": getattr(
                resolver, "evidence_source", None
            ),
            "paper_type_classifier_version": getattr(
                resolver, "classifier_version", None
            ),
            "paper_type_model": getattr(resolver, "model", None),
        }
        self.writer.append("query_results.jsonl", summary)
        return summary
