#!/usr/bin/env python3
"""Text-only deep-retrieval controls matched to per-subquery graph pools.

The baseline retriever supplies the deep pools.  The graph postprocessor is
used only to define each event budget and the comparison sets; no graph
feature is leaked into the deep rerank.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

import config
from graph_methods import (
    CandidateIndex,
    DEFAULT_FEATURE_WEIGHTS,
    RERANK_FORMULA_ID,
    month_key,
    normalize_arxiv_id,
)
from structures import Paper


DEEP_EVENT_METHOD = "deep_event_offset_matched"
DEEP_MERGED_METHOD = "deep_merged_subquery_sum_budget"
DEEP_FEATURE_WEIGHTS = dict(DEFAULT_FEATURE_WEIGHTS)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ordered_unique(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen: Set[str] = set()
    for value in values:
        paper_id = normalize_arxiv_id(value)
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            output.append(paper_id)
    return output


def id_set_comparison(graph_ids: Sequence[str], deep_ids: Sequence[str]) -> Dict[str, Any]:
    """Ordered overlap/difference record for later candidate-set analysis."""
    graph = ordered_unique(graph_ids)
    deep = ordered_unique(deep_ids)
    graph_set, deep_set = set(graph), set(deep)
    intersection = [paper_id for paper_id in graph if paper_id in deep_set]
    union_count = len(graph_set | deep_set)
    return {
        "graph_count": len(graph),
        "deep_count": len(deep),
        "intersection_count": len(intersection),
        "union_count": union_count,
        "jaccard": float(len(intersection)) / float(union_count) if union_count else 0.0,
        "graph_arxiv_ids": graph,
        "deep_arxiv_ids": deep,
        "intersection_arxiv_ids": intersection,
        "graph_only_arxiv_ids": [paper_id for paper_id in graph if paper_id not in deep_set],
        "deep_only_arxiv_ids": [paper_id for paper_id in deep if paper_id not in graph_set],
    }


class DeepRetrievalProcessor:
    """Prepare shared deep pools and apply the fixed four-factor rerank."""

    def __init__(
        self,
        rag_system: Any,
        paper_db: Mapping[str, Mapping[str, Any]],
        *,
        scoring_backend: str,
        embedding_provider: Any,
    ) -> None:
        if scoring_backend not in {"bm25", "embedding"}:
            raise ValueError(f"unsupported deep-retrieval backend: {scoring_backend}")
        self.rag = rag_system
        self.paper_db = dict(paper_db)
        self.backend = scoring_backend
        self.embedding_provider = embedding_provider
        self.weights = dict(DEEP_FEATURE_WEIGHTS)

    @staticmethod
    def _request_states(requests: Mapping[Any, Mapping[str, Any]]) -> Dict[Any, Dict[str, Any]]:
        states: Dict[Any, Dict[str, Any]] = {}
        for key, request in requests.items():
            states[key] = {
                "offset": max(0, _as_int(request.get("offset"), 0)),
                "count": max(0, _as_int(request.get("count"), 0)),
                "exclude": set(ordered_unique(request.get("exclude_arxiv_ids") or [])),
                "eligible_rank": 0,
                "completion_global_rank": None,
            }
        return states

    @staticmethod
    def _diagnostics(
        states: Mapping[Any, Mapping[str, Any]],
        outputs: Mapping[Any, Sequence[Mapping[str, Any]]],
        *,
        ranked_document_count: int,
        pre_dedup_document_count: int,
    ) -> Dict[Any, Dict[str, Any]]:
        result: Dict[Any, Dict[str, Any]] = {}
        for key, state in states.items():
            requested = _as_int(state.get("count"), 0)
            actual = len(outputs.get(key) or [])
            result[key] = {
                "requested_count": requested,
                "actual_count": actual,
                "budget_fulfillment_rate": float(actual) / float(requested) if requested else 0.0,
                "retrieval_offset": _as_int(state.get("offset"), 0),
                "retrieval_exclusion_count": len(state.get("exclude") or set()),
                "eligible_rank_scanned_for_request": _as_int(state.get("eligible_rank"), 0),
                "completion_global_date_valid_unique_rank": state.get("completion_global_rank"),
                "date_valid_ranked_document_count": ranked_document_count,
                "date_valid_document_count_before_arxiv_dedup": pre_dedup_document_count,
                "canonical_arxiv_deduplication_before_exclusion_offset": True,
            }
        return result

    def _materialize_requests(
        self,
        ranked: Sequence[Mapping[str, Any]],
        requests: Mapping[Any, Mapping[str, Any]],
        *,
        pre_dedup_document_count: int,
    ) -> Tuple[Dict[Any, List[Dict[str, Any]]], Dict[Any, Dict[str, Any]]]:
        states = self._request_states(requests)
        outputs: Dict[Any, List[Dict[str, Any]]] = {key: [] for key in requests}
        for item in ranked:
            paper_id = normalize_arxiv_id(item.get("paper_arxiv_id"))
            if not paper_id:
                continue
            global_rank = _as_int(item.get("deep_retrieval_rank_global_date_valid"), 0)
            for key, state in states.items():
                if state["count"] <= 0 or state["eligible_rank"] >= state["offset"] + state["count"]:
                    continue
                if paper_id in state["exclude"]:
                    continue
                state["eligible_rank"] += 1
                eligible_rank = state["eligible_rank"]
                if state["offset"] < eligible_rank <= state["offset"] + state["count"]:
                    outputs[key].append(
                        {
                            **dict(item),
                            "deep_retrieval_rank_after_exclusion": eligible_rank,
                            "deep_retrieval_rank_in_local_pool": eligible_rank - state["offset"],
                        }
                    )
                if eligible_rank >= state["offset"] + state["count"]:
                    state["completion_global_rank"] = global_rank
            if all(
                state["count"] <= 0 or state["eligible_rank"] >= state["offset"] + state["count"]
                for state in states.values()
            ):
                break
        return outputs, self._diagnostics(
            states,
            outputs,
            ranked_document_count=len(ranked),
            pre_dedup_document_count=pre_dedup_document_count,
        )

    def _retrieve_bm25_requests(
        self,
        *,
        subquery: str,
        before_date: Optional[str],
        requests: Mapping[Any, Mapping[str, Any]],
    ) -> Tuple[Dict[Any, List[Dict[str, Any]]], Dict[Any, Dict[str, Any]]]:
        if self.rag.bm25_index is None:
            raise ValueError("BM25 index is not loaded")
        tokens = self.rag._preprocess_text_for_bm25(subquery)
        if not tokens:
            empty = {key: [] for key in requests}
            states = self._request_states(requests)
            return empty, self._diagnostics(
                states, empty, ranked_document_count=0, pre_dedup_document_count=0
            )
        scores = self.rag.bm25_index.get_scores(tokens)
        positive_indices = np.where(scores > 0)[0]
        cutoff = month_key(before_date)
        date_valid: List[int] = []
        for raw_index in positive_indices:
            index = int(raw_index)
            raw_id = self.rag.bm25_index_to_id.get(index)
            metadata = self.rag.paper_metadata.get(raw_id, {}) if raw_id else {}
            paper_date = month_key(metadata.get("date"))
            if not raw_id or (cutoff and (not paper_date or paper_date > cutoff)):
                continue
            date_valid.append(index)
        sorted_indices = sorted(date_valid, key=lambda index: (-float(scores[index]), index))
        ranked: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        for index in sorted_indices:
            raw_id = self.rag.bm25_index_to_id.get(index)
            metadata = dict(self.rag.paper_metadata.get(raw_id, {}) or {})
            paper_id = normalize_arxiv_id(metadata.get("arxiv_id") or raw_id)
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            metadata["arxiv_id"] = paper_id
            ranked.append(
                {
                    "paper_arxiv_id": paper_id,
                    "deep_retrieval_score_raw": float(scores[index]),
                    "deep_retrieval_rank_global_date_valid": len(ranked) + 1,
                    "_metadata": metadata,
                }
            )
        return self._materialize_requests(
            ranked, requests, pre_dedup_document_count=len(sorted_indices)
        )

    def _retrieve_vector_requests(
        self,
        *,
        subquery: str,
        before_date: Optional[str],
        requests: Mapping[Any, Mapping[str, Any]],
    ) -> Tuple[Dict[Any, List[Dict[str, Any]]], Dict[Any, Dict[str, Any]]]:
        store = getattr(self.rag, "qdrant_vector_store", None)
        if store is None:
            raise ValueError("Qdrant vector store is not loaded")
        maximum_needed = max(
            (
                max(0, _as_int(request.get("offset"), 0))
                + max(0, _as_int(request.get("count"), 0))
                + len(ordered_unique(request.get("exclude_arxiv_ids") or []))
                for request in requests.values()
            ),
            default=0,
        )
        initial_fetch_k = max(getattr(config, "GT_RANK_CUTOFF", 100), maximum_needed) * 5
        max_fetch_k = max(maximum_needed, int(getattr(config, "DEEP_VECTOR_MAX_FETCH_K", 20000)))
        fetch_k = min(initial_fetch_k, max_fetch_k)
        cutoff = month_key(before_date)
        attempts = 0
        while True:
            attempts += 1
            raw_results = store.similarity_search_with_score(query=subquery, k=max(1, fetch_k))
            ranked: List[Dict[str, Any]] = []
            seen: Set[str] = set()
            date_valid_pre_dedup = 0
            for document, score in raw_results:
                metadata = dict(getattr(document, "metadata", {}) or {})
                paper_id = normalize_arxiv_id(metadata.get("arxiv_id") or metadata.get("id"))
                paper_date = month_key(metadata.get("date"))
                if not paper_id or (cutoff and (not paper_date or paper_date > cutoff)):
                    continue
                date_valid_pre_dedup += 1
                if paper_id in seen:
                    continue
                seen.add(paper_id)
                metadata["arxiv_id"] = paper_id
                ranked.append(
                    {
                        "paper_arxiv_id": paper_id,
                        "deep_retrieval_score_raw": float(score),
                        "deep_retrieval_rank_global_date_valid": len(ranked) + 1,
                        "_metadata": metadata,
                    }
                )
            outputs, diagnostics = self._materialize_requests(
                ranked, requests, pre_dedup_document_count=date_valid_pre_dedup
            )
            complete = all(
                len(outputs.get(key) or []) >= max(0, _as_int(request.get("count"), 0))
                for key, request in requests.items()
            )
            exhausted = len(raw_results) < fetch_k
            if complete or exhausted or fetch_k >= max_fetch_k:
                for item in diagnostics.values():
                    item.update(
                        {
                            "vector_fetch_k": fetch_k,
                            "vector_fetch_attempts": attempts,
                            "vector_fetch_cap": max_fetch_k,
                            "vector_source_exhausted": exhausted,
                        }
                    )
                return outputs, diagnostics
            fetch_k = min(max_fetch_k, fetch_k * 2)

    def _retrieve_requests(
        self,
        *,
        subquery: str,
        before_date: Optional[str],
        requests: Mapping[Any, Mapping[str, Any]],
    ) -> Tuple[Dict[Any, List[Dict[str, Any]]], Dict[Any, Dict[str, Any]]]:
        if self.backend == "bm25":
            return self._retrieve_bm25_requests(
                subquery=subquery, before_date=before_date, requests=requests
            )
        return self._retrieve_vector_requests(
            subquery=subquery, before_date=before_date, requests=requests
        )

    def prepare_pools(
        self,
        graph_events: Sequence[Mapping[str, Any]],
        *,
        include_event: bool,
        include_merged: bool,
    ) -> Dict[str, Any]:
        """Score a stable subquery once and serve both deep-control requests."""
        grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        group_order: List[str] = []
        for item in graph_events:
            event = item["event"]
            subquery_id = str(event.get("subquery_id"))
            if subquery_id not in grouped:
                group_order.append(subquery_id)
            grouped[subquery_id].append(item)

        groups: List[Dict[str, Any]] = []
        pools: Dict[Any, List[Dict[str, Any]]] = {}
        diagnostics: Dict[Any, Dict[str, Any]] = {}
        for subquery_id in group_order:
            items = grouped[subquery_id]
            texts = {str(item["event"].get("subquery") or "") for item in items}
            cutoffs = {month_key(item["event"].get("subquery_before_date") or item["event"].get("query_date")) for item in items}
            if len(texts) != 1 or len(cutoffs) != 1:
                raise ValueError(f"subquery_id={subquery_id} changed text/date across continue events")
            event_records: List[Dict[str, Any]] = []
            for item in items:
                event = item["event"]
                graph_rows = list(item.get("rows") or [])
                graph_top_rows = list(item.get("top_rows") or [])
                event_records.append(
                    {
                        **dict(item),
                        "graph_local_pool_arxiv_ids": ordered_unique(
                            row.get("paper_arxiv_id") for row in graph_rows
                        ),
                        "graph_topk_arxiv_ids": ordered_unique(
                            row.get("paper_arxiv_id") for row in graph_top_rows
                        ),
                        "source_local_graph_pool_size": len(
                            ordered_unique(row.get("paper_arxiv_id") for row in graph_rows)
                        ),
                        "retrieval_event_id": event.get("retrieval_event_id"),
                    }
                )
            graph_occurrence_budget = sum(
                item["source_local_graph_pool_size"] for item in event_records
            )
            graph_union = ordered_unique(
                paper_id
                for item in event_records
                for paper_id in item["graph_local_pool_arxiv_ids"]
            )
            first_event = event_records[0]["event"]
            group = {
                "subquery_key": subquery_id,
                "subquery_id": first_event.get("subquery_id"),
                "subquery": next(iter(texts)),
                "subquery_before_date": next(iter(cutoffs)),
                "events": event_records,
                "source_graph_pool_occurrence_budget": graph_occurrence_budget,
                "source_graph_pool_union_arxiv_ids": graph_union,
                "source_graph_pool_union_count": len(graph_union),
                "source_graph_pool_overlap_occurrence_count": graph_occurrence_budget - len(graph_union),
                "frozen_first_event_exclusion_arxiv_ids": ordered_unique(
                    first_event.get("retrieval_exclusion_arxiv_ids") or []
                ),
            }
            groups.append(group)
            requests: Dict[Any, Dict[str, Any]] = {}
            if include_event:
                for item in event_records:
                    event = item["event"]
                    requests[(DEEP_EVENT_METHOD, item["retrieval_event_id"])] = {
                        "offset": event.get("retrieval_offset") or 0,
                        "count": item["source_local_graph_pool_size"],
                        "exclude_arxiv_ids": event.get("retrieval_exclusion_arxiv_ids") or [],
                    }
            if include_merged:
                requests[(DEEP_MERGED_METHOD, subquery_id)] = {
                    "offset": 0,
                    "count": graph_occurrence_budget,
                    "exclude_arxiv_ids": group["frozen_first_event_exclusion_arxiv_ids"],
                }
            group_pools, group_diagnostics = self._retrieve_requests(
                subquery=group["subquery"],
                before_date=group["subquery_before_date"],
                requests=requests,
            )
            pools.update(group_pools)
            diagnostics.update(group_diagnostics)
        return {"groups": groups, "pools": pools, "diagnostics": diagnostics}

    def materialize_pool_features(
        self,
        pool: Sequence[Mapping[str, Any]],
        *,
        query: str,
        subquery: str,
        cutoff: Optional[str],
    ) -> Dict[str, Any]:
        """Materialize retriever-order deep-pool component features only."""
        retrieval_order = ordered_unique(item.get("paper_arxiv_id") for item in pool)
        retrieval_by_id = {
            normalize_arxiv_id(item.get("paper_arxiv_id")): item
            for item in pool
            if normalize_arxiv_id(item.get("paper_arxiv_id"))
        }
        metadata: Dict[str, Dict[str, Any]] = {}
        for paper_id in retrieval_order:
            hit_metadata = dict((retrieval_by_id[paper_id].get("_metadata") or {}))
            metadata[paper_id] = dict(self.paper_db.get(paper_id) or hit_metadata)
        index = CandidateIndex(
            retrieval_order,
            metadata,
            self.backend,
            self.embedding_provider,
        )
        query_raw, query_norm, query_rank = index.score(query)
        subquery_raw, subquery_norm, subquery_rank = index.score(subquery)
        scorable = [paper_id for paper_id in retrieval_order if paper_id in query_raw and paper_id in subquery_raw]
        scorable_set = set(scorable)
        unscorable = [paper_id for paper_id in retrieval_order if paper_id not in scorable_set]
        rows: List[Dict[str, Any]] = []
        for materialization_rank, paper_id in enumerate(retrieval_order, start=1):
            hit = retrieval_by_id[paper_id]
            is_scorable = paper_id in scorable_set
            rows.append(
                {
                    "paper_arxiv_id": paper_id,
                    "candidate_type": "deep_retrieval",
                    "retrieval_backend": self.backend,
                    "passed_date_cutoff": True,
                    "date_cutoff_month": month_key(cutoff),
                    "deep_retrieval_score_raw": float(hit.get("deep_retrieval_score_raw") or 0.0),
                    "deep_retrieval_rank_global_date_valid": hit.get("deep_retrieval_rank_global_date_valid"),
                    "deep_retrieval_rank_after_exclusion": hit.get("deep_retrieval_rank_after_exclusion"),
                    "deep_retrieval_rank_in_local_pool": hit.get("deep_retrieval_rank_in_local_pool"),
                    "deep_retrieval_rank_scope": "date_valid_canonical_retriever_order_after_frozen_request_exclusion",
                    "query_score_raw": float(query_raw.get(paper_id, 0.0)) if is_scorable else None,
                    "query_score_normalized": float(query_norm.get(paper_id, 0.0)) if is_scorable else None,
                    "query_component_rank": query_rank.get(paper_id) if is_scorable else None,
                    "subquery_score_raw": float(subquery_raw.get(paper_id, 0.0)) if is_scorable else None,
                    "subquery_score_normalized": float(subquery_norm.get(paper_id, 0.0)) if is_scorable else None,
                    "subquery_component_rank": subquery_rank.get(paper_id) if is_scorable else None,
                    "component_rank_scope": "closed_deep_retrieval_pool",
                    "normalization_scope": "closed_deep_retrieval_pool_minmax",
                    "intent_labels": [],
                    "intent_score": 0.0,
                    "path_count": 0,
                    "path_count_normalized": 0.0,
                    "feature_scorable": is_scorable,
                    "feature_unavailable_reason": (
                        None if is_scorable else "missing non-empty title/abstract"
                    ),
                    "materialization_order_rank": materialization_rank,
                    "materialization_order_scope": "deep_retrieval_order",
                }
            )
        return {
            "retrieval_order_arxiv_ids": retrieval_order,
            "scorable_arxiv_ids": scorable,
            "unscorable_arxiv_ids": unscorable,
            "rows": rows,
            "metadata": metadata,
            "features_materialized": True,
            "legacy_rerank_applied": False,
        }

    def rerank_pool(
        self,
        pool: Sequence[Mapping[str, Any]],
        *,
        query: str,
        subquery: str,
        cutoff: Optional[str],
    ) -> Dict[str, Any]:
        """Apply the shared four-factor formula to a deep pool.

        Deep controls have no graph edges, so their intent/path features are
        explicitly zero while the formula and manifest schema stay identical
        to the graph arm.
        """
        materialized = self.materialize_pool_features(
            pool,
            query=query,
            subquery=subquery,
            cutoff=cutoff,
        )
        row_by_id = {
            row["paper_arxiv_id"]: dict(row) for row in materialized["rows"]
        }
        scorable = list(materialized["scorable_arxiv_ids"])
        rerank_scores = {
            paper_id: (
                self.weights["query_score_normalized"]
                * float(row_by_id[paper_id].get("query_score_normalized") or 0.0)
                + self.weights["subquery_score_normalized"]
                * float(row_by_id[paper_id].get("subquery_score_normalized") or 0.0)
                + self.weights["intent_score"]
                * float(row_by_id[paper_id].get("intent_score") or 0.0)
                + self.weights["path_count_normalized"]
                * float(row_by_id[paper_id].get("path_count_normalized") or 0.0)
            )
            for paper_id in scorable
        }
        ordered = sorted(
            scorable,
            key=lambda paper_id: (
                -rerank_scores[paper_id],
                _as_int(
                    row_by_id[paper_id].get("deep_retrieval_rank_in_local_pool"),
                    10**12,
                ),
                paper_id,
            ),
        )
        rerank_rank = {
            paper_id: rank for rank, paper_id in enumerate(ordered, start=1)
        }
        rows: List[Dict[str, Any]] = []
        for paper_id in ordered + list(materialized["unscorable_arxiv_ids"]):
            row = row_by_id[paper_id]
            is_scorable = bool(row.pop("feature_scorable", False))
            unavailable_reason = row.pop("feature_unavailable_reason", None)
            row.pop("materialization_order_rank", None)
            row.pop("materialization_order_scope", None)
            row.update(
                {
                    "rerank_formula_id": RERANK_FORMULA_ID,
                    "feature_weights": dict(self.weights),
                    "rerank_score": (
                        float(rerank_scores[paper_id]) if is_scorable else None
                    ),
                    "rerank_rank": rerank_rank.get(paper_id),
                    "rerankable": is_scorable,
                    "rerank_drop_reason": unavailable_reason,
                }
            )
            rows.append(row)
        return {
            "retrieval_order_arxiv_ids": materialized["retrieval_order_arxiv_ids"],
            "rerank_order_arxiv_ids": ordered,
            "unscorable_arxiv_ids": materialized["unscorable_arxiv_ids"],
            "rows": rows,
            "metadata": materialized["metadata"],
            "features_materialized": True,
            "legacy_rerank_applied": True,
            "rerank_formula_id": RERANK_FORMULA_ID,
        }

    @staticmethod
    def papers_for_ids(
        ids: Sequence[str],
        rows: Sequence[Mapping[str, Any]],
        metadata: Mapping[str, Mapping[str, Any]],
    ) -> List[Paper]:
        row_by_id = {row["paper_arxiv_id"]: row for row in rows}
        papers: List[Paper] = []
        for paper_id in ids:
            item = metadata.get(paper_id) or {}
            row = row_by_id.get(paper_id) or {}
            papers.append(
                Paper(
                    id=paper_id,
                    arxiv_id=paper_id,
                    title=str(item.get("title") or "N/A"),
                    abstract=str(item.get("abstract") or "N/A"),
                    date=item.get("date") or "",
                    score=float(row.get("rerank_score") or 0.0),
                )
            )
        return papers


COMPACT_DEEP_ROW_FIELDS = (
    "paper_arxiv_id",
    "candidate_type",
    "deep_retrieval_score_raw",
    "deep_retrieval_rank_global_date_valid",
    "deep_retrieval_rank_after_exclusion",
    "deep_retrieval_rank_in_local_pool",
    "query_score_raw",
    "query_score_normalized",
    "query_component_rank",
    "subquery_score_raw",
    "subquery_score_normalized",
    "subquery_component_rank",
    "component_rank_scope",
    "normalization_scope",
    "intent_score",
    "path_count",
    "path_count_normalized",
    "feature_scorable",
    "feature_unavailable_reason",
    "materialization_order_rank",
    "materialization_order_scope",
    "rerank_formula_id",
    "rerank_score",
    "rerank_rank",
    "rerankable",
    "in_selector_topk",
    "selector_input_rank",
    "selector_selected",
    "in_source_graph_local_pool",
    "in_source_graph_topk",
    "source_graph_query_score_normalized",
    "source_graph_subquery_score_normalized",
    "source_graph_intent_score",
    "source_graph_path_count_normalized",
    "source_graph_event_ids",
    "source_graph_event_features",
    "source_graph_rerank_score",
    "source_graph_rerank_rank",
    "in_matching_event_graph_local_pool",
    "in_matching_event_graph_topk",
)


def compact_deep_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {key: row.get(key) for key in COMPACT_DEEP_ROW_FIELDS if key in row}
        for row in rows
    ]
