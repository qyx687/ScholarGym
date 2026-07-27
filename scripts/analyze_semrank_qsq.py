#!/usr/bin/env python3
"""Analyze SemRank-QSQ online results and optional static/ours runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence


def artifact_dir(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / "query_results.jsonl").is_file():
        return path
    if (path / "online_artifacts" / "query_results.jsonl").is_file():
        return path / "online_artifacts"
    raise FileNotFoundError(f"query_results.jsonl not found below {path}")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if isinstance(value, dict):
                yield value


def latest_by_query(path: Path) -> Dict[str, Dict[str, Any]]:
    output = {}
    for row in iter_jsonl(path):
        query_id = str(row.get("query_id") or "")
        if query_id:
            output[query_id] = row
    return output


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def end_to_end(rows: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    values = list(rows.values())
    if not values:
        return {"query_count": 0}
    retrieval_recall = mean(
        safe_div(
            len(set(row.get("candidate_gt_ids") or [])),
            int(row.get("gt_count") or 0),
        )
        for row in values
    )
    retrieval_precision = mean(
        safe_div(
            len(set(row.get("candidate_gt_ids") or [])),
            int(row.get("candidate_count") or 0),
        )
        for row in values
    )
    selection_recall = mean(
        safe_div(
            len(set(row.get("selected_gt_ids") or [])),
            int(row.get("gt_count") or 0),
        )
        for row in values
    )
    selection_precision = mean(
        safe_div(
            len(set(row.get("selected_gt_ids") or [])),
            int(row.get("selected_count") or 0),
        )
        for row in values
    )
    retrieved_gt = sum(
        len(set(row.get("candidate_gt_ids") or [])) for row in values
    )
    selected_gt = sum(
        len(set(row.get("selected_gt_ids") or [])) for row in values
    )
    return {
        "query_count": len(values),
        "selection_recall": selection_recall,
        "selection_precision": selection_precision,
        "selection_f1": f1(selection_precision, selection_recall),
        "retrieval_recall": retrieval_recall,
        "retrieval_precision": retrieval_precision,
        "retrieval_f1": f1(retrieval_precision, retrieval_recall),
        "retrieved_gt": retrieved_gt,
        "selected_gt": selected_gt,
        "selector_gap": retrieved_gt - selected_gt,
        "gt_conversion": safe_div(selected_gt, retrieved_gt),
    }


def binary_metrics(ranking: Sequence[str], relevant: set[str]) -> Dict[str, float]:
    ranks = [
        rank
        for rank, paper_id in enumerate(ranking, start=1)
        if paper_id in relevant
    ]
    output: Dict[str, float] = {}
    for k in (5, 10, 20):
        hits = sum(rank <= k for rank in ranks)
        output[f"recall@{k}"] = safe_div(hits, len(relevant))
        seen = 0
        precision_sum = 0.0
        for rank, paper_id in enumerate(ranking[:k], start=1):
            if paper_id in relevant:
                seen += 1
                precision_sum += seen / rank
        output[f"map@{k}"] = safe_div(
            precision_sum,
            min(len(relevant), k),
        )
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, paper_id in enumerate(ranking[:k], start=1)
            if paper_id in relevant
        )
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(relevant), k) + 1)
        )
        output[f"ndcg@{k}"] = safe_div(dcg, idcg)
        output[f"hit@{k}"] = float(bool(ranks and min(ranks) <= k))
    output["mrr"] = 1.0 / min(ranks) if ranks else 0.0
    return output


def finish_event_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, float] | None:
    if not rows:
        return None
    ranking = [
        str(row.get("paper_arxiv_id") or "")
        for row in sorted(
            rows,
            key=lambda row: (
                int(row.get("rerank_rank") or 10**12),
                str(row.get("paper_arxiv_id") or ""),
            ),
        )
    ]
    relevant = {
        str(row.get("paper_arxiv_id") or "")
        for row in rows
        if row.get("is_ground_truth")
    }
    return binary_metrics(ranking, relevant)


def rerank_metrics(path: Path) -> Dict[str, Any]:
    events: Dict[str, Dict[str, float]] = {}
    current_event = ""
    event_rows: List[Dict[str, Any]] = []
    for row in iter_jsonl(path):
        event_id = str(row.get("retrieval_event_id") or "")
        starts_repeated_event = bool(
            event_rows
            and event_id == current_event
            and int(row.get("rerank_rank") or 0) == 1
        )
        if current_event and (
            event_id != current_event or starts_repeated_event
        ):
            value = finish_event_metrics(event_rows)
            if value is not None:
                events[current_event] = value
            event_rows = []
        current_event = event_id
        event_rows.append(row)
    value = finish_event_metrics(event_rows)
    if value is not None:
        events[current_event] = value
    metrics = list(events.values())
    if not metrics:
        return {"retrieval_event_count": 0}
    return {
        "retrieval_event_count": len(metrics),
        **{
            key: mean(item[key] for item in metrics)
            for key in metrics[0]
        },
    }


def graph_unique_survival(path: Path) -> Dict[str, Any]:
    unique_pool = set()
    unique_topk = set()
    unique_selected = set()
    semantic_pool = set()
    semantic_topk = set()
    semantic_selected = set()
    for row in iter_jsonl(path):
        if not row.get("is_ground_truth"):
            continue
        key = (
            str(row.get("query_id") or ""),
            str(row.get("paper_arxiv_id") or ""),
        )
        if row.get("is_expanded") and not row.get("is_seed"):
            unique_pool.add(key)
            if row.get("in_selector_topk"):
                unique_topk.add(key)
            if row.get("selector_selected"):
                unique_selected.add(key)
        if row.get("is_seed"):
            semantic_pool.add(key)
            if row.get("in_selector_topk"):
                semantic_topk.add(key)
            if row.get("selector_selected"):
                semantic_selected.add(key)
    return {
        "graph_pool_unique_gt_count": len(unique_pool),
        "graph_pool_unique_gt_entering_topk": len(unique_topk),
        "graph_pool_unique_gt_selected": len(unique_selected),
        "graph_pool_unique_topk_survival": safe_div(
            len(unique_topk), len(unique_pool)
        ),
        "graph_pool_unique_selection_survival": safe_div(
            len(unique_selected), len(unique_pool)
        ),
        "semantic_seed_gt_count": len(semantic_pool),
        "semantic_seed_gt_entering_topk": len(semantic_topk),
        "semantic_seed_gt_selected": len(semantic_selected),
    }


def semrank_cost(
    rows: Mapping[str, Mapping[str, Any]],
    artifacts: Path | None = None,
) -> Dict[str, Any]:
    totals: Counter[str] = Counter()
    wall = 0.0
    for row in rows.values():
        for key, value in (row.get("semrank_stats_delta") or {}).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            totals[str(key)] += numeric
            if key == "reranker.rerank_wall_seconds":
                wall += numeric
    hits = totals["paper_profile.paper_concept_cache_hits"]
    misses = totals["paper_profile.paper_concept_cache_misses"]
    query_count = max(len(rows), 1)
    output = {
        "query_concept_llm_calls": totals["llm.query_concept_llm_calls"],
        "paper_concept_llm_calls": totals["llm.paper_concept_llm_calls"],
        "llm_transport_failures": totals["llm.llm_failures"],
        "llm_parse_failures": totals["llm.llm_parse_failures"],
        "topic_classifier_service_calls": totals[
            "paper_profile.topic_classifier_calls"
        ],
        "topic_classifier_forward_batches": totals[
            "classifier.forward_batches"
        ],
        "topic_classifier_forward_papers": totals[
            "classifier.papers_scored"
        ],
        "topic_classifier_papers": totals[
            "paper_profile.topic_classifier_papers"
        ],
        "paper_concept_cache_hits": hits,
        "paper_concept_cache_misses": misses,
        "paper_concept_cache_hit_rate": safe_div(hits, hits + misses),
        "concept_embedding_cache_hits": totals["encoder.hits"],
        "concept_embedding_cache_misses": totals["encoder.misses"],
        "semrank_rerank_wall_seconds": wall,
        "fallback_rerank_events": totals["reranker.fallback_events"],
        "auxiliary_initial_retrieval_papers": totals[
            "query_profile.auxiliary_initial_retrieval_papers"
        ],
        "auxiliary_initial_retrieval_calls": totals[
            "query_profile.auxiliary_initial_retrieval_calls"
        ],
        "retrieval_embedding_api_calls": totals[
            "retrieval_embedding.api_calls"
        ],
        "retrieval_embedding_encoded_texts": totals[
            "retrieval_embedding.encoded_texts"
        ],
        "retrieval_embedding_cache_hits": totals[
            "retrieval_embedding.cache_hits"
        ],
        "retrieval_embedding_cache_misses": totals[
            "retrieval_embedding.cache_misses"
        ],
        "llm_calls_per_query": safe_div(
            totals["llm.query_concept_llm_calls"]
            + totals["llm.paper_concept_llm_calls"],
            query_count,
        ),
    }
    if artifacts is not None:
        query_statuses: Counter[str] = Counter()
        query_fallback_reasons: Counter[str] = Counter()
        query_profiles = {
            str(profile.get("query_id") or ""): profile
            for profile in iter_jsonl(
                artifacts / "semrank_query_profiles.jsonl"
            )
        }
        query_counts = [
            int(
                profile.get("selected_concept_count")
                or len(profile.get("selected_concepts") or [])
            )
            for profile in query_profiles.values()
        ]
        for profile in query_profiles.values():
            query_statuses[str(profile.get("selection_status") or "unknown")] += 1
            if profile.get("fallback_reason"):
                query_fallback_reasons[str(profile["fallback_reason"])] += 1
        paper_statuses: Counter[str] = Counter()
        paper_fallback_reasons: Counter[str] = Counter()
        paper_profiles = {
            str(profile.get("profile_id") or ""): profile
            for profile in iter_jsonl(
                artifacts / "semrank_paper_concepts.jsonl"
            )
        }
        paper_counts = [
            int(
                profile.get("concept_count")
                or len(profile.get("concepts") or [])
            )
            for profile in paper_profiles.values()
        ]
        for profile in paper_profiles.values():
            paper_statuses[str(profile.get("status") or "unknown")] += 1
            if profile.get("fallback_reason"):
                paper_fallback_reasons[str(profile["fallback_reason"])] += 1
        event_total = 0
        event_candidates = 0
        for event in iter_jsonl(
            artifacts / "semrank_event_profiles.jsonl"
        ):
            event_total += int(event.get("paper_concept_count_total") or 0)
            event_candidates += int(event.get("candidate_count") or 0)
        output.update(
            {
                "average_query_concepts": (
                    mean(query_counts) if query_counts else 0.0
                ),
                "average_unique_paper_profile_concepts": (
                    mean(paper_counts) if paper_counts else 0.0
                ),
                "average_candidate_paper_concepts": safe_div(
                    event_total, event_candidates
                ),
                "query_profile_status_counts": dict(query_statuses),
                "query_profile_fallback_reason_counts": dict(
                    query_fallback_reasons
                ),
                "paper_profile_status_counts": dict(paper_statuses),
                "paper_profile_fallback_reason_counts": dict(
                    paper_fallback_reasons
                ),
            }
        )
    return output


def scientific_audit(artifacts: Path) -> Dict[str, Any]:
    checks = Counter()
    failures = Counter()
    query_profiles_by_qid: Dict[str, set[str]] = defaultdict(set)
    query_profile_rows: Counter[str] = Counter()
    for profile in iter_jsonl(artifacts / "semrank_query_profiles.jsonl"):
        query_id = str(profile.get("query_id") or "")
        profile_id = str(profile.get("query_profile_id") or "")
        query_profile_rows[query_id] += 1
        query_profiles_by_qid[query_id].add(profile_id)
        vocabulary = {
            str(item.get("concept") or "").strip().lower()
            for item in list(profile.get("candidate_topics") or [])
            + list(profile.get("candidate_keyphrases") or [])
        }
        selected = {
            str(value).strip().lower()
            for value in profile.get("selected_concepts") or []
        }
        checks["query_selection_vocab"] += 1
        failures["query_selection_vocab"] += int(not selected <= vocabulary)
        checks["auxiliary_date_cutoff_present"] += 1
        failures["auxiliary_date_cutoff_present"] += int(
            not profile.get("date_cutoff")
        )

    ranks: Dict[str, List[int]] = defaultdict(list)
    rows_by_event: Dict[
        str,
        List[tuple[str, float, float, float, float, str]],
    ] = defaultdict(list)
    event_candidate_counts: Counter[str] = Counter()
    for row in iter_jsonl(artifacts / "paper_rows.jsonl"):
        event_id = str(row.get("retrieval_event_id") or "")
        rank = row.get("rerank_rank")
        if (
            event_id in rows_by_event
            and rank is not None
            and int(rank) == 1
        ):
            rows_by_event[event_id] = []
            ranks[event_id] = []
            event_candidate_counts[event_id] = 0
        rows_by_event[event_id].append(
            (
                str(row.get("paper_arxiv_id") or ""),
                float(row.get("semrank_base_score") or 0.0),
                float(row.get("semrank_base_score_z") or 0.0),
                float(row.get("semrank_concept_score") or 0.0),
                float(row.get("semrank_concept_score_z") or 0.0),
                str(row.get("candidate_pool_signature") or ""),
            )
        )
        event_candidate_counts[event_id] += 1
        fallback = bool(row.get("semrank_fallback_used"))
        expected_final = float(row.get("semrank_base_score_z") or 0.0)
        if not fallback:
            expected_final += float(
                row.get("semrank_concept_score_z") or 0.0
            )
        expected_base = (
            0.4 * float(row.get("query_score_normalized") or 0.0)
            + 0.6
            * float(row.get("subquery_score_normalized") or 0.0)
        )
        checks["exact_final_formula"] += 1
        failures["exact_final_formula"] += int(
            not math.isclose(
                float(row.get("rerank_score") or 0.0),
                expected_final,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
        checks["exact_base_formula"] += 1
        failures["exact_base_formula"] += int(
            not math.isclose(
                float(row.get("semrank_base_score") or 0.0),
                expected_base,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        )
        checks["method"] += 1
        failures["method"] += int(
            row.get("rerank_method") != "semrank_qsq"
        )
        checks["date_flag"] += 1
        failures["date_flag"] += int(
            row.get("passed_date_cutoff") is not True
        )
        if rank is not None:
            ranks[event_id].append(int(rank))
    for event_id, values in ranks.items():
        checks["consecutive_ranks"] += 1
        failures["consecutive_ranks"] += int(
            sorted(values) != list(range(1, len(values) + 1))
        )
    event_profiles = {
        str(event.get("retrieval_event_id") or ""): event
        for event in iter_jsonl(
            artifacts / "semrank_event_profiles.jsonl"
        )
    }
    for event_id, event in event_profiles.items():
        checks["candidate_count_preserved"] += 1
        failures["candidate_count_preserved"] += int(
            int(event.get("candidate_count") or 0)
            != event_candidate_counts[event_id]
            or event.get("candidate_ids_preserved") is not True
        )
    for event_id, rows in rows_by_event.items():
        if not event_id or not rows:
            continue
        base_values = [row[1] for row in rows]
        concept_values = [row[3] for row in rows]

        def expected_z(values: Sequence[float]) -> tuple[List[float], float, float]:
            value_mean = mean(values)
            variance = mean(
                (value - value_mean) ** 2 for value in values
            )
            value_std = math.sqrt(max(variance, 0.0))
            if value_std <= 1e-12:
                return [0.0] * len(values), value_mean, value_std
            return (
                [
                    (value - value_mean) / value_std
                    for value in values
                ],
                value_mean,
                value_std,
            )

        base_z, base_mean, base_std = expected_z(base_values)
        concept_z, concept_mean, concept_std = expected_z(concept_values)
        checks["population_zscore_ddof0"] += len(rows) * 2
        for index, row in enumerate(rows):
            failures["population_zscore_ddof0"] += int(
                not math.isclose(
                    row[2],
                    base_z[index],
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )
            )
            failures["population_zscore_ddof0"] += int(
                not math.isclose(
                    row[4],
                    concept_z[index],
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )
            )
        profile = event_profiles.get(event_id) or {}
        checks["event_distribution_stats"] += 1
        failures["event_distribution_stats"] += int(
            not (
                math.isclose(
                    float(profile.get("base_mean") or 0.0),
                    base_mean,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )
                and math.isclose(
                    float(profile.get("base_std") or 0.0),
                    base_std,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )
                and math.isclose(
                    float(profile.get("concept_mean") or 0.0),
                    concept_mean,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )
                and math.isclose(
                    float(profile.get("concept_std") or 0.0),
                    concept_std,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )
            )
        )
        candidate_ids = sorted(
            row[0] for row in rows
        )
        expected_signature = hashlib.sha256(
            json.dumps(
                candidate_ids,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        row_signatures = {row[5] for row in rows}
        checks["candidate_pool_signature"] += 1
        failures["candidate_pool_signature"] += int(
            len(row_signatures) != 1
            or next(iter(row_signatures), "") != expected_signature
            or str(profile.get("candidate_pool_signature") or "")
            != expected_signature
        )
    for query_id, profiles in query_profiles_by_qid.items():
        checks["one_profile_per_original_query"] += 1
        failures["one_profile_per_original_query"] += int(
            len(profiles) != 1
        )
    results = {
        key: {
            "checked": int(checks[key]),
            "failed": int(failures[key]),
            "passed": failures[key] == 0 and checks[key] > 0,
        }
        for key in sorted(checks)
    }
    return {
        "checks": results,
        "all_checks_passed": bool(results)
        and all(value["passed"] for value in results.values()),
        "scoring_signal_assertion": (
            "exact reconstruction uses only query/subquery normalized "
            "scores and concept score; intent/path/type are absent"
        ),
        "duplicate_query_profile_rows": int(
            sum(
                max(0, count - 1)
                for count in query_profile_rows.values()
            )
        ),
    }


def event_candidate_sets(path: Path) -> Dict[str, set[str]]:
    output: Dict[str, set[str]] = defaultdict(set)
    for row in iter_jsonl(path):
        event_id = str(row.get("retrieval_event_id") or "")
        paper_id = str(row.get("paper_arxiv_id") or "")
        if event_id and paper_id:
            output[event_id].add(paper_id)
    return output


def query_gt_rank_stats(path: Path) -> Dict[str, Dict[str, float]]:
    event_rows: Dict[str, List[int]] = defaultdict(list)
    event_query: Dict[str, str] = {}
    for row in iter_jsonl(path):
        event_id = str(row.get("retrieval_event_id") or "")
        query_id = str(row.get("query_id") or "")
        if event_id:
            event_query[event_id] = query_id
        if row.get("is_ground_truth") and row.get("rerank_rank") is not None:
            event_rows[event_id].append(int(row["rerank_rank"]))
    by_query: Dict[str, List[int]] = defaultdict(list)
    for event_id, ranks in event_rows.items():
        if ranks:
            by_query[event_query.get(event_id, "")].append(min(ranks))
    return {
        query_id: {
            "mean_best_gt_rank": mean(values),
            "events_with_gt": len(values),
        }
        for query_id, values in by_query.items()
        if query_id
    }


def compare_event_pools(
    left: Path, right: Path
) -> Dict[str, Any]:
    first = event_candidate_sets(left)
    second = event_candidate_sets(right)
    common = sorted(set(first) & set(second))
    equal = sum(first[event_id] == second[event_id] for event_id in common)
    return {
        "common_event_count": len(common),
        "equal_candidate_set_count": equal,
        "candidate_set_equality_rate": safe_div(equal, len(common)),
        "left_only_event_count": len(set(first) - set(second)),
        "right_only_event_count": len(set(second) - set(first)),
        "note": (
            "Full online trajectories may diverge after reranking changes "
            "Selector memory and later Planner outputs."
        ),
    }


def query_rows(
    methods: Mapping[str, Mapping[str, Mapping[str, Any]]],
    rank_stats: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> List[Dict[str, Any]]:
    query_ids = sorted(set().union(*(set(rows) for rows in methods.values())))
    output = []
    for query_id in query_ids:
        row: Dict[str, Any] = {"query_id": query_id}
        for name, values in methods.items():
            value = values.get(query_id) or {}
            gt_count = int(value.get("gt_count") or 0)
            candidate_gt = len(set(value.get("candidate_gt_ids") or []))
            selected_gt = len(set(value.get("selected_gt_ids") or []))
            row[f"{name}_retrieval_recall"] = safe_div(
                candidate_gt, gt_count
            )
            row[f"{name}_selection_recall"] = safe_div(
                selected_gt, gt_count
            )
            row[f"{name}_candidate_count"] = int(
                value.get("candidate_count") or 0
            )
            row[f"{name}_selected_count"] = int(
                value.get("selected_count") or 0
            )
            row[f"{name}_mean_best_gt_rank"] = (
                rank_stats.get(name, {})
                .get(query_id, {})
                .get("mean_best_gt_rank")
            )
            row[f"{name}_events_with_gt"] = int(
                rank_stats.get(name, {})
                .get(query_id, {})
                .get("events_with_gt", 0)
            )
        semrank = row.get("semrank_selection_recall", 0.0)
        if "static_selection_recall" in row:
            delta = semrank - row["static_selection_recall"]
            row["semrank_vs_static"] = (
                "improved"
                if delta > 1e-12
                else ("degraded" if delta < -1e-12 else "unchanged")
            )
            semrank_ids = set(
                (methods["semrank"].get(query_id) or {}).get(
                    "candidate_arxiv_ids"
                )
                or []
            )
            static_ids = set(
                (methods["static"].get(query_id) or {}).get(
                    "candidate_arxiv_ids"
                )
                or []
            )
            row["semrank_vs_static_candidate_set_equal"] = (
                semrank_ids == static_ids
            )
            left_rank = row.get("semrank_mean_best_gt_rank")
            right_rank = row.get("static_mean_best_gt_rank")
            row["semrank_minus_static_mean_best_gt_rank"] = (
                float(left_rank) - float(right_rank)
                if left_rank is not None and right_rank is not None
                else None
            )
        if "ours_selection_recall" in row:
            delta = semrank - row["ours_selection_recall"]
            row["semrank_vs_ours"] = (
                "improved"
                if delta > 1e-12
                else ("degraded" if delta < -1e-12 else "unchanged")
            )
            semrank_ids = set(
                (methods["semrank"].get(query_id) or {}).get(
                    "candidate_arxiv_ids"
                )
                or []
            )
            ours_ids = set(
                (methods["ours"].get(query_id) or {}).get(
                    "candidate_arxiv_ids"
                )
                or []
            )
            row["semrank_vs_ours_candidate_set_equal"] = (
                semrank_ids == ours_ids
            )
            left_rank = row.get("semrank_mean_best_gt_rank")
            right_rank = row.get("ours_mean_best_gt_rank")
            row["semrank_minus_ours_mean_best_gt_rank"] = (
                float(left_rank) - float(right_rank)
                if left_rank is not None and right_rank is not None
                else None
            )
        output.append(row)
    return output


def write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted(set().union(*(set(value) for value in values))) if values else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semrank_run", required=True)
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        metavar="NAME=RUN",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    semrank = artifact_dir(args.semrank_run)
    comparisons = {}
    for value in args.comparison:
        if "=" not in value:
            parser.error("--comparison must be NAME=RUN")
        name, run = value.split("=", 1)
        comparisons[name.strip()] = artifact_dir(run)
    methods = {"semrank": semrank, **comparisons}
    query_data = {
        name: latest_by_query(path / "query_results.jsonl")
        for name, path in methods.items()
    }
    rank_data = {
        name: query_gt_rank_stats(path / "paper_rows.jsonl")
        for name, path in methods.items()
    }
    summary = {
        "methods": {
            name: {
                "artifact_dir": str(path),
                "end_to_end": end_to_end(query_data[name]),
                "rerank": rerank_metrics(path / "paper_rows.jsonl"),
                "graph_unique_gt": graph_unique_survival(
                    path / "paper_rows.jsonl"
                ),
            }
            for name, path in methods.items()
        },
        "semrank_cost": semrank_cost(query_data["semrank"], semrank),
        "semrank_scientific_audit": scientific_audit(semrank),
        "online_event_pool_comparisons": {
            f"semrank_vs_{name}": compare_event_pools(
                semrank / "paper_rows.jsonl",
                path / "paper_rows.jsonl",
            )
            for name, path in comparisons.items()
        },
    }
    rows = query_rows(query_data, rank_data)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(output / "query_level.jsonl", rows)
    write_csv(output / "query_level.csv", rows)
    markdown = [
        "# SemRank-QSQ analysis",
        "",
        f"Scientific audit passed: **{summary['semrank_scientific_audit']['all_checks_passed']}**",
        "",
        "| Method | Sel R | Sel P | Sel F1 | Ret R | Ret P | GT conversion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in summary["methods"].items():
        metric = value["end_to_end"]
        markdown.append(
            f"| {name} | {metric.get('selection_recall', 0):.6f} | "
            f"{metric.get('selection_precision', 0):.6f} | "
            f"{metric.get('selection_f1', 0):.6f} | "
            f"{metric.get('retrieval_recall', 0):.6f} | "
            f"{metric.get('retrieval_precision', 0):.6f} | "
            f"{metric.get('gt_conversion', 0):.6f} |"
        )
    markdown.extend(
        [
            "",
            "Full-online candidate-set differences after the first divergent "
            "Selector/Planner decisions are expected; inspect "
            "`online_event_pool_comparisons` separately from frozen replay.",
        ]
    )
    (output / "report.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
