#!/usr/bin/env python3
"""Analyze graph expansion against Deep merged on the full candidate pool.

This is a post-annotation analysis over exactly two retrieval arms:

* ``graph_only`` = graph and not deep_merged
* ``deep_merged_only`` = deep_merged and not graph
* ``graph_and_deep_merged`` = graph and deep_merged

Other saved retrieval sources do not participate in the candidate universe or
partitioning.  The historical ``source_partition`` field is left untouched for
backward compatibility.  All statistics here are candidate-level (query-paper
pairs), never occurrence-weighted.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


INFORMATION_RELATIONSHIPS = {"direct", "partial", "contextual"}
DIRECT_OR_PARTIAL = {"direct", "partial"}
RELEVANCE_SCORES = {
    "direct": 1.0,
    "partial": 2.0 / 3.0,
    "contextual": 1.0 / 3.0,
    "unrelated": 0.0,
}

SEMANTIC_COMPLEMENT_DEFINITIONS: Sequence[Mapping[str, str]] = (
    {
        "category": "historical_predecessor_proxy",
        "field": "information_added",
        "label": "historical_context",
        "measurement": "proxy",
        "interpretation": "Historical context or predecessor-like background.",
        "limitation": "Does not by itself prove a direct predecessor relationship.",
    },
    {
        "category": "method_component",
        "field": "scholarly_roles",
        "label": "method_component",
        "measurement": "direct_label",
        "interpretation": "A component used by, or relevant to, the queried method.",
        "limitation": "Component importance is not graded.",
    },
    {
        "category": "direct_target",
        "field": "scholarly_roles",
        "label": "direct_target",
        "measurement": "direct_label",
        "interpretation": "The paper directly targets the requested relationship.",
        "limitation": "This role may co-occur with method- or task-level roles.",
    },
    {
        "category": "task_or_application",
        "field": "scholarly_roles",
        "label": "task_or_application",
        "measurement": "direct_label",
        "interpretation": "The paper matches the requested task or application.",
        "limitation": "A task match alone may still be only partial relevance.",
    },
    {
        "category": "direct_task_method_match",
        "field": "derived",
        "label": "direct_task_method_match",
        "measurement": "derived",
        "interpretation": "Direct relationship plus a target, method-component, or task/application role.",
        "limitation": "Derived from the existing relationship and role labels, not a separate judgment.",
    },
    {
        "category": "mechanism_or_theory_proxy",
        "field": "information_added",
        "label": "mechanism_or_theory",
        "measurement": "proxy",
        "interpretation": "Mechanistic or theoretical explanation.",
        "limitation": "The current label does not distinguish explicit from implicit mechanisms.",
    },
    {
        "category": "application_domain_proxy",
        "field": "information_added",
        "label": "application_domain",
        "measurement": "proxy",
        "interpretation": "An application-domain contribution.",
        "limitation": "Does not establish that the application crosses domains; that needs a secondary label.",
    },
    {
        "category": "background_or_foundation",
        "field": "scholarly_roles",
        "label": "background_or_foundation",
        "measurement": "direct_label",
        "interpretation": "Background or foundational support.",
        "limitation": "May overlap historical or theoretical contributions.",
    },
    {
        "category": "dataset_or_domain_bridge",
        "field": "scholarly_roles",
        "label": "dataset_or_domain",
        "measurement": "direct_label",
        "interpretation": "Dataset or domain bridge to the query.",
        "limitation": "Does not imply cross-domain transfer.",
    },
    {
        "category": "survey_or_taxonomy",
        "field": "scholarly_roles",
        "label": "survey_or_taxonomy",
        "measurement": "direct_label",
        "interpretation": "Survey, taxonomy, or synthesis support.",
        "limitation": "Secondary evidence may not answer a primary-study request.",
    },
    {
        "category": "metric_or_benchmark",
        "field": "information_added",
        "label": "metric_or_benchmark",
        "measurement": "direct_label",
        "interpretation": "Metric or benchmark contribution.",
        "limitation": "Benchmark relevance still depends on the query scope.",
    },
)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _safe_rate(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def primary_partition(sources: Iterable[str]) -> Optional[str]:
    source_set = set(sources)
    in_graph = "graph" in source_set
    in_deep_merged = "deep_merged" in source_set
    if in_graph and in_deep_merged:
        return "graph_and_deep_merged"
    if in_graph:
        return "graph_only"
    if in_deep_merged:
        return "deep_merged_only"
    return None


def _analysis_groups(sources: Iterable[str]) -> List[str]:
    source_set = set(sources)
    partition = primary_partition(source_set)
    if partition is None:
        return []
    groups = ["union", partition]
    if "graph" in source_set:
        groups.append("graph")
    if "deep_merged" in source_set:
        groups.append("deep_merged")
    return groups


def _annotation_field(
    annotation: Mapping[str, Any], canonical_name: str, legacy_name: str
) -> Any:
    """Read the canonical annotation field while accepting completed v1 data."""

    if canonical_name in annotation:
        return annotation[canonical_name]
    return annotation.get(legacy_name)


def _relevance_grade(annotation: Mapping[str, Any]) -> str:
    return str(_annotation_field(annotation, "relevance_grade", "relationship"))


def _scholarly_roles(annotation: Mapping[str, Any]) -> List[str]:
    return list(_annotation_field(annotation, "scholarly_roles", "relation_roles") or [])


def _information_added(annotation: Mapping[str, Any]) -> List[str]:
    return list(
        _annotation_field(annotation, "information_added", "contribution_types") or []
    )


def _update_counter(
    counts: MutableMapping[str, float], annotation: Mapping[str, Any], is_ground_truth: bool
) -> None:
    relevance_grade = _relevance_grade(annotation)
    distance = str(annotation["semantic_distance"])
    paper_type = str(annotation["paper_type"])
    matched_aspects = list(annotation.get("matched_aspect_ids") or [])
    scholarly_roles = set(_scholarly_roles(annotation))
    counts["candidate_count"] += 1
    counts[f"relevance_grade:{relevance_grade}"] += 1
    counts[f"distance:{distance}"] += 1
    counts[f"paper_type:{paper_type}"] += 1
    counts["ground_truth_count"] += int(is_ground_truth)
    counts["confidence_sum"] += float(annotation["confidence"])
    counts["needs_full_text_count"] += int(bool(annotation.get("needs_full_text")))
    counts["matched_aspect_sum"] += len(matched_aspects)
    counts["matched_aspect_any_count"] += int(bool(matched_aspects))
    counts["matched_aspect_ge3_count"] += int(len(matched_aspects) >= 3)
    if relevance_grade in DIRECT_OR_PARTIAL:
        counts["direct_or_partial_count"] += 1
    if relevance_grade == "direct" and scholarly_roles.intersection(
        {"direct_target", "method_component", "task_or_application"}
    ):
        counts["direct_task_method_match_count"] += 1
    if relevance_grade in INFORMATION_RELATIONSHIPS:
        counts["information_bearing_count"] += 1
        counts["information_matched_aspect_sum"] += len(matched_aspects)
        for value in scholarly_roles:
            counts[f"information_role:{value}"] += 1
        for value in _information_added(annotation):
            counts[f"information_contribution:{value}"] += 1
        counts[f"information_paper_type:{paper_type}"] += 1
    if relevance_grade in RELEVANCE_SCORES:
        counts["scored_count"] += 1
        counts["relevance_score_sum"] += RELEVANCE_SCORES[relevance_grade]
    for value in scholarly_roles:
        counts[f"role:{value}"] += 1
    for value in _information_added(annotation):
        counts[f"contribution:{value}"] += 1
    for value in annotation.get("exclusion_reasons") or []:
        counts[f"exclusion:{value}"] += 1


def _summary_row(group: str, counts: Mapping[str, float]) -> Dict[str, Any]:
    total = counts.get("candidate_count", 0)
    information = counts.get("information_bearing_count", 0)
    exact_near = counts.get("distance:exact", 0) + counts.get("distance:near", 0)
    return {
        "group": group,
        "candidate_count": int(total),
        "ground_truth_count": int(counts.get("ground_truth_count", 0)),
        "ground_truth_rate": _safe_rate(counts.get("ground_truth_count", 0), total),
        "direct_count": int(counts.get("relevance_grade:direct", 0)),
        "partial_count": int(counts.get("relevance_grade:partial", 0)),
        "contextual_count": int(counts.get("relevance_grade:contextual", 0)),
        "unrelated_count": int(counts.get("relevance_grade:unrelated", 0)),
        "direct_rate": _safe_rate(counts.get("relevance_grade:direct", 0), total),
        "direct_or_partial_count": int(counts.get("direct_or_partial_count", 0)),
        "direct_or_partial_rate": _safe_rate(counts.get("direct_or_partial_count", 0), total),
        "direct_task_method_match_count": int(
            counts.get("direct_task_method_match_count", 0)
        ),
        "direct_task_method_match_rate": _safe_rate(
            counts.get("direct_task_method_match_count", 0), total
        ),
        "information_bearing_count": int(information),
        "information_bearing_rate": _safe_rate(information, total),
        "exact_or_near_count": int(exact_near),
        "exact_or_near_rate": _safe_rate(exact_near, total),
        "mean_relevance_score": _safe_rate(
            counts.get("relevance_score_sum", 0), counts.get("scored_count", 0)
        ),
        "mean_confidence": _safe_rate(counts.get("confidence_sum", 0), total),
        "needs_full_text_rate": _safe_rate(counts.get("needs_full_text_count", 0), total),
        "mean_matched_aspects": _safe_rate(counts.get("matched_aspect_sum", 0), total),
        "mean_matched_aspects_given_information": _safe_rate(
            counts.get("information_matched_aspect_sum", 0), information
        ),
        "matched_aspect_any_rate": _safe_rate(counts.get("matched_aspect_any_count", 0), total),
        "matched_aspect_ge3_rate": _safe_rate(counts.get("matched_aspect_ge3_count", 0), total),
    }


def _semantic_count_key(definition: Mapping[str, str]) -> str:
    field = definition["field"]
    label = definition["label"]
    if field == "derived":
        return f"{label}_count"
    prefix = (
        "information_role" if field == "scholarly_roles" else "information_contribution"
    )
    return f"{prefix}:{label}"


def _semantic_complement_rows(
    group_counts: Mapping[str, Mapping[str, float]],
    query_counts: Mapping[Tuple[str, str], Mapping[str, float]],
    *,
    bootstrap_samples: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    graph = group_counts["graph_only"]
    deep = group_counts["deep_merged_only"]
    overlap = group_counts["graph_and_deep_merged"]
    graph_queries = {query_id for group, query_id in query_counts if group == "graph_only"}
    deep_queries = {
        query_id for group, query_id in query_counts if group == "deep_merged_only"
    }
    paired_queries = sorted(graph_queries & deep_queries)
    for definition_index, definition in enumerate(SEMANTIC_COMPLEMENT_DEFINITIONS):
        key = _semantic_count_key(definition)
        graph_count = graph.get(key, 0)
        deep_count = deep.get(key, 0)
        overlap_count = overlap.get(key, 0)
        graph_den = graph.get("information_bearing_count", 0)
        deep_den = deep.get("information_bearing_count", 0)
        overlap_den = overlap.get("information_bearing_count", 0)
        graph_rate = _safe_rate(graph_count, graph_den)
        deep_rate = _safe_rate(deep_count, deep_den)
        overlap_rate = _safe_rate(overlap_count, overlap_den)
        macro_share_pairs = []
        macro_yield_pairs = []
        for query_id in paired_queries:
            graph_query = query_counts[("graph_only", query_id)]
            deep_query = query_counts[("deep_merged_only", query_id)]
            graph_query_information = graph_query.get("information_bearing_count", 0)
            deep_query_information = deep_query.get("information_bearing_count", 0)
            if graph_query_information and deep_query_information:
                macro_share_pairs.append(
                    (
                        graph_query.get(key, 0) / graph_query_information,
                        deep_query.get(key, 0) / deep_query_information,
                    )
                )
            macro_yield_pairs.append(
                (
                    graph_query.get(key, 0) / graph_query["candidate_count"],
                    deep_query.get(key, 0) / deep_query["candidate_count"],
                )
            )
        share_differences = [graph_value - deep_value for graph_value, deep_value in macro_share_pairs]
        yield_differences = [graph_value - deep_value for graph_value, deep_value in macro_yield_pairs]
        share_ci = _bootstrap_mean_ci(
            share_differences,
            samples=bootstrap_samples,
            seed=20260800 + definition_index,
        )
        yield_ci = _bootstrap_mean_ci(
            yield_differences,
            samples=bootstrap_samples,
            seed=20260900 + definition_index,
        )
        rows.append(
            {
                **definition,
                "denominator": "information_bearing_candidates",
                "graph_only_count": int(graph_count),
                "graph_only_rate": graph_rate,
                "graph_only_pool_yield": _safe_rate(
                    graph_count, graph.get("candidate_count", 0)
                ),
                "deep_merged_only_count": int(deep_count),
                "deep_merged_only_rate": deep_rate,
                "deep_merged_only_pool_yield": _safe_rate(
                    deep_count, deep.get("candidate_count", 0)
                ),
                "graph_and_deep_merged_count": int(overlap_count),
                "graph_and_deep_merged_rate": overlap_rate,
                "graph_and_deep_merged_pool_yield": _safe_rate(
                    overlap_count, overlap.get("candidate_count", 0)
                ),
                "graph_minus_deep_rate": (
                    graph_rate - deep_rate
                    if graph_rate is not None and deep_rate is not None
                    else None
                ),
                "graph_over_deep_rate_ratio": (
                    graph_rate / deep_rate
                    if graph_rate is not None and deep_rate not in (None, 0)
                    else None
                ),
                "graph_minus_deep_pool_yield": (
                    _safe_rate(graph_count, graph.get("candidate_count", 0))
                    - _safe_rate(deep_count, deep.get("candidate_count", 0))
                    if graph.get("candidate_count", 0) and deep.get("candidate_count", 0)
                    else None
                ),
                "macro_share_paired_query_count": len(macro_share_pairs),
                "graph_only_macro_share": sum(value[0] for value in macro_share_pairs)
                / len(macro_share_pairs),
                "deep_merged_only_macro_share": sum(
                    value[1] for value in macro_share_pairs
                )
                / len(macro_share_pairs),
                "macro_share_graph_minus_deep": sum(share_differences)
                / len(share_differences),
                "macro_share_bootstrap_95_ci": list(share_ci) if share_ci else None,
                "macro_share_graph_wins": sum(value > 0 for value in share_differences),
                "macro_share_deep_wins": sum(value < 0 for value in share_differences),
                "macro_share_ties": sum(value == 0 for value in share_differences),
                "graph_only_macro_pool_yield": sum(value[0] for value in macro_yield_pairs)
                / len(macro_yield_pairs),
                "deep_merged_only_macro_pool_yield": sum(
                    value[1] for value in macro_yield_pairs
                )
                / len(macro_yield_pairs),
                "macro_pool_yield_graph_minus_deep": sum(yield_differences)
                / len(yield_differences),
                "macro_pool_yield_bootstrap_95_ci": list(yield_ci) if yield_ci else None,
            }
        )
    return rows


def _load_query_aspects(
    work_dir: Path, query_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for query in query_rows:
        query_id = str(query["query_id"])
        rubric_path = work_dir / "outputs" / "rubrics" / f"{query['query_token']}.json"
        rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
        if rubric.get("query_id") != query_id:
            raise ValueError(f"Rubric/query mismatch at {rubric_path}")
        aspects = {
            str(row["aspect_id"]): str(row.get("label") or row["aspect_id"])
            for row in rubric.get("aspects") or []
        }
        if not aspects:
            raise ValueError(f"Rubric has no aspects at {rubric_path}")
        result[query_id] = aspects
    return result


def _aspect_coverage(
    query_aspects: Mapping[str, Mapping[str, str]],
    observed: Mapping[Tuple[str, str], set[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for query_id in sorted(query_aspects):
        rubric_ids = set(query_aspects[query_id])
        deep = set(observed.get(("deep_merged", query_id), set())) & rubric_ids
        graph_added = set(observed.get(("graph_only", query_id), set())) & rubric_ids
        novel = graph_added - deep
        union = deep | graph_added
        row = {
            "query_id": query_id,
            "rubric_aspect_count": len(rubric_ids),
            "deep_merged_information_aspect_count": len(deep),
            "graph_only_information_aspect_count": len(graph_added),
            "novel_graph_information_aspect_count": len(novel),
            "union_information_aspect_count": len(union),
            "deep_merged_aspect_coverage_rate": _safe_rate(len(deep), len(rubric_ids)),
            "graph_only_aspect_coverage_rate": _safe_rate(
                len(graph_added), len(rubric_ids)
            ),
            "union_aspect_coverage_rate": _safe_rate(len(union), len(rubric_ids)),
            "aspect_coverage_gain": _safe_rate(len(novel), len(rubric_ids)),
            "novel_graph_information_aspect_ids": sorted(novel),
            "novel_graph_information_aspect_labels": [
                query_aspects[query_id][aspect_id] for aspect_id in sorted(novel)
            ],
        }
        rows.append(row)
        totals["rubric"] += len(rubric_ids)
        totals["deep"] += len(deep)
        totals["graph_added"] += len(graph_added)
        totals["novel"] += len(novel)
        totals["union"] += len(union)
        totals["queries_with_novel"] += int(bool(novel))
    query_count = len(rows)
    summary = {
        "unit": "query-local rubric aspect",
        "candidate_filter": "information-bearing only",
        "baseline": "inclusive deep_merged",
        "addition": "graph_only",
        "query_count": query_count,
        "rubric_query_aspect_count": totals["rubric"],
        "deep_merged_information_query_aspect_count": totals["deep"],
        "graph_only_information_query_aspect_count": totals["graph_added"],
        "novel_graph_information_query_aspect_count": totals["novel"],
        "union_information_query_aspect_count": totals["union"],
        "deep_merged_micro_aspect_coverage_rate": _safe_rate(
            totals["deep"], totals["rubric"]
        ),
        "graph_only_micro_aspect_coverage_rate": _safe_rate(
            totals["graph_added"], totals["rubric"]
        ),
        "union_micro_aspect_coverage_rate": _safe_rate(
            totals["union"], totals["rubric"]
        ),
        "micro_aspect_coverage_gain": _safe_rate(totals["novel"], totals["rubric"]),
        "queries_with_novel_graph_aspects": totals["queries_with_novel"],
        "queries_with_novel_graph_aspects_rate": _safe_rate(
            totals["queries_with_novel"], query_count
        ),
        "deep_merged_macro_aspect_coverage_rate": sum(
            row["deep_merged_aspect_coverage_rate"] for row in rows
        )
        / query_count,
        "union_macro_aspect_coverage_rate": sum(
            row["union_aspect_coverage_rate"] for row in rows
        )
        / query_count,
        "macro_aspect_coverage_gain": sum(row["aspect_coverage_gain"] for row in rows)
        / query_count,
    }
    return rows, summary


def _semantic_categories(annotation: Mapping[str, Any]) -> set[str]:
    relevance_grade = _relevance_grade(annotation)
    if relevance_grade not in INFORMATION_RELATIONSHIPS:
        return set()
    roles = set(_scholarly_roles(annotation))
    contributions = set(_information_added(annotation))
    result = set()
    for definition in SEMANTIC_COMPLEMENT_DEFINITIONS:
        field = definition["field"]
        label = definition["label"]
        if field == "scholarly_roles" and label in roles:
            result.add(definition["category"])
        elif field == "information_added" and label in contributions:
            result.add(definition["category"])
        elif (
            field == "derived"
            and relevance_grade == "direct"
            and roles.intersection(
                {"direct_target", "method_component", "task_or_application"}
            )
        ):
            result.add(definition["category"])
    return result


def _sample_profile_metrics(
    profiles: Sequence[Tuple[bool, bool, frozenset[str], frozenset[str]]],
    rubric_aspect_count: int,
) -> Dict[str, float]:
    information_count = sum(profile[0] for profile in profiles)
    strong_count = sum(profile[1] for profile in profiles)
    aspects: set[str] = set()
    categories: set[str] = set()
    for is_information, _, matched_aspects, semantic_categories in profiles:
        if is_information:
            aspects.update(matched_aspects)
            categories.update(semantic_categories)
    return {
        "information_bearing_rate": information_count / len(profiles),
        "direct_or_partial_rate": strong_count / len(profiles),
        "aspect_coverage_rate": len(aspects) / rubric_aspect_count,
        "semantic_category_richness": float(len(categories)),
    }


def _percentile_interval(values: Sequence[float]) -> Optional[List[float]]:
    if not values:
        return None
    ordered = sorted(values)
    low = ordered[max(0, int(len(ordered) * 0.025))]
    high = ordered[min(len(ordered) - 1, int(len(ordered) * 0.975) - 1)]
    return [low, high]


def _equal_budget_rarefaction(
    profiles: Mapping[
        Tuple[str, str],
        Sequence[Tuple[bool, bool, frozenset[str], frozenset[str]]],
    ],
    query_aspects: Mapping[str, Mapping[str, str]],
    *,
    samples: int,
    seed: int = 20260716,
) -> Dict[str, Any]:
    if samples <= 0:
        return {"enabled": False, "samples": samples}
    paired_queries = sorted(
        query_id
        for query_id in query_aspects
        if profiles.get(("graph_only", query_id))
        and profiles.get(("deep_merged_only", query_id))
    )
    if not paired_queries:
        return {
            "enabled": False,
            "samples": samples,
            "reason": "No queries contain both exclusive pools.",
        }
    metric_names = (
        "information_bearing_rate",
        "direct_or_partial_rate",
        "aspect_coverage_rate",
        "semantic_category_richness",
    )
    budget_specs: Sequence[Optional[int]] = (20, 50, 100, 200, 500, None)
    budget_curve: List[Dict[str, Any]] = []
    for budget_index, target_budget in enumerate(budget_specs):
        rng = random.Random(seed + budget_index)
        observations: Dict[str, Dict[str, List[float]]] = {
            metric: {"graph": [], "deep": [], "difference": []}
            for metric in metric_names
        }
        actual_budgets = {
            query_id: min(
                len(profiles[("graph_only", query_id)]),
                len(profiles[("deep_merged_only", query_id)]),
                target_budget if target_budget is not None else float("inf"),
            )
            for query_id in paired_queries
        }
        for _ in range(samples):
            per_query: Dict[str, Dict[str, List[float]]] = {
                metric: {"graph": [], "deep": []} for metric in metric_names
            }
            for query_id in paired_queries:
                graph_pool = profiles[("graph_only", query_id)]
                deep_pool = profiles[("deep_merged_only", query_id)]
                budget = int(actual_budgets[query_id])
                graph_sample = (
                    graph_pool
                    if len(graph_pool) == budget
                    else rng.sample(graph_pool, budget)
                )
                deep_sample = (
                    deep_pool if len(deep_pool) == budget else rng.sample(deep_pool, budget)
                )
                graph_metrics = _sample_profile_metrics(
                    graph_sample, len(query_aspects[query_id])
                )
                deep_metrics = _sample_profile_metrics(
                    deep_sample, len(query_aspects[query_id])
                )
                for metric in metric_names:
                    per_query[metric]["graph"].append(graph_metrics[metric])
                    per_query[metric]["deep"].append(deep_metrics[metric])
            for metric in metric_names:
                graph_macro = sum(per_query[metric]["graph"]) / len(paired_queries)
                deep_macro = sum(per_query[metric]["deep"]) / len(paired_queries)
                observations[metric]["graph"].append(graph_macro)
                observations[metric]["deep"].append(deep_macro)
                observations[metric]["difference"].append(graph_macro - deep_macro)
        metric_rows = []
        for metric in metric_names:
            graph_values = observations[metric]["graph"]
            deep_values = observations[metric]["deep"]
            differences = observations[metric]["difference"]
            metric_rows.append(
                {
                    "metric": metric,
                    "graph_only_mean": sum(graph_values) / samples,
                    "deep_merged_only_mean": sum(deep_values) / samples,
                    "graph_minus_deep_mean": sum(differences) / samples,
                    "sampling_95_interval": _percentile_interval(differences),
                }
            )
        budget_curve.append(
            {
                "target_candidate_budget_per_query": (
                    target_budget if target_budget is not None else "max_equal"
                ),
                "minimum_actual_budget": min(actual_budgets.values()),
                "maximum_actual_budget": max(actual_budgets.values()),
                "sampled_candidates_per_arm_per_repeat": sum(actual_budgets.values()),
                "metrics": metric_rows,
            }
        )
    max_equal = budget_curve[-1]
    return {
        "enabled": True,
        "method": "within-query equal-candidate-budget repeated subsampling",
        "comparison": "graph_only versus deep_merged_only",
        "uncertainty_scope": "subsampling variation only; not a query-bootstrap confidence interval",
        "samples": samples,
        "seed": seed,
        "paired_query_count": len(paired_queries),
        "fixed_budget_note": (
            "Fixed budgets are random full-pool samples, not reranked Top-K; budgets are capped "
            "by the smaller exclusive pool when necessary."
        ),
        "sampled_candidates_per_arm_per_repeat": max_equal[
            "sampled_candidates_per_arm_per_repeat"
        ],
        "metrics": max_equal["metrics"],
        "budget_curve": budget_curve,
    }


def _bucket_at_least(value: Optional[float], boundaries: Sequence[Tuple[float, str]]) -> str:
    if value is None:
        return "missing"
    for maximum, label in boundaries:
        if value <= maximum:
            return label
    return boundaries[-1][1].replace("<=", ">")


def _edge_bucket(source_stats: Mapping[str, Any]) -> str:
    edges = set(source_stats.get("edge_types") or [])
    if edges == {"reference"}:
        return "reference_only"
    if edges == {"citation"}:
        return "citation_only"
    if edges == {"citation", "reference"}:
        return "citation_and_reference"
    if not edges:
        return "missing"
    return "+".join(sorted(str(value) for value in edges))


def _graph_role_bucket(source_stats: Mapping[str, Any]) -> str:
    is_seed = bool(source_stats.get("is_seed"))
    is_expanded = bool(source_stats.get("is_expanded"))
    if is_seed and is_expanded:
        return "seed_and_expanded"
    if is_seed:
        return "seed_only"
    if is_expanded:
        return "expanded_only"
    return "neither"


def _structure_bucket_rows(
    structure_counts: Mapping[Tuple[str, str], Mapping[str, float]],
    structure_query_counts: Mapping[Tuple[str, str, str], Mapping[str, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    all_summary = _summary_row("graph_only", structure_counts[("all", "all")])
    for (dimension, bucket), counts in sorted(structure_counts.items()):
        row = _summary_row("graph_only", counts)
        row.pop("group")
        query_summaries = [
            _summary_row("graph_only", query_counts)
            for (query_dimension, query_bucket, _), query_counts in structure_query_counts.items()
            if query_dimension == dimension and query_bucket == bucket
        ]
        information_rate = row["information_bearing_rate"]
        strong_rate = row["direct_or_partial_rate"]
        relevance = row["mean_relevance_score"]
        rows.append(
            {
                "scope": "graph_only",
                "dimension": dimension,
                "bucket": bucket,
                **row,
                "query_count": len(query_summaries),
                "macro_information_bearing_rate": sum(
                    value["information_bearing_rate"] for value in query_summaries
                )
                / len(query_summaries),
                "macro_direct_or_partial_rate": sum(
                    value["direct_or_partial_rate"] for value in query_summaries
                )
                / len(query_summaries),
                "macro_mean_relevance_score": sum(
                    value["mean_relevance_score"] for value in query_summaries
                )
                / len(query_summaries),
                "information_bearing_lift_vs_graph_only": (
                    information_rate / all_summary["information_bearing_rate"]
                    if all_summary["information_bearing_rate"] not in (None, 0)
                    else None
                ),
                "direct_or_partial_lift_vs_graph_only": (
                    strong_rate / all_summary["direct_or_partial_rate"]
                    if all_summary["direct_or_partial_rate"] not in (None, 0)
                    else None
                ),
                "relevance_lift_vs_graph_only": (
                    relevance / all_summary["mean_relevance_score"]
                    if all_summary["mean_relevance_score"] not in (None, 0)
                    else None
                ),
            }
        )
    return rows


def _count_field_for_metric(metric: str) -> str:
    return {
        "candidate": "candidate_count",
        "ground_truth": "ground_truth_count",
        "direct": "direct_count",
        "direct_or_partial": "direct_or_partial_count",
        "contextual": "contextual_count",
        "information_bearing": "information_bearing_count",
    }[metric]


def _marginal_coverage(
    group_rows: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    base = group_rows["deep_merged"]
    added = group_rows["graph_only"]
    union = group_rows["union"]
    overlap = group_rows["graph_and_deep_merged"]
    candidate_growth = _safe_rate(added["candidate_count"], base["candidate_count"])
    metric_rows = []
    for metric in (
        "candidate",
        "ground_truth",
        "direct",
        "direct_or_partial",
        "contextual",
        "information_bearing",
    ):
        key = _count_field_for_metric(metric)
        base_count = int(base[key])
        added_count = int(added[key])
        union_count = int(union[key])
        relative_gain = _safe_rate(added_count, base_count)
        metric_rows.append(
            {
                "metric": metric,
                "deep_merged_base_count": base_count,
                "graph_only_added_count": added_count,
                "union_count": union_count,
                "relative_gain_over_deep_merged": relative_gain,
                "added_share_of_union": _safe_rate(added_count, union_count),
                "marginal_yield_per_added_candidate": _safe_rate(
                    added_count, added["candidate_count"]
                ),
                "gain_per_1000_added_candidates": _safe_rate(
                    1000.0 * added_count, added["candidate_count"]
                ),
                "gain_efficiency_vs_pool_growth": (
                    relative_gain / candidate_growth
                    if relative_gain is not None and candidate_growth not in (None, 0)
                    else None
                ),
            }
        )
    graph_count = group_rows["graph"]["candidate_count"]
    deep_count = base["candidate_count"]
    return {
        "base": "deep_merged",
        "addition": "graph_only",
        "candidate_pool_growth_rate": candidate_growth,
        "graph_candidate_count": graph_count,
        "deep_merged_candidate_count": deep_count,
        "overlap_candidate_count": overlap["candidate_count"],
        "overlap_share_of_graph": _safe_rate(overlap["candidate_count"], graph_count),
        "overlap_share_of_deep_merged": _safe_rate(overlap["candidate_count"], deep_count),
        "jaccard_graph_deep_merged": _safe_rate(overlap["candidate_count"], union["candidate_count"]),
        "metrics": metric_rows,
    }


def _query_row(query_id: str, counts: Mapping[str, float]) -> Dict[str, Any]:
    total = counts.get("candidate_count", 0)
    return {
        "query_id": query_id,
        "candidate_count": int(total),
        "direct_rate": _safe_rate(counts.get("relevance_grade:direct", 0), total),
        "direct_or_partial_rate": _safe_rate(counts.get("direct_or_partial_count", 0), total),
        "information_bearing_rate": _safe_rate(counts.get("information_bearing_count", 0), total),
        "mean_relevance_score": _safe_rate(
            counts.get("relevance_score_sum", 0), counts.get("scored_count", 0)
        ),
    }


def _bootstrap_mean_ci(
    values: Sequence[float], *, samples: int, seed: int
) -> Optional[Tuple[float, float]]:
    if not values or samples <= 0:
        return None
    rng = random.Random(seed)
    means = []
    size = len(values)
    for _ in range(samples):
        means.append(sum(rng.choice(values) for _ in range(size)) / size)
    means.sort()
    low_index = max(0, int(samples * 0.025))
    high_index = min(samples - 1, int(samples * 0.975) - 1)
    return means[low_index], means[high_index]


def _query_comparison(
    query_counts: Mapping[Tuple[str, str], Mapping[str, float]],
    query_text: Mapping[str, str],
    *,
    bootstrap_samples: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    graph_queries = {query_id for group, query_id in query_counts if group == "graph_only"}
    deep_queries = {query_id for group, query_id in query_counts if group == "deep_merged_only"}
    paired_queries = sorted(graph_queries & deep_queries)
    rows: List[Dict[str, Any]] = []
    metrics = (
        "direct_rate",
        "direct_or_partial_rate",
        "information_bearing_rate",
        "mean_relevance_score",
    )
    for query_id in paired_queries:
        graph = _query_row(query_id, query_counts[("graph_only", query_id)])
        deep = _query_row(query_id, query_counts[("deep_merged_only", query_id)])
        row: Dict[str, Any] = {
            "query_id": query_id,
            "query": query_text.get(query_id),
            "graph_only_candidate_count": graph["candidate_count"],
            "deep_merged_only_candidate_count": deep["candidate_count"],
        }
        for metric in metrics:
            row[f"graph_only_{metric}"] = graph[metric]
            row[f"deep_merged_only_{metric}"] = deep[metric]
            row[f"graph_minus_deep_{metric}"] = graph[metric] - deep[metric]
        rows.append(row)

    paired_summary: Dict[str, Any] = {
        "paired_query_count": len(rows),
        "bootstrap_samples": bootstrap_samples,
        "difference_direction": "graph_only_minus_deep_merged_only",
        "metrics": [],
    }
    for metric_index, metric in enumerate(metrics):
        differences = [row[f"graph_minus_deep_{metric}"] for row in rows]
        ci = _bootstrap_mean_ci(
            differences, samples=bootstrap_samples, seed=20260716 + metric_index
        )
        paired_summary["metrics"].append(
            {
                "metric": metric,
                "graph_macro_mean": sum(row[f"graph_only_{metric}"] for row in rows) / len(rows),
                "deep_merged_macro_mean": sum(
                    row[f"deep_merged_only_{metric}"] for row in rows
                )
                / len(rows),
                "mean_difference": sum(differences) / len(differences),
                "bootstrap_95_ci": list(ci) if ci else None,
                "graph_wins": sum(value > 0 for value in differences),
                "deep_merged_wins": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }
        )
    return rows, paired_summary


def _pct(value: Optional[float]) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def _report_markdown(
    group_rows: Mapping[str, Mapping[str, Any]],
    semantic_rows: Sequence[Mapping[str, Any]],
    structure_rows: Sequence[Mapping[str, Any]],
    marginal: Mapping[str, Any],
    paired: Mapping[str, Any],
    aspect_summary: Mapping[str, Any],
    rarefaction: Mapping[str, Any],
    analysis_scope: Optional[Mapping[str, Any]] = None,
) -> str:
    scope = dict(analysis_scope or {})
    title = str(
        scope.get("report_title")
        or "Graph expansion vs Deep merged — full candidate pool"
    )
    unit_description = str(
        scope.get("unit_description")
        or "The unit is a deduplicated query-paper candidate, before rerank/Top-K."
    )
    interpretation_boundary = str(
        scope.get("interpretation_boundary")
        or (
            "These are descriptive, model-annotated full-pool results. They do not establish "
            "causal effects of an edge type and do not substitute for Top-K/Novel Recall "
            "analysis or human auditing of model-labeled novel direct candidates."
        )
    )
    lines = [
        f"# {title}",
        "",
        "The analysis universe contains exactly **Graph** and **Deep merged** candidates.",
        unit_description,
        "",
        "## Primary partitions",
        "",
        "| Group | Candidates | Direct | Direct task/method | Direct+Partial | Information-bearing | Mean relevance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("graph_only", "deep_merged_only", "graph_and_deep_merged"):
        row = group_rows[group]
        lines.append(
            f"| {group} | {row['candidate_count']:,} | {_pct(row['direct_rate'])} | "
            f"{_pct(row['direct_task_method_match_rate'])} | "
            f"{_pct(row['direct_or_partial_rate'])} | {_pct(row['information_bearing_rate'])} | "
            f"{row['mean_relevance_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Semantic complement profile (conditional on information-bearing)",
            "",
            "| Category | Measurement | Graph share | Deep share | Macro Graph−Deep (95% CI) | Graph pool yield | Deep pool yield |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in semantic_rows:
        macro_ci = row["macro_share_bootstrap_95_ci"]
        macro_text = (
            f"{_pct(row['macro_share_graph_minus_deep'])} "
            f"[{_pct(macro_ci[0])}, {_pct(macro_ci[1])}]"
        )
        lines.append(
            f"| {row['category']} | {row['measurement']} | {_pct(row['graph_only_rate'])} | "
            f"{_pct(row['deep_merged_only_rate'])} | {macro_text} | "
            f"{_pct(row['graph_only_pool_yield'])} | "
            f"{_pct(row['deep_merged_only_pool_yield'])} |"
        )
    lines.extend(
        [
            "",
            "`historical_predecessor_proxy`, `mechanism_or_theory_proxy`, and "
            "`application_domain_proxy` are proxies. In particular, the current labels cannot prove "
            "cross-domain transfer or distinguish implicit from explicit mechanisms.",
            "",
            "## Graph structure quality (graph-only, candidate-level deduplication)",
            "",
            "| Dimension | Bucket | Candidates | Direct+Partial | Information-bearing | Info lift | Macro info |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    preferred_structure = {
        ("edge_type", "reference_only"),
        ("edge_type", "citation_only"),
        ("edge_type", "citation_and_reference"),
        ("path_count", "1"),
        ("path_count", "2"),
        ("path_count", "3-4"),
        ("path_count", "5+"),
        ("source_seed_count", "1"),
        ("source_seed_count", "2"),
        ("source_seed_count", "3-4"),
        ("source_seed_count", "5+"),
    }
    for row in structure_rows:
        if (row["dimension"], row["bucket"]) not in preferred_structure:
            continue
        lines.append(
            f"| {row['dimension']} | {row['bucket']} | {row['candidate_count']:,} | "
            f"{_pct(row['direct_or_partial_rate'])} | {_pct(row['information_bearing_rate'])} | "
            f"{row['information_bearing_lift_vs_graph_only']:.2f}× | "
            f"{_pct(row['macro_information_bearing_rate'])} |"
        )

    lines.extend(
        [
            "",
            "Edge-type buckets are candidate-level unions across saved graph occurrences; "
            "`citation_and_reference` is multi-edge support, not necessarily one mixed path.",
            "",
            "## Query-aspect novelty added by graph-only",
            "",
            f"Deep merged covers **{aspect_summary['deep_merged_information_query_aspect_count']:,} / "
            f"{aspect_summary['rubric_query_aspect_count']:,}** query-local rubric aspects "
            f"({_pct(aspect_summary['deep_merged_micro_aspect_coverage_rate'])}).",
            f"Adding graph-only contributes **{aspect_summary['novel_graph_information_query_aspect_count']:,}** "
            f"previously uncovered query-aspects, raising coverage to "
            f"{_pct(aspect_summary['union_micro_aspect_coverage_rate'])} "
            f"(+{_pct(aspect_summary['micro_aspect_coverage_gain'])}).",
            f"Queries with at least one novel graph aspect: "
            f"**{aspect_summary['queries_with_novel_graph_aspects']:,} / {aspect_summary['query_count']:,}**.",
        ]
    )

    marginal_rows = {row["metric"]: row for row in marginal["metrics"]}
    lines.extend(
        [
            "",
            "## Marginal coverage from adding graph-only to Deep merged",
            "",
            f"Candidate-pool growth: **{_pct(marginal['candidate_pool_growth_rate'])}**.",
            "",
            "| Metric | Deep merged base | Graph-only added | Relative gain | Marginal yield |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in ("ground_truth", "direct", "direct_or_partial", "contextual", "information_bearing"):
        row = marginal_rows[metric]
        lines.append(
            f"| {metric} | {row['deep_merged_base_count']:,} | {row['graph_only_added_count']:,} | "
            f"{_pct(row['relative_gain_over_deep_merged'])} | "
            f"{_pct(row['marginal_yield_per_added_candidate'])} |"
        )

    lines.extend(
        [
            "",
            "## Query-paired macro comparison",
            "",
            "| Metric | Graph macro | Deep merged macro | Graph−Deep | 95% query-bootstrap CI |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired["metrics"]:
        ci = row["bootstrap_95_ci"]
        ci_text = "NA" if ci is None else f"[{_pct(ci[0])}, {_pct(ci[1])}]"
        lines.append(
            f"| {row['metric']} | {_pct(row['graph_macro_mean'])} | "
            f"{_pct(row['deep_merged_macro_mean'])} | {_pct(row['mean_difference'])} | {ci_text} |"
        )
    if rarefaction.get("enabled"):
        lines.extend(
            [
                "",
                "## Equal-budget sensitivity analysis",
                "",
                f"Within each of {rarefaction['paired_query_count']} paired queries, both exclusive pools "
                f"are repeatedly sampled to the same candidate count "
                f"({rarefaction['samples']} repetitions).",
                "",
                "| Metric | Graph-only | Deep-merged-only | Graph−Deep | Sampling interval |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in rarefaction["metrics"]:
            interval = row["sampling_95_interval"]
            if row["metric"] == "semantic_category_richness":
                graph_text = f"{row['graph_only_mean']:.3f}"
                deep_text = f"{row['deep_merged_only_mean']:.3f}"
                difference_text = f"{row['graph_minus_deep_mean']:.3f}"
                interval_text = f"[{interval[0]:.3f}, {interval[1]:.3f}]"
            else:
                graph_text = _pct(row["graph_only_mean"])
                deep_text = _pct(row["deep_merged_only_mean"])
                difference_text = _pct(row["graph_minus_deep_mean"])
                interval_text = f"[{_pct(interval[0])}, {_pct(interval[1])}]"
            lines.append(
                f"| {row['metric']} | {graph_text} | {deep_text} | "
                f"{difference_text} | {interval_text} |"
            )
        lines.extend(
            [
                "",
                "Fixed budgets below are random samples from the unranked pools, not reranked Top-K.",
                "",
                "| Candidate budget/query | Graph info | Deep info | Graph aspect coverage | Deep aspect coverage | Graph category richness | Deep category richness |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for point in rarefaction["budget_curve"]:
            metrics = {row["metric"]: row for row in point["metrics"]}
            information = metrics["information_bearing_rate"]
            aspects = metrics["aspect_coverage_rate"]
            richness = metrics["semantic_category_richness"]
            lines.append(
                f"| {point['target_candidate_budget_per_query']} | "
                f"{_pct(information['graph_only_mean'])} | "
                f"{_pct(information['deep_merged_only_mean'])} | "
                f"{_pct(aspects['graph_only_mean'])} | "
                f"{_pct(aspects['deep_merged_only_mean'])} | "
                f"{richness['graph_only_mean']:.3f} | "
                f"{richness['deep_merged_only_mean']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            interpretation_boundary,
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    work_dir: Path,
    *,
    output_dir: Optional[Path] = None,
    bootstrap_samples: int = 10000,
    rarefaction_samples: int = 200,
    analysis_scope: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    work_dir = work_dir.resolve()
    aggregate_summary_path = work_dir / "analysis" / "summary.json"
    if not aggregate_summary_path.exists():
        raise FileNotFoundError("Run strict aggregate before Deep merged primary analysis")
    aggregate_summary = json.loads(aggregate_summary_path.read_text(encoding="utf-8"))
    if not aggregate_summary.get("complete"):
        raise RuntimeError("Deep merged primary analysis requires a complete strict aggregation")
    output_dir = (output_dir or work_dir / "analysis" / "deep_merged_primary").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    query_rows = list(_iter_jsonl(work_dir / "manifest" / "queries.jsonl"))
    query_text = {row["query_id"]: row.get("query", "") for row in query_rows}
    query_aspects = _load_query_aspects(work_dir, query_rows)
    group_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    query_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    structure_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    structure_query_counts: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    observed_aspects: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    profiles: Dict[
        Tuple[str, str],
        List[Tuple[bool, bool, frozenset[str], frozenset[str]]],
    ] = defaultdict(list)
    candidate_path = work_dir / "manifest" / "candidates.jsonl"
    annotation_path = work_dir / "analysis" / "annotations.jsonl"

    candidate_iter = _iter_jsonl(candidate_path)
    annotation_iter = _iter_jsonl(annotation_path)
    for candidate, annotation_row in itertools.zip_longest(candidate_iter, annotation_iter):
        if candidate is None or annotation_row is None:
            raise ValueError("Candidate and annotation files have different lengths")
        if candidate["candidate_id"] != annotation_row["candidate_id"]:
            raise ValueError("Candidate and annotation order/IDs do not match")
        annotation = annotation_row.get("annotation")
        if annotation is None:
            raise RuntimeError(f"Missing annotation for {candidate['candidate_id']}")
        groups = _analysis_groups(candidate.get("sources") or [])
        if not groups:
            continue
        partition = primary_partition(candidate.get("sources") or [])
        for group in groups:
            _update_counter(group_counts[group], annotation, bool(candidate["is_ground_truth"]))
        if partition in {"graph_only", "deep_merged_only", "graph_and_deep_merged"}:
            _update_counter(
                query_counts[(partition, candidate["query_id"])],
                annotation,
                bool(candidate["is_ground_truth"]),
            )
        relevance_grade = _relevance_grade(annotation)
        if relevance_grade in INFORMATION_RELATIONSHIPS:
            matched_aspects = set(annotation.get("matched_aspect_ids") or [])
            if "deep_merged" in groups:
                observed_aspects[("deep_merged", candidate["query_id"])].update(
                    matched_aspects
                )
            if partition == "graph_only":
                observed_aspects[("graph_only", candidate["query_id"])].update(
                    matched_aspects
                )
        if partition in {"graph_only", "deep_merged_only"}:
            profiles[(partition, candidate["query_id"])].append(
                (
                    relevance_grade in INFORMATION_RELATIONSHIPS,
                    relevance_grade in DIRECT_OR_PARTIAL,
                    frozenset(annotation.get("matched_aspect_ids") or []),
                    frozenset(_semantic_categories(annotation)),
                )
            )
        if partition == "graph_only":
            graph_stats = (candidate.get("source_stats") or {}).get("graph") or {}
            dimensions = {
                "graph_role": _graph_role_bucket(graph_stats),
                "edge_type": _edge_bucket(graph_stats),
                "path_count": _bucket_at_least(
                    graph_stats.get("max_path_count"),
                    ((1, "1"), (2, "2"), (4, "3-4"), (float("inf"), "5+")),
                ),
                "source_seed_count": _bucket_at_least(
                    graph_stats.get("source_seed_count"),
                    ((1, "1"), (2, "2"), (4, "3-4"), (float("inf"), "5+")),
                ),
                "event_count": _bucket_at_least(
                    graph_stats.get("event_count"),
                    ((1, "1"), (2, "2"), (4, "3-4"), (float("inf"), "5+")),
                ),
            }
            for key in (("all", "all"), *dimensions.items()):
                _update_counter(
                    structure_counts[key], annotation, bool(candidate["is_ground_truth"])
                )
                _update_counter(
                    structure_query_counts[(key[0], key[1], candidate["query_id"])],
                    annotation,
                    bool(candidate["is_ground_truth"]),
                )

    required_groups = {
        "union",
        "graph",
        "deep_merged",
        "graph_only",
        "deep_merged_only",
        "graph_and_deep_merged",
    }
    missing_groups = required_groups - set(group_counts)
    if missing_groups:
        raise RuntimeError("Missing primary analysis groups: " + ", ".join(sorted(missing_groups)))

    group_rows = [_summary_row(group, group_counts[group]) for group in sorted(group_counts)]
    group_by_name = {row["group"]: row for row in group_rows}
    semantic_rows = _semantic_complement_rows(
        group_counts, query_counts, bootstrap_samples=bootstrap_samples
    )
    structure_rows = _structure_bucket_rows(structure_counts, structure_query_counts)
    marginal = _marginal_coverage(group_by_name)
    query_rows, paired_summary = _query_comparison(
        query_counts, query_text, bootstrap_samples=bootstrap_samples
    )
    aspect_rows, aspect_summary = _aspect_coverage(query_aspects, observed_aspects)
    rarefaction = _equal_budget_rarefaction(
        profiles, query_aspects, samples=rarefaction_samples
    )
    definitions = {
        "primary_semantic_retrieval": "deep_merged",
        "primary_partitions": {
            "graph_only": "graph and not deep_merged",
            "deep_merged_only": "deep_merged and not graph",
            "graph_and_deep_merged": "graph and deep_merged",
        },
        "unit": "deduplicated query-paper candidate",
        "candidate_scope": (
            analysis_scope.get("candidate_scope")
            if analysis_scope
            else "complete saved artifact union before rerank/Top-K"
        ),
        "information_bearing": sorted(INFORMATION_RELATIONSHIPS),
        "semantic_complement_definitions": list(SEMANTIC_COMPLEMENT_DEFINITIONS),
        "secondary_annotation_needed_for_strict_claims": {
            "implicit_mechanism": "Current mechanism_or_theory label does not encode implicitness.",
            "cross_domain_application": "Current application_domain label does not encode source/target domain transfer.",
            "historical_predecessor": "Current historical_context label does not prove lineage.",
        },
        "structure_statistics": "candidate-level union across graph occurrences; not occurrence-weighted",
        "aspect_novelty": (
            "Query-local rubric aspects matched by information-bearing candidates; "
            "baseline is inclusive Deep merged and addition is graph-only."
        ),
        "equal_budget_sensitivity": rarefaction.get("method"),
    }
    summary = {
        "complete": True,
        "primary_semantic_retrieval": "deep_merged",
        "group_summary": group_rows,
        "marginal_coverage": marginal,
        "paired_summary": paired_summary,
        "aspect_coverage_summary": aspect_summary,
        "equal_budget_rarefaction": rarefaction,
        "analysis_scope": dict(analysis_scope or {}),
        "output_dir": str(output_dir),
    }

    _write_jsonl(output_dir / "group_summary.jsonl", group_rows)
    _write_csv(output_dir / "group_summary.csv", group_rows)
    _write_jsonl(output_dir / "semantic_complement.jsonl", semantic_rows)
    _write_csv(output_dir / "semantic_complement.csv", semantic_rows)
    _write_jsonl(output_dir / "graph_structure_quality.jsonl", structure_rows)
    _write_csv(output_dir / "graph_structure_quality.csv", structure_rows)
    _write_jsonl(output_dir / "query_comparison.jsonl", query_rows)
    _write_csv(output_dir / "query_comparison.csv", query_rows)
    _write_jsonl(output_dir / "aspect_novelty.jsonl", aspect_rows)
    _write_csv(output_dir / "aspect_novelty.csv", aspect_rows)
    _atomic_write_json(output_dir / "marginal_coverage.json", marginal)
    _atomic_write_json(output_dir / "paired_summary.json", paired_summary)
    _atomic_write_json(output_dir / "aspect_coverage_summary.json", aspect_summary)
    _atomic_write_json(output_dir / "equal_budget_rarefaction.json", rarefaction)
    _atomic_write_json(output_dir / "metric_definitions.json", definitions)
    _atomic_write_json(output_dir / "summary.json", summary)
    report = _report_markdown(
        group_by_name,
        semantic_rows,
        structure_rows,
        marginal,
        paired_summary,
        aspect_summary,
        rarefaction,
        analysis_scope,
    )
    report_path = output_dir / "report.md"
    temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--rarefaction-samples", type=int, default=200)
    parser.add_argument(
        "--analysis-scope-json",
        default=None,
        help="Optional JSON file overriding report title/scope wording.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    analysis_scope = (
        json.loads(Path(args.analysis_scope_json).read_text(encoding="utf-8"))
        if args.analysis_scope_json
        else None
    )
    summary = analyze(
        Path(args.work_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        bootstrap_samples=args.bootstrap_samples,
        rarefaction_samples=args.rarefaction_samples,
        analysis_scope=analysis_scope,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
