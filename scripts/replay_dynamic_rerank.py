#!/usr/bin/env python3
"""Replay query-conditioned reranking on saved OnePass candidate pools.

This script never retrieves papers, expands the graph, changes date cutoffs, or
calls Selector.  It reuses each event's materialized features and original
``selector_top_k``, then compares the query-level union of reranked slices with
the legacy static formula under the existing OnePass candidate-F1 definition.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from paper_type import (  # noqa: E402
    DEFAULT_EXCLUDE_HARD_FILTER_MIN_CONFIDENCE,
    DEFAULT_REQUIRE_HARD_FILTER_MIN_CONFIDENCE,
    evaluate_paper_type_rules,
    load_paper_type_cache,
    normalize_paper_id,
)
from runtime_env import load_env_file  # noqa: E402
from rerank_skill import (  # noqa: E402
    CATALOG_VERSION,
    DEFAULT_MAX_NEGATIVE_MASS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_NEGATIVE_WEIGHT,
    DEFAULT_SEMANTIC_MIN_MASS,
    PROMPT_VERSION,
    CompiledPolicy,
    RerankPolicy,
    RerankSkill,
)


IMPLEMENTATION_VERSION = "dynamic_rerank_replay_v2"
METHODS = ("legacy_static", "dynamic_policy")


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(recall: float, precision: float) -> float:
    return 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
class BenchmarkQuery:
    benchmark_idx: int
    query_id: str
    query: str
    gt_ids: frozenset[str]


@dataclass
class MethodAccumulator:
    selected_ids: Set[str] = field(default_factory=set)
    selected_occurrence_count: int = 0
    best_rows: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hard_filtered_occurrence_count: int = 0

    def add_event(
        self,
        scored_rows: Sequence[Mapping[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        eligible = [row for row in scored_rows if not row.get("hard_filtered")]
        selected = [dict(row) for row in eligible[: max(0, int(top_k))]]
        self.selected_occurrence_count += len(selected)
        self.selected_ids.update(str(row["paper_arxiv_id"]) for row in selected)
        self.hard_filtered_occurrence_count += sum(
            int(bool(row.get("hard_filtered"))) for row in scored_rows
        )
        for row in eligible:
            paper_id = str(row["paper_arxiv_id"])
            current = self.best_rows.get(paper_id)
            if current is None or _best_occurrence_key(row) < _best_occurrence_key(current):
                self.best_rows[paper_id] = dict(row)
        return selected


@dataclass
class QueryAccumulator:
    benchmark: BenchmarkQuery
    event_count: int = 0
    methods: Dict[str, MethodAccumulator] = field(
        default_factory=lambda: {name: MethodAccumulator() for name in METHODS}
    )


def _best_occurrence_key(row: Mapping[str, Any]) -> Tuple[float, int, int, str]:
    observed_rank = row.get("observed_retrieval_rank")
    try:
        rank = int(observed_rank) if observed_rank is not None else 10**12
    except (TypeError, ValueError):
        rank = 10**12
    return (
        -float(row.get("rerank_score") or 0.0),
        -int(bool(row.get("is_seed"))),
        rank,
        str(row.get("paper_arxiv_id") or ""),
    )


def _extract_gt_ids(record: Mapping[str, Any]) -> Set[str]:
    papers = record.get("cited_paper") or record.get("ground_truth_papers") or []
    labels = record.get("gt_label") or record.get("gt_labels")
    if not isinstance(labels, list):
        labels = [1] * len(papers) if isinstance(papers, list) else []
    output: Set[str] = set()
    for paper, label in zip(papers if isinstance(papers, list) else [], labels):
        if label != 1:
            continue
        value = paper.get("arxiv_id") if isinstance(paper, Mapping) else paper
        paper_id = normalize_paper_id(value)
        if paper_id:
            output.add(paper_id)
    return output


def load_benchmark(path: Path) -> Tuple[List[BenchmarkQuery], Dict[str, BenchmarkQuery]]:
    ordered: List[BenchmarkQuery] = []
    by_id: Dict[str, BenchmarkQuery] = {}
    with path.open("r", encoding="utf-8") as handle:
        for benchmark_idx, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid benchmark JSONL at logical row {benchmark_idx}: {exc}"
                ) from exc
            query_id = str(value.get("qid") or value.get("query_id") or f"idx-{benchmark_idx}")
            if query_id in by_id:
                raise ValueError(f"duplicate benchmark query_id={query_id}")
            item = BenchmarkQuery(
                benchmark_idx=benchmark_idx,
                query_id=query_id,
                query=str(value.get("query") or ""),
                gt_ids=frozenset(_extract_gt_ids(value)),
            )
            ordered.append(item)
            by_id[query_id] = item
    if not ordered:
        raise ValueError(f"no benchmark rows found in {path}")
    return ordered, by_id


def iter_pool_records(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid pool JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value.get("local_pool_rows"), list):
                raise ValueError(f"pool line {line_number} has no local_pool_rows array")
            yield line_number, value


def _pool_materializes_intent_labels(path: Path) -> bool:
    for _, record in iter_pool_records(path):
        rows = record.get("local_pool_rows") or []
        return not rows or "intent_labels" in rows[0]
    return False


def load_intent_label_lookup(
    path: Path,
    allowed_queries: Set[str],
) -> Dict[Tuple[str, str, str], List[str]]:
    lookup: Dict[Tuple[str, str, str], List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid paper_rows JSONL at line {line_number}: {exc}"
                ) from exc
            query_id = str(row.get("query_id") or "")
            if query_id not in allowed_queries:
                continue
            event_id = str(row.get("retrieval_event_id") or "")
            paper_id = normalize_paper_id(row.get("paper_arxiv_id"))
            if not event_id or not paper_id:
                continue
            lookup[(query_id, event_id, paper_id)] = [
                str(label).strip().lower()
                for label in (row.get("intent_labels") or [])
                if str(label).strip()
            ]
    return lookup


def _hydrate_candidate(
    candidate: Dict[str, Any],
    *,
    query_id: str,
    event_id: str,
    intent_label_lookup: Mapping[Tuple[str, str, str], Sequence[str]],
) -> None:
    paper_id = normalize_paper_id(candidate.get("paper_arxiv_id"))
    candidate["paper_arxiv_id"] = paper_id
    if "intent_labels" not in candidate:
        candidate["intent_labels"] = list(
            intent_label_lookup.get((query_id, event_id, paper_id), [])
        )
    candidate_type = str(candidate.get("candidate_type") or "")
    if "is_seed" not in candidate:
        candidate["is_seed"] = candidate_type in {"seed", "seed_and_expanded"}
    if "is_expanded" not in candidate:
        candidate["is_expanded"] = candidate_type in {
            "expanded",
            "seed_and_expanded",
        }


def _validate_candidate_features(row: Mapping[str, Any], line_number: int) -> None:
    required = {
        "paper_arxiv_id",
        "query_score_normalized",
        "subquery_score_normalized",
        "intent_score",
        "path_count_normalized",
    }
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(
            f"pool line {line_number} candidate is missing materialized features: {missing}"
        )


def _rank_metrics(ordered_ids: Sequence[str], gt_ids: Set[str]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for cutoff in (10, 20, 50):
        hits = len(set(ordered_ids[:cutoff]) & gt_ids)
        output[f"recall@{cutoff}"] = _safe_div(hits, len(gt_ids))
    dcg = 0.0
    for index, paper_id in enumerate(ordered_ids[:20], start=1):
        if paper_id in gt_ids:
            dcg += 1.0 / math.log2(index + 1)
    ideal_count = min(20, len(gt_ids))
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_count + 1))
    output["ndcg@20"] = dcg / idcg if idcg else 0.0
    reciprocal_rank = 0.0
    for index, paper_id in enumerate(ordered_ids, start=1):
        if paper_id in gt_ids:
            reciprocal_rank = 1.0 / index
            break
    output["mrr"] = reciprocal_rank
    return output


def _query_method_metrics(
    accumulator: MethodAccumulator,
    gt_ids: Set[str],
    *,
    skill: RerankSkill,
    dynamic_policy: CompiledPolicy,
) -> Dict[str, Any]:
    selected_ids = set(accumulator.selected_ids)
    hits = selected_ids & gt_ids
    recall = _safe_div(len(hits), len(gt_ids))
    precision = _safe_div(len(hits), len(selected_ids))
    ranked_rows = sorted(accumulator.best_rows.values(), key=_best_occurrence_key)
    ranked_ids = [str(row["paper_arxiv_id"]) for row in ranked_rows]

    exclusion_violations: List[str] = []
    rules = [rule.to_dict() for rule in dynamic_policy.paper_type_rules]
    if rules:
        for paper_id in sorted(selected_ids):
            row = accumulator.best_rows.get(paper_id) or {"paper_arxiv_id": paper_id}
            type_result = evaluate_paper_type_rules(
                skill._type_record_for_candidate(row),
                rules,
                exclude_hard_filter_min_confidence=(
                    skill.exclude_hard_filter_min_confidence
                ),
                require_hard_filter_min_confidence=(
                    skill.require_hard_filter_min_confidence
                ),
                exclude_threshold=skill.exclude_threshold,
                require_threshold=skill.require_threshold,
            )
            if type_result.get("paper_type_filter_action") == "exclude":
                exclusion_violations.append(paper_id)

    return {
        "gt_count": len(gt_ids),
        "candidate_count": len(selected_ids),
        "candidate_arxiv_ids": sorted(selected_ids),
        "candidate_gt_count": len(hits),
        "candidate_gt_ids": sorted(hits),
        "candidate_recall": recall,
        "candidate_precision": precision,
        "candidate_f1": _f1(recall, precision),
        "candidate_occurrence_count": accumulator.selected_occurrence_count,
        "query_ranked_candidate_count": len(ranked_ids),
        "query_ranked_arxiv_ids": ranked_ids,
        "hard_filtered_occurrence_count": accumulator.hard_filtered_occurrence_count,
        "paper_type_exclusion_violation_count": len(exclusion_violations),
        "paper_type_exclusion_violation_ids": exclusion_violations,
        "paper_type_exclusion_violation_rate": _safe_div(
            len(exclusion_violations), len(selected_ids)
        ),
        **_rank_metrics(ranked_ids, gt_ids),
    }


def _aggregate(method: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metrics = (
        "candidate_count",
        "candidate_recall",
        "candidate_precision",
        "candidate_f1",
        "recall@10",
        "recall@20",
        "recall@50",
        "ndcg@20",
        "mrr",
        "paper_type_exclusion_violation_rate",
    )
    total_gt = sum(int(row.get("gt_count") or 0) for row in rows)
    total_candidates = sum(int(row.get("candidate_count") or 0) for row in rows)
    total_hits = sum(int(row.get("candidate_gt_count") or 0) for row in rows)
    micro_recall = _safe_div(total_hits, total_gt)
    micro_precision = _safe_div(total_hits, total_candidates)
    output: Dict[str, Any] = {
        "method": method,
        "evaluated_query_count": len(rows),
        "total_gt_count": total_gt,
        "total_candidate_count": total_candidates,
        "total_candidate_gt_count": total_hits,
        "total_candidate_occurrence_count": sum(
            int(row.get("candidate_occurrence_count") or 0) for row in rows
        ),
        "total_hard_filtered_occurrence_count": sum(
            int(row.get("hard_filtered_occurrence_count") or 0) for row in rows
        ),
        "total_paper_type_exclusion_violation_count": sum(
            int(row.get("paper_type_exclusion_violation_count") or 0) for row in rows
        ),
        "micro_candidate_recall": micro_recall,
        "micro_candidate_precision": micro_precision,
        "micro_candidate_f1": _f1(micro_recall, micro_precision),
    }
    for metric in metrics:
        output[f"avg_{metric}"] = _mean(
            [float(row.get(metric) or 0.0) for row in rows]
        )
    output["mean_query_candidate_f1"] = output["avg_candidate_f1"]
    output["macro_candidate_f1_from_avg_recall_precision"] = _f1(
        output["avg_candidate_recall"], output["avg_candidate_precision"]
    )
    output["main_table_candidate_f1"] = output[
        "macro_candidate_f1_from_avg_recall_precision"
    ]
    return output


def _numeric_delta(dynamic: Mapping[str, Any], legacy: Mapping[str, Any]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key in sorted(set(dynamic) & set(legacy)):
        left, right = dynamic[key], legacy[key]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            output[key] = float(left) - float(right)
    return output


def _compact_candidate(row: Mapping[str, Any], selected: bool) -> Dict[str, Any]:
    keys = (
        "paper_arxiv_id",
        "is_seed",
        "is_expanded",
        "observed_retrieval_rank",
        "query_score_normalized",
        "subquery_score_normalized",
        "intent_labels",
        "intent_score",
        "intent_background",
        "intent_method",
        "intent_result",
        "path_count",
        "path_count_normalized",
        "paper_type_probs",
        "paper_type_classifier_confidence",
        "paper_type_evidence_source",
        "paper_type_publication_types",
        "paper_type_supported_types",
        "paper_type_negative_evidence_types",
        "s2_publication_types",
        "paper_type_alignment",
        "paper_type_soft_penalty",
        "paper_type_filter_action",
        "paper_type_filter_reason",
        "hard_filtered",
        "component_contributions",
        "compiled_feature_weights",
        "rerank_score",
        "rerank_rank",
        "artifact_rank",
        "rerank_policy_id",
        "rerank_used_fallback",
    )
    return {**{key: row.get(key) for key in keys}, "selected_at_event_top_k": selected}


def replay(
    pool_records_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    *,
    model: str = "qwen3-30b-a3b-instruct-2507",
    policy_cache_path: Optional[Path] = None,
    paper_type_cache_path: Optional[Path] = None,
    paper_rows_path: Optional[Path] = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    semantic_min_mass: float = DEFAULT_SEMANTIC_MIN_MASS,
    negative_weight: float = DEFAULT_NEGATIVE_WEIGHT,
    max_negative_mass: float = DEFAULT_MAX_NEGATIVE_MASS,
    exclude_hard_filter_min_confidence: float = DEFAULT_EXCLUDE_HARD_FILTER_MIN_CONFIDENCE,
    require_hard_filter_min_confidence: float = DEFAULT_REQUIRE_HARD_FILTER_MIN_CONFIDENCE,
    catalog_version: str = CATALOG_VERSION,
    prompt_version: str = PROMPT_VERSION,
    is_local: bool = False,
    llm_call: Any = None,
    limit: Optional[int] = None,
    artifact_level: str = "full",
    generate_policies: bool = True,
    retry_cached_fallbacks: bool = False,
) -> Dict[str, Any]:
    pool_records_path = pool_records_path.resolve()
    benchmark_path = benchmark_path.resolve()
    output_dir = output_dir.resolve()
    if artifact_level not in {"full", "selected"}:
        raise ValueError("artifact_level must be full or selected")
    if not pool_records_path.exists():
        raise FileNotFoundError(pool_records_path)
    if not benchmark_path.exists():
        raise FileNotFoundError(benchmark_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_benchmark, benchmark_by_id = load_benchmark(benchmark_path)
    allowed_ids = {
        item.query_id for item in (ordered_benchmark[:limit] if limit else ordered_benchmark)
    }
    labels_materialized_in_pool = _pool_materializes_intent_labels(pool_records_path)
    if not labels_materialized_in_pool and paper_rows_path is None:
        inferred_paper_rows = pool_records_path.parent / "paper_rows.jsonl"
        if inferred_paper_rows.exists():
            paper_rows_path = inferred_paper_rows
    intent_label_lookup: Dict[Tuple[str, str, str], List[str]] = {}
    if not labels_materialized_in_pool and paper_rows_path is not None:
        paper_rows_path = paper_rows_path.resolve()
        if not paper_rows_path.exists():
            raise FileNotFoundError(paper_rows_path)
        intent_label_lookup = load_intent_label_lookup(paper_rows_path, allowed_ids)
    paper_type_cache = load_paper_type_cache(paper_type_cache_path)
    policy_cache_path = policy_cache_path or output_dir / "query_rerank_policy_cache.jsonl"
    skill = RerankSkill(
        model,
        is_local=is_local,
        llm_call=llm_call,
        policy_cache_path=policy_cache_path,
        retry_cached_fallbacks=retry_cached_fallbacks,
        paper_type_cache=paper_type_cache,
        min_confidence=min_confidence,
        catalog_version=catalog_version,
        prompt_version=prompt_version,
        semantic_min_mass=semantic_min_mass,
        negative_weight=negative_weight,
        max_negative_mass=max_negative_mass,
        exclude_hard_filter_min_confidence=exclude_hard_filter_min_confidence,
        require_hard_filter_min_confidence=require_hard_filter_min_confidence,
    )
    static_policy = skill.legacy_compiled_policy()
    policies: Dict[str, Tuple[RerankPolicy, CompiledPolicy]] = {}
    accumulators: Dict[str, QueryAccumulator] = {}
    policy_artifacts: List[Dict[str, Any]] = []
    event_count = 0
    candidate_occurrence_count = 0
    stored_topk_event_count = 0
    legacy_stored_topk_mismatch_count = 0
    legacy_stored_topk_mismatch_examples: List[Dict[str, Any]] = []

    ranked_tmp = output_dir / "ranked_candidates.jsonl.tmp"
    event_tmp = output_dir / "event_results.jsonl.tmp"
    with ranked_tmp.open("w", encoding="utf-8") as ranked_handle, event_tmp.open(
        "w", encoding="utf-8"
    ) as event_handle:
        for line_number, record in iter_pool_records(pool_records_path):
            query_id = str(record.get("query_id") or "")
            if query_id not in allowed_ids:
                continue
            benchmark = benchmark_by_id[query_id]
            original_query = str(record.get("query") or benchmark.query)
            if query_id not in policies:
                policy = (
                    skill.build_policy(original_query)
                    if generate_policies
                    else skill.legacy_policy_for_query(original_query)
                )
                compiled = skill.compile_weights(
                    policy, paper_type_available=bool(paper_type_cache)
                )
                policies[query_id] = (policy, compiled)
                policy_artifacts.append(
                    skill.artifact_record(
                        query_id=query_id,
                        original_query=original_query,
                        policy=policy,
                        compiled=compiled,
                    )
                )
                accumulators[query_id] = QueryAccumulator(benchmark=benchmark)
            policy, dynamic_policy = policies[query_id]
            accumulator = accumulators[query_id]
            accumulator.event_count += 1
            event_count += 1

            candidates = [dict(row) for row in record["local_pool_rows"]]
            event_id = str(record.get("retrieval_event_id") or "")
            for candidate in candidates:
                _hydrate_candidate(
                    candidate,
                    query_id=query_id,
                    event_id=event_id,
                    intent_label_lookup=intent_label_lookup,
                )
                _validate_candidate_features(candidate, line_number)
            candidate_occurrence_count += len(candidates)
            top_k = int(record.get("selector_top_k") or 0)
            event_methods: Dict[str, Any] = {}
            for method, compiled_policy in (
                ("legacy_static", static_policy),
                ("dynamic_policy", dynamic_policy),
            ):
                scored = skill.score_candidates(candidates, compiled_policy)
                selected = accumulator.methods[method].add_event(scored, top_k)
                selected_ids = {str(row["paper_arxiv_id"]) for row in selected}
                selected_gt_ids = sorted(selected_ids & set(benchmark.gt_ids))
                compact_rows = [
                    _compact_candidate(row, str(row["paper_arxiv_id"]) in selected_ids)
                    for row in scored
                    if (
                        artifact_level == "full"
                        or str(row["paper_arxiv_id"]) in selected_ids
                        or bool(row.get("hard_filtered"))
                    )
                ]
                ranked_handle.write(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "benchmark_idx": benchmark.benchmark_idx,
                            "retrieval_event_id": record.get("retrieval_event_id"),
                            "iteration_idx": record.get("iteration_idx"),
                            "subquery_id": record.get("subquery_id"),
                            "subquery": record.get("subquery"),
                            "method": method,
                            "selector_top_k": top_k,
                            "rerank_policy_id": compiled_policy.policy_id,
                            "ranked_candidates": compact_rows,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                event_methods[method] = {
                    "rerank_policy_id": compiled_policy.policy_id,
                    "input_candidate_count": len(candidates),
                    "eligible_candidate_count": sum(
                        1 for row in scored if not row.get("hard_filtered")
                    ),
                    "hard_filtered_count": sum(
                        1 for row in scored if row.get("hard_filtered")
                    ),
                    "selected_arxiv_ids": [str(row["paper_arxiv_id"]) for row in selected],
                    "selected_gt_arxiv_ids": selected_gt_ids,
                }
            event_handle.write(
                json.dumps(
                    {
                        "query_id": query_id,
                        "benchmark_idx": benchmark.benchmark_idx,
                        "retrieval_event_id": record.get("retrieval_event_id"),
                        "iteration_idx": record.get("iteration_idx"),
                        "subquery_id": record.get("subquery_id"),
                        "subquery": record.get("subquery"),
                        "selector_top_k": top_k,
                        "methods": event_methods,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            if candidates and all("in_selector_topk" in row for row in candidates):
                stored_topk_event_count += 1
                stored_ids = {
                    str(row["paper_arxiv_id"])
                    for row in candidates
                    if row.get("in_selector_topk")
                }
                replayed_ids = set(
                    event_methods["legacy_static"]["selected_arxiv_ids"]
                )
                if stored_ids != replayed_ids:
                    legacy_stored_topk_mismatch_count += 1
                    if len(legacy_stored_topk_mismatch_examples) < 10:
                        legacy_stored_topk_mismatch_examples.append(
                            {
                                "query_id": query_id,
                                "retrieval_event_id": record.get("retrieval_event_id"),
                                "missing_from_replay": sorted(stored_ids - replayed_ids),
                                "extra_in_replay": sorted(replayed_ids - stored_ids),
                            }
                        )
    os.replace(ranked_tmp, output_dir / "ranked_candidates.jsonl")
    os.replace(event_tmp, output_dir / "event_results.jsonl")

    per_query_results: List[Dict[str, Any]] = []
    method_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in METHODS}
    for benchmark in ordered_benchmark:
        if benchmark.query_id not in allowed_ids or benchmark.query_id not in accumulators:
            continue
        accumulator = accumulators[benchmark.query_id]
        policy, dynamic_policy = policies[benchmark.query_id]
        methods: Dict[str, Any] = {}
        for method in METHODS:
            metrics = _query_method_metrics(
                accumulator.methods[method],
                set(benchmark.gt_ids),
                skill=skill,
                dynamic_policy=dynamic_policy,
            )
            methods[method] = metrics
            method_rows[method].append(metrics)
        per_query_results.append(
            {
                "benchmark_idx": benchmark.benchmark_idx,
                "query_id": benchmark.query_id,
                "query": benchmark.query,
                "gt_count": len(benchmark.gt_ids),
                "event_count": accumulator.event_count,
                "rerank_policy_id": policy.policy_id,
                "policy_used_fallback": policy.used_fallback,
                "methods": methods,
                "dynamic_minus_legacy": _numeric_delta(
                    methods["dynamic_policy"], methods["legacy_static"]
                ),
            }
        )

    legacy_summary = _aggregate("legacy_static", method_rows["legacy_static"])
    dynamic_summary = _aggregate("dynamic_policy", method_rows["dynamic_policy"])
    fallback_count = sum(
        int(policy.used_fallback) for policy, _ in policies.values()
    )
    paper_type_evidence_source_counts: Dict[str, int] = {}
    for record in paper_type_cache.values():
        source = str(record.get("evidence_source") or "unknown")
        paper_type_evidence_source_counts[source] = (
            paper_type_evidence_source_counts.get(source, 0) + 1
        )
    summary = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "pool_records": str(pool_records_path),
        "benchmark": str(benchmark_path),
        "paper_type_cache": str(paper_type_cache_path) if paper_type_cache_path else None,
        "paper_rows": str(paper_rows_path) if paper_rows_path else None,
        "intent_labels_source": (
            "pool_records"
            if labels_materialized_in_pool
            else ("paper_rows" if paper_rows_path else "missing_treated_as_empty")
        ),
        "paper_type_cache_record_count": len(paper_type_cache),
        "paper_type_evidence_source_counts": paper_type_evidence_source_counts,
        "paper_type_supported_types": sorted(skill.paper_type_supported_types),
        "policy_cache": str(policy_cache_path),
        "model": model,
        "catalog_version": catalog_version,
        "prompt_version": prompt_version,
        "compiler_config": {
            "min_confidence": min_confidence,
            "semantic_min_mass": semantic_min_mass,
            "negative_weight": negative_weight,
            "max_negative_mass": max_negative_mass,
            "exclude_hard_filter_min_confidence": (
                exclude_hard_filter_min_confidence
            ),
            "require_hard_filter_min_confidence": (
                require_hard_filter_min_confidence
            ),
        },
        "policy_generation_enabled": generate_policies,
        "retry_cached_fallbacks": retry_cached_fallbacks,
        "evaluated_query_count": len(per_query_results),
        "event_count": event_count,
        "candidate_occurrence_count": candidate_occurrence_count,
        "stored_topk_event_count": stored_topk_event_count,
        "legacy_stored_topk_mismatch_count": legacy_stored_topk_mismatch_count,
        "legacy_stored_topk_mismatch_examples": legacy_stored_topk_mismatch_examples,
        "candidate_scope": (
            "per-query deduplicated union of each OnePass event's reranked "
            "original selector_top_k"
        ),
        "query_ranking_scope": (
            "all eligible pool candidates collapsed by best event score per query"
        ),
        "policy_fallback_count": fallback_count,
        "policy_fallback_rate": _safe_div(fallback_count, len(policies)),
        "legacy_static": legacy_summary,
        "dynamic_policy": dynamic_summary,
        "dynamic_minus_legacy": _numeric_delta(dynamic_summary, legacy_summary),
    }
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_jsonl(output_dir / "per_query_results.jsonl", per_query_results)
    atomic_write_jsonl(output_dir / "query_rerank_policies.jsonl", policy_artifacts)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_records", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--paper_type_cache", type=Path, default=None)
    parser.add_argument("--paper_rows", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument(
        "--env_file",
        type=Path,
        default=None,
        help="Optional local KEY=value file loaded without logging secret values.",
    )
    parser.add_argument("--policy_cache", type=Path, default=None)
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--semantic_min_mass", type=float, default=DEFAULT_SEMANTIC_MIN_MASS)
    parser.add_argument("--negative_weight", type=float, default=DEFAULT_NEGATIVE_WEIGHT)
    parser.add_argument("--max_negative_mass", type=float, default=DEFAULT_MAX_NEGATIVE_MASS)
    parser.add_argument(
        "--exclude_hard_filter_min_confidence",
        type=float,
        default=DEFAULT_EXCLUDE_HARD_FILTER_MIN_CONFIDENCE,
    )
    parser.add_argument(
        "--require_hard_filter_min_confidence",
        type=float,
        default=DEFAULT_REQUIRE_HARD_FILTER_MIN_CONFIDENCE,
    )
    parser.add_argument("--catalog_version", default=CATALOG_VERSION)
    parser.add_argument("--prompt_version", default=PROMPT_VERSION)
    parser.add_argument("--is_local", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--artifact_level", choices=("full", "selected"), default="full")
    parser.add_argument(
        "--generate_policies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable for a no-API legacy smoke/baseline run.",
    )
    parser.add_argument(
        "--retry_cached_fallbacks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retry only cached fallback policies; successful policies remain cached.",
    )
    parser.add_argument(
        "--compare_legacy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Accepted for task-command compatibility; legacy comparison is always emitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    summary = replay(
        args.pool_records,
        args.benchmark,
        args.output,
        model=args.model,
        policy_cache_path=args.policy_cache,
        paper_type_cache_path=args.paper_type_cache,
        paper_rows_path=args.paper_rows,
        min_confidence=args.min_confidence,
        semantic_min_mass=args.semantic_min_mass,
        negative_weight=args.negative_weight,
        max_negative_mass=args.max_negative_mass,
        exclude_hard_filter_min_confidence=(
            args.exclude_hard_filter_min_confidence
        ),
        require_hard_filter_min_confidence=(
            args.require_hard_filter_min_confidence
        ),
        catalog_version=args.catalog_version,
        prompt_version=args.prompt_version,
        is_local=args.is_local,
        limit=args.limit,
        artifact_level=args.artifact_level,
        generate_policies=args.generate_policies,
        retry_cached_fallbacks=args.retry_cached_fallbacks,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
