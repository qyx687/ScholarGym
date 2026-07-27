"""Pure SemRank-QSQ scoring and online candidate-row adapter."""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .encoder import ConceptEncoder
from .models import (
    QueryConceptProfile,
    SEMRANK_FORMULA_ID,
    SEMRANK_METHOD,
    SemRankConfig,
    SemRankEventProfile,
    stable_hash,
)
from .paper_concepts import PaperConceptProfile, PaperConceptService
from .query_profile import DateValidInitialRetriever, QueryProfileBuilder


def population_zscore(
    values: Sequence[float],
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return np.zeros(0, dtype=np.float64), 0.0, 0.0
    finite = np.isfinite(array)
    if not bool(finite.all()):
        replacement = (
            float(np.min(array[finite])) if bool(finite.any()) else 0.0
        )
        array = np.where(finite, array, replacement)
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    if not math.isfinite(std) or std <= float(eps):
        return np.zeros_like(array), mean, std if math.isfinite(std) else 0.0
    return (array - mean) / std, mean, std


def base_semantic_scores(
    query_normalized: Sequence[float],
    subquery_normalized: Sequence[float],
    *,
    query_weight: float = 0.4,
    subquery_weight: float = 0.6,
) -> np.ndarray:
    query = np.asarray(list(query_normalized), dtype=np.float64)
    subquery = np.asarray(list(subquery_normalized), dtype=np.float64)
    if query.shape != subquery.shape:
        raise ValueError("SemRank query/subquery channels must have equal shape")
    return float(query_weight) * query + float(subquery_weight) * subquery


def mean_max_concept_cosine(
    query_embeddings: np.ndarray,
    paper_embeddings: np.ndarray,
) -> float:
    query = np.asarray(query_embeddings, dtype=np.float32)
    paper = np.asarray(paper_embeddings, dtype=np.float32)
    if query.size == 0 or paper.size == 0:
        return 0.0
    if query.ndim != 2 or paper.ndim != 2:
        raise ValueError("SemRank concept embeddings must be matrices")
    if query.shape[1] != paper.shape[1]:
        raise ValueError("SemRank concept embedding dimensions do not match")
    similarities = query @ paper.T
    return float(np.max(similarities, axis=1).mean())


def semrank_scores(
    query_normalized: Sequence[float],
    subquery_normalized: Sequence[float],
    concept_scores: Sequence[float],
    *,
    query_weight: float = 0.4,
    subquery_weight: float = 0.6,
    eps: float = 1e-12,
    base_only: bool = False,
) -> Dict[str, Any]:
    base = base_semantic_scores(
        query_normalized,
        subquery_normalized,
        query_weight=query_weight,
        subquery_weight=subquery_weight,
    )
    concepts = np.asarray(list(concept_scores), dtype=np.float64)
    if concepts.shape != base.shape:
        raise ValueError("SemRank base/concept channels must have equal shape")
    base_z, base_mean, base_std = population_zscore(base, eps=eps)
    concept_z, concept_mean, concept_std = population_zscore(
        concepts, eps=eps
    )
    final = base_z if base_only else base_z + concept_z
    return {
        "base": base,
        "base_z": base_z,
        "base_mean": base_mean,
        "base_std": base_std,
        "concept": concepts,
        "concept_z": concept_z,
        "concept_mean": concept_mean,
        "concept_std": concept_std,
        "final": final,
    }


class SemRankQSQReranker:
    def __init__(
        self,
        config: SemRankConfig,
        query_profiles: QueryProfileBuilder,
        paper_concepts: PaperConceptService,
        encoder: ConceptEncoder,
    ) -> None:
        self.config = config
        self.query_profiles = query_profiles
        self.paper_concepts = paper_concepts
        self.encoder = encoder
        self.active_query_profile: Optional[QueryConceptProfile] = None
        self._stats: Counter[str] = Counter()
        self._wall_seconds = 0.0

    def start_query(
        self,
        *,
        query_id: str,
        query_text: str,
        before_date: str,
        retriever: DateValidInitialRetriever,
    ) -> QueryConceptProfile:
        profile = self.query_profiles.start_query(
            query_id=query_id,
            query=query_text,
            date_cutoff=before_date,
            retriever=retriever,
        )
        self.active_query_profile = profile
        return profile

    def rerank(
        self,
        rows: Sequence[Mapping[str, Any]],
        paper_metadata: Mapping[str, Mapping[str, Any]],
        *,
        retrieval_event_id: str,
    ) -> Tuple[list[Dict[str, Any]], SemRankEventProfile]:
        started = time.perf_counter()
        if self.active_query_profile is None:
            raise RuntimeError(
                "SemRank-QSQ rerank called before start_query constructed C(q)"
            )
        original_ids = [str(row["paper_arxiv_id"]) for row in rows]
        if len(original_ids) != len(set(original_ids)):
            raise ValueError("SemRank candidate rows contain duplicate IDs")
        profiles = self.paper_concepts.get_or_build(
            {
                paper_id: paper_metadata.get(paper_id) or {}
                for paper_id in original_ids
            }
        )
        query_concepts = list(
            self.active_query_profile.selected_concepts
        )
        paper_concept_lists = [
            list(profiles[paper_id].concepts) for paper_id in original_ids
        ]
        paper_counts = [len(values) for values in paper_concept_lists]
        concept_values = [0.0] * len(original_ids)
        if query_concepts:
            flattened = list(
                dict.fromkeys(
                    query_concepts
                    + [
                        concept
                        for concepts in paper_concept_lists
                        for concept in concepts
                    ]
                )
            )
            all_embeddings = self.encoder.encode(flattened)
            vectors = {
                concept: all_embeddings[index]
                for index, concept in enumerate(flattened)
            }
            query_embeddings = np.stack(
                [vectors[concept] for concept in query_concepts]
            ).astype(np.float32)
            for index, concepts in enumerate(paper_concept_lists):
                if not concepts:
                    continue
                paper_embeddings = np.stack(
                    [vectors[concept] for concept in concepts]
                ).astype(np.float32)
                concept_values[index] = mean_max_concept_cosine(
                    query_embeddings,
                    paper_embeddings,
                )

        query_values = [
            float(row.get("query_score_normalized") or 0.0) for row in rows
        ]
        subquery_values = [
            float(row.get("subquery_score_normalized") or 0.0) for row in rows
        ]
        fallback_used = not bool(query_concepts)
        fallback_reason = (
            self.active_query_profile.fallback_reason
            or "query_concepts_empty"
            if fallback_used
            else None
        )
        scores = semrank_scores(
            query_values,
            subquery_values,
            concept_values,
            query_weight=self.config.base_query_weight,
            subquery_weight=self.config.base_subquery_weight,
            eps=self.config.zscore_eps,
            base_only=fallback_used,
        )
        pool_signature = stable_hash(sorted(original_ids))
        updated: list[Dict[str, Any]] = []
        for index, row_value in enumerate(rows):
            row = dict(row_value)
            row.update(
                {
                    "legacy_static_rerank_score": row.get("rerank_score"),
                    "rerank_method": SEMRANK_METHOD,
                    "rerank_formula_id": SEMRANK_FORMULA_ID,
                    "feature_weights": {
                        "query_score_normalized": (
                            self.config.base_query_weight
                        ),
                        "subquery_score_normalized": (
                            self.config.base_subquery_weight
                        ),
                        "concept_score_z": 0.0 if fallback_used else 1.0,
                    },
                    "query_profile_id": (
                        self.active_query_profile.query_profile_id
                    ),
                    "semrank_query_concept_count": len(query_concepts),
                    "semrank_paper_concept_count": paper_counts[index],
                    "semrank_base_score": float(scores["base"][index]),
                    "semrank_base_score_z": float(scores["base_z"][index]),
                    "semrank_concept_score": float(
                        scores["concept"][index]
                    ),
                    "semrank_concept_score_z": float(
                        scores["concept_z"][index]
                    ),
                    "rerank_score": float(scores["final"][index]),
                    "candidate_pool_signature": pool_signature,
                    "semrank_fallback_used": fallback_used,
                    "semrank_fallback_reason": fallback_reason,
                }
            )
            updated.append(row)

        def order_key(row: Mapping[str, Any]) -> Tuple[float, int, int, str]:
            observed = row.get("observed_retrieval_rank")
            return (
                -float(row["rerank_score"]),
                -int(bool(row.get("is_seed"))),
                int(observed if observed is not None else 10**12),
                str(row["paper_arxiv_id"]),
            )

        updated.sort(key=order_key)
        for rank, row in enumerate(updated, start=1):
            row["rerank_rank"] = rank
        if {row["paper_arxiv_id"] for row in updated} != set(original_ids):
            raise AssertionError("SemRank changed the candidate ID set")

        elapsed = time.perf_counter() - started
        self._stats["rerank_events"] += 1
        self._stats["rerank_candidates"] += len(updated)
        self._stats["fallback_events"] += int(fallback_used)
        self._wall_seconds += elapsed
        event_profile = SemRankEventProfile(
            retrieval_event_id=str(retrieval_event_id),
            query_profile_id=self.active_query_profile.query_profile_id,
            candidate_count=len(updated),
            query_concept_count=len(query_concepts),
            paper_concept_count_total=sum(paper_counts),
            paper_concept_count_mean=(
                float(np.mean(paper_counts)) if paper_counts else 0.0
            ),
            base_mean=float(scores["base_mean"]),
            base_std=float(scores["base_std"]),
            concept_mean=float(scores["concept_mean"]),
            concept_std=float(scores["concept_std"]),
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            rerank_wall_seconds=elapsed,
            candidate_pool_signature=pool_signature,
            extra={
                "base_query_weight": self.config.base_query_weight,
                "base_subquery_weight": self.config.base_subquery_weight,
                "zscore_eps": self.config.zscore_eps,
                "candidate_ids_preserved": True,
            },
        )
        return updated, event_profile

    def snapshot_stats(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            key: int(count) for key, count in self._stats.items()
        }
        value["rerank_wall_seconds"] = float(self._wall_seconds)
        return value
