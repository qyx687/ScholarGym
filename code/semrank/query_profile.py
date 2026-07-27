"""One cached SemRank concept profile per original benchmark query."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence

from .cache import SemRankCache
from .encoder import ConceptEncoder
from .models import (
    AuxiliaryPaper,
    QueryConceptProfile,
    SEMRANK_CONCEPT_NORMALIZATION_VERSION,
    SEMRANK_PAPER_PROMPT_VERSION,
    SEMRANK_QUERY_PROMPT_VERSION,
    SEMRANK_TOPIC_PIPELINE_VERSION,
    SemRankConfig,
    normalize_concept,
    query_text_profile_identity,
    stable_hash,
)
from .paper_concepts import PaperConceptService
from .prompts import SemRankLLMClient


class DateValidInitialRetriever(Protocol):
    retriever_identity: str

    def retrieve_date_valid(
        self,
        query: str,
        *,
        top_k: int,
        before_date: str,
    ) -> Sequence[AuxiliaryPaper]:
        ...


def _rank_frequencies(
    counter: Counter[str], limit: int
) -> list[Dict[str, Any]]:
    return [
        {"concept": concept, "frequency": int(frequency)}
        for concept, frequency in sorted(
            counter.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[:limit]
    ]


class QueryProfileBuilder:
    def __init__(
        self,
        config: SemRankConfig,
        cache: SemRankCache,
        paper_concepts: PaperConceptService,
        llm: SemRankLLMClient,
        encoder: ConceptEncoder,
    ) -> None:
        self.config = config
        self.cache = cache
        self.paper_concepts = paper_concepts
        self.llm = llm
        self.encoder = encoder
        self.active_profile: Optional[QueryConceptProfile] = None
        self._active_identity_key: Optional[str] = None
        self._stats: Counter[str] = Counter()

    def _identity(
        self,
        query_id: str,
        query: str,
        date_cutoff: str,
        retriever: DateValidInitialRetriever,
    ) -> Dict[str, Any]:
        return query_text_profile_identity({
            "query_id": str(query_id),
            "original_query_hash": stable_hash(str(query or "")),
            "date_cutoff": str(date_cutoff or "")[:7],
            "initial_retriever_identity": retriever.retriever_identity,
            "initial_top_m": self.config.initial_top_m,
            "feedback_top_n": self.config.feedback_top_n,
            "prompt_top_papers": self.config.prompt_top_papers,
            "candidate_topic_k": self.config.candidate_topic_k,
            "candidate_phrase_k": self.config.candidate_phrase_k,
            "paper_concept_pipeline_version": (
                self.paper_concepts.pipeline_version
            ),
            "paper_classifier_topic_k": self.config.classifier_topic_k,
            "paper_prompt_version": getattr(
                self.paper_concepts,
                "paper_prompt_version",
                SEMRANK_PAPER_PROMPT_VERSION,
            ),
            "paper_topic_classifier": (
                self.paper_concepts.classifier.classifier_id
            ),
            "paper_topic_label_space": (
                self.paper_concepts.classifier.label_space_id
            ),
            "paper_extraction_llm": getattr(
                self.paper_concepts,
                "extraction_llm_id",
                self.paper_concepts.llm.llm_id,
            ),
            "concept_normalization": SEMRANK_CONCEPT_NORMALIZATION_VERSION,
            "query_prompt_version": SEMRANK_QUERY_PROMPT_VERSION,
            "query_llm": self.llm.llm_id,
        })

    def start_query(
        self,
        *,
        query_id: str,
        query: str,
        date_cutoff: str,
        retriever: DateValidInitialRetriever,
    ) -> QueryConceptProfile:
        identity = self._identity(
            query_id,
            query,
            date_cutoff,
            retriever,
        )
        cache_key = stable_hash(identity)
        if (
            self._active_identity_key == cache_key
            and self.active_profile is not None
        ):
            self._stats["active_query_profile_reuses"] += 1
            return self.active_profile

        self.active_profile = None
        self._active_identity_key = cache_key
        if not self.config.rebuild_query_profile:
            cached = self.cache.get_query_profile(cache_key)
            if cached is not None:
                cached = replace(
                    cached,
                    concept_encoder=self.encoder.encoder_id,
                )
                retryable_failure = (
                    cached.selection_status == "fallback"
                    and str(cached.fallback_reason or "").startswith(
                        "query_concept_selection_failed:"
                    )
                )
                if (
                    self.config.retry_failed_query_profiles
                    and retryable_failure
                ):
                    self._stats["query_profile_failed_cache_retries"] += 1
                else:
                    if cached.selected_concepts:
                        self.encoder.encode(cached.selected_concepts)
                    self.active_profile = cached
                    self._stats["query_profile_cache_hits"] += 1
                    return cached

        self._stats["query_profile_cache_misses"] += 1
        if self.config.cache_only:
            raise RuntimeError(
                "SemRank cache-only evaluation is missing the original-query "
                f"profile for query_id={query_id!r}"
            )

        retrieved = list(
            retriever.retrieve_date_valid(
                query,
                top_k=self.config.initial_top_m,
                before_date=str(date_cutoff or ""),
            )
        )
        if len(retrieved) < self.config.initial_top_m:
            raise RuntimeError(
                "SemRank auxiliary retrieval returned only "
                f"{len(retrieved)} date-valid papers; "
                f"{self.config.initial_top_m} are required"
            )
        cutoff_month = str(date_cutoff or "")[:7]
        for paper in retrieved:
            paper_month = str(paper.date or "")[:7]
            if (
                not re.fullmatch(r"\d{4}-\d{2}", paper_month)
                or paper_month > cutoff_month
            ):
                raise AssertionError(
                    "SemRank auxiliary retriever returned a paper outside "
                    "the strict date-valid set"
                )
        self._stats["auxiliary_initial_retrieval_calls"] += 1
        self._stats["auxiliary_initial_retrieval_papers"] += len(retrieved)
        feedback = retrieved[: self.config.feedback_top_n]
        paper_metadata = {
            item.paper_arxiv_id: {
                "title": item.title,
                "abstract": item.abstract,
                "date": item.date,
            }
            for item in feedback
        }
        profiles = self.paper_concepts.get_or_build(paper_metadata)
        failed_feedback = [
            item.paper_arxiv_id
            for item in feedback
            if profiles[item.paper_arxiv_id].status == "failed"
        ]
        if failed_feedback:
            self._stats["query_profile_failed_feedback_papers"] += len(
                failed_feedback
            )
            preview = ", ".join(failed_feedback[:10])
            raise RuntimeError(
                "SemRank query concepts cannot be constructed from transiently "
                f"failed feedback profiles ({len(failed_feedback)} failed; "
                f"first IDs: {preview}). Retry the query after the failed "
                "paper-profile cache entries are rebuilt."
            )

        topic_frequency: Counter[str] = Counter()
        keyphrase_frequency: Counter[str] = Counter()
        for item in feedback:
            profile = profiles[item.paper_arxiv_id]
            topic_frequency.update(
                {
                    normalize_concept(value)
                    for value in profile.selected_topics
                    if normalize_concept(value)
                }
            )
            keyphrase_frequency.update(
                {
                    normalize_concept(value)
                    for value in profile.keyphrases
                    if normalize_concept(value)
                }
            )
        candidate_topics = _rank_frequencies(
            topic_frequency,
            self.config.candidate_topic_k,
        )
        candidate_keyphrases = _rank_frequencies(
            keyphrase_frequency,
            self.config.candidate_phrase_k,
        )

        selected: list[str] = []
        raw_output: Optional[str] = None
        status = "fallback"
        fallback_reason: Optional[str] = None
        if not candidate_topics and not candidate_keyphrases:
            fallback_reason = "no_candidate_concepts_from_feedback"
        else:
            try:
                selected, raw_output = self.llm.select_query_concepts(
                    query,
                    retrieved[: self.config.prompt_top_papers],
                    candidate_topics,
                    candidate_keyphrases,
                )
            except Exception as exc:
                fallback_reason = (
                    "query_concept_selection_failed:"
                    f"{type(exc).__name__}"
                )
            if selected:
                status = "ok"
                fallback_reason = None
            elif fallback_reason is None:
                fallback_reason = "query_concept_selection_empty"

        profile = QueryConceptProfile(
            query_id=str(query_id),
            query_profile_id=cache_key,
            query=str(query or ""),
            date_cutoff=cutoff_month,
            initial_retrieval_paper_ids=[
                item.paper_arxiv_id for item in retrieved
            ],
            feedback_paper_ids=[
                item.paper_arxiv_id for item in feedback
            ],
            candidate_topics=candidate_topics,
            candidate_keyphrases=candidate_keyphrases,
            selected_concepts=selected,
            selection_status=status,
            fallback_reason=fallback_reason,
            prompt_version=SEMRANK_QUERY_PROMPT_VERSION,
            llm_model=self.llm.model,
            concept_encoder=self.encoder.encoder_id,
            topic_pipeline_version=self.paper_concepts.pipeline_version,
            retriever_identity=retriever.retriever_identity,
            initial_retrieval_target=self.config.initial_top_m,
            initial_retrieval_count=len(retrieved),
            initial_retrieval_complete=(
                len(retrieved) >= self.config.initial_top_m
            ),
            cache_hit=False,
            raw_llm_output=raw_output,
        )
        # Cache text selection before vectorization so an embedding-service
        # failure cannot trigger another query LLM call on resume.
        self.cache.put_query_profile(cache_key, identity, profile)
        if selected:
            self.encoder.encode(selected)
        self.active_profile = profile
        self._stats["query_profiles_built"] += 1
        self._stats["query_concepts_total"] += len(selected)
        return profile

    def snapshot_stats(self) -> Dict[str, int]:
        return {key: int(value) for key, value in self._stats.items()}
