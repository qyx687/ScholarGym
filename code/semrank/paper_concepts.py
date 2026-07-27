"""Global, query-independent SemRank paper concept profiles."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from threading import Lock
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .cache import SemRankCache
from .classifier import TopicClassifier
from .encoder import ConceptEncoder
from .models import (
    PaperConceptProfile,
    SEMRANK_CONCEPT_NORMALIZATION_VERSION,
    SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY,
    SEMRANK_PAPER_PROMPT_VERSION,
    SEMRANK_TOPIC_PIPELINE_VERSION,
    SemRankConfig,
    TopicCandidate,
    normalize_concept,
    paper_text_profile_identity,
    semrank_paper_prompt_version,
    semrank_topic_pipeline_version,
    stable_hash,
    stable_unique,
)
from .prompts import SemRankLLMClient


class SemRankCacheMissError(RuntimeError):
    pass


def paper_text(title: Any, abstract: Any) -> str:
    return f"{str(title or '').strip()}. {str(abstract or '').strip()}".strip()


class PaperConceptService:
    def __init__(
        self,
        config: SemRankConfig,
        cache: SemRankCache,
        classifier: TopicClassifier,
        llm: SemRankLLMClient,
        encoder: ConceptEncoder,
    ) -> None:
        self.config = config
        self.cache = cache
        self.classifier = classifier
        self.llm = llm
        self.encoder = encoder
        self.paper_concept_mode = config.paper_concept_mode
        self.pipeline_version = (
            semrank_topic_pipeline_version(self.paper_concept_mode)
            if self.paper_concept_mode
            == SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY
            else SEMRANK_TOPIC_PIPELINE_VERSION
        )
        self.paper_prompt_version = (
            semrank_paper_prompt_version(self.paper_concept_mode)
            if self.paper_concept_mode
            == SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY
            else SEMRANK_PAPER_PROMPT_VERSION
        )
        self.extraction_llm_id = (
            "none"
            if self.paper_concept_mode
            == SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY
            else self.llm.llm_id
        )
        self._stats: Counter[str] = Counter()
        self._audit_records: List[Dict[str, Any]] = []
        self._emitted_profile_versions = set()
        self._lock = Lock()

    def _identity(
        self, paper_id: str, title: str, abstract: str
    ) -> Dict[str, Any]:
        return paper_text_profile_identity({
            "paper_arxiv_id": str(paper_id),
            "title_abstract_hash": stable_hash(
                {"title": str(title or ""), "abstract": str(abstract or "")}
            ),
            "classifier_id": self.classifier.classifier_id,
            "label_space_id": self.classifier.label_space_id,
            "classifier_topic_k": self.config.classifier_topic_k,
            "paper_prompt_version": self.paper_prompt_version,
            "extraction_llm": self.extraction_llm_id,
            "concept_normalization": SEMRANK_CONCEPT_NORMALIZATION_VERSION,
            "paper_text_serialization": "title_period_space_abstract_v1",
            "pipeline_version": self.pipeline_version,
        })

    @staticmethod
    def _cache_key(identity: Mapping[str, Any]) -> str:
        return stable_hash(dict(identity))

    def _record_audit(self, profile: PaperConceptProfile) -> None:
        version = (
            profile.profile_id,
            profile.status,
            profile.fallback_reason,
            tuple(profile.selected_topics),
            tuple(profile.keyphrases),
        )
        with self._lock:
            if version in self._emitted_profile_versions:
                return
            self._emitted_profile_versions.add(version)
            value = profile.to_dict(include_raw=True)
            value["concept_count"] = len(profile.concepts)
            self._audit_records.append(value)

    def drain_audit_records(self) -> List[Dict[str, Any]]:
        with self._lock:
            output = list(self._audit_records)
            self._audit_records.clear()
        return output

    def get_or_build(
        self,
        papers: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, PaperConceptProfile]:
        ordered_ids = list(dict.fromkeys(str(value) for value in papers))
        profiles: Dict[str, PaperConceptProfile] = {}
        concepts_to_encode: List[str] = []
        misses: List[
            Tuple[str, str, str, Dict[str, Any], str]
        ] = []

        for paper_id in ordered_ids:
            metadata = papers.get(paper_id) or {}
            title = str(metadata.get("title") or "")
            abstract = str(metadata.get("abstract") or "")
            identity = self._identity(paper_id, title, abstract)
            cache_key = self._cache_key(identity)
            cached = None
            if not self.config.rebuild_paper_concepts:
                cached = self.cache.get_paper_profile(cache_key)
                if (
                    cached is not None
                    and self.paper_concept_mode
                    != SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY
                    and self.config.retry_failed_paper_concepts
                    and cached.status == "failed"
                ):
                    parser = getattr(
                        self.llm,
                        "parse_paper_response",
                        None,
                    )
                    repaired = None
                    if (
                        callable(parser)
                        and cached.raw_llm_output
                        and cached.fallback_reason
                        == "paper_concept_llm_parse_failed"
                    ):
                        candidates = [
                            TopicCandidate(
                                concept=str(
                                    item.get("concept") or ""
                                ),
                                score=float(item.get("score") or 0.0),
                                label_id=str(
                                    item.get("label_id") or ""
                                ),
                            )
                            for item in cached.candidate_topics
                        ]
                        try:
                            selected_topics, keyphrases = parser(
                                cached.raw_llm_output,
                                title,
                                abstract,
                                candidates,
                            )
                        except (ValueError, json.JSONDecodeError):
                            repaired = None
                        else:
                            concepts = stable_unique(
                                selected_topics + keyphrases
                            )
                            repaired = replace(
                                cached,
                                selected_topics=stable_unique(
                                    selected_topics
                                ),
                                keyphrases=stable_unique(keyphrases),
                                concepts=concepts,
                                status="ok" if concepts else "empty",
                                fallback_reason=(
                                    None
                                    if concepts
                                    else "paper_concepts_empty"
                                ),
                                concept_encoder=self.encoder.encoder_id,
                                cache_hit=False,
                            )
                            self.cache.put_paper_profile(
                                cache_key,
                                identity,
                                repaired,
                            )
                            self._stats[
                                "paper_concept_failed_cache_local_repairs"
                            ] += 1
                    if repaired is not None:
                        cached = repaired
                    else:
                        self._stats[
                            "paper_concept_failed_cache_retries"
                        ] += 1
                        cached = None
            if cached is not None:
                cached = replace(
                    cached,
                    concept_encoder=self.encoder.encoder_id,
                )
                if cached.concepts:
                    concepts_to_encode.extend(cached.concepts)
                profiles[paper_id] = cached
                self._stats["paper_concept_cache_hits"] += 1
                self._record_audit(cached)
            else:
                self._stats["paper_concept_cache_misses"] += 1
                misses.append(
                    (paper_id, title, abstract, identity, cache_key)
                )

        if misses and self.config.cache_only:
            preview = ", ".join(item[0] for item in misses[:10])
            raise SemRankCacheMissError(
                "SemRank cache-only evaluation is missing paper concept "
                f"profiles ({len(misses)} missing; first IDs: {preview})"
            )
        if misses and not self.config.allow_lazy_paper_concepts:
            raise SemRankCacheMissError(
                "SemRank lazy paper concept construction is disabled and "
                f"{len(misses)} profiles are missing"
            )
        if not misses:
            if concepts_to_encode:
                self.encoder.encode(stable_unique(concepts_to_encode))
            return profiles

        texts = [paper_text(title, abstract) for _, title, abstract, _, _ in misses]
        candidates = self.classifier.predict_batch(
            texts,
            top_k=self.config.classifier_topic_k,
        )
        self._stats["topic_classifier_calls"] += 1
        self._stats["topic_classifier_papers"] += len(misses)
        if len(candidates) != len(misses):
            raise RuntimeError(
                "SemRank topic classifier returned an unexpected batch size"
            )
        if (
            self.paper_concept_mode
            == SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY
        ):
            for miss, topic_candidates in zip(misses, candidates):
                paper_id, _, _, identity, cache_key = miss
                selected_topics = stable_unique(
                    item.concept
                    if isinstance(item, TopicCandidate)
                    else item.get("concept")
                    for item in topic_candidates
                )
                profile = PaperConceptProfile(
                    paper_arxiv_id=paper_id,
                    profile_id=cache_key,
                    title_abstract_hash=str(identity["title_abstract_hash"]),
                    candidate_topics=[
                        item.to_dict()
                        if isinstance(item, TopicCandidate)
                        else dict(item)
                        for item in topic_candidates
                    ],
                    selected_topics=selected_topics,
                    keyphrases=[],
                    concepts=selected_topics,
                    status="ok" if selected_topics else "empty",
                    fallback_reason=(
                        None
                        if selected_topics
                        else "classifier_topics_empty"
                    ),
                    pipeline_version=self.pipeline_version,
                    classifier_id=self.classifier.classifier_id,
                    label_space_id=self.classifier.label_space_id,
                    llm_model="none",
                    prompt_version=self.paper_prompt_version,
                    concept_encoder=self.encoder.encoder_id,
                    cache_hit=False,
                    raw_llm_output=None,
                )
                concepts_to_encode.extend(selected_topics)
                self.cache.put_paper_profile(
                    cache_key,
                    identity,
                    profile,
                )
                profiles[paper_id] = profile
                self._stats[
                    "paper_classifier_only_profiles_built"
                ] += 1
                self._stats["paper_concepts_total"] += len(
                    selected_topics
                )
                self._record_audit(profile)
            if concepts_to_encode:
                self.encoder.encode(stable_unique(concepts_to_encode))
            return {
                paper_id: profiles[paper_id] for paper_id in ordered_ids
            }

        llm_requests = [
            (title, abstract, topic_candidates)
            for (
                _,
                title,
                abstract,
                _,
                _,
            ), topic_candidates in zip(misses, candidates)
        ]
        refinements = self.llm.refine_papers(llm_requests)
        if len(refinements) != len(misses):
            raise RuntimeError(
                "SemRank paper concept LLM returned an unexpected batch size"
            )

        for miss, topic_candidates, refinement in zip(
            misses, candidates, refinements
        ):
            paper_id, _, _, identity, cache_key = miss
            if len(refinement) == 4:
                selected_topics, keyphrases, raw, refinement_error = (
                    refinement
                )
            else:  # Backward-compatible fake clients used by local tests.
                selected_topics, keyphrases, raw = refinement
                refinement_error = None
            selected_topics = stable_unique(selected_topics)
            keyphrases = stable_unique(keyphrases)
            concepts = stable_unique(selected_topics + keyphrases)
            status = (
                "failed"
                if refinement_error
                else ("ok" if concepts else "empty")
            )
            reason = (
                refinement_error
                if refinement_error
                else (None if concepts else "paper_concepts_empty")
            )
            profile = PaperConceptProfile(
                paper_arxiv_id=paper_id,
                profile_id=cache_key,
                title_abstract_hash=str(identity["title_abstract_hash"]),
                candidate_topics=[
                    item.to_dict()
                    if isinstance(item, TopicCandidate)
                    else dict(item)
                    for item in topic_candidates
                ],
                selected_topics=selected_topics,
                keyphrases=keyphrases,
                concepts=concepts,
                status=status,
                fallback_reason=reason,
                pipeline_version=self.pipeline_version,
                classifier_id=self.classifier.classifier_id,
                label_space_id=self.classifier.label_space_id,
                llm_model=self.llm.model,
                prompt_version=self.paper_prompt_version,
                concept_encoder=self.encoder.encoder_id,
                cache_hit=False,
                raw_llm_output=raw,
            )
            concepts_to_encode.extend(concepts)
            # Persist the vector-independent text result before embedding it.
            # A concept-service outage must never force classifier/LLM work
            # to be repeated.
            self.cache.put_paper_profile(
                cache_key,
                identity,
                profile,
            )
            profiles[paper_id] = profile
            self._stats["paper_concept_profiles_built"] += 1
            self._stats["paper_concepts_total"] += len(concepts)
            self._record_audit(profile)

        if concepts_to_encode:
            self.encoder.encode(stable_unique(concepts_to_encode))
        return {paper_id: profiles[paper_id] for paper_id in ordered_ids}

    def snapshot_stats(self) -> Dict[str, int]:
        return {key: int(value) for key, value in self._stats.items()}
