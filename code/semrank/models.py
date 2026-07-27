"""Data models and immutable identities for the SemRank-QSQ adaptation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SEMRANK_METHOD = "semrank_qsq"
SEMRANK_METHOD_DISPLAY_NAME = (
    "SemRank-QSQ: query-level cached concepts + query/subquery-aware "
    "semantic reranking"
)
SEMRANK_IMPLEMENTATION_VERSION = (
    "semrank_qsq_v4_qwen3_full_and_classifier_only"
)
SEMRANK_FORMULA_ID = "zscore_pool_ddof0_base_q040_sq060_plus_concept_v1"
SEMRANK_TOPIC_PIPELINE_VERSION = "official_semrank_classifier_llm_adapted_v1"
SEMRANK_PAPER_PROMPT_VERSION = "official_semrank_paper_topic_keyphrase_json_v1"
SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION = (
    "official_semrank_classifier_only_topics_v1"
)
SEMRANK_CLASSIFIER_ONLY_PAPER_PROMPT_VERSION = (
    "classifier_only_no_paper_llm_v1"
)
SEMRANK_PAPER_CONCEPT_MODE_FULL = "full"
SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY = "classifier_only"
SEMRANK_PAPER_CONCEPT_MODES = {
    SEMRANK_PAPER_CONCEPT_MODE_FULL,
    SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY,
}
SEMRANK_QUERY_PROMPT_VERSION = "official_semrank_query_concept_json_v1"
SEMRANK_CONCEPT_NORMALIZATION_VERSION = "nfkc_lower_space_v1"
SEMRANK_PAPER_TEXT_PROFILE_IDENTITY_VERSION = (
    "paper_text_profile_encoder_independent_v1"
)
SEMRANK_QUERY_TEXT_PROFILE_IDENTITY_VERSION = (
    "query_text_profile_encoder_independent_v1"
)
SEMRANK_ZSCORE_EPS = 1e-12


def normalize_concept(value: Any) -> str:
    """Normalize for exact vocabulary matching without fuzzy expansion."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text.strip(" \t\r\n,;")


def stable_unique(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        normalized = normalize_concept(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paper_text_profile_identity(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the vector-encoder-independent paper text-profile identity."""

    identity = dict(value)
    identity.pop("concept_encoder", None)
    identity["text_profile_identity_version"] = (
        SEMRANK_PAPER_TEXT_PROFILE_IDENTITY_VERSION
    )
    return identity


def query_text_profile_identity(
    value: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the vector-encoder-independent query text-profile identity."""

    identity = dict(value)
    identity.pop("paper_concept_encoder", None)
    identity.pop("concept_encoder", None)
    identity["text_profile_identity_version"] = (
        SEMRANK_QUERY_TEXT_PROFILE_IDENTITY_VERSION
    )
    return identity


def semrank_topic_pipeline_version(paper_concept_mode: str) -> str:
    if paper_concept_mode == SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY:
        return SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION
    return SEMRANK_TOPIC_PIPELINE_VERSION


def semrank_paper_prompt_version(paper_concept_mode: str) -> str:
    if paper_concept_mode == SEMRANK_PAPER_CONCEPT_MODE_CLASSIFIER_ONLY:
        return SEMRANK_CLASSIFIER_ONLY_PAPER_PROMPT_VERSION
    return SEMRANK_PAPER_PROMPT_VERSION


@dataclass(frozen=True)
class SemRankConfig:
    initial_top_m: int = 1000
    feedback_top_n: int = 100
    prompt_top_papers: int = 50
    candidate_topic_k: int = 50
    candidate_phrase_k: int = 50
    classifier_topic_k: int = 100
    paper_concept_mode: str = SEMRANK_PAPER_CONCEPT_MODE_FULL
    base_query_weight: float = 0.4
    base_subquery_weight: float = 0.6
    zscore_eps: float = SEMRANK_ZSCORE_EPS
    concept_encoder_backend: str = "ollama"
    concept_encoder: str = "qwen3-embedding:0.6b"
    concept_encoder_base_url: str = "http://127.0.0.1:11434"
    concept_encoder_revision: str = ""
    concept_encoder_max_length: int = 512
    concept_encoder_batch_size: int = 64
    concept_encoder_device: str = "cuda:0"
    llm_model: str = "qwen3-30b-a3b-instruct-2507"
    llm_is_local: bool = False
    llm_enable_thinking: bool = False
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_max_tokens: int = 4096
    llm_workers: int = 8
    topic_classifier_checkpoint: str = ""
    topic_labels_path: str = ""
    topic_classifier_encoder: str = "allenai/specter2_base"
    topic_classifier_encoder_revision: str = (
        "3447645e1def9117997203454fa4495937bfbd83"
    )
    topic_classifier_device: str = "cuda:0"
    topic_classifier_batch_size: int = 4
    cache_path: str = "cache/semrank/semrank.sqlite3"
    allow_lazy_paper_concepts: bool = True
    cache_only: bool = False
    rebuild_query_profile: bool = False
    rebuild_paper_concepts: bool = False
    retry_failed_query_profiles: bool = True
    retry_failed_paper_concepts: bool = True

    def __post_init__(self) -> None:
        for name in (
            "initial_top_m",
            "feedback_top_n",
            "prompt_top_papers",
            "candidate_topic_k",
            "candidate_phrase_k",
            "classifier_topic_k",
            "concept_encoder_max_length",
            "concept_encoder_batch_size",
            "llm_workers",
            "topic_classifier_batch_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if abs(
            float(self.base_query_weight)
            + float(self.base_subquery_weight)
            - 1.0
        ) > 1e-12:
            raise ValueError("SemRank base query/subquery weights must sum to 1")
        if (
            abs(float(self.base_query_weight) - 0.4) > 1e-12
            or abs(float(self.base_subquery_weight) - 0.6) > 1e-12
        ):
            raise ValueError(
                "SemRank-QSQ main mode fixes base weights at 0.4 query / "
                "0.6 subquery"
            )
        if float(self.zscore_eps) <= 0:
            raise ValueError("zscore_eps must be positive")
        if self.concept_encoder_backend not in {
            "ollama",
            "api",
            "specter2",
        }:
            raise ValueError(
                "concept_encoder_backend must be ollama, api, or specter2"
            )
        if not str(self.concept_encoder).strip():
            raise ValueError("concept_encoder must be non-empty")
        if self.concept_encoder_backend in {"ollama", "api"} and not str(
            self.concept_encoder_base_url
        ).strip():
            raise ValueError(
                "concept_encoder_base_url is required for service encoders"
            )
        if not str(self.topic_classifier_encoder).strip():
            raise ValueError("topic_classifier_encoder must be non-empty")
        if self.paper_concept_mode not in SEMRANK_PAPER_CONCEPT_MODES:
            raise ValueError(
                "paper_concept_mode must be full or classifier_only"
            )
        if self.cache_only and (
            self.rebuild_query_profile or self.rebuild_paper_concepts
        ):
            raise ValueError(
                "SemRank cache-only mode cannot rebuild cached profiles"
            )

    def identity(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "implementation_version": SEMRANK_IMPLEMENTATION_VERSION,
            "formula_id": SEMRANK_FORMULA_ID,
            "topic_pipeline_version": semrank_topic_pipeline_version(
                self.paper_concept_mode
            ),
            "paper_prompt_version": semrank_paper_prompt_version(
                self.paper_concept_mode
            ),
            "query_prompt_version": SEMRANK_QUERY_PROMPT_VERSION,
            "concept_normalization_version": (
                SEMRANK_CONCEPT_NORMALIZATION_VERSION
            ),
        }


@dataclass(frozen=True)
class TopicCandidate:
    concept: str
    score: float
    label_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": normalize_concept(self.concept),
            "score": float(self.score),
            "label_id": str(self.label_id or ""),
        }


@dataclass(frozen=True)
class AuxiliaryPaper:
    paper_arxiv_id: str
    score: float
    title: str
    abstract: str
    date: str
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperConceptProfile:
    paper_arxiv_id: str
    profile_id: str
    title_abstract_hash: str
    candidate_topics: Sequence[Mapping[str, Any]]
    selected_topics: Sequence[str]
    keyphrases: Sequence[str]
    concepts: Sequence[str]
    status: str
    fallback_reason: Optional[str]
    pipeline_version: str
    classifier_id: str
    label_space_id: str
    llm_model: str
    prompt_version: str
    concept_encoder: str
    cache_hit: bool = False
    raw_llm_output: Optional[str] = None

    def to_dict(self, *, include_raw: bool = True) -> Dict[str, Any]:
        value = asdict(self)
        value["candidate_topics"] = [
            dict(item) for item in self.candidate_topics
        ]
        value["selected_topics"] = list(self.selected_topics)
        value["keyphrases"] = list(self.keyphrases)
        value["concepts"] = list(self.concepts)
        if not include_raw:
            value.pop("raw_llm_output", None)
        return value

    def with_cache_hit(self, value: bool) -> "PaperConceptProfile":
        return replace(self, cache_hit=bool(value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PaperConceptProfile":
        accepted = {item.name for item in fields(cls)}
        field_values = {
            key: item for key, item in value.items() if key in accepted
        }
        for key in (
            "candidate_topics",
            "selected_topics",
            "keyphrases",
            "concepts",
        ):
            field_values[key] = list(field_values.get(key) or [])
        field_values.setdefault("cache_hit", False)
        field_values.setdefault("raw_llm_output", None)
        return cls(**field_values)


@dataclass(frozen=True)
class QueryConceptProfile:
    query_id: str
    query_profile_id: str
    query: str
    date_cutoff: str
    initial_retrieval_paper_ids: Sequence[str]
    feedback_paper_ids: Sequence[str]
    candidate_topics: Sequence[Mapping[str, Any]]
    candidate_keyphrases: Sequence[Mapping[str, Any]]
    selected_concepts: Sequence[str]
    selection_status: str
    fallback_reason: Optional[str]
    prompt_version: str
    llm_model: str
    concept_encoder: str
    topic_pipeline_version: str
    retriever_identity: str
    initial_retrieval_target: int
    initial_retrieval_count: int
    initial_retrieval_complete: bool
    cache_hit: bool = False
    raw_llm_output: Optional[str] = None

    @property
    def selected_concept_count(self) -> int:
        return len(self.selected_concepts)

    def to_dict(self, *, include_raw: bool = True) -> Dict[str, Any]:
        value = asdict(self)
        value["initial_retrieval_paper_ids"] = list(
            self.initial_retrieval_paper_ids
        )
        value["feedback_paper_ids"] = list(self.feedback_paper_ids)
        value["candidate_topics"] = [
            dict(item) for item in self.candidate_topics
        ]
        value["candidate_keyphrases"] = [
            dict(item) for item in self.candidate_keyphrases
        ]
        value["selected_concepts"] = list(self.selected_concepts)
        value["selected_concept_count"] = self.selected_concept_count
        if not include_raw:
            value.pop("raw_llm_output", None)
        return value

    def with_cache_hit(self, value: bool) -> "QueryConceptProfile":
        return replace(self, cache_hit=bool(value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QueryConceptProfile":
        accepted = {item.name for item in fields(cls)}
        field_values = {
            key: item for key, item in value.items() if key in accepted
        }
        for key in (
            "initial_retrieval_paper_ids",
            "feedback_paper_ids",
            "candidate_topics",
            "candidate_keyphrases",
            "selected_concepts",
        ):
            field_values[key] = list(field_values.get(key) or [])
        field_values.setdefault("cache_hit", False)
        field_values.setdefault("raw_llm_output", None)
        return cls(**field_values)


@dataclass
class SemRankEventProfile:
    retrieval_event_id: str
    query_profile_id: str
    candidate_count: int
    query_concept_count: int
    paper_concept_count_total: int
    paper_concept_count_mean: float
    base_mean: float
    base_std: float
    concept_mean: float
    concept_std: float
    fallback_used: bool
    fallback_reason: Optional[str]
    rerank_wall_seconds: float
    candidate_pool_signature: str
    formula_id: str = SEMRANK_FORMULA_ID
    method: str = SEMRANK_METHOD
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        extra = value.pop("extra", {})
        value.update(extra)
        return value
