#!/usr/bin/env python3
"""Fixed feature and paper-type catalog for query-conditioned reranking."""

from __future__ import annotations

from typing import Dict, Tuple


CATALOG_VERSION = "v1"
POLICY_VERSION = "dynamic_rerank_v1"
PROMPT_VERSION = "v3"

POSITIVE_LEVELS: Dict[str, float] = {
    "very_high": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
    "off": 0.0,
}
NEGATIVE_LEVEL = "negative"

DIMENSIONS = {
    "query_similarity": {
        "source": "query_score_normalized",
        "allowed_levels": ["very_high", "high", "medium", "low", "off"],
    },
    "subquery_similarity": {
        "source": "subquery_score_normalized",
        "allowed_levels": ["very_high", "high", "medium", "low", "off"],
    },
    "intent_background": {
        "source": "intent_labels",
        "allowed_levels": [
            "very_high", "high", "medium", "low", "off", "negative",
        ],
    },
    "intent_method": {
        "source": "intent_labels",
        "allowed_levels": [
            "very_high", "high", "medium", "low", "off", "negative",
        ],
    },
    "intent_result": {
        "source": "intent_labels",
        "allowed_levels": [
            "very_high", "high", "medium", "low", "off", "negative",
        ],
    },
    "path_count": {
        "source": "path_count_normalized",
        "allowed_levels": [
            "very_high", "high", "medium", "low", "off", "negative",
        ],
    },
    "paper_type_alignment": {
        "source": "paper_type_probs + paper_type_rules",
        "allowed_levels": ["very_high", "high", "medium", "low", "off"],
    },
}

DIMENSION_NAMES: Tuple[str, ...] = tuple(DIMENSIONS)
SEMANTIC_DIMENSIONS: Tuple[str, ...] = (
    "query_similarity",
    "subquery_similarity",
)

PAPER_TYPES: Tuple[str, ...] = (
    "primary_method",
    "empirical_study",
    "theory_analysis",
    "dataset_benchmark",
    "application_case_study",
    "system_resource",
    "survey_review",
    "taxonomy_tutorial",
    "position_perspective",
)

# Conservative lexical aliases observed from instruction-tuned policy models.
# Each maps unambiguously to one catalog value; no paper type is inferred here.
PAPER_TYPE_ALIASES = {
    "survey": "survey_review",
    "review": "survey_review",
    "survey_paper": "survey_review",
    "review_paper": "survey_review",
    "tutorial": "taxonomy_tutorial",
    "taxonomy": "taxonomy_tutorial",
    "position": "position_perspective",
    "perspective": "position_perspective",
    "case_study": "application_case_study",
}

PAPER_TYPE_ACTIONS: Tuple[str, ...] = (
    "require",
    "prefer",
    "avoid",
    "exclude",
)
PAPER_TYPE_LOGICS: Tuple[str, ...] = ("any", "all")
PAPER_TYPE_STRENGTHS: Tuple[str, ...] = (
    "very_high",
    "high",
    "medium",
    "low",
)


def intent_dimension_values(intent_labels: object) -> Dict[str, float]:
    """Map Semantic Scholar intent labels to the three dynamic features."""

    labels = {
        str(value).strip().lower()
        for value in (intent_labels or [])
        if str(value).strip()
    }
    return {
        "intent_background": 1.0 if "background" in labels else 0.0,
        "intent_method": 1.0 if labels & {"method", "methodology"} else 0.0,
        "intent_result": 1.0 if "result" in labels else 0.0,
    }
