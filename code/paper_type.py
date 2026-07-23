#!/usr/bin/env python3
"""Semantic Scholar publication-type metadata and rule evaluation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

S2_CLASSIFIER_VERSION = "s2_publication_types_v1"
S2_EVIDENCE_SOURCE = "semantic_scholar"

_STRENGTH_VALUES = {
    "very_high": 1.0,
    "high": 0.75,
    "medium": 0.50,
    "low": 0.25,
}

S2_PUBLICATION_TYPES = (
    "Review",
    "JournalArticle",
    "CaseReport",
    "ClinicalTrial",
    "Conference",
    "Dataset",
    "Editorial",
    "LettersAndComments",
    "MetaAnalysis",
    "News",
    "Study",
    "Book",
    "BookSection",
)

_S2_PUBLICATION_TYPE_ALIASES = {
    re.sub(r"[^a-z0-9]+", "", name.lower()): name
    for name in S2_PUBLICATION_TYPES
}


def normalize_paper_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"(?i)^arxiv\s*:\s*", "", text)
    text = re.sub(r"(?i)\.pdf$", "", text)
    match = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", text)
    if match:
        return match.group(1)
    return re.sub(r"v\d+$", "", text, flags=re.IGNORECASE).lower()


def _finite_probability(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


def normalize_s2_publication_type(value: Any) -> str:
    """Normalize documented S2 type spellings while preserving future values."""

    text = str(value or "").strip()
    if not text:
        return ""
    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    return _S2_PUBLICATION_TYPE_ALIASES.get(key, text)


def _validated_s2_type_list(value: Any, field: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    output = []
    for item in value:
        type_name = normalize_s2_publication_type(item)
        if type_name not in S2_PUBLICATION_TYPES:
            raise ValueError(f"unknown paper type in {field}: {type_name!r}")
        if type_name not in output:
            output.append(type_name)
    return output


def s2_publication_types_to_record(
    paper_arxiv_id: Any,
    publication_types: Any,
    *,
    resolved: bool = True,
) -> Dict[str, Any]:
    """Convert S2 bibliographic types into positive-only native evidence."""

    if publication_types is None:
        raw_types: Sequence[Any] = []
    elif isinstance(publication_types, (list, tuple, set)) and not isinstance(
        publication_types, (str, bytes)
    ):
        raw_types = list(publication_types)
    else:
        raise ValueError("publication_types must be an array or null")
    normalized_types = []
    for value in raw_types:
        normalized = normalize_s2_publication_type(value)
        if normalized and normalized not in normalized_types:
            normalized_types.append(normalized)
    probs = {
        publication_type: 1.0
        for publication_type in normalized_types
        if publication_type in S2_PUBLICATION_TYPES
    }
    return validate_type_record(
        {
            "paper_arxiv_id": paper_arxiv_id,
            "type_probs": probs,
            "confidence": 1.0 if resolved else 0.0,
            "classifier_version": S2_CLASSIFIER_VERSION,
            "evidence_source": S2_EVIDENCE_SOURCE,
            "publication_types": normalized_types,
            "supported_types": list(S2_PUBLICATION_TYPES),
            # Native S2 publicationTypes are positive-only evidence. An absent
            # tag is unknown rather than reliable negative evidence.
            "negative_evidence_types": [],
        }
    )


def validate_type_record(value: Mapping[str, Any]) -> Dict[str, Any]:
    allowed_keys = {
        "paper_arxiv_id",
        "type_probs",
        "confidence",
        "classifier_version",
        "evidence_source",
        "publication_types",
        "supported_types",
        "negative_evidence_types",
    }
    extra = set(value) - allowed_keys
    if extra:
        raise ValueError(f"unexpected paper-type keys: {sorted(extra)}")
    paper_id = normalize_paper_id(value.get("paper_arxiv_id"))
    if not paper_id:
        raise ValueError("paper_arxiv_id is required")
    confidence = _finite_probability(value.get("confidence"), "confidence")
    classifier_version = str(
        value.get("classifier_version") or S2_CLASSIFIER_VERSION
    )
    evidence_source = str(value.get("evidence_source") or "").strip()
    if not evidence_source:
        # Old Qwen cache rows predate explicit provenance. Keep detecting them
        # so they cannot be silently accepted as native S2 evidence.
        evidence_source = (
            S2_EVIDENCE_SOURCE
            if classifier_version.startswith("s2_")
            else "qwen"
        )
    raw_publication_types = value.get("publication_types") or []
    if not isinstance(raw_publication_types, (list, tuple, set)) or isinstance(
        raw_publication_types, (str, bytes)
    ):
        raise ValueError("publication_types must be an array")
    publication_types = []
    for item in raw_publication_types:
        normalized = normalize_s2_publication_type(item)
        if normalized and normalized not in publication_types:
            publication_types.append(normalized)
    raw_probs = value.get("type_probs")
    if not isinstance(raw_probs, Mapping):
        raise ValueError("type_probs must be an object")
    if evidence_source == S2_EVIDENCE_SOURCE:
        # publication_types is authoritative. This also upgrades old mapped S2
        # cache rows in memory without requiring an expensive API rebuild.
        probs = {
            type_name: 1.0
            for type_name in publication_types
            if type_name in S2_PUBLICATION_TYPES
        }
        supported_types = list(S2_PUBLICATION_TYPES)
        negative_evidence_types: List[str] = []
    else:
        illegal = set(raw_probs) - set(S2_PUBLICATION_TYPES)
        if illegal:
            raise ValueError(f"unknown native S2 paper types: {sorted(illegal)}")
        probs = {
            str(name): _finite_probability(probability, f"type_probs.{name}")
            for name, probability in raw_probs.items()
        }
        supported_types = _validated_s2_type_list(
            value.get("supported_types", S2_PUBLICATION_TYPES), "supported_types"
        )
        negative_evidence_types = _validated_s2_type_list(
            value.get("negative_evidence_types", []),
            "negative_evidence_types",
        )
    if not set(negative_evidence_types).issubset(supported_types):
        raise ValueError("negative_evidence_types must be a subset of supported_types")
    return {
        "paper_arxiv_id": paper_id,
        "type_probs": probs,
        "confidence": confidence,
        "classifier_version": classifier_version,
        "evidence_source": evidence_source,
        "publication_types": publication_types,
        "supported_types": supported_types,
        "negative_evidence_types": negative_evidence_types,
    }


def load_paper_type_cache(path: str | Path | None) -> Dict[str, Dict[str, Any]]:
    """Load the latest valid JSONL record per arXiv ID."""

    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("paper-type cache record must be an object")
                record = validate_type_record(value)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid paper-type cache line {line_number}: {exc}"
                ) from exc
            output[record["paper_arxiv_id"]] = record
    return output


def rule_match_probability(
    type_probs: Mapping[str, Any],
    types: Sequence[str],
    logic: str,
) -> float:
    values = [float(type_probs.get(type_name, 0.0) or 0.0) for type_name in types]
    if not values:
        return 0.0
    if logic == "any":
        return max(values)
    if logic == "all":
        return min(values)
    raise ValueError(f"unsupported paper-type logic: {logic}")


def evaluate_paper_type_rules(
    record: Optional[Mapping[str, Any]],
    rules: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate native S2 prefer/avoid/exclude rules for one paper."""

    type_probs = dict((record or {}).get("type_probs") or {})
    confidence = float((record or {}).get("confidence") or 0.0)
    evidence_source = str((record or {}).get("evidence_source") or "").strip()
    if not evidence_source and record:
        evidence_source = S2_EVIDENCE_SOURCE
    default_supported = S2_PUBLICATION_TYPES if record else ()
    supported_types = set((record or {}).get("supported_types", default_supported) or [])
    default_negative = () if evidence_source == S2_EVIDENCE_SOURCE else supported_types
    negative_evidence_types = set(
        (record or {}).get("negative_evidence_types", default_negative) or []
    )
    publication_types = list((record or {}).get("publication_types") or [])
    positive_types = set(publication_types)
    exclude_known = False
    known_for_relevant_hard_filter = False
    prefer_reward = 0.0
    avoid_penalty = 0.0
    filter_actions: List[str] = []
    filter_reasons: List[str] = []
    matches: List[Dict[str, Any]] = []

    for rule in rules:
        types = list(rule.get("types") or [])
        action = str(rule.get("action") or "")
        logic = str(rule.get("logic") or "any")
        match = rule_match_probability(type_probs, types, logic)
        requested_types = set(types)
        supported_for_rule = (
            bool(requested_types & supported_types)
            if logic == "any"
            else requested_types.issubset(supported_types)
        )
        native_membership_match = (
            bool(requested_types & positive_types)
            if logic == "any"
            else bool(requested_types) and requested_types.issubset(positive_types)
        )
        matches.append(
            {
                "types": types,
                "action": action,
                "logic": logic,
                "match_probability": match,
                "native_membership_match": native_membership_match,
                "supported_by_source": supported_for_rule,
                "evidence_source": evidence_source or None,
            }
        )
        if action == "prefer":
            prefer_reward += _STRENGTH_VALUES[str(rule.get("strength") or "medium")] * match
        elif action == "avoid":
            avoid_penalty += _STRENGTH_VALUES[str(rule.get("strength") or "medium")] * match
        elif action == "exclude":
            # S2 publicationTypes is a discrete positive tag set. A hard
            # exclusion therefore needs neither a probability threshold nor a
            # confidence threshold: an exact native tag match is sufficient.
            exclude_known_for_rule = bool(record) and native_membership_match
            exclude_known = exclude_known or exclude_known_for_rule
            known_for_relevant_hard_filter = (
                known_for_relevant_hard_filter or exclude_known_for_rule
            )
            if exclude_known_for_rule:
                filter_actions.append("exclude")
                filter_reasons.append(
                    f"exclude({','.join(types)}): native S2 type match; "
                    f"source={evidence_source or 'unknown'}"
                )

    alignment = max(-1.0, min(1.0, prefer_reward - avoid_penalty))
    return {
        "paper_type_probs": type_probs,
        "paper_type_classifier_confidence": confidence,
        "paper_type_evidence_source": evidence_source or None,
        "paper_type_publication_types": publication_types,
        "paper_type_supported_types": sorted(supported_types),
        "paper_type_negative_evidence_types": sorted(negative_evidence_types),
        "paper_type_known_for_hard_filter": known_for_relevant_hard_filter,
        "paper_type_known_for_exclude_filter": exclude_known,
        "paper_type_alignment": alignment,
        "paper_type_filter_action": "exclude" if filter_actions else None,
        "paper_type_filter_reason": "; ".join(filter_reasons) or None,
        "paper_type_rule_matches": matches,
        "hard_filtered": bool(filter_actions),
    }
