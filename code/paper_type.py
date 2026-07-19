#!/usr/bin/env python3
"""Query-independent paper-type classification and rule evaluation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from dimension_catalog import (
    PAPER_TYPE_ACTIONS,
    PAPER_TYPE_LOGICS,
    PAPER_TYPE_STRENGTHS,
    PAPER_TYPES,
)


CLASSIFIER_VERSION = "qwen30b_paper_type_v1"
S2_CLASSIFIER_VERSION = "s2_publication_types_v1"
S2_EVIDENCE_SOURCE = "semantic_scholar"
QWEN_EVIDENCE_SOURCE = "qwen"
DEFAULT_HARD_FILTER_MIN_CONFIDENCE = 0.70
DEFAULT_EXCLUDE_HARD_FILTER_MIN_CONFIDENCE = 0.80
DEFAULT_REQUIRE_HARD_FILTER_MIN_CONFIDENCE = 0.95
DEFAULT_EXCLUDE_THRESHOLD = 0.80
DEFAULT_REQUIRE_THRESHOLD = 0.55
MAX_UNKNOWN_REQUIRE_PENALTY = 0.10

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

# S2 publication types are bibliographic metadata, not a full functional-role
# taxonomy. Only mappings with a defensible semantic relationship are exposed
# to the existing canonical catalog. Dataset deliberately does not map to
# dataset_benchmark: a data record is not necessarily a benchmark paper.
S2_CANONICAL_TYPE_SCORES = {
    "Review": {"survey_review": 1.0},
    "MetaAnalysis": {"survey_review": 0.95},
    "CaseReport": {"application_case_study": 1.0},
    "ClinicalTrial": {"empirical_study": 1.0},
    "Study": {"empirical_study": 0.85},
    "Editorial": {"position_perspective": 0.90},
    "LettersAndComments": {"position_perspective": 0.80},
}
S2_SUPPORTED_CANONICAL_TYPES = tuple(
    sorted(
        {
            canonical_type
            for mapping in S2_CANONICAL_TYPE_SCORES.values()
            for canonical_type in mapping
        }
    )
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


def _validated_canonical_type_list(value: Any, field: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple, set)) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    output = []
    for item in value:
        type_name = str(item or "").strip()
        if type_name not in PAPER_TYPES:
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
    """Convert S2 bibliographic types into positive-only canonical evidence."""

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
    probs: Dict[str, float] = {}
    for publication_type in normalized_types:
        for type_name, score in S2_CANONICAL_TYPE_SCORES.get(
            publication_type, {}
        ).items():
            probs[type_name] = max(probs.get(type_name, 0.0), float(score))
    return validate_type_record(
        {
            "paper_arxiv_id": paper_arxiv_id,
            "type_probs": probs,
            "confidence": 1.0 if resolved else 0.0,
            "classifier_version": S2_CLASSIFIER_VERSION,
            "evidence_source": S2_EVIDENCE_SOURCE,
            "publication_types": normalized_types,
            "supported_types": list(S2_SUPPORTED_CANONICAL_TYPES),
            # Absence from S2 is not reliable negative evidence. This prevents
            # an untagged paper from failing a require rule.
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
    raw_probs = value.get("type_probs")
    if not isinstance(raw_probs, Mapping):
        raise ValueError("type_probs must be an object")
    illegal = set(raw_probs) - set(PAPER_TYPES)
    if illegal:
        raise ValueError(f"unknown paper types: {sorted(illegal)}")
    probs = {
        str(name): _finite_probability(probability, f"type_probs.{name}")
        for name, probability in raw_probs.items()
    }
    confidence = _finite_probability(value.get("confidence"), "confidence")
    classifier_version = str(
        value.get("classifier_version") or CLASSIFIER_VERSION
    )
    evidence_source = str(value.get("evidence_source") or "").strip()
    if not evidence_source:
        evidence_source = (
            S2_EVIDENCE_SOURCE
            if classifier_version.startswith("s2_")
            else QWEN_EVIDENCE_SOURCE
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
    default_supported = (
        S2_SUPPORTED_CANONICAL_TYPES
        if evidence_source == S2_EVIDENCE_SOURCE
        else PAPER_TYPES
    )
    supported_types = _validated_canonical_type_list(
        value.get("supported_types", default_supported), "supported_types"
    )
    default_negative = [] if evidence_source == S2_EVIDENCE_SOURCE else supported_types
    negative_evidence_types = _validated_canonical_type_list(
        value.get("negative_evidence_types", default_negative),
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
    *,
    hard_filter_min_confidence: Optional[float] = None,
    exclude_hard_filter_min_confidence: float = DEFAULT_EXCLUDE_HARD_FILTER_MIN_CONFIDENCE,
    require_hard_filter_min_confidence: float = DEFAULT_REQUIRE_HARD_FILTER_MIN_CONFIDENCE,
    exclude_threshold: float = DEFAULT_EXCLUDE_THRESHOLD,
    require_threshold: float = DEFAULT_REQUIRE_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate general require/prefer/avoid/exclude rules for one paper."""

    type_probs = dict((record or {}).get("type_probs") or {})
    confidence = float((record or {}).get("confidence") or 0.0)
    classifier_version = str((record or {}).get("classifier_version") or "")
    evidence_source = str((record or {}).get("evidence_source") or "").strip()
    if not evidence_source and record:
        evidence_source = (
            S2_EVIDENCE_SOURCE
            if classifier_version.startswith("s2_")
            else QWEN_EVIDENCE_SOURCE
        )
    default_supported = (
        S2_SUPPORTED_CANONICAL_TYPES
        if evidence_source == S2_EVIDENCE_SOURCE
        else (PAPER_TYPES if record else ())
    )
    supported_types = set((record or {}).get("supported_types", default_supported) or [])
    default_negative = () if evidence_source == S2_EVIDENCE_SOURCE else supported_types
    negative_evidence_types = set(
        (record or {}).get("negative_evidence_types", default_negative) or []
    )
    publication_types = list((record or {}).get("publication_types") or [])
    # A single threshold remains supported for callers of the v1 helper. New
    # rerank code uses action-specific thresholds because a false negative on
    # ``require`` is more destructive than a missed ``exclude``.
    if hard_filter_min_confidence is not None:
        exclude_hard_filter_min_confidence = hard_filter_min_confidence
        require_hard_filter_min_confidence = hard_filter_min_confidence
    exclude_confident = bool(record) and confidence >= exclude_hard_filter_min_confidence
    require_confident = bool(record) and confidence >= require_hard_filter_min_confidence
    exclude_known = False
    require_known = False
    known_for_relevant_hard_filter = False
    prefer_reward = 0.0
    avoid_penalty = 0.0
    soft_penalty = 0.0
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
        # To prove a require violation conservatively, every requested type
        # must have a source that provides reliable negative evidence. S2 is
        # positive-only, while the Qwen multi-label classifier is closed-world.
        negative_known_for_rule = bool(requested_types) and requested_types.issubset(
            negative_evidence_types
        )
        matches.append(
            {
                "types": types,
                "action": action,
                "logic": logic,
                "match_probability": match,
                "supported_by_source": supported_for_rule,
                "negative_evidence_known": negative_known_for_rule,
                "evidence_source": evidence_source or None,
            }
        )
        if action == "prefer":
            prefer_reward += _STRENGTH_VALUES[str(rule.get("strength") or "medium")] * match
        elif action == "avoid":
            avoid_penalty += _STRENGTH_VALUES[str(rule.get("strength") or "medium")] * match
        elif action == "exclude":
            exclude_known_for_rule = (
                exclude_confident and supported_for_rule
            )
            exclude_known = exclude_known or exclude_known_for_rule
            known_for_relevant_hard_filter = (
                known_for_relevant_hard_filter or exclude_known_for_rule
            )
            threshold = float(rule.get("threshold", exclude_threshold))
            if exclude_known_for_rule and match >= threshold:
                filter_actions.append("exclude")
                filter_reasons.append(
                    f"exclude({','.join(types)}): match={match:.4f} >= "
                    f"{threshold:.4f}; source={evidence_source or 'unknown'}"
                )
        elif action == "require":
            require_known_for_rule = (
                require_confident and negative_known_for_rule
            )
            require_known = require_known or require_known_for_rule
            known_for_relevant_hard_filter = (
                known_for_relevant_hard_filter or require_known_for_rule
            )
            threshold = float(rule.get("threshold", require_threshold))
            if require_known_for_rule and match < threshold:
                filter_actions.append("require")
                filter_reasons.append(
                    f"require({','.join(types)}): match={match:.4f} < {threshold:.4f}"
                )
            elif not require_known_for_rule and match < threshold:
                # Unknown classifications never cause a hard drop.  The bounded
                # soft penalty makes unresolved requirements visible in scoring.
                shortfall = (threshold - match) / max(threshold, 1e-12)
                soft_penalty = min(
                    soft_penalty,
                    -MAX_UNKNOWN_REQUIRE_PENALTY * min(1.0, shortfall),
                )

    alignment = max(-1.0, min(1.0, prefer_reward - avoid_penalty))
    action = None
    if filter_actions:
        action = "exclude" if "exclude" in filter_actions else "require"
    return {
        "paper_type_probs": type_probs,
        "paper_type_classifier_confidence": confidence,
        "paper_type_evidence_source": evidence_source or None,
        "paper_type_publication_types": publication_types,
        "paper_type_supported_types": sorted(supported_types),
        "paper_type_negative_evidence_types": sorted(negative_evidence_types),
        "paper_type_known_for_hard_filter": known_for_relevant_hard_filter,
        "paper_type_known_for_exclude_filter": exclude_known,
        "paper_type_known_for_require_filter": require_known,
        "paper_type_alignment": alignment,
        "paper_type_soft_penalty": soft_penalty,
        "paper_type_filter_action": action,
        "paper_type_filter_reason": "; ".join(filter_reasons) or None,
        "paper_type_rule_matches": matches,
        "hard_filtered": bool(filter_actions),
    }


def _extract_first_json_value(text: str, opening: str, closing: str) -> Any:
    start = text.find(opening)
    if start < 0:
        raise ValueError("model output contains no JSON value")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("model output contains incomplete JSON")


class PaperTypeClassifier:
    """Batch Qwen classifier whose prompt contains title and abstract only."""

    def __init__(
        self,
        model: str,
        *,
        is_local: bool = False,
        llm_call: Optional[Callable[..., Any]] = None,
        classifier_version: str = CLASSIFIER_VERSION,
    ) -> None:
        self.model = model
        self.is_local = is_local
        self.llm_call = llm_call
        self.classifier_version = classifier_version

    def _call(self, prompt: str) -> str:
        if self.llm_call is None:
            from api import _call_llm

            call = _call_llm
        else:
            call = self.llm_call
        output = call(
            prompt,
            self.model,
            {"max_tokens": 8192, "temperature": 0, "top_p": 1, "stream": False},
            self.is_local,
            False,
        )
        if isinstance(output, tuple):
            output = output[-1]
        return str(output or "")

    def _prompt(self, papers: Sequence[Mapping[str, Any]]) -> str:
        payload = [
            {
                "paper_arxiv_id": normalize_paper_id(paper.get("paper_arxiv_id")),
                "title": str(paper.get("title") or ""),
                "abstract": str(paper.get("abstract") or ""),
            }
            for paper in papers
        ]
        return (
            "You are a scholarly paper-type classifier. Classify every paper "
            "independently of any search query. Use only its title and abstract. "
            "Types are multi-label probabilities and do not need to sum to one.\n\n"
            f"Allowed types: {', '.join(PAPER_TYPES)}\n"
            "Return one JSON array only. Each item must contain exactly "
            "paper_arxiv_id, type_probs, confidence, and classifier_version. "
            f"classifier_version must be {self.classifier_version}. "
            "All probabilities and confidence values must be in [0, 1].\n\n"
            "Papers:\n"
            + json.dumps(payload, ensure_ascii=False)
        )

    def _parse(self, raw_output: str, expected_ids: Iterable[str]) -> List[Dict[str, Any]]:
        try:
            value = _extract_first_json_value(raw_output, "[", "]")
        except ValueError:
            single = _extract_first_json_value(raw_output, "{", "}")
            value = single.get("papers") if isinstance(single, Mapping) else None
        if not isinstance(value, list):
            raise ValueError("paper-type response must be a JSON array")
        records = []
        model_keys = {
            "paper_arxiv_id",
            "type_probs",
            "confidence",
            "classifier_version",
        }
        for item in value:
            if not isinstance(item, Mapping):
                continue
            if set(item) != model_keys:
                raise ValueError(
                    "paper-type response items must contain exactly "
                    f"{sorted(model_keys)}"
                )
            # Provider provenance is controlled by the caller, never trusted
            # from model-generated text. Qwen is a closed-world classifier over
            # the full canonical taxonomy, so it supplies both positive and
            # negative evidence for every catalog type.
            normalized = dict(item)
            normalized.update(
                {
                    "classifier_version": self.classifier_version,
                    "evidence_source": QWEN_EVIDENCE_SOURCE,
                    "publication_types": [],
                    "supported_types": list(PAPER_TYPES),
                    "negative_evidence_types": list(PAPER_TYPES),
                }
            )
            records.append(validate_type_record(normalized))
        expected = set(expected_ids)
        actual = {record["paper_arxiv_id"] for record in records}
        if actual != expected or len(actual) != len(records):
            raise ValueError(
                f"paper-type response IDs mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        return records

    def classify_batch(self, papers: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        if not papers:
            return []
        expected_ids = [normalize_paper_id(paper.get("paper_arxiv_id")) for paper in papers]
        if not all(expected_ids) or len(set(expected_ids)) != len(expected_ids):
            raise ValueError("paper batch must contain unique, non-empty paper_arxiv_id values")
        prompt = self._prompt(papers)
        raw_output = self._call(prompt)
        try:
            return self._parse(raw_output, expected_ids)
        except ValueError as exc:
            repair_prompt = (
                "Repair the following paper-type classifier response. Return only a valid "
                "JSON array that follows the original schema. Do not add or remove paper IDs.\n"
                f"Validation error: {exc}\n"
                f"Expected IDs: {json.dumps(expected_ids)}\n"
                f"Invalid response:\n{raw_output}"
            )
            repaired = self._call(repair_prompt)
            return self._parse(repaired, expected_ids)
