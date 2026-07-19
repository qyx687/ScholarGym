#!/usr/bin/env python3
"""Query-conditioned rerank policy generation, compilation, and scoring."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from dimension_catalog import (
    CATALOG_VERSION,
    DIMENSIONS,
    DIMENSION_NAMES,
    NEGATIVE_LEVEL,
    PAPER_TYPE_ACTIONS,
    PAPER_TYPE_ALIASES,
    PAPER_TYPE_LOGICS,
    PAPER_TYPE_STRENGTHS,
    PAPER_TYPES,
    POLICY_VERSION,
    POSITIVE_LEVELS,
    PROMPT_VERSION,
    SEMANTIC_DIMENSIONS,
    intent_dimension_values,
)
from paper_type import (
    DEFAULT_EXCLUDE_THRESHOLD,
    DEFAULT_EXCLUDE_HARD_FILTER_MIN_CONFIDENCE,
    DEFAULT_REQUIRE_THRESHOLD,
    DEFAULT_REQUIRE_HARD_FILTER_MIN_CONFIDENCE,
    S2_SUPPORTED_CANONICAL_TYPES,
    evaluate_paper_type_rules,
    normalize_paper_id,
    s2_publication_types_to_record,
)


LEGACY_FORMULA_ID = "q030_sq040_intent015_path015_closed_pool_minmax_v1"
LEGACY_FEATURE_WEIGHTS = {
    "query_score_normalized": 0.30,
    "subquery_score_normalized": 0.40,
    "intent_score": 0.15,
    "path_count_normalized": 0.15,
}
DEFAULT_MIN_CONFIDENCE = 0.60
# Tuned only on ScholarGym tune100, then frozen for PASA transfer. This keeps
# query-conditioned graph/type signals as a conservative residual around the
# strong semantic baseline.
DEFAULT_SEMANTIC_MIN_MASS = 0.90
DEFAULT_NEGATIVE_WEIGHT = 0.15
DEFAULT_MAX_NEGATIVE_MASS = 0.30


class PolicyValidationError(ValueError):
    pass


@dataclass(frozen=True)
class PaperTypeRule:
    types: Tuple[str, ...]
    action: str
    logic: str
    strength: Optional[str] = None
    threshold: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "types": list(self.types),
            "action": self.action,
            "logic": self.logic,
        }
        if self.strength is not None:
            output["strength"] = self.strength
        if self.threshold is not None:
            output["threshold"] = self.threshold
        return output


@dataclass(frozen=True)
class RerankPolicy:
    policy_version: str
    query_intent: str
    weight_levels: Mapping[str, str]
    paper_type_rules: Tuple[PaperTypeRule, ...]
    confidence: float
    policy_id: str
    raw_model_output: str = ""
    used_fallback: bool = False
    fallback_reason: Optional[str] = None
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        if self.used_fallback:
            return {}
        return {
            "policy_version": self.policy_version,
            "query_intent": self.query_intent,
            "weight_levels": dict(self.weight_levels),
            "paper_type_rules": [rule.to_dict() for rule in self.paper_type_rules],
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CompiledPolicy:
    policy_id: str
    query_intent: str
    feature_weights: Mapping[str, float]
    weight_levels: Mapping[str, str]
    paper_type_rules: Tuple[PaperTypeRule, ...]
    confidence: float
    used_fallback: bool
    fallback_reason: Optional[str]
    raw_model_output: str
    paper_type_alignment_enabled: bool
    semantic_mass: float
    total_negative_mass: float
    adjustments: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "query_intent": self.query_intent,
            "feature_weights": dict(self.feature_weights),
            "weight_levels": dict(self.weight_levels),
            "paper_type_rules": [rule.to_dict() for rule in self.paper_type_rules],
            "confidence": self.confidence,
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "paper_type_alignment_enabled": self.paper_type_alignment_enabled,
            "semantic_mass": self.semantic_mass,
            "total_negative_mass": self.total_negative_mass,
            "adjustments": list(self.adjustments),
        }


def extract_first_json_object(text: str) -> Dict[str, Any]:
    """Extract the first complete JSON object, ignoring prose/Markdown wrappers."""

    start = text.find("{")
    if start < 0:
        raise PolicyValidationError("model output contains no JSON object")
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
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise PolicyValidationError(f"invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise PolicyValidationError("policy JSON must be an object")
                return value
    raise PolicyValidationError("model output contains an incomplete JSON object")


def _finite_number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise PolicyValidationError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PolicyValidationError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise PolicyValidationError(f"{name} must be in [{low}, {high}]")
    return result


def _canonical_paper_type(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if key in PAPER_TYPES:
        return key
    return PAPER_TYPE_ALIASES.get(key, value)


def validate_policy_object(
    value: Mapping[str, Any],
    *,
    policy_id: str,
    raw_model_output: str = "",
) -> RerankPolicy:
    required_keys = {
        "policy_version",
        "query_intent",
        "weight_levels",
        "paper_type_rules",
        "confidence",
    }
    if set(value) != required_keys:
        raise PolicyValidationError(
            f"policy keys must be exactly {sorted(required_keys)}; got {sorted(value)}"
        )
    if value.get("policy_version") != POLICY_VERSION:
        raise PolicyValidationError(f"policy_version must be {POLICY_VERSION}")
    query_intent = value.get("query_intent")
    if not isinstance(query_intent, str) or not query_intent.strip():
        raise PolicyValidationError("query_intent must be a non-empty string")
    if len(query_intent) > 160:
        raise PolicyValidationError("query_intent is too long")

    raw_levels = value.get("weight_levels")
    if not isinstance(raw_levels, Mapping) or set(raw_levels) != set(DIMENSION_NAMES):
        actual = sorted(raw_levels) if isinstance(raw_levels, Mapping) else type(raw_levels).__name__
        raise PolicyValidationError(
            f"weight_levels must contain exactly {sorted(DIMENSION_NAMES)}; got {actual}"
        )
    levels: Dict[str, str] = {}
    for dimension in DIMENSION_NAMES:
        level = raw_levels.get(dimension)
        if not isinstance(level, str) or level not in DIMENSIONS[dimension]["allowed_levels"]:
            raise PolicyValidationError(
                f"invalid level {level!r} for {dimension}; allowed="
                f"{DIMENSIONS[dimension]['allowed_levels']}"
            )
        levels[dimension] = level
    if all(levels[name] == "off" for name in SEMANTIC_DIMENSIONS):
        raise PolicyValidationError(
            "query_similarity and subquery_similarity cannot both be off"
        )

    raw_rules = value.get("paper_type_rules")
    if not isinstance(raw_rules, list):
        raise PolicyValidationError("paper_type_rules must be an array")
    if len(raw_rules) > 16:
        raise PolicyValidationError("paper_type_rules cannot contain more than 16 rules")
    rules: List[PaperTypeRule] = []
    for index, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, Mapping):
            raise PolicyValidationError(f"paper_type_rules[{index}] must be an object")
        allowed_rule_keys = {"types", "action", "logic", "strength", "threshold"}
        extra = set(raw_rule) - allowed_rule_keys
        if extra:
            raise PolicyValidationError(
                f"paper_type_rules[{index}] has unexpected keys: {sorted(extra)}"
            )
        raw_types = raw_rule.get("types")
        if not isinstance(raw_types, list) or not raw_types or not all(isinstance(item, str) for item in raw_types):
            raise PolicyValidationError(
                f"paper_type_rules[{index}].types must be a non-empty string array"
            )
        types = [_canonical_paper_type(item) for item in raw_types]
        if len(set(types)) != len(types):
            raise PolicyValidationError(f"paper_type_rules[{index}].types contains duplicates")
        illegal_types = set(types) - set(PAPER_TYPES)
        if illegal_types:
            raise PolicyValidationError(
                f"paper_type_rules[{index}] has unknown types: {sorted(illegal_types)}"
            )
        action = raw_rule.get("action")
        if action not in PAPER_TYPE_ACTIONS:
            raise PolicyValidationError(
                f"paper_type_rules[{index}].action must be one of {PAPER_TYPE_ACTIONS}"
            )
        logic = raw_rule.get("logic")
        if logic not in PAPER_TYPE_LOGICS:
            raise PolicyValidationError(
                f"paper_type_rules[{index}].logic must be one of {PAPER_TYPE_LOGICS}"
            )
        strength = raw_rule.get("strength")
        threshold = raw_rule.get("threshold")
        if action in {"prefer", "avoid"}:
            if strength not in PAPER_TYPE_STRENGTHS:
                raise PolicyValidationError(
                    f"paper_type_rules[{index}].strength must be one of "
                    f"{PAPER_TYPE_STRENGTHS} for {action}"
                )
            if threshold is not None:
                raise PolicyValidationError(
                    f"paper_type_rules[{index}].threshold is not allowed for {action}"
                )
        else:
            if strength is not None:
                raise PolicyValidationError(
                    f"paper_type_rules[{index}].strength is not allowed for {action}"
                )
            if threshold is not None:
                threshold = _finite_number(
                    threshold,
                    f"paper_type_rules[{index}].threshold",
                    0.0,
                    1.0,
                )
        rules.append(
            PaperTypeRule(
                types=tuple(types),
                action=str(action),
                logic=str(logic),
                strength=str(strength) if strength is not None else None,
                threshold=float(threshold) if threshold is not None else None,
            )
        )

    confidence = _finite_number(value.get("confidence"), "confidence", 0.0, 1.0)
    return RerankPolicy(
        policy_version=POLICY_VERSION,
        query_intent=query_intent.strip(),
        weight_levels=levels,
        paper_type_rules=tuple(rules),
        confidence=confidence,
        policy_id=policy_id,
        raw_model_output=raw_model_output,
    )


class RerankSkill:
    """Generate one query policy, then apply it deterministically to every pool."""

    def __init__(
        self,
        model: str,
        *,
        is_local: bool = False,
        llm_call: Optional[Callable[..., Any]] = None,
        policy_cache_path: str | Path | None = None,
        retry_cached_fallbacks: bool = False,
        paper_type_cache: Optional[Mapping[str, Mapping[str, Any]]] = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        catalog_version: str = CATALOG_VERSION,
        prompt_version: str = PROMPT_VERSION,
        semantic_min_mass: float = DEFAULT_SEMANTIC_MIN_MASS,
        negative_weight: float = DEFAULT_NEGATIVE_WEIGHT,
        max_negative_mass: float = DEFAULT_MAX_NEGATIVE_MASS,
        hard_filter_min_confidence: Optional[float] = None,
        exclude_hard_filter_min_confidence: float = DEFAULT_EXCLUDE_HARD_FILTER_MIN_CONFIDENCE,
        require_hard_filter_min_confidence: float = DEFAULT_REQUIRE_HARD_FILTER_MIN_CONFIDENCE,
        exclude_threshold: float = DEFAULT_EXCLUDE_THRESHOLD,
        require_threshold: float = DEFAULT_REQUIRE_THRESHOLD,
    ) -> None:
        self.model = model
        self.is_local = is_local
        self.llm_call = llm_call
        self.policy_cache_path = Path(policy_cache_path) if policy_cache_path else None
        self.retry_cached_fallbacks = bool(retry_cached_fallbacks)
        self.paper_type_cache = {
            normalize_paper_id(paper_id): dict(record)
            for paper_id, record in (paper_type_cache or {}).items()
            if normalize_paper_id(paper_id)
        }
        self.paper_type_supported_types = set()
        for record in self.paper_type_cache.values():
            explicit_supported = record.get("supported_types")
            if explicit_supported is not None:
                self.paper_type_supported_types.update(explicit_supported or [])
            elif str(record.get("classifier_version") or "").startswith("s2_"):
                self.paper_type_supported_types.update(S2_SUPPORTED_CANONICAL_TYPES)
            else:
                # Legacy/Qwen records predate explicit provenance metadata.
                self.paper_type_supported_types.update(PAPER_TYPES)
        self.min_confidence = _finite_number(min_confidence, "min_confidence", 0.0, 1.0)
        self.catalog_version = catalog_version
        self.prompt_version = prompt_version
        self.semantic_min_mass = _finite_number(
            semantic_min_mass, "semantic_min_mass", 0.0, 1.0
        )
        self.negative_weight = _finite_number(negative_weight, "negative_weight", 0.0, 1.0)
        self.max_negative_mass = _finite_number(
            max_negative_mass, "max_negative_mass", 0.0, 1.0
        )
        if hard_filter_min_confidence is not None:
            shared_hard_filter_confidence = _finite_number(
                hard_filter_min_confidence,
                "hard_filter_min_confidence",
                0.0,
                1.0,
            )
            exclude_hard_filter_min_confidence = shared_hard_filter_confidence
            require_hard_filter_min_confidence = shared_hard_filter_confidence
        self.exclude_hard_filter_min_confidence = _finite_number(
            exclude_hard_filter_min_confidence,
            "exclude_hard_filter_min_confidence",
            0.0,
            1.0,
        )
        self.require_hard_filter_min_confidence = _finite_number(
            require_hard_filter_min_confidence,
            "require_hard_filter_min_confidence",
            0.0,
            1.0,
        )
        self.exclude_threshold = _finite_number(
            exclude_threshold, "exclude_threshold", 0.0, 1.0
        )
        self.require_threshold = _finite_number(
            require_threshold, "require_threshold", 0.0, 1.0
        )
        self._memory_cache: Dict[str, RerankPolicy] = {}
        self._load_cache()

    def _cache_key(self, query: str) -> str:
        payload = json.dumps(
            {
                "query": str(query),
                "model": self.model,
                "catalog_version": self.catalog_version,
                "prompt_version": self.prompt_version,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _fallback_policy(
        self,
        policy_id: str,
        reason: str,
        *,
        raw_model_output: str = "",
        cache_hit: bool = False,
    ) -> RerankPolicy:
        return RerankPolicy(
            policy_version="legacy_static_v1",
            query_intent="legacy_fallback",
            weight_levels={},
            paper_type_rules=(),
            confidence=0.0,
            policy_id=policy_id,
            raw_model_output=raw_model_output,
            used_fallback=True,
            fallback_reason=reason,
            cache_hit=cache_hit,
        )

    def _load_cache(self) -> None:
        path = self.policy_cache_path
        if path is None or not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if (
                        record.get("model") != self.model
                        or record.get("catalog_version") != self.catalog_version
                        or record.get("prompt_version") != self.prompt_version
                    ):
                        continue
                    policy_id = str(record.get("policy_id") or record.get("cache_key") or "")
                    if not policy_id:
                        continue
                    if record.get("used_fallback") and self.retry_cached_fallbacks:
                        continue
                    if record.get("used_fallback"):
                        policy = self._fallback_policy(
                            policy_id,
                            str(record.get("fallback_reason") or "cached fallback"),
                            raw_model_output=str(record.get("raw_model_output") or ""),
                            cache_hit=True,
                        )
                    else:
                        policy = validate_policy_object(
                            record.get("validated_policy") or {},
                            policy_id=policy_id,
                            raw_model_output=str(record.get("raw_model_output") or ""),
                        )
                        policy = RerankPolicy(**{**policy.__dict__, "cache_hit": True})
                    self._memory_cache[policy_id] = policy
                except (json.JSONDecodeError, PolicyValidationError, TypeError, ValueError):
                    # A damaged line cannot poison the rest of an append-only cache.
                    continue

    def _append_cache(self, query: str, policy: RerankPolicy) -> None:
        path = self.policy_cache_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "policy_id": policy.policy_id,
            "cache_key": policy.policy_id,
            "query": query,
            "model": self.model,
            "catalog_version": self.catalog_version,
            "prompt_version": self.prompt_version,
            "raw_model_output": policy.raw_model_output,
            "validated_policy": policy.to_dict(),
            "used_fallback": policy.used_fallback,
            "fallback_reason": policy.fallback_reason,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    def build_prompt(self, query: str) -> str:
        dimension_lines = [
            "- query_similarity: overall topical relevance",
            "- subquery_similarity: relevance to a retrieval aspect",
            "- intent_background: candidate is used as background or foundation",
            "- intent_method: candidate method is used or extended",
            "- intent_result: candidate result is compared, supported, or discussed",
            "- path_count: structural support from multiple seed-to-paper paths",
            "- paper_type_alignment: alignment with required or excluded paper types",
        ]
        return (
            "You are a scholarly reranking policy generator.\n\n"
            "Generate one fixed reranking policy for the original research-paper query. "
            "This policy will be reused for all subqueries and candidates.\n\n"
            "Available dimensions:\n"
            + "\n".join(dimension_lines)
            + "\n\nMost queries have no structural requirement. Treat citation intent as "
            "the relationship between a seed paper and an expanded paper, NOT as the "
            "topic or contribution type of the desired paper. A generic request for "
            "papers that propose, use, analyze, compare, or evaluate a method/result "
            "does not by itself justify any citation-intent dimension. Use citation "
            "intent or path_count only when the query explicitly asks for technical "
            "foundations, origins, lineage, follow-up works, common/core papers, "
            "multiple independent support, historical development, or knowledge "
            "synthesis. Otherwise set all four graph dimensions to off.\n\n"
            "Choose levels only from: very_high, high, medium, low, off, negative. "
            "query_similarity and subquery_similarity cannot be negative or both off.\n\n"
            f"Allowed paper types: {', '.join(PAPER_TYPES)}.\n"
            "Use these exact canonical names. In particular, write survey_review, "
            "never survey or review.\n"
            "paper_type_rules MUST be a JSON array. Generate rules only when the query "
            "explicitly requests, prefers, avoids, or excludes a paper genre/type "
            "(for example survey, tutorial, benchmark, theory, empirical study, or "
            "application case study). For an ordinary topical or method search, use [] "
            "and set paper_type_alignment to off; do not infer a primary_method rule "
            "merely because the query asks for methods. Each array item uses the plural "
            "key types and has "
            "exactly types, action, logic, plus the action-specific field. "
            "Paper type actions: require, prefer, avoid, exclude. Logic: any, all. "
            "prefer/avoid items require strength from very_high, high, medium, low; "
            "require/exclude items may use threshold as a number in [0,1] and MUST "
            "NOT use strength. Never use a rules wrapper or a singular type key.\n\n"
            "Return one valid JSON object only, with exactly these top-level keys: "
            "policy_version, query_intent, weight_levels, paper_type_rules, confidence. "
            f"policy_version must be {POLICY_VERSION}. weight_levels must contain every "
            f"catalog dimension exactly once: {', '.join(DIMENSION_NAMES)}. "
            "confidence MUST be a JSON number from 0 to 1, never a word.\n\n"
            "Exact shape example for an ordinary topical search (shape is mandatory; "
            "choose different levels only when justified):\n"
            "{\"policy_version\":\"dynamic_rerank_v1\","
            "\"query_intent\":\"topical_search\",\"weight_levels\":{"
            "\"query_similarity\":\"high\",\"subquery_similarity\":\"very_high\","
            "\"intent_background\":\"off\",\"intent_method\":\"off\","
            "\"intent_result\":\"off\",\"path_count\":\"off\","
            "\"paper_type_alignment\":\"off\"},\"paper_type_rules\":[],"
            "\"confidence\":0.9}\n\n"
            "Original query:\n"
            + str(query)
        )

    def _call(self, prompt: str) -> str:
        if self.llm_call is None:
            from api import _call_llm

            call = _call_llm
        else:
            call = self.llm_call
        output = call(
            prompt,
            self.model,
            {"max_tokens": 4096, "temperature": 0, "top_p": 1, "stream": False},
            self.is_local,
            False,
        )
        if isinstance(output, tuple):
            output = output[-1]
        return str(output or "")

    def _parse_policy(self, raw_output: str, policy_id: str) -> RerankPolicy:
        return validate_policy_object(
            extract_first_json_object(raw_output),
            policy_id=policy_id,
            raw_model_output=raw_output,
        )

    def build_policy(self, query: str) -> RerankPolicy:
        query = str(query or "").strip()
        if not query:
            raise ValueError("query must be non-empty")
        policy_id = self._cache_key(query)
        if policy_id in self._memory_cache:
            return self._memory_cache[policy_id]

        raw_output = ""
        try:
            raw_output = self._call(self.build_prompt(query))
            try:
                policy = self._parse_policy(raw_output, policy_id)
            except PolicyValidationError as initial_error:
                repair_prompt = (
                    self.build_prompt(query)
                    + "\n\nYour previous response was invalid. Repair it and return exactly "
                    "one valid JSON object and no prose. Preserve the intended meaning "
                    "while obeying every schema requirement above.\n\n"
                    f"Validation error: {initial_error}\n\n"
                    f"Invalid response:\n{raw_output}"
                )
                repaired_output = self._call(repair_prompt)
                combined_output = raw_output + "\n\n--- JSON REPAIR ---\n" + repaired_output
                raw_output = combined_output
                policy = self._parse_policy(repaired_output, policy_id)
                policy = RerankPolicy(
                    **{
                        **policy.__dict__,
                        "raw_model_output": combined_output,
                    }
                )
            if policy.confidence < self.min_confidence:
                policy = self._fallback_policy(
                    policy_id,
                    f"policy confidence {policy.confidence:.4f} < {self.min_confidence:.4f}",
                    raw_model_output=policy.raw_model_output,
                )
        except Exception as exc:
            policy = self._fallback_policy(
                policy_id,
                f"policy generation failed: {type(exc).__name__}: {exc}",
                raw_model_output=raw_output,
            )

        self._memory_cache[policy_id] = policy
        self._append_cache(query, policy)
        return policy

    def legacy_policy_for_query(
        self,
        query: str,
        reason: str = "dynamic policy generation disabled",
    ) -> RerankPolicy:
        """Return a query-keyed legacy policy without calling or mutating a cache."""

        query = str(query or "").strip()
        if not query:
            raise ValueError("query must be non-empty")
        return self._fallback_policy(self._cache_key(query), reason)

    def compile_weights(
        self,
        policy: RerankPolicy,
        *,
        paper_type_available: Optional[bool] = None,
    ) -> CompiledPolicy:
        if policy.used_fallback:
            return CompiledPolicy(
                policy_id=policy.policy_id,
                query_intent=policy.query_intent,
                feature_weights=dict(LEGACY_FEATURE_WEIGHTS),
                weight_levels={},
                paper_type_rules=(),
                confidence=policy.confidence,
                used_fallback=True,
                fallback_reason=policy.fallback_reason,
                raw_model_output=policy.raw_model_output,
                paper_type_alignment_enabled=False,
                semantic_mass=0.70,
                total_negative_mass=0.0,
                adjustments=(),
            )

        levels = dict(policy.weight_levels)
        adjustments: List[str] = []
        type_available = (
            bool(self.paper_type_cache)
            if paper_type_available is None
            else bool(paper_type_available)
        )
        supported_types = set(self.paper_type_supported_types)
        soft_rules = [
            rule
            for rule in policy.paper_type_rules
            if rule.action in {"prefer", "avoid"}
        ]

        def rule_is_supported(rule: PaperTypeRule) -> bool:
            requested = set(rule.types)
            if rule.logic == "any":
                return bool(requested & supported_types)
            return bool(requested) and requested.issubset(supported_types)

        soft_type_available = (
            type_available
            and bool(supported_types)
            and any(rule_is_supported(rule) for rule in soft_rules)
        )
        if not soft_type_available:
            if levels.get("paper_type_alignment") != "off":
                if not type_available:
                    reason = "paper_type_alignment_forced_off_no_cache"
                elif not soft_rules:
                    reason = "paper_type_alignment_forced_off_no_soft_rules"
                else:
                    reason = "paper_type_alignment_forced_off_unsupported_rules"
                adjustments.append(reason)
            levels["paper_type_alignment"] = "off"

        positive_raw = {
            name: POSITIVE_LEVELS.get(level, 0.0)
            for name, level in levels.items()
            if level != NEGATIVE_LEVEL
        }
        total_positive = sum(positive_raw.values())
        semantic_raw = sum(positive_raw.get(name, 0.0) for name in SEMANTIC_DIMENSIONS)
        if total_positive <= 0.0 or semantic_raw <= 0.0:
            raise PolicyValidationError("compiled policy must have positive semantic weight")
        weights = {
            name: value / total_positive
            for name, value in positive_raw.items()
        }
        semantic_mass = sum(weights.get(name, 0.0) for name in SEMANTIC_DIMENSIONS)
        if semantic_mass + 1e-12 < self.semantic_min_mass:
            nonsemantic_mass = 1.0 - semantic_mass
            for name in SEMANTIC_DIMENSIONS:
                weights[name] = (
                    weights.get(name, 0.0) * self.semantic_min_mass / semantic_mass
                )
            if nonsemantic_mass > 0.0:
                target_nonsemantic = 1.0 - self.semantic_min_mass
                for name in weights:
                    if name not in SEMANTIC_DIMENSIONS:
                        weights[name] = weights[name] * target_nonsemantic / nonsemantic_mass
            semantic_mass = self.semantic_min_mass
            adjustments.append("semantic_mass_raised_to_minimum")

        negative_names = [name for name, level in levels.items() if level == NEGATIVE_LEVEL]
        nominal_negative_mass = self.negative_weight * len(negative_names)
        negative_scale = 1.0
        if nominal_negative_mass > self.max_negative_mass and nominal_negative_mass > 0.0:
            negative_scale = self.max_negative_mass / nominal_negative_mass
            adjustments.append("negative_weights_scaled_to_mass_cap")
        for name in negative_names:
            weights[name] = -self.negative_weight * negative_scale
        for name in DIMENSION_NAMES:
            weights.setdefault(name, 0.0)
        total_negative_mass = sum(abs(value) for value in weights.values() if value < 0.0)
        return CompiledPolicy(
            policy_id=policy.policy_id,
            query_intent=policy.query_intent,
            feature_weights=weights,
            weight_levels=levels,
            paper_type_rules=policy.paper_type_rules,
            confidence=policy.confidence,
            used_fallback=False,
            fallback_reason=None,
            raw_model_output=policy.raw_model_output,
            paper_type_alignment_enabled=(
                levels.get("paper_type_alignment") != "off" and soft_type_available
            ),
            semantic_mass=semantic_mass,
            total_negative_mass=total_negative_mass,
            adjustments=tuple(adjustments),
        )

    def build_and_compile(self, query: str) -> Tuple[RerankPolicy, CompiledPolicy]:
        policy = self.build_policy(query)
        return policy, self.compile_weights(policy)

    @staticmethod
    def legacy_compiled_policy(policy_id: str = "legacy-static") -> CompiledPolicy:
        return CompiledPolicy(
            policy_id=policy_id,
            query_intent="legacy_static",
            feature_weights=dict(LEGACY_FEATURE_WEIGHTS),
            weight_levels={},
            paper_type_rules=(),
            confidence=1.0,
            used_fallback=True,
            fallback_reason="explicit legacy comparison",
            raw_model_output="",
            paper_type_alignment_enabled=False,
            semantic_mass=0.70,
            total_negative_mass=0.0,
            adjustments=(),
        )

    def _type_record_for_candidate(self, candidate: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        paper_id = normalize_paper_id(candidate.get("paper_arxiv_id"))
        provided_probs = candidate.get("paper_type_probs")
        provided_confidence = (
            candidate.get("paper_type_classifier_confidence")
            if candidate.get("paper_type_classifier_confidence") is not None
            else candidate.get("paper_type_confidence")
        )
        if isinstance(provided_probs, Mapping) and (
            provided_probs or float(provided_confidence or 0.0) > 0.0
        ):
            provided_record = {
                "paper_arxiv_id": paper_id,
                "type_probs": dict(provided_probs),
                "confidence": float(
                    provided_confidence or 0.0
                ),
                "evidence_source": candidate.get("paper_type_evidence_source"),
                "publication_types": candidate.get(
                    "paper_type_publication_types"
                )
                or [],
            }
            for candidate_key, record_key in (
                ("paper_type_supported_types", "supported_types"),
                (
                    "paper_type_negative_evidence_types",
                    "negative_evidence_types",
                ),
            ):
                if candidate.get(candidate_key) is not None:
                    provided_record[record_key] = candidate.get(candidate_key)
            return provided_record
        cached = self.paper_type_cache.get(paper_id)
        if cached:
            return dict(cached)
        if isinstance(candidate.get("s2_publication_types"), (list, tuple, set)):
            return s2_publication_types_to_record(
                paper_id,
                candidate.get("s2_publication_types"),
                resolved=True,
            )
        if isinstance(provided_probs, Mapping):
            return {
                "paper_arxiv_id": paper_id,
                "type_probs": dict(provided_probs),
                "confidence": 0.0,
            }
        return None

    @staticmethod
    def _feature_values(candidate: Mapping[str, Any]) -> Dict[str, float]:
        values = {
            "query_similarity": float(candidate.get("query_score_normalized") or 0.0),
            "subquery_similarity": float(candidate.get("subquery_score_normalized") or 0.0),
            "path_count": float(candidate.get("path_count_normalized") or 0.0),
        }
        values.update(intent_dimension_values(candidate.get("intent_labels") or []))
        return values

    @staticmethod
    def _original_rank(candidate: Mapping[str, Any]) -> int:
        for key in (
            "observed_retrieval_rank",
            "observed_retrieval_rank_after_exclusion",
            "materialization_order_rank",
            "retrieval_rank",
        ):
            value = candidate.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return 10**12

    def score_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        policy: CompiledPolicy,
    ) -> List[Dict[str, Any]]:
        """Score one closed pool and retain hard-filtered rows for artifacts."""

        scored: List[Dict[str, Any]] = []
        rule_dicts = [rule.to_dict() for rule in policy.paper_type_rules]
        for original in candidates:
            row = dict(original)
            paper_id = normalize_paper_id(row.get("paper_arxiv_id"))
            if not paper_id:
                raise ValueError("candidate is missing paper_arxiv_id")
            row["paper_arxiv_id"] = paper_id
            if policy.used_fallback:
                contributions = {
                    "query_similarity": LEGACY_FEATURE_WEIGHTS["query_score_normalized"]
                    * float(row.get("query_score_normalized") or 0.0),
                    "subquery_similarity": LEGACY_FEATURE_WEIGHTS["subquery_score_normalized"]
                    * float(row.get("subquery_score_normalized") or 0.0),
                    "intent_score": LEGACY_FEATURE_WEIGHTS["intent_score"]
                    * float(row.get("intent_score") or 0.0),
                    "path_count": LEGACY_FEATURE_WEIGHTS["path_count_normalized"]
                    * float(row.get("path_count_normalized") or 0.0),
                    "paper_type_alignment": 0.0,
                    "paper_type_soft_penalty": 0.0,
                }
                type_result = {
                    "paper_type_probs": {},
                    "paper_type_classifier_confidence": 0.0,
                    "paper_type_evidence_source": None,
                    "paper_type_publication_types": [],
                    "paper_type_supported_types": [],
                    "paper_type_negative_evidence_types": [],
                    "paper_type_known_for_hard_filter": False,
                    "paper_type_alignment": 0.0,
                    "paper_type_soft_penalty": 0.0,
                    "paper_type_filter_action": None,
                    "paper_type_filter_reason": None,
                    "paper_type_rule_matches": [],
                    "hard_filtered": False,
                }
            else:
                feature_values = self._feature_values(row)
                type_result = evaluate_paper_type_rules(
                    self._type_record_for_candidate(row),
                    rule_dicts,
                    exclude_hard_filter_min_confidence=(
                        self.exclude_hard_filter_min_confidence
                    ),
                    require_hard_filter_min_confidence=(
                        self.require_hard_filter_min_confidence
                    ),
                    exclude_threshold=self.exclude_threshold,
                    require_threshold=self.require_threshold,
                )
                contributions = {
                    name: float(policy.feature_weights.get(name, 0.0)) * value
                    for name, value in feature_values.items()
                }
                contributions["paper_type_alignment"] = (
                    float(policy.feature_weights.get("paper_type_alignment", 0.0))
                    * float(type_result["paper_type_alignment"])
                )
                contributions["paper_type_soft_penalty"] = (
                    abs(float(policy.feature_weights.get("paper_type_alignment", 0.0)))
                    * float(type_result["paper_type_soft_penalty"])
                )
                row.update(feature_values)
            score = sum(contributions.values())
            row.update(type_result)
            row.update(
                {
                    "rerank_policy_id": policy.policy_id,
                    "dynamic_rerank_enabled": not policy.used_fallback,
                    "rerank_used_fallback": policy.used_fallback,
                    "compiled_feature_weights": dict(policy.feature_weights),
                    "component_contributions": contributions,
                    "rerank_formula_id": (
                        LEGACY_FORMULA_ID if policy.used_fallback else POLICY_VERSION
                    ),
                    "rerank_score": score,
                }
            )
            scored.append(row)

        def sort_key(row: Mapping[str, Any]) -> Tuple[int, float, int, int, str]:
            return (
                int(bool(row.get("hard_filtered"))),
                -float(row.get("rerank_score") or 0.0),
                -int(bool(row.get("is_seed"))),
                self._original_rank(row),
                str(row.get("paper_arxiv_id") or ""),
            )

        ordered = sorted(scored, key=sort_key)
        eligible_rank = 0
        for artifact_rank, row in enumerate(ordered, start=1):
            row["artifact_rank"] = artifact_rank
            if row.get("hard_filtered"):
                row["rerank_rank"] = None
            else:
                eligible_rank += 1
                row["rerank_rank"] = eligible_rank
        return ordered

    def artifact_record(
        self,
        *,
        query_id: str,
        original_query: str,
        policy: RerankPolicy,
        compiled: CompiledPolicy,
    ) -> Dict[str, Any]:
        return {
            "query_id": query_id,
            "original_query": original_query,
            "rerank_policy_id": policy.policy_id,
            "raw_model_output": policy.raw_model_output,
            "validated_policy": policy.to_dict(),
            "compiled_weights": dict(compiled.feature_weights),
            "compiled_policy": compiled.to_dict(),
            "used_fallback": policy.used_fallback,
            "fallback_reason": policy.fallback_reason,
            "cache_hit": policy.cache_hit,
            "model": self.model,
            "catalog_version": self.catalog_version,
            "prompt_version": self.prompt_version,
        }
