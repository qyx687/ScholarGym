import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from dimension_catalog import DIMENSION_NAMES, PAPER_TYPES, intent_dimension_values
from paper_type import (
    CLASSIFIER_VERSION,
    PaperTypeClassifier,
    evaluate_paper_type_rules,
    s2_publication_types_to_record,
)
from rerank_skill import (
    LEGACY_FEATURE_WEIGHTS,
    PolicyValidationError,
    RerankSkill,
    validate_policy_object,
)


def valid_policy(**overrides):
    value = {
        "policy_version": "dynamic_rerank_v1",
        "query_intent": "method_search",
        "weight_levels": {
            "query_similarity": "high",
            "subquery_similarity": "very_high",
            "intent_background": "off",
            "intent_method": "medium",
            "intent_result": "low",
            "path_count": "off",
            "paper_type_alignment": "high",
        },
        "paper_type_rules": [
            {
                "types": ["primary_method"],
                "action": "prefer",
                "logic": "any",
                "strength": "high",
            }
        ],
        "confidence": 0.9,
    }
    value.update(overrides)
    return value


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []

    def __call__(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        if not self.outputs:
            raise AssertionError("unexpected LLM call")
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def test_qwen_paper_type_classifier_controls_provenance_and_full_taxonomy():
    raw = json.dumps(
        [
            {
                "paper_arxiv_id": "2001.00001",
                "type_probs": {"dataset_benchmark": 0.95},
                "confidence": 0.96,
                "classifier_version": "model-invented-version",
            }
        ]
    )
    fake = FakeLLM([raw])
    classifier = PaperTypeClassifier("qwen-test", llm_call=fake)

    records = classifier.classify_batch(
        [
            {
                "paper_arxiv_id": "2001.00001",
                "title": "A benchmark",
                "abstract": "A dataset and evaluation benchmark.",
            }
        ]
    )

    assert records[0]["classifier_version"] == CLASSIFIER_VERSION
    assert records[0]["evidence_source"] == "qwen"
    assert set(records[0]["supported_types"]) == set(PAPER_TYPES)
    assert set(records[0]["negative_evidence_types"]) == set(PAPER_TYPES)


def test_markdown_wrapped_policy_is_parsed_and_cached_once(tmp_path):
    raw = "Here is the policy:\n```json\n" + json.dumps(valid_policy()) + "\n```\nextra"
    fake = FakeLLM([raw])
    skill = RerankSkill(
        "qwen-test",
        llm_call=fake,
        policy_cache_path=tmp_path / "policies.jsonl",
    )

    first = skill.build_policy("find primary methods")
    second = skill.build_policy("find primary methods")

    assert first.policy_id == second.policy_id
    assert not first.used_fallback
    assert len(fake.prompts) == 1
    assert first.to_dict() == valid_policy()


def test_invalid_json_gets_exactly_one_repair_attempt():
    fake = FakeLLM(["```json\n{bad}\n```", json.dumps(valid_policy())])
    skill = RerankSkill("qwen-test", llm_call=fake)

    policy = skill.build_policy("query")

    assert not policy.used_fallback
    assert len(fake.prompts) == 2
    assert "JSON REPAIR" in policy.raw_model_output


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["weight_levels"].update({"made_up_dimension": "high"}),
        lambda value: value["paper_type_rules"][0].update({"types": ["survey_magic"]}),
        lambda value: value["paper_type_rules"][0].update({"action": "boost"}),
    ],
)
def test_illegal_dimension_type_and_action_are_rejected(mutate):
    value = valid_policy()
    mutate(value)

    with pytest.raises(PolicyValidationError):
        validate_policy_object(value, policy_id="p")


def test_unambiguous_paper_type_alias_is_canonicalized():
    value = valid_policy(
        paper_type_rules=[
            {"types": ["survey"], "action": "exclude", "logic": "any"}
        ]
    )

    policy = validate_policy_object(value, policy_id="p")

    assert policy.paper_type_rules[0].types == ("survey_review",)


def test_low_confidence_automatically_falls_back_to_legacy():
    fake = FakeLLM([json.dumps(valid_policy(confidence=0.59))])
    skill = RerankSkill("qwen-test", llm_call=fake, min_confidence=0.60)

    policy = skill.build_policy("query")
    compiled = skill.compile_weights(policy)

    assert policy.used_fallback
    assert compiled.used_fallback
    assert compiled.feature_weights == LEGACY_FEATURE_WEIGHTS


def test_cached_fallback_can_be_explicitly_retried(tmp_path):
    cache = tmp_path / "policies.jsonl"
    first_llm = FakeLLM([json.dumps(valid_policy(confidence=0.59))])
    first = RerankSkill(
        "qwen-test",
        llm_call=first_llm,
        min_confidence=0.60,
        policy_cache_path=cache,
    ).build_policy("query")
    retry_llm = FakeLLM([json.dumps(valid_policy(confidence=0.95))])
    retried = RerankSkill(
        "qwen-test",
        llm_call=retry_llm,
        min_confidence=0.60,
        policy_cache_path=cache,
        retry_cached_fallbacks=True,
    ).build_policy("query")

    assert first.used_fallback
    assert not retried.used_fallback
    assert len(retry_llm.prompts) == 1


def test_semantic_mass_positive_normalization_and_negative_cap():
    value = valid_policy(
        weight_levels={
            "query_similarity": "low",
            "subquery_similarity": "low",
            "intent_background": "negative",
            "intent_method": "negative",
            "intent_result": "negative",
            "path_count": "negative",
            "paper_type_alignment": "off",
        },
        paper_type_rules=[],
    )
    policy = validate_policy_object(value, policy_id="p")
    skill = RerankSkill("qwen-test")

    compiled = skill.compile_weights(policy, paper_type_available=False)

    positive_sum = sum(weight for weight in compiled.feature_weights.values() if weight > 0)
    negative_mass = sum(-weight for weight in compiled.feature_weights.values() if weight < 0)
    assert positive_sum == pytest.approx(1.0)
    assert compiled.semantic_mass >= 0.40
    assert negative_mass == pytest.approx(0.30)
    assert all(
        compiled.feature_weights[name] == pytest.approx(-0.075)
        for name in ("intent_background", "intent_method", "intent_result", "path_count")
    )


def test_methodology_maps_to_intent_method():
    assert intent_dimension_values(["methodology", "background"]) == {
        "intent_background": 1.0,
        "intent_method": 1.0,
        "intent_result": 0.0,
    }


def test_require_prefer_avoid_exclude_and_hard_filter_confidence():
    rules = [
        {"types": ["primary_method"], "action": "require", "logic": "any"},
        {
            "types": ["primary_method"],
            "action": "prefer",
            "logic": "any",
            "strength": "high",
        },
        {
            "types": ["survey_review"],
            "action": "avoid",
            "logic": "any",
            "strength": "medium",
        },
        {
            "types": ["survey_review", "taxonomy_tutorial"],
            "action": "exclude",
            "logic": "any",
            "threshold": 0.8,
        },
    ]

    primary = evaluate_paper_type_rules(
        {"type_probs": {"primary_method": 0.9, "survey_review": 0.1}, "confidence": 0.95},
        rules,
    )
    survey = evaluate_paper_type_rules(
        {"type_probs": {"primary_method": 0.1, "survey_review": 0.9}, "confidence": 0.95},
        rules,
    )
    unknown = evaluate_paper_type_rules(None, rules)
    medium_confidence_require = evaluate_paper_type_rules(
        {"type_probs": {"primary_method": 0.0}, "confidence": 0.90},
        [{"types": ["primary_method"], "action": "require", "logic": "any"}],
    )
    medium_confidence_exclude = evaluate_paper_type_rules(
        {"type_probs": {"survey_review": 0.9}, "confidence": 0.90},
        [{"types": ["survey_review"], "action": "exclude", "logic": "any"}],
    )

    assert primary["paper_type_alignment"] > 0
    assert not primary["hard_filtered"]
    assert survey["paper_type_alignment"] < 0
    assert survey["hard_filtered"]
    assert survey["paper_type_filter_action"] == "exclude"
    assert not unknown["hard_filtered"]
    assert -0.10 <= unknown["paper_type_soft_penalty"] < 0
    assert not medium_confidence_require["hard_filtered"]
    assert medium_confidence_require["paper_type_soft_penalty"] < 0
    assert medium_confidence_exclude["hard_filtered"]


def test_s2_publication_types_are_positive_only_canonical_evidence():
    review = s2_publication_types_to_record(
        "2307.13721", ["Journal Article", "Review"]
    )
    assert review["publication_types"] == ["JournalArticle", "Review"]
    assert review["type_probs"]["survey_review"] == pytest.approx(1.0)
    assert review["negative_evidence_types"] == []

    excluded = evaluate_paper_type_rules(
        review,
        [{"types": ["survey_review"], "action": "exclude", "logic": "any"}],
    )
    assert excluded["hard_filtered"]
    assert excluded["paper_type_evidence_source"] == "semantic_scholar"
    assert "source=semantic_scholar" in excluded["paper_type_filter_reason"]

    untagged = s2_publication_types_to_record(
        "2000.00001", ["JournalArticle"]
    )
    required = evaluate_paper_type_rules(
        untagged,
        [{"types": ["survey_review"], "action": "require", "logic": "any"}],
    )
    assert not required["hard_filtered"]
    assert not required["paper_type_known_for_require_filter"]
    assert required["paper_type_soft_penalty"] < 0


def test_s2_capabilities_disable_only_unsupported_soft_type_alignment():
    s2_cache = {
        "paper": s2_publication_types_to_record(
            "paper", ["JournalArticle", "Review"]
        )
    }
    skill = RerankSkill("qwen-test", paper_type_cache=s2_cache)
    unsupported_value = valid_policy(
        paper_type_rules=[
            {
                "types": ["dataset_benchmark"],
                "action": "prefer",
                "logic": "any",
                "strength": "high",
            }
        ]
    )
    unsupported = skill.compile_weights(
        validate_policy_object(unsupported_value, policy_id="unsupported")
    )
    assert not unsupported.paper_type_alignment_enabled
    assert unsupported.feature_weights["paper_type_alignment"] == 0.0
    assert "paper_type_alignment_forced_off_unsupported_rules" in unsupported.adjustments

    supported_value = valid_policy(
        paper_type_rules=[
            {
                "types": ["survey_review"],
                "action": "prefer",
                "logic": "any",
                "strength": "high",
            }
        ]
    )
    supported = skill.compile_weights(
        validate_policy_object(supported_value, policy_id="supported")
    )
    assert supported.paper_type_alignment_enabled


def test_hard_filtered_rows_do_not_enter_ranking_and_ties_are_reproducible():
    policy_value = valid_policy(
        paper_type_rules=[
            {
                "types": ["survey_review"],
                "action": "exclude",
                "logic": "any",
                "threshold": 0.8,
            }
        ]
    )
    policy = validate_policy_object(policy_value, policy_id="p")
    cache = {
        "survey": {
            "type_probs": {"survey_review": 0.95},
            "confidence": 0.95,
        },
        "seed": {"type_probs": {"primary_method": 0.8}, "confidence": 0.9},
        "other": {"type_probs": {"primary_method": 0.8}, "confidence": 0.9},
    }
    skill = RerankSkill("qwen-test", paper_type_cache=cache)
    compiled = skill.compile_weights(policy)
    base = {
        "query_score_normalized": 0.5,
        "subquery_score_normalized": 0.5,
        "intent_labels": [],
        "intent_score": 0.0,
        "path_count_normalized": 0.0,
    }
    rows = skill.score_candidates(
        [
            {**base, "paper_arxiv_id": "other", "is_seed": False},
            {**base, "paper_arxiv_id": "survey", "is_seed": False},
            {**base, "paper_arxiv_id": "seed", "is_seed": True},
        ],
        compiled,
    )

    assert [row["paper_arxiv_id"] for row in rows] == ["seed", "other", "survey"]
    assert rows[-1]["hard_filtered"]
    assert rows[-1]["rerank_rank"] is None
    assert skill._type_record_for_candidate(
        {
            "paper_arxiv_id": "survey",
            "paper_type_probs": {},
            "paper_type_classifier_confidence": 0.0,
        }
    )["type_probs"]["survey_review"] == pytest.approx(0.95)


def test_hard_type_rule_is_independent_of_soft_alignment_weight():
    value = valid_policy(
        weight_levels={
            **valid_policy()["weight_levels"],
            "paper_type_alignment": "off",
        },
        paper_type_rules=[
            {
                "types": ["survey_review"],
                "action": "exclude",
                "logic": "any",
            }
        ],
    )
    policy = validate_policy_object(value, policy_id="p")
    skill = RerankSkill(
        "qwen-test",
        paper_type_cache={
            "survey": {
                "type_probs": {"survey_review": 0.95},
                "confidence": 0.95,
            }
        },
    )
    compiled = skill.compile_weights(policy)
    row = skill.score_candidates(
        [
            {
                "paper_arxiv_id": "survey",
                "query_score_normalized": 0.5,
                "subquery_score_normalized": 0.5,
                "intent_labels": [],
                "intent_score": 0.0,
                "path_count_normalized": 0.0,
            }
        ],
        compiled,
    )[0]

    assert not compiled.paper_type_alignment_enabled
    assert row["hard_filtered"]


def test_explicit_legacy_score_is_exact():
    skill = RerankSkill("qwen-test")
    row = {
        "paper_arxiv_id": "x",
        "query_score_normalized": 0.2,
        "subquery_score_normalized": 0.4,
        "intent_labels": ["methodology"],
        "intent_score": 1.0,
        "path_count_normalized": 0.5,
    }

    scored = skill.score_candidates([row], skill.legacy_compiled_policy())[0]

    assert scored["rerank_score"] == pytest.approx(
        0.30 * 0.2 + 0.40 * 0.4 + 0.15 * 1.0 + 0.15 * 0.5
    )


def test_policy_prompt_receives_only_original_query_context():
    fake = FakeLLM([json.dumps(valid_policy())])
    skill = RerankSkill("qwen-test", llm_call=fake)
    query = "find methods and exclude surveys"

    skill.build_policy(query)

    assert query in fake.prompts[0]
    for forbidden in ("ground_truth", "cited_paper", "query_date", "candidate title"):
        assert forbidden not in fake.prompts[0]
    assert all(name in skill.build_prompt(query) for name in DIMENSION_NAMES)
