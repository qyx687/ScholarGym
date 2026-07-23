import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from dimension_catalog import DIMENSION_NAMES, intent_dimension_values
from paper_type import (
    S2_PUBLICATION_TYPES,
    evaluate_paper_type_rules,
    s2_publication_types_to_record,
    validate_type_record,
)
from rerank_skill import (
    LEGACY_FEATURE_WEIGHTS,
    S2_NATIVE_PAPER_TYPE_NAMESPACE,
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
                "types": ["Conference"],
                "action": "prefer",
                "logic": "any",
                "strength": "high",
            }
        ],
        "confidence": 0.9,
    }
    value.update(overrides)
    return value


def weight_levels_with_type_alignment(level):
    return {
        **valid_policy()["weight_levels"],
        "paper_type_alignment": level,
    }


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
        lambda value: value["paper_type_rules"][0].update({"action": "require"}),
        lambda value: value["paper_type_rules"][0].update({"threshold": 0.8}),
    ],
)
def test_illegal_dimension_type_and_action_are_rejected(mutate):
    value = valid_policy()
    mutate(value)

    with pytest.raises(PolicyValidationError):
        validate_policy_object(value, policy_id="p")


def test_native_paper_type_case_is_normalized():
    value = valid_policy(
        weight_levels=weight_levels_with_type_alignment("off"),
        paper_type_rules=[
            {"types": ["review"], "action": "exclude", "logic": "any"}
        ]
    )

    policy = validate_policy_object(value, policy_id="p")

    assert policy.paper_type_rules[0].types == ("Review",)


def test_canonical_functional_type_is_rejected():
    canonical_invalid = valid_policy(
        paper_type_rules=[
            {
                "types": ["primary_method"],
                "action": "prefer",
                "logic": "any",
                "strength": "low",
            }
        ]
    )
    with pytest.raises(PolicyValidationError, match="unknown types"):
        validate_policy_object(canonical_invalid, policy_id="native-only")


def test_type_alignment_is_enabled_exactly_for_soft_type_rules():
    hard_only = [
        {"types": ["Review"], "action": "exclude", "logic": "any"}
    ]
    soft = [
        {
            "types": ["Review"],
            "action": "avoid",
            "logic": "any",
            "strength": "high",
        }
    ]

    validate_policy_object(
        valid_policy(
            weight_levels=weight_levels_with_type_alignment("off"),
            paper_type_rules=hard_only,
        ),
        policy_id="hard-only",
    )
    with pytest.raises(PolicyValidationError, match="must be non-off exactly"):
        validate_policy_object(
            valid_policy(paper_type_rules=hard_only), policy_id="hard-with-weight"
        )
    with pytest.raises(PolicyValidationError, match="must be non-off exactly"):
        validate_policy_object(
            valid_policy(
                weight_levels=weight_levels_with_type_alignment("off"),
                paper_type_rules=soft,
            ),
            policy_id="soft-without-weight",
        )


def test_s2_native_prompt_exposes_only_native_s2_vocabulary():
    skill = RerankSkill("qwen-test")

    prompt = skill.build_prompt("Exclude survey papers")

    assert "Semantic Scholar's native publicationTypes" in prompt
    assert "Review, JournalArticle" in prompt
    assert "survey_review" not in prompt
    assert "Dataset is a dataset record, not a paper" in prompt
    assert "never use Dataset for a query asking for papers" in prompt
    assert "subject-matter mention" in prompt
    assert "Paper type actions: prefer, avoid, exclude" in prompt
    assert "represented as prefer" in prompt
    assert "threshold" not in prompt
    assert skill.paper_type_backend == "s2"
    assert skill.paper_type_namespace == S2_NATIVE_PAPER_TYPE_NAMESPACE


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


def test_policy_cache_is_bound_to_minimum_confidence(tmp_path):
    cache = tmp_path / "policies.jsonl"
    first_llm = FakeLLM([json.dumps(valid_policy(confidence=0.65))])
    first = RerankSkill(
        "qwen-test",
        llm_call=first_llm,
        min_confidence=0.60,
        policy_cache_path=cache,
    ).build_policy("query")
    second_llm = FakeLLM([json.dumps(valid_policy(confidence=0.85))])
    second = RerankSkill(
        "qwen-test",
        llm_call=second_llm,
        min_confidence=0.70,
        policy_cache_path=cache,
    ).build_policy("query")

    assert not first.used_fallback
    assert not second.used_fallback
    assert first.policy_id != second.policy_id
    assert len(second_llm.prompts) == 1


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


def test_prefer_avoid_and_exclude_use_direct_native_s2_membership():
    def native_record(*publication_types, confidence=1.0):
        return {
            "type_probs": {type_name: 1.0 for type_name in publication_types},
            "confidence": confidence,
            "evidence_source": "semantic_scholar",
            "publication_types": list(publication_types),
            "supported_types": list(S2_PUBLICATION_TYPES),
            "negative_evidence_types": [],
        }

    rules = [
        {
            "types": ["Conference"],
            "action": "prefer",
            "logic": "any",
            "strength": "high",
        },
        {
            "types": ["Review"],
            "action": "avoid",
            "logic": "any",
            "strength": "medium",
        },
        {
            "types": ["Review", "MetaAnalysis"],
            "action": "exclude",
            "logic": "any",
        },
    ]

    primary = evaluate_paper_type_rules(native_record("Conference"), rules)
    survey = evaluate_paper_type_rules(native_record("Review"), rules)
    unknown = evaluate_paper_type_rules(None, rules)
    low_confidence_exclude = evaluate_paper_type_rules(
        native_record("Review", confidence=0.01),
        [{"types": ["Review"], "action": "exclude", "logic": "any"}],
    )
    review_only_all = evaluate_paper_type_rules(
        native_record("Review"),
        [
            {
                "types": ["Review", "MetaAnalysis"],
                "action": "exclude",
                "logic": "all",
            }
        ],
    )
    review_and_meta_all = evaluate_paper_type_rules(
        native_record("Review", "MetaAnalysis"),
        [
            {
                "types": ["Review", "MetaAnalysis"],
                "action": "exclude",
                "logic": "all",
            }
        ],
    )

    assert primary["paper_type_alignment"] > 0
    assert not primary["hard_filtered"]
    assert survey["paper_type_alignment"] < 0
    assert survey["hard_filtered"]
    assert survey["paper_type_filter_action"] == "exclude"
    assert not unknown["hard_filtered"]
    assert "paper_type_soft_penalty" not in unknown
    assert low_confidence_exclude["hard_filtered"]
    assert not review_only_all["hard_filtered"]
    assert review_and_meta_all["hard_filtered"]


def test_s2_publication_types_preserve_raw_positive_only_metadata():
    review = s2_publication_types_to_record(
        "2307.13721", ["Journal Article", "Review"]
    )
    assert review["publication_types"] == ["JournalArticle", "Review"]
    assert review["type_probs"] == {"JournalArticle": 1.0, "Review": 1.0}
    assert set(review["supported_types"]) == set(S2_PUBLICATION_TYPES)
    assert review["negative_evidence_types"] == []
    projected = RerankSkill(
        "qwen-test", paper_type_cache={"2307.13721": review}
    )._type_record_for_candidate({"paper_arxiv_id": "2307.13721"})
    assert projected["type_probs"] == {"JournalArticle": 1.0, "Review": 1.0}
    assert projected["negative_evidence_types"] == []


def test_legacy_mapped_s2_cache_record_is_upgraded_to_native_in_memory():
    upgraded = validate_type_record(
        {
            "paper_arxiv_id": "2307.13721",
            "type_probs": {"survey_review": 1.0},
            "confidence": 1.0,
            "classifier_version": "s2_publication_types_v1",
            "evidence_source": "semantic_scholar",
            "publication_types": ["JournalArticle", "Review"],
            "supported_types": ["survey_review"],
            "negative_evidence_types": [],
        }
    )

    assert upgraded["type_probs"] == {"JournalArticle": 1.0, "Review": 1.0}
    assert set(upgraded["supported_types"]) == set(S2_PUBLICATION_TYPES)


def test_native_catalog_is_fixed_even_before_cache_is_populated():
    skill = RerankSkill("qwen-test")
    policy = validate_policy_object(valid_policy(), policy_id="native-supported")

    available = skill.compile_weights(policy, paper_type_available=True)
    unavailable = skill.compile_weights(policy, paper_type_available=False)

    assert skill.paper_type_supported_types == set(S2_PUBLICATION_TYPES)
    assert available.paper_type_alignment_enabled
    assert not unavailable.paper_type_alignment_enabled
    assert unavailable.feature_weights["paper_type_alignment"] == 0.0


def test_s2_native_scoring_uses_raw_publication_types_without_mapping():
    cache = {
        "review": s2_publication_types_to_record(
            "review", ["JournalArticle", "Review"]
        ),
        "journal": s2_publication_types_to_record(
            "journal", ["JournalArticle"]
        ),
    }
    skill = RerankSkill("qwen-test", paper_type_cache=cache)
    policy = validate_policy_object(
        valid_policy(
            weight_levels=weight_levels_with_type_alignment("off"),
            paper_type_rules=[
                {
                    "types": ["Review"],
                    "action": "exclude",
                    "logic": "any",
                },
            ]
        ),
        policy_id="native",
    )
    base = {
        "query_score_normalized": 0.5,
        "subquery_score_normalized": 0.5,
        "intent_labels": [],
        "path_count_normalized": 0.0,
    }
    rows = skill.score_candidates(
        [
            {**base, "paper_arxiv_id": "review"},
            {**base, "paper_arxiv_id": "journal"},
        ],
        skill.compile_weights(policy),
    )
    by_id = {row["paper_arxiv_id"]: row for row in rows}

    assert by_id["review"]["paper_type_probs"] == {
        "JournalArticle": 1.0,
        "Review": 1.0,
    }
    assert "survey_review" not in by_id["review"]["paper_type_probs"]
    assert by_id["review"]["hard_filtered"]
    assert not by_id["journal"]["hard_filtered"]
    assert "paper_type_soft_penalty" not in by_id["journal"]
    assert by_id["journal"]["paper_type_namespace"] == "s2_native"


def test_s2_native_positive_type_evidence_supports_soft_preference():
    cache = {
        "conference": s2_publication_types_to_record(
            "conference", ["JournalArticle", "Conference"]
        ),
        "journal": s2_publication_types_to_record(
            "journal", ["JournalArticle"]
        ),
    }
    skill = RerankSkill("qwen-test", paper_type_cache=cache)
    policy = validate_policy_object(
        valid_policy(
            paper_type_rules=[
                {
                    "types": ["Conference"],
                    "action": "prefer",
                    "logic": "any",
                    "strength": "high",
                }
            ]
        ),
        policy_id="native-prefer",
    )
    base = {
        "query_score_normalized": 0.5,
        "subquery_score_normalized": 0.5,
        "intent_labels": [],
        "path_count_normalized": 0.0,
    }

    rows = skill.score_candidates(
        [
            {**base, "paper_arxiv_id": "journal"},
            {**base, "paper_arxiv_id": "conference"},
        ],
        skill.compile_weights(policy),
    )

    assert rows[0]["paper_arxiv_id"] == "conference"
    assert (
        rows[0]["component_contributions"]["paper_type_alignment"]
        > rows[1]["component_contributions"]["paper_type_alignment"]
    )


def test_hard_filtered_rows_do_not_enter_ranking_and_ties_are_reproducible():
    policy_value = valid_policy(
        weight_levels=weight_levels_with_type_alignment("off"),
        paper_type_rules=[
            {
                "types": ["Review"],
                "action": "exclude",
                "logic": "any",
            }
        ]
    )
    policy = validate_policy_object(policy_value, policy_id="p")
    cache = {
        "survey": s2_publication_types_to_record("survey", ["Review"]),
        "seed": s2_publication_types_to_record("seed", ["Conference"]),
        "other": s2_publication_types_to_record("other", ["Conference"]),
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
    )["type_probs"]["Review"] == pytest.approx(1.0)


def test_hard_type_rule_is_independent_of_soft_alignment_weight():
    value = valid_policy(
        weight_levels={
            **valid_policy()["weight_levels"],
            "paper_type_alignment": "off",
        },
        paper_type_rules=[
            {
                "types": ["Review"],
                "action": "exclude",
                "logic": "any",
            }
        ],
    )
    policy = validate_policy_object(value, policy_id="p")
    skill = RerankSkill(
        "qwen-test",
        paper_type_cache={
            "survey": s2_publication_types_to_record("survey", ["Review"])
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
