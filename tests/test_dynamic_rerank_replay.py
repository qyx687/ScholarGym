import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "replay_dynamic_rerank.py"
SPEC = importlib.util.spec_from_file_location("replay_dynamic_rerank", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

S2_BUILD_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_s2_paper_type_cache.py"
)
S2_BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_s2_paper_type_cache", S2_BUILD_SCRIPT_PATH
)
S2_BUILD_MODULE = importlib.util.module_from_spec(S2_BUILD_SPEC)
assert S2_BUILD_SPEC.loader is not None
sys.modules[S2_BUILD_SPEC.name] = S2_BUILD_MODULE
S2_BUILD_SPEC.loader.exec_module(S2_BUILD_MODULE)

MERGE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "merge_paper_type_caches.py"
)
MERGE_SPEC = importlib.util.spec_from_file_location(
    "merge_paper_type_caches", MERGE_SCRIPT_PATH
)
MERGE_MODULE = importlib.util.module_from_spec(MERGE_SPEC)
assert MERGE_SPEC.loader is not None
sys.modules[MERGE_SPEC.name] = MERGE_MODULE
MERGE_SPEC.loader.exec_module(MERGE_MODULE)


def policy_json():
    return json.dumps(
        {
            "policy_version": "dynamic_rerank_v1",
            "query_intent": "method_search",
            "weight_levels": {
                "query_similarity": "low",
                "subquery_similarity": "low",
                "intent_background": "off",
                "intent_method": "very_high",
                "intent_result": "off",
                "path_count": "off",
                "paper_type_alignment": "off",
            },
            "paper_type_rules": [],
            "confidence": 0.95,
        }
    )


class CountingLLM:
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return policy_json()


def write_fixture(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "qid": "q0",
                "query": "find a primary method",
                "cited_paper": [{"arxiv_id": "gt"}],
                "gt_label": [1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    base = {
        "query_id": "q0",
        "benchmark_idx": 0,
        "query": "find a primary method",
        "selector_top_k": 1,
        "iteration_idx": 1,
        "subquery_id": 1,
        "subquery": "methods",
    }
    semantic = {
        "paper_arxiv_id": "semantic",
        "is_seed": True,
        "is_expanded": False,
        "observed_retrieval_rank": 1,
        "query_score_normalized": 0.9,
        "subquery_score_normalized": 0.9,
        "intent_labels": [],
        "intent_score": 0.0,
        "path_count": 0,
        "path_count_normalized": 0.0,
    }
    gt = {
        "paper_arxiv_id": "gt",
        "is_seed": False,
        "is_expanded": True,
        "observed_retrieval_rank": None,
        "query_score_normalized": 0.5,
        "subquery_score_normalized": 0.5,
        "intent_labels": ["methodology"],
        "intent_score": 1.0,
        "path_count": 1,
        "path_count_normalized": 0.0,
    }
    pool = tmp_path / "pool.jsonl"
    with pool.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    **base,
                    "retrieval_event_id": "q0-e1",
                    "local_pool_rows": [semantic, gt],
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {
                    **base,
                    "retrieval_event_id": "q0-e2",
                    "iteration_idx": 2,
                    "local_pool_rows": [semantic, gt],
                }
            )
            + "\n"
        )
    return pool, benchmark


def test_fixture_replay_improves_f1_and_generates_one_policy_per_query(tmp_path):
    pool, benchmark = write_fixture(tmp_path)
    output = tmp_path / "out"
    llm = CountingLLM()

    summary = MODULE.replay(
        pool,
        benchmark,
        output,
        model="qwen-test",
        llm_call=llm,
        artifact_level="selected",
        semantic_min_mass=0.40,
    )

    assert llm.count == 1
    assert summary["legacy_static"]["avg_candidate_f1"] == 0.0
    assert summary["dynamic_policy"]["avg_candidate_f1"] == 1.0
    assert summary["dynamic_minus_legacy"]["avg_candidate_f1"] == 1.0
    assert summary["policy_fallback_rate"] == 0.0
    assert summary["compiler_config"]["semantic_min_mass"] == pytest.approx(0.40)
    for name in (
        "summary.json",
        "per_query_results.jsonl",
        "query_rerank_policies.jsonl",
        "ranked_candidates.jsonl",
        "event_results.jsonl",
    ):
        assert (output / name).exists()


def test_replay_supports_native_s2_publication_type_exclusion(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps(
            {
                "qid": "q-native",
                "query": "Find multimodal methods; exclude survey papers.",
                "cited_paper": [{"arxiv_id": "method"}],
                "gt_label": [1],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    base_row = {
        "is_seed": True,
        "is_expanded": False,
        "intent_labels": [],
        "intent_score": 0.0,
        "path_count": 0,
        "path_count_normalized": 0.0,
    }
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {
                "query_id": "q-native",
                "benchmark_idx": 0,
                "query": "Find multimodal methods; exclude survey papers.",
                "selector_top_k": 1,
                "iteration_idx": 1,
                "subquery_id": 1,
                "subquery": "multimodal methods",
                "retrieval_event_id": "q-native-e1",
                "local_pool_rows": [
                    {
                        **base_row,
                        "paper_arxiv_id": "review",
                        "observed_retrieval_rank": 1,
                        "query_score_normalized": 1.0,
                        "subquery_score_normalized": 1.0,
                    },
                    {
                        **base_row,
                        "paper_arxiv_id": "method",
                        "observed_retrieval_rank": 2,
                        "query_score_normalized": 0.8,
                        "subquery_score_normalized": 0.8,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    type_cache = tmp_path / "s2-types.jsonl"
    type_cache.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                S2_BUILD_MODULE.s2_publication_types_to_record(
                    "review", ["JournalArticle", "Review"]
                ),
                S2_BUILD_MODULE.s2_publication_types_to_record(
                    "method", ["JournalArticle", "Conference"]
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    def native_policy(*args, **kwargs):
        return json.dumps(
            {
                "policy_version": "dynamic_rerank_v1",
                "query_intent": "method_search_exclude_reviews",
                "weight_levels": {
                    "query_similarity": "high",
                    "subquery_similarity": "very_high",
                    "intent_background": "off",
                    "intent_method": "off",
                    "intent_result": "off",
                    "path_count": "off",
                    "paper_type_alignment": "off",
                },
                "paper_type_rules": [
                    {"types": ["Review"], "action": "exclude", "logic": "any"}
                ],
                "confidence": 0.95,
            }
        )

    summary = MODULE.replay(
        pool,
        benchmark,
        tmp_path / "out-native",
        model="qwen-test",
        paper_type_cache_path=type_cache,
        llm_call=native_policy,
        artifact_level="selected",
    )

    assert summary["paper_type_namespace"] == "s2_native"
    assert summary["legacy_static"]["avg_candidate_f1"] == 0.0
    assert summary["dynamic_policy"]["avg_candidate_f1"] == 1.0
    ranked = [
        json.loads(line)
        for line in (tmp_path / "out-native/ranked_candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    dynamic_rows = next(
        row["ranked_candidates"]
        for row in ranked
        if row["method"] == "dynamic_policy"
    )
    review = next(row for row in dynamic_rows if row["paper_arxiv_id"] == "review")
    assert review["paper_type_probs"]["Review"] == 1.0
    assert review["paper_type_namespace"] == "s2_native"
    assert review["hard_filtered"] is True


def test_replay_policy_cache_prevents_new_llm_call(tmp_path):
    pool, benchmark = write_fixture(tmp_path)
    cache = tmp_path / "policy_cache.jsonl"
    first_llm = CountingLLM()
    MODULE.replay(
        pool,
        benchmark,
        tmp_path / "first",
        model="qwen-test",
        policy_cache_path=cache,
        llm_call=first_llm,
        artifact_level="selected",
    )
    second_llm = CountingLLM()

    MODULE.replay(
        pool,
        benchmark,
        tmp_path / "second",
        model="qwen-test",
        policy_cache_path=cache,
        llm_call=second_llm,
        artifact_level="selected",
    )

    assert first_llm.count == 1
    assert second_llm.count == 0


def test_selected_artifact_retains_hard_filtered_rows_with_reason(tmp_path):
    pool, benchmark = write_fixture(tmp_path)
    type_cache = tmp_path / "types.jsonl"
    type_cache.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                S2_BUILD_MODULE.s2_publication_types_to_record(
                    "semantic", ["Review"]
                ),
                S2_BUILD_MODULE.s2_publication_types_to_record(
                    "gt", ["Conference"]
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    policy = json.loads(policy_json())
    policy["paper_type_rules"] = [
        {"types": ["Review"], "action": "exclude", "logic": "any"}
    ]
    output = tmp_path / "out"
    MODULE.replay(
        pool,
        benchmark,
        output,
        model="qwen-test",
        paper_type_cache_path=type_cache,
        llm_call=lambda *args, **kwargs: json.dumps(policy),
        artifact_level="selected",
        semantic_min_mass=0.40,
    )

    artifact_rows = [
        json.loads(line)
        for line in (output / "ranked_candidates.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    dynamic_events = [row for row in artifact_rows if row["method"] == "dynamic_policy"]
    assert dynamic_events
    for event in dynamic_events:
        filtered = [
            row
            for row in event["ranked_candidates"]
            if row["paper_arxiv_id"] == "semantic"
        ]
        assert len(filtered) == 1
        assert filtered[0]["hard_filtered"] is True
        assert filtered[0]["selected_at_event_top_k"] is False
        assert filtered[0]["paper_type_filter_action"] == "exclude"
        assert filtered[0]["paper_type_filter_reason"]


def test_s2_type_cache_builder_is_batched_positive_only_and_resumable(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {
                "local_pool_rows": [
                    {"paper_arxiv_id": "2307.13721"},
                    {"paper_arxiv_id": "2000.00001"},
                    {"paper_arxiv_id": "2307.13721"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return [
                None,
                {
                    "paperId": "s2-review",
                    "publicationTypes": ["JournalArticle", "Review"],
                },
            ]

    class Session:
        def __init__(self):
            self.headers = {}
            self.calls = []

        def post(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return Response()

    session = Session()
    client = S2_BUILD_MODULE.S2PublicationTypeClient(
        requests_per_second=0,
        session=session,
    )
    output = tmp_path / "s2-types.jsonl"
    first = S2_BUILD_MODULE.build_cache(
        pool,
        output,
        batch_size=2,
        client=client,
    )
    cache = MODULE.load_paper_type_cache(output)

    assert len(session.calls) == 1
    assert first["unique_candidate_count"] == 2
    assert first["s2_resolved_count"] == 1
    assert first["remaining_candidate_count"] == 0
    assert cache["2307.13721"]["publication_types"] == [
        "JournalArticle",
        "Review",
    ]
    assert cache["2307.13721"]["negative_evidence_types"] == []
    assert cache["2000.00001"]["confidence"] == 0.0

    second = S2_BUILD_MODULE.build_cache(
        pool,
        output,
        batch_size=2,
        resume=True,
        client=S2_BUILD_MODULE.S2PublicationTypeClient(
            requests_per_second=0,
            session=Session(),
        ),
    )
    assert second["requested_pending_count"] == 0
    assert second["classified_and_written_count"] == 0


def test_native_s2_type_caches_merge_without_duplicates(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        "\n".join(
            json.dumps(S2_BUILD_MODULE.s2_publication_types_to_record(paper_id, types))
            for paper_id, types in (
                ("1000.00001", ["Review"]),
                ("1000.00002", ["Conference"]),
            )
        ) + "\n",
        encoding="utf-8",
    )
    second.write_text(
        "\n".join(
            json.dumps(S2_BUILD_MODULE.s2_publication_types_to_record(paper_id, types))
            for paper_id, types in (
                ("1000.00002", ["Conference"]),
                ("1000.00003", ["JournalArticle"]),
            )
        ) + "\n",
        encoding="utf-8",
    )

    merged = tmp_path / "merged.jsonl"
    merge_summary = MERGE_MODULE.merge_caches([first, second], merged)

    assert merge_summary["merged_unique_record_count"] == 3
    assert set(MODULE.load_paper_type_cache(merged)) == {
        "1000.00001",
        "1000.00002",
        "1000.00003",
    }
