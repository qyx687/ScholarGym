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

BUILD_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_paper_type_cache.py"
BUILD_SPEC = importlib.util.spec_from_file_location("build_paper_type_cache", BUILD_SCRIPT_PATH)
BUILD_MODULE = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
sys.modules[BUILD_SPEC.name] = BUILD_MODULE
BUILD_SPEC.loader.exec_module(BUILD_MODULE)

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
                {
                    "paper_arxiv_id": "semantic",
                    "type_probs": {"survey_review": 0.95},
                    "confidence": 0.95,
                    "classifier_version": "qwen30b_paper_type_v1",
                },
                {
                    "paper_arxiv_id": "gt",
                    "type_probs": {"primary_method": 0.95},
                    "confidence": 0.95,
                    "classifier_version": "qwen30b_paper_type_v1",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    policy = json.loads(policy_json())
    policy["paper_type_rules"] = [
        {"types": ["survey_review"], "action": "exclude", "logic": "any"}
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


def test_paper_type_cache_builder_is_unique_batched_query_independent_and_resumable(tmp_path):
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {
                "query": "SECRET QUERY MUST NOT REACH CLASSIFIER",
                "local_pool_rows": [
                    {"paper_arxiv_id": "1000.00001"},
                    {"paper_arxiv_id": "1000.00002"},
                    {"paper_arxiv_id": "1000.00001"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paper_db = tmp_path / "papers.json"
    paper_db.write_text(
        json.dumps(
            {
                "1000.00001": {"title": "A method", "abstract": "Introduces a model."},
                "1000.00002": {"title": "A survey", "abstract": "Reviews prior work."},
                "unused": {"title": "Unused", "abstract": "Must not be classified."},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    prompts = []

    def fake_classifier(prompt, *args, **kwargs):
        prompts.append(prompt)
        payload = json.loads(prompt.split("Papers:\n", 1)[1])
        return json.dumps(
            [
                {
                    "paper_arxiv_id": paper["paper_arxiv_id"],
                    "type_probs": {"primary_method": 0.9},
                    "confidence": 0.95,
                    "classifier_version": "qwen30b_paper_type_v1",
                }
                for paper in payload
            ]
        )

    output = tmp_path / "types.jsonl"
    first = BUILD_MODULE.build_cache(
        pool,
        paper_db,
        output,
        model="qwen-test",
        batch_size=2,
        llm_call=fake_classifier,
    )
    first_record = output.read_text(encoding="utf-8").splitlines()[0]
    with output.open("a", encoding="utf-8") as handle:
        handle.write(first_record + "\n")
    second = BUILD_MODULE.build_cache(
        pool,
        paper_db,
        output,
        model="qwen-test",
        batch_size=2,
        resume=True,
        llm_call=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    assert first["unique_candidate_count"] == 2
    assert first["classified_and_written_count"] == 2
    assert second["classified_and_written_count"] == 0
    assert second["duplicate_records_removed"] == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    assert len(prompts) == 1
    assert "SECRET QUERY" not in prompts[0]
    assert "Unused" not in prompts[0]


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
    cache = BUILD_MODULE.load_paper_type_cache(output)

    assert len(session.calls) == 1
    assert first["unique_candidate_count"] == 2
    assert first["s2_resolved_count"] == 1
    assert first["remaining_candidate_count"] == 0
    assert cache["2307.13721"]["type_probs"]["survey_review"] == pytest.approx(1.0)
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


def test_paper_type_cache_shards_exclude_base_and_merge_without_duplicates(tmp_path):
    paper_ids = ["1000.00001", "1000.00002", "1000.00003", "1000.00004"]
    pool = tmp_path / "pool.jsonl"
    pool.write_text(
        json.dumps(
            {"local_pool_rows": [{"paper_arxiv_id": paper_id} for paper_id in paper_ids]}
        )
        + "\n",
        encoding="utf-8",
    )
    paper_db = tmp_path / "papers.json"
    paper_db.write_text(
        json.dumps(
            {
                paper_id: {"title": f"Paper {paper_id}", "abstract": "A method."}
                for paper_id in paper_ids
            }
        ),
        encoding="utf-8",
    )

    def fake_classifier(prompt, *args, **kwargs):
        payload = json.loads(prompt.split("Papers:\n", 1)[1])
        return json.dumps(
            [
                {
                    "paper_arxiv_id": paper["paper_arxiv_id"],
                    "type_probs": {"primary_method": 0.9},
                    "confidence": 0.95,
                    "classifier_version": "qwen30b_paper_type_v1",
                }
                for paper in payload
            ]
        )

    base = tmp_path / "base.jsonl"
    base.write_text(
        json.dumps(
            {
                "paper_arxiv_id": paper_ids[0],
                "type_probs": {"primary_method": 0.9},
                "confidence": 0.95,
                "classifier_version": "qwen30b_paper_type_v1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    shards = [tmp_path / "shard0.jsonl", tmp_path / "shard1.jsonl"]
    summaries = [
        BUILD_MODULE.build_cache(
            pool,
            paper_db,
            shards[index],
            model="qwen-test",
            batch_size=2,
            exclude_cache_path=base,
            shard_count=2,
            shard_index=index,
            llm_call=fake_classifier,
        )
        for index in range(2)
    ]

    shard_ids = [
        set(BUILD_MODULE.load_paper_type_cache(path))
        for path in shards
        if path.exists()
    ]
    assert shard_ids[0].isdisjoint(shard_ids[1])
    assert sum(summary["classified_and_written_count"] for summary in summaries) == 3

    merged = tmp_path / "merged.jsonl"
    merge_summary = MERGE_MODULE.merge_caches([base, *shards], merged)
    assert merge_summary["merged_unique_record_count"] == 4
    assert set(BUILD_MODULE.load_paper_type_cache(merged)) == set(paper_ids)
