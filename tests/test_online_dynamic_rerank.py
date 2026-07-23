import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from graph_methods import ArtifactWriter, PerSubqueryProcessor
from online_paper_type import S2PublicationTypeResolver
from online_per_subquery import OnlinePerSubqueryManager
from paper_type import S2_PUBLICATION_TYPES, s2_publication_types_to_record
from rerank_skill import RerankSkill


def _policy_json():
    return json.dumps(
        {
            "policy_version": "dynamic_rerank_v1",
            "query_intent": "exclude_surveys",
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
                {
                    "types": ["Review"],
                    "action": "exclude",
                    "logic": "any",
                }
            ],
            "confidence": 0.95,
        }
    )


class CountingPolicyLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return _policy_json()


class FakeGraphS2:
    def expand(self, seed_ids, method, limit):
        return [
            {
                "seed_arxiv_id": "2001.00001",
                "expanded_arxiv_id": "2001.00003",
                "edge_type": "reference",
                "edge_rank": 1,
                "intents": ["background"],
                "is_influential": False,
            }
        ]

    def snapshot_stats(self):
        return {}


class FakeTypeResolver:
    backend = "s2"
    evidence_source = "semantic_scholar"
    classifier_version = "s2_publication_types_v1"
    supported_types = S2_PUBLICATION_TYPES
    model = None

    def __init__(self):
        self.calls = 0

    def snapshot_stats(self):
        return {"resolve_calls": self.calls}

    def resolve(self, paper_ids):
        self.calls += 1
        return {
            paper_id: s2_publication_types_to_record(
                paper_id,
                ["Review"] if paper_id == "2001.00001" else ["JournalArticle"],
            )
            for paper_id in paper_ids
        }


def _event(event_id="q1:e1", iteration=1):
    return {
        "schema_version": "1.0",
        "query_id": "q1",
        "benchmark_idx": 0,
        "query": "Find multimodal models; exclude survey papers.",
        "query_date": "2002-01",
        "iteration_idx": iteration,
        "subquery_id": iteration,
        "subquery": "multimodal models",
        "subquery_before_date": "2002-01",
        "retrieval_event_id": event_id,
        "selector_top_k": 2,
        "seed_papers": [
            {
                "paper_arxiv_id": "2001.00001",
                "observed_retrieval_score": 5.0,
                "observed_retrieval_rank": 1,
            },
            {
                "paper_arxiv_id": "2001.00002",
                "observed_retrieval_score": 4.0,
                "observed_retrieval_rank": 2,
            },
        ],
    }


def test_dynamic_topk_score_reaches_memory_and_policy_is_built_once(tmp_path):
    paper_db = {
        "2001.00001": {
            "title": "A survey",
            "abstract": "A review of multimodal models.",
            "date": "2001-01",
        },
        "2001.00002": {
            "title": "A model",
            "abstract": "Visual and audio pretraining.",
            "date": "2001-02",
        },
        "2001.00003": {
            "title": "A foundation",
            "abstract": "Multimodal model foundations.",
            "date": "2001-03",
        },
    }
    policy_llm = CountingPolicyLLM()
    skill = RerankSkill(
        "qwen-test",
        llm_call=policy_llm,
        policy_cache_path=tmp_path / "policies.jsonl",
    )
    resolver = FakeTypeResolver()
    processor = PerSubqueryProcessor(
        paper_db,
        FakeGraphS2(),
        scoring_backend="bm25",
        embedding_provider=None,
        rerank_skill=skill,
        paper_type_resolver=resolver,
    )
    writer = ArtifactWriter(str(tmp_path / "artifacts"), save_level="full")
    manager = OnlinePerSubqueryManager(processor, writer, run_id="run")
    query = "Find multimodal models; exclude survey papers."

    manager.start_query(query, query_id="q1", benchmark_idx=0)
    first = manager.process_event(_event(), set())
    second = manager.process_event(_event("q1:e2", 2), set())

    assert policy_llm.calls == 1
    assert processor.active_original_query == query
    survey = next(
        row for row in first["rows"] if row["paper_arxiv_id"] == "2001.00001"
    )
    assert survey["hard_filtered"] is True
    assert survey["rerank_rank"] is None
    assert "2001.00001" not in {
        row["paper_arxiv_id"] for row in first["top_rows"]
    }
    assert [paper.score for paper in first["papers"]] == [
        row["rerank_score"] for row in first["top_rows"]
    ]
    assert first["rerank_policy_id"] == second["rerank_policy_id"]

    selected_id = first["top_rows"][0]["paper_arxiv_id"]
    manager.finish_event(
        _event(),
        first,
        [selected_id],
        "kept a relevant non-survey paper",
    )
    transition = json.loads(
        (tmp_path / "artifacts" / "memory_transitions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[-1]
    )
    assert transition["affects_next_iteration"] is True
    assert transition["rerank_policy_id"] == first["rerank_policy_id"]
    assert transition["retrieved_memory_rerank_scores"][selected_id] == next(
        row["rerank_score"]
        for row in first["top_rows"]
        if row["paper_arxiv_id"] == selected_id
    )


class FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        ids = kwargs["json"]["ids"]
        return FakeResponse(
            [
                {
                    "paperId": f"s2-{index}",
                    "publicationTypes": ["Review"] if index == 0 else [],
                }
                for index, _ in enumerate(ids)
            ]
        )


def test_online_s2_type_cache_is_resumable_and_offline_safe(tmp_path):
    cache_path = tmp_path / "types.jsonl"
    session = FakeSession()
    resolver = S2PublicationTypeResolver(
        cache_path,
        requests_per_second=0,
        session=session,
    )

    first = resolver.resolve(["2001.00001", "2001.00002"])
    second = resolver.resolve(["2001.00001", "2001.00002"])

    assert len(session.calls) == 1
    assert first == second
    assert first["2001.00001"]["publication_types"] == ["Review"]
    assert set(resolver.supported_types) == set(S2_PUBLICATION_TYPES)

    offline = S2PublicationTypeResolver(cache_path, offline=True)
    cached = offline.resolve(["2001.00001", "2099.99999"])
    assert set(cached) == {"2001.00001"}
    assert offline.snapshot_stats()["offline_misses"] == 1


def test_online_s2_batch_failure_is_deferred_without_recursive_requests(tmp_path):
    class FailingSession:
        def __init__(self):
            self.headers = {}
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError("provider unavailable")

    session = FailingSession()
    resolver = S2PublicationTypeResolver(
        tmp_path / "types.jsonl",
        requests_per_second=0,
        max_retries=1,
        session=session,
    )

    assert resolver.resolve(["2001.00001", "2001.00002"]) == {}
    assert resolver.resolve(["2001.00001", "2001.00002"]) == {}

    stats = resolver.snapshot_stats()
    assert session.calls == 1
    assert stats["failed_batches"] == 1
    assert stats["failed_papers"] == 2
    assert stats["deferred_misses"] == 2


def test_online_s2_cache_rejects_non_s2_records(tmp_path):
    cache_path = tmp_path / "mixed-types.jsonl"
    cache_path.write_text(
        json.dumps(
            {
                "paper_arxiv_id": "2001.00001",
                "type_probs": {},
                "confidence": 1.0,
                "classifier_version": "paper_type_v1",
                "evidence_source": "qwen",
                "publication_types": [],
                "supported_types": [],
                "negative_evidence_types": [],
            }
        ) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="native S2 cache contains non-S2 record"):
        S2PublicationTypeResolver(cache_path, offline=True)
