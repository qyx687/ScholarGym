import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from graph_methods import ArtifactWriter, PerSubqueryProcessor
from dimension_catalog import PAPER_TYPES
from online_paper_type import QwenPaperTypeResolver, S2PublicationTypeResolver
from online_per_subquery import OnlinePerSubqueryManager
from paper_type import CLASSIFIER_VERSION, s2_publication_types_to_record
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
                    "types": ["survey_review"],
                    "action": "exclude",
                    "logic": "any",
                }
            ],
            "confidence": 0.95,
        }
    )


def _dataset_policy_json():
    return json.dumps(
        {
            "policy_version": "dynamic_rerank_v1",
            "query_intent": "dataset_search",
            "weight_levels": {
                "query_similarity": "high",
                "subquery_similarity": "very_high",
                "intent_background": "off",
                "intent_method": "off",
                "intent_result": "off",
                "path_count": "off",
                "paper_type_alignment": "high",
            },
            "paper_type_rules": [
                {
                    "types": ["dataset_benchmark"],
                    "action": "prefer",
                    "logic": "any",
                    "strength": "high",
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
    supported_types = (
        "application_case_study",
        "empirical_study",
        "position_perspective",
        "survey_review",
    )
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
    assert first["2001.00001"]["type_probs"]["survey_review"] == 1.0

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


class FakeQwenTypeClassifier:
    classifier_version = CLASSIFIER_VERSION

    def __init__(self):
        self.calls = []

    def classify_batch(self, papers):
        self.calls.append(list(papers))
        return [
            {
                "paper_arxiv_id": paper["paper_arxiv_id"],
                "type_probs": {
                    "dataset_benchmark": 0.95,
                    "primary_method": 0.75,
                },
                "confidence": 0.96,
                "classifier_version": self.classifier_version,
            }
            for paper in papers
        ]


def test_online_qwen_type_cache_is_batched_resumable_and_full_taxonomy(tmp_path):
    cache_path = tmp_path / "qwen-types.jsonl"
    paper_db = {
        "2001.00001": {"title": "Dataset one", "abstract": "A benchmark."},
        "2001.00002": {"title": "Dataset two", "abstract": "A benchmark."},
        "2001.00003": {"title": "Dataset three", "abstract": "A benchmark."},
    }
    classifier = FakeQwenTypeClassifier()
    resolver = QwenPaperTypeResolver(
        cache_path,
        paper_db,
        "qwen-test",
        batch_size=2,
        classifier=classifier,
    )

    first = resolver.resolve(paper_db)
    second = resolver.resolve(paper_db)

    assert len(classifier.calls) == 2
    assert first == second
    assert set(first) == set(paper_db)
    assert first["2001.00001"]["evidence_source"] == "qwen"
    assert set(first["2001.00001"]["supported_types"]) == set(PAPER_TYPES)
    assert set(first["2001.00001"]["negative_evidence_types"]) == set(
        PAPER_TYPES
    )

    offline_classifier = FakeQwenTypeClassifier()
    offline = QwenPaperTypeResolver(
        cache_path,
        paper_db,
        "qwen-test",
        offline=True,
        classifier=offline_classifier,
    )
    cached = offline.resolve(["2001.00001", "2099.99999"])
    assert set(cached) == {"2001.00001"}
    assert not offline_classifier.calls
    assert offline.snapshot_stats()["offline_misses"] == 1


def test_online_type_caches_reject_records_from_the_other_backend(tmp_path):
    cache_path = tmp_path / "mixed-types.jsonl"
    cache_path.write_text(
        json.dumps(s2_publication_types_to_record("2001.00001", ["Review"]))
        + "\n",
        encoding="utf-8",
    )
    classifier = FakeQwenTypeClassifier()
    resolver = QwenPaperTypeResolver(
        cache_path,
        {"2001.00001": {"title": "Review", "abstract": "Review."}},
        "qwen-test",
        offline=True,
        classifier=classifier,
    )

    assert resolver.resolve(["2001.00001"]) == {}
    stats = resolver.snapshot_stats()
    assert stats["cache_backend_mismatch_lines"] == 1
    assert stats["offline_misses"] == 1


def test_online_qwen_cache_is_bound_to_the_configured_model(tmp_path):
    cache_path = tmp_path / "qwen-types.jsonl"
    paper_db = {
        "2001.00001": {"title": "Dataset one", "abstract": "A benchmark."}
    }
    first = QwenPaperTypeResolver(
        cache_path,
        paper_db,
        "qwen-model-a",
        classifier=FakeQwenTypeClassifier(),
    )
    assert set(first.resolve(paper_db)) == {"2001.00001"}

    same_model = QwenPaperTypeResolver(
        cache_path,
        paper_db,
        "qwen-model-a",
        offline=True,
        classifier=FakeQwenTypeClassifier(),
    )
    other_model = QwenPaperTypeResolver(
        cache_path,
        paper_db,
        "qwen-model-b",
        offline=True,
        classifier=FakeQwenTypeClassifier(),
    )

    assert set(same_model.resolve(paper_db)) == {"2001.00001"}
    assert other_model.resolve(paper_db) == {}
    assert other_model.snapshot_stats()["cache_backend_mismatch_lines"] == 1


def test_qwen_backend_enables_full_taxonomy_while_s2_stays_conservative(tmp_path):
    paper_db = {
        "2001.00001": {"title": "Robot benchmark", "abstract": "Dataset."}
    }
    qwen_resolver = QwenPaperTypeResolver(
        tmp_path / "qwen.jsonl",
        paper_db,
        "qwen-test",
        offline=True,
        classifier=FakeQwenTypeClassifier(),
    )
    qwen_skill = RerankSkill(
        "qwen-test",
        llm_call=lambda *args, **kwargs: _dataset_policy_json(),
    )
    qwen_processor = PerSubqueryProcessor(
        paper_db,
        FakeGraphS2(),
        scoring_backend="bm25",
        embedding_provider=None,
        rerank_skill=qwen_skill,
        paper_type_resolver=qwen_resolver,
    )
    _, qwen_compiled = qwen_processor.configure_query("Find robot benchmarks")

    s2_skill = RerankSkill(
        "qwen-test",
        llm_call=lambda *args, **kwargs: _dataset_policy_json(),
    )
    s2_processor = PerSubqueryProcessor(
        paper_db,
        FakeGraphS2(),
        scoring_backend="bm25",
        embedding_provider=None,
        rerank_skill=s2_skill,
        paper_type_resolver=FakeTypeResolver(),
    )
    _, s2_compiled = s2_processor.configure_query("Find robot benchmarks")

    assert qwen_compiled is not None
    assert qwen_compiled.paper_type_alignment_enabled is True
    assert qwen_compiled.feature_weights["paper_type_alignment"] > 0.0
    assert s2_compiled is not None
    assert s2_compiled.paper_type_alignment_enabled is False
    assert s2_compiled.feature_weights["paper_type_alignment"] == 0.0
    assert "paper_type_alignment_forced_off_unsupported_rules" in (
        s2_compiled.adjustments
    )
