import json
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import config
from graph_methods import (
    ArtifactWriter,
    BoundedEmbeddingProvider,
    CandidateIndex,
    PerSubqueryProcessor,
    QueryScopedEmbeddingCache,
    RERANK_FORMULA_ID,
    S2GraphClient,
)
from rerank_skill import RerankSkill


class FakeS2:
    def expand(self, seed_ids, method, limit):
        return [
            {
                "seed_arxiv_id": "2001.00001",
                "expanded_arxiv_id": "2001.00003",
                "seed_s2_paper_id": "S1",
                "expanded_s2_paper_id": "E3",
                "edge_type": "citation",
                "edge_rank": 1,
                "intents": ["methodology"],
                "is_influential": True,
                "seed_publication_types": ["JournalArticle"],
                "expanded_publication_types": ["JournalArticle", "Review"],
            },
            {
                "seed_arxiv_id": "2001.00002",
                "expanded_arxiv_id": "2001.00003",
                "seed_s2_paper_id": "S2",
                "expanded_s2_paper_id": "E3",
                "edge_type": "reference",
                "edge_rank": 2,
                "intents": ["background"],
                "is_influential": False,
                "seed_publication_types": ["Conference"],
                "expanded_publication_types": ["JournalArticle", "Review"],
            },
            {
                "seed_arxiv_id": "2001.00001",
                "expanded_arxiv_id": "2003.00004",
                "edge_type": "citation",
                "edge_rank": 3,
                "intents": [],
                "is_influential": False,
            },
        ]

    def snapshot_stats(self):
        return {}


class FakeOllamaEmbeddings:
    """LangChain-compatible surface; deliberately has no custom ``embed`` method."""

    def __init__(self):
        self.document_batch_sizes = []
        self.document_texts = []

    @staticmethod
    def _vector(text):
        if "alpha" in text and "gamma" not in text:
            return [2.0, 0.0]
        if "beta" in text:
            return [0.0, 3.0]
        return [1.0, 1.0]

    def embed_documents(self, texts):
        self.document_batch_sizes.append(len(texts))
        self.document_texts.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def test_embedding_candidate_index_accepts_baseline_ollama_interface_and_batches(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_RERANK_EMBEDDING_BATCH_SIZE", 2)
    provider = FakeOllamaEmbeddings()
    metadata = {
        "a": {"title": "alpha", "abstract": "paper"},
        "b": {"title": "beta", "abstract": "paper"},
        "c": {"title": "gamma", "abstract": "paper"},
    }

    index = CandidateIndex(["a", "b", "c"], metadata, "embedding", provider)
    raw, normalized, ranks = index.score("alpha query")

    assert provider.document_batch_sizes == [2, 1]
    assert provider.document_texts[0] == "title: alpha\n abstract: paper"
    assert raw["a"] == 1.0
    assert raw["b"] == 0.0
    assert 0.70 < raw["c"] < 0.71
    assert normalized["a"] == 1.0
    assert ranks == {"a": 1, "c": 2, "b": 3}


def test_embedding_graph_materialization_stops_before_legacy_formula():
    provider = FakeOllamaEmbeddings()
    paper_db = {
        "2001.00001": {
            "title": "alpha seed",
            "abstract": "paper",
            "date": "2001-01",
        },
        "2001.00002": {
            "title": "beta seed",
            "abstract": "paper",
            "date": "2001-02",
        },
        "2001.00003": {
            "title": "gamma expanded",
            "abstract": "paper",
            "date": "2001-03",
        },
        "2003.00004": {
            "title": "future",
            "abstract": "paper",
            "date": "2003-01",
        },
    }
    processor = PerSubqueryProcessor(
        paper_db,
        FakeS2(),
        scoring_backend="embedding",
        embedding_provider=provider,
    )
    result = processor.materialize(
        {
            "query_id": "dense-stage-a",
            "query": "alpha query",
            "query_date": "2002-01",
            "subquery_id": 1,
            "subquery": "beta query",
            "subquery_before_date": "2002-01",
            "retrieval_event_id": "dense-stage-a:event-1",
            "retrieval_offset": 0,
            "selector_top_k": 1,
            "seed_papers": [
                {
                    "paper_arxiv_id": "2001.00001",
                    "observed_retrieval_rank": 1,
                },
                {
                    "paper_arxiv_id": "2001.00002",
                    "observed_retrieval_rank": 2,
                },
            ],
        }
    )

    assert result["features_materialized"] is True
    assert result["legacy_rerank_applied"] is False
    assert [row["paper_arxiv_id"] for row in result["rows"]] == [
        "2001.00001",
        "2001.00002",
        "2001.00003",
    ]
    assert all(row["retrieval_backend"] == "embedding" for row in result["rows"])
    assert all("query_score_normalized" in row for row in result["rows"])
    assert all("subquery_score_normalized" in row for row in result["rows"])
    assert all("rerank_score" not in row for row in result["rows"])
    assert all("rerank_rank" not in row for row in result["rows"])


def test_graph_runtime_rerank_uses_requested_four_factor_formula():
    provider = FakeOllamaEmbeddings()
    paper_db = {
        "2001.00001": {
            "title": "alpha seed",
            "abstract": "paper",
            "date": "2001-01",
        },
        "2001.00002": {
            "title": "beta seed",
            "abstract": "paper",
            "date": "2001-02",
        },
        "2001.00003": {
            "title": "gamma expanded",
            "abstract": "paper",
            "date": "2001-03",
        },
    }
    processor = PerSubqueryProcessor(
        paper_db,
        FakeS2(),
        scoring_backend="embedding",
        embedding_provider=provider,
    )
    result = processor.process(
        {
            "query_id": "dense-full",
            "query": "alpha query",
            "query_date": "2002-01",
            "subquery_id": 1,
            "subquery": "beta query",
            "subquery_before_date": "2002-01",
            "retrieval_event_id": "dense-full:event-1",
            "retrieval_offset": 0,
            "selector_top_k": 1,
            "seed_papers": [
                {
                    "paper_arxiv_id": "2001.00001",
                    "observed_retrieval_rank": 1,
                },
                {
                    "paper_arxiv_id": "2001.00002",
                    "observed_retrieval_rank": 2,
                },
            ],
        }
    )

    assert result["rerank_formula_id"] == RERANK_FORMULA_ID
    for row in result["rows"]:
        assert row["rerank_formula_id"] == RERANK_FORMULA_ID
        assert row["feature_weights"] == {
            "query_score_normalized": 0.30,
            "subquery_score_normalized": 0.40,
            "intent_score": 0.15,
            "path_count_normalized": 0.15,
        }
        assert row["rerank_score"] == (
            0.30 * row["query_score_normalized"]
            + 0.40 * row["subquery_score_normalized"]
            + 0.15 * row["intent_score"]
            + 0.15 * row["path_count_normalized"]
        )
    expanded = next(
        row for row in result["rows"] if row["paper_arxiv_id"] == "2001.00003"
    )
    assert expanded["intent_score"] > 0.0
    assert expanded["path_count_normalized"] > 0.0


def test_graph_runtime_dynamic_policy_reuses_query_policy_and_hard_filters_type():
    policy_calls = []

    def policy_llm(prompt, *args, **kwargs):
        policy_calls.append(prompt)
        return json.dumps(
            {
                "policy_version": "dynamic_rerank_v1",
                "query_intent": "method_search_excluding_surveys",
                "weight_levels": {
                    "query_similarity": "high",
                    "subquery_similarity": "high",
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
                        "threshold": 0.8,
                    }
                ],
                "confidence": 0.95,
            }
        )

    class FakeTypeResolver:
        backend = "qwen"
        evidence_source = "qwen"
        classifier_version = "qwen30b_paper_type_v1"
        supported_types = ("survey_review",)
        model = "qwen-test"

        def __init__(self):
            self.calls = []

        def resolve(self, paper_ids):
            ids = list(paper_ids)
            self.calls.append(ids)
            return {
                "2001.00003": {
                    "paper_arxiv_id": "2001.00003",
                    "type_probs": {"survey_review": 0.99},
                    "confidence": 0.96,
                    "classifier_version": self.classifier_version,
                    "evidence_source": self.evidence_source,
                    "publication_types": [],
                    "supported_types": ["survey_review"],
                    "negative_evidence_types": ["survey_review"],
                }
            }

        def snapshot_stats(self):
            return {"resolve_calls": len(self.calls)}

        def snapshot_cache(self):
            return {}

    paper_db = {
        "2001.00001": {
            "title": "alpha seed",
            "abstract": "paper",
            "date": "2001-01",
        },
        "2001.00002": {
            "title": "beta seed",
            "abstract": "paper",
            "date": "2001-02",
        },
        "2001.00003": {
            "title": "survey expanded",
            "abstract": "review",
            "date": "2001-03",
        },
    }
    resolver = FakeTypeResolver()
    processor = PerSubqueryProcessor(
        paper_db,
        FakeS2(),
        scoring_backend="embedding",
        embedding_provider=FakeOllamaEmbeddings(),
        rerank_skill=RerankSkill("qwen-test", llm_call=policy_llm),
        paper_type_resolver=resolver,
    )
    event = {
        "query_id": "dynamic-full",
        "query": "find methods; exclude survey papers",
        "query_date": "2002-01",
        "subquery_id": 1,
        "subquery": "alpha beta methods",
        "subquery_before_date": "2002-01",
        "retrieval_event_id": "dynamic-full:event-1",
        "retrieval_offset": 0,
        "selector_top_k": 3,
        "seed_papers": [
            {"paper_arxiv_id": "2001.00001", "observed_retrieval_rank": 1},
            {"paper_arxiv_id": "2001.00002", "observed_retrieval_rank": 2},
        ],
    }

    first = processor.process(event)
    second = processor.process({**event, "retrieval_event_id": "dynamic-full:event-2"})

    survey = next(
        row for row in first["rows"] if row["paper_arxiv_id"] == "2001.00003"
    )
    assert survey["hard_filtered"] is True
    assert survey["rerank_rank"] is None
    assert "2001.00003" not in {
        row["paper_arxiv_id"] for row in first["top_rows"]
    }
    assert first["legacy_rerank_applied"] is False
    assert first["rerank_formula_id"] == "dynamic_rerank_v1"
    assert all(paper.score is not None for paper in first["papers"])
    assert len(policy_calls) == 1
    assert len(resolver.calls) == 2
    assert second["rerank_policy_id"] == first["rerank_policy_id"]


def test_postprocess_embedding_wrapper_enforces_bound():
    class TrackingProvider:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def embed_documents(self, texts):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, _text):
            return [1.0, 0.0]

    provider = TrackingProvider()
    bounded = BoundedEmbeddingProvider(provider, max_concurrency=1)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: bounded.embed_documents(["paper"]), range(4)))
    assert provider.max_active == 1


def test_query_scoped_embedding_cache_reuses_exact_documents_and_queries(monkeypatch):
    monkeypatch.setattr(config, "LOCAL_RERANK_EMBEDDING_BATCH_SIZE", 64)
    provider = FakeOllamaEmbeddings()
    cache = QueryScopedEmbeddingCache(provider)
    metadata = {
        "a": {"title": "alpha", "abstract": "paper"},
        "b": {"title": "beta", "abstract": "paper"},
        "c": {"title": "gamma", "abstract": "paper"},
    }

    cache.begin_query_scope("q1")
    first = CandidateIndex(["a", "b"], metadata, "embedding", cache)
    first_raw, _, _ = first.score("alpha query")
    second = CandidateIndex(["b", "c"], metadata, "embedding", cache)
    second_raw, _, _ = second.score("alpha query")
    stats = cache.snapshot_query_stats()

    assert first_raw["a"] == 1.0
    assert second_raw["b"] == 0.0
    assert provider.document_batch_sizes == [2, 1]
    assert stats["document_request_count"] == 4
    assert stats["document_backend_text_count"] == 3
    assert stats["query_request_count"] == 2
    assert stats["query_backend_call_count"] == 1
    assert stats["saved_embedding_count"] == 2
    assert stats["document_cache_entry_count"] == 3
    assert stats["query_cache_entry_count"] == 1

    cache.end_query_scope()
    cache.begin_query_scope("q2")
    CandidateIndex(["a"], metadata, "embedding", cache).score("alpha query")
    assert provider.document_batch_sizes == [2, 1, 1]
    assert cache.snapshot_query_stats()["backend_embedding_count"] == 2
    cache.end_query_scope()


def test_query_scoped_embedding_cache_singleflights_concurrent_misses():
    class TrackingProvider:
        def __init__(self):
            self.lock = threading.Lock()
            self.document_counts = {}
            self.query_calls = 0

        def embed_documents(self, texts):
            with self.lock:
                for text in texts:
                    self.document_counts[text] = self.document_counts.get(text, 0) + 1
            time.sleep(0.03)
            return [[float(len(text)), 1.0] for text in texts]

        def embed_query(self, text):
            with self.lock:
                self.query_calls += 1
            time.sleep(0.03)
            return [float(len(text)), 1.0]

    provider = TrackingProvider()
    cache = QueryScopedEmbeddingCache(provider)
    cache.begin_query_scope("concurrent")
    with ThreadPoolExecutor(max_workers=4) as executor:
        document_results = list(
            executor.map(
                lambda index: cache.embed_documents(["shared", f"unique-{index}"]),
                range(4),
            )
        )
    with ThreadPoolExecutor(max_workers=4) as executor:
        query_results = list(executor.map(cache.embed_query, ["same query"] * 4))

    assert provider.document_counts["shared"] == 1
    assert all(provider.document_counts[f"unique-{index}"] == 1 for index in range(4))
    assert provider.query_calls == 1
    assert all(len(rows) == 2 for rows in document_results)
    assert all(row.tolist() == query_results[0].tolist() for row in query_results)
    stats = cache.snapshot_query_stats()
    assert stats["document_request_count"] == 8
    assert stats["document_backend_text_count"] == 5
    assert stats["query_request_count"] == 4
    assert stats["query_backend_call_count"] == 1
    assert stats["saved_embedding_count"] == 6
    cache.end_query_scope()


def test_s2_cache_singleflight_avoids_duplicate_concurrent_api_calls(tmp_path):
    assert "publicationTypes" in S2GraphClient.PAPER_FIELDS
    assert "publicationTypes" in S2GraphClient.EDGE_PAPER_FIELDS

    class Response:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"paperId": "S2", "externalIds": {"ArXiv": "2001.00001"}}

    class Session:
        def __init__(self):
            self.calls = 0
            self.lock = threading.Lock()

        def get(self, *_args, **_kwargs):
            with self.lock:
                self.calls += 1
            time.sleep(0.03)
            return Response()

    client = S2GraphClient(str(tmp_path), rate_limit_rps=0)
    session = Session()
    client._session_for_thread = lambda: session
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: client.resolve("2001.00001"), range(4)))

    assert session.calls == 1
    assert [row[0]["paperId"] for row in results] == ["S2"] * 4
    assert sum(bool(row[1]) for row in results) == 3


def test_expanded_candidate_has_scores_rank_and_all_seed_origins():
    paper_db = {
        "2001.00001": {"title": "graph retrieval", "abstract": "query method", "date": "2001-01"},
        "2001.00002": {"title": "retrieval", "abstract": "subquery method", "date": "2001-02"},
        "2001.00003": {"title": "expanded graph method", "abstract": "query subquery", "date": "2001-03"},
        "2003.00004": {"title": "future paper", "abstract": "query", "date": "2003-01"},
    }
    processor = PerSubqueryProcessor(
        paper_db,
        FakeS2(),
        scoring_backend="bm25",
        embedding_provider=None,
    )
    event = {
        "query_id": "q1",
        "query": "graph query",
        "query_date": "2002-01",
        "iteration_idx": 1,
        "subquery_id": 7,
        "subquery": "retrieval method",
        "subquery_before_date": "2002-01",
        "results_per_query": 2,
        "seed_papers": [
            {"paper_arxiv_id": "2001.00001", "observed_retrieval_score": 5.0, "observed_retrieval_rank": 1},
            {"paper_arxiv_id": "2001.00002", "observed_retrieval_score": 4.0, "observed_retrieval_rank": 2},
        ],
    }
    result = processor.process(event)
    rows = {row["paper_arxiv_id"]: row for row in result["rows"]}
    expanded = rows["2001.00003"]
    assert expanded["retrieval_score_raw"] is not None
    assert expanded["retrieval_rank"] is not None
    assert expanded["source_seed_arxiv_ids"] == ["2001.00001", "2001.00002"]
    assert expanded["path_count"] == 2
    assert expanded["s2_publication_types"] == ["JournalArticle", "Review"]
    assert rows["2001.00001"]["s2_publication_types"] == ["JournalArticle"]
    assert len([edge for edge in result["edges"] if edge["expanded_arxiv_id"] == "2001.00003"]) == 2
    assert "2003.00004" not in rows


def test_artifact_query_staging_and_checkpoint_reconciliation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = ArtifactWriter(tmp, "full")

        writer.begin_query("q1", 1)
        writer.append("per_subquery/paper_rows.jsonl", {"query_id": "q1", "benchmark_idx": 1, "paper": "a"})
        assert not (root / "per_subquery/paper_rows.jsonl").exists()
        writer.commit_query()
        assert json.loads((root / "per_subquery/paper_rows.jsonl").read_text()) == {
            "benchmark_idx": 1,
            "paper": "a",
            "query_id": "q1",
        }

        # Simulate a crash during canonical flush for an uncommitted query.
        writer.append("per_subquery/paper_rows.jsonl", {"query_id": "q2", "benchmark_idx": 2, "paper": "b"})
        writer.append("per_subquery/filter_stats.jsonl", {"query_id": "q2", "kept": 3})
        with (root / "per_subquery/filter_stats.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"query_id":"q2"')
        stale_stage = root / ".staging/stale/per_subquery"
        stale_stage.mkdir(parents=True)
        (stale_stage / "paper_rows.jsonl").write_text('{"query_id":"q3"}\n')

        stats = writer.reconcile_with_checkpoint({1}, {"q1"})

        rows = [json.loads(line) for line in (root / "per_subquery/paper_rows.jsonl").read_text().splitlines()]
        assert rows == [{"benchmark_idx": 1, "paper": "a", "query_id": "q1"}]
        assert (root / "per_subquery/filter_stats.jsonl").read_text() == ""
        assert not (root / ".staging").exists()
        assert stats["removed_rows"] == 2
        assert stats["removed_malformed_rows"] == 1
        assert stats["removed_staging_directories"] == 1


def test_aborted_artifact_query_never_reaches_canonical_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ArtifactWriter(tmp, "full")
        writer.begin_query("q-abort", 4)
        writer.append("deep_merged/paper_rows.jsonl", {"query_id": "q-abort", "benchmark_idx": 4})
        writer.abort_query()

        assert not (Path(tmp) / "deep_merged/paper_rows.jsonl").exists()
