import asyncio
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from deep_retrieval import DeepRetrievalProcessor
from graph_methods import (
    ArtifactWriter,
    BoundedEmbeddingProvider,
    PerSubqueryProcessor,
    QueryScopedEmbeddingCache,
)
from onepass_postprocess import OnePassPostprocessor, aggregate_postprocess_metrics
from rerank_skill import RerankSkill


class FakeS2:
    def expand(self, seed_ids, method, limit):
        rows = []
        for index, seed in enumerate(seed_ids, start=1):
            rows.append(
                {
                    "seed_arxiv_id": seed,
                    "expanded_arxiv_id": "2001.00003",
                    "seed_s2_paper_id": f"S{index}",
                    "expanded_s2_paper_id": "E3",
                    "edge_type": "citation",
                    "edge_rank": index,
                    "intents": ["methodology"],
                    "is_influential": True,
                }
            )
        return rows

    def snapshot_stats(self):
        return {"api_calls": 0}


class FakeSelector:
    def __init__(self):
        self.subqueries = []
        self.calls = []

    async def decide_for_subquery(self, papers, return_details=False, **kwargs):
        self.subqueries.append(kwargs["sub_query"])
        self.calls.append(
            {
                "paper_ids": [paper.id for paper in papers],
                "checklist": kwargs["planner_checklist"],
                "iteration_index": kwargs["iteration_index"],
                "old_overview": kwargs["old_overview"],
            }
        )
        selected = list(papers[:1])
        details = {"reasons": {selected[0].id: "kept"}} if selected else {"reasons": {}}
        return selected, "overview", {}, details


class ConcurrentS2(FakeS2):
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def expand(self, seed_ids, method, limit):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.04)
            return super().expand(seed_ids, method, limit)
        finally:
            with self.lock:
                self.active -= 1


class ConcurrentSelector(FakeSelector):
    def __init__(self):
        super().__init__()
        self.active = 0
        self.max_active = 0

    async def decide_for_subquery(self, papers, return_details=False, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.04)
            return await super().decide_for_subquery(
                papers=papers, return_details=return_details, **kwargs
            )
        finally:
            self.active -= 1


class ConcurrentDeepProcessor(DeepRetrievalProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def rerank_pool(self, *args, **kwargs):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.04)
            return super().rerank_pool(*args, **kwargs)
        finally:
            with self.lock:
                self.active -= 1


class FailFirstSelector(FakeSelector):
    def __init__(self):
        super().__init__()
        self.attempts = 0

    async def decide_for_subquery(self, *args, **kwargs):
        self.attempts += 1
        if self.attempts == 1:
            raise RuntimeError("per-subquery selector failed")
        return await super().decide_for_subquery(*args, **kwargs)


class NeverCallSelector:
    async def decide_for_subquery(self, *args, **kwargs):
        raise AssertionError("Stage A must not call a shadow Selector")


class FixedScores:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def get_scores(self, _tokens):
        return self.scores


class FakeRAG:
    def __init__(self, paper_db):
        ids = list(paper_db)
        self.bm25_index = FixedScores(range(len(ids), 0, -1))
        self.bm25_index_to_id = dict(enumerate(ids))
        self.paper_metadata = {
            paper_id: {**metadata, "arxiv_id": paper_id}
            for paper_id, metadata in paper_db.items()
        }

    @staticmethod
    def _preprocess_text_for_bm25(_text):
        return ["query"]


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_graph_events_and_selectors_are_bounded_concurrent_with_stable_artifact_order(tmp_path):
    paper_db = {
        f"2001.0000{index}": {
            "title": f"seed {index}",
            "abstract": "graph retrieval",
            "date": "2001-01",
        }
        for index in range(1, 6)
    }
    s2 = ConcurrentS2()
    selector = ConcurrentSelector()
    processor = PerSubqueryProcessor(
        paper_db, s2, scoring_backend="bm25", embedding_provider=None
    )
    events = []
    for index in range(1, 5):
        events.append(
            {
                "query_id": "q-concurrent",
                "benchmark_idx": 0,
                "query": "graph retrieval",
                "query_date": "2002-01",
                "iteration_idx": index,
                "subquery_id": index,
                "subquery": f"graph retrieval {index}",
                "subquery_before_date": "2002-01",
                "subquery_target_k": 1,
                "retrieval_event_id": f"event-{index}",
                "retrieval_page_idx": 1,
                "retrieval_offset": 0,
                "selector_top_k": 1,
                "planner_checklist": f"check-{index}",
                "retrieval_exclusion_arxiv_ids": [],
                "seed_papers": [
                    {
                        "paper_arxiv_id": f"2001.0000{index}",
                        "observed_retrieval_score": 1.0,
                        "observed_retrieval_rank": 1,
                    }
                ],
                "baseline_selected_arxiv_ids": [],
            }
        )

    manager = OnePassPostprocessor(
        selector=selector,
        paper_db=paper_db,
        writer=ArtifactWriter(str(tmp_path), "full"),
        s2_client=s2,
        per_subquery_processor=processor,
        deep_retrieval_processor=None,
        scoring_backend="bm25",
        embedding_provider=None,
        run_per_subquery=True,
        event_workers=4,
        selector_concurrency=3,
    )
    manager.process_query(
        {"query": "graph retrieval", "date": "2002-01"}, [], events, set()
    )

    assert s2.max_active >= 2
    assert selector.max_active == 3
    pool_records = _read_jsonl(tmp_path / "per_subquery/pool_records.jsonl")
    selector_records = _read_jsonl(tmp_path / "per_subquery/selector_decisions.jsonl")
    expected_order = [f"event-{index}" for index in range(1, 5)]
    assert [row["retrieval_event_id"] for row in pool_records] == expected_order
    assert [row["retrieval_event_id"] for row in selector_records] == expected_order


def test_onepass_dynamic_policy_is_built_once_and_recorded_for_all_events(tmp_path):
    calls = []

    def policy_llm(prompt, *args, **kwargs):
        calls.append(prompt)
        return json.dumps(
            {
                "policy_version": "dynamic_rerank_v1",
                "query_intent": "method_search",
                "weight_levels": {
                    "query_similarity": "high",
                    "subquery_similarity": "very_high",
                    "intent_background": "off",
                    "intent_method": "medium",
                    "intent_result": "off",
                    "path_count": "low",
                    "paper_type_alignment": "off",
                },
                "paper_type_rules": [],
                "confidence": 0.95,
            }
        )

    paper_db = {
        "2001.00001": {
            "title": "seed",
            "abstract": "graph retrieval",
            "date": "2001-01",
        },
        "2001.00003": {
            "title": "expanded",
            "abstract": "method graph",
            "date": "2001-03",
        },
    }
    processor = PerSubqueryProcessor(
        paper_db,
        FakeS2(),
        scoring_backend="bm25",
        embedding_provider=None,
        rerank_skill=RerankSkill("qwen-test", llm_call=policy_llm),
    )
    event_base = {
        "query_id": "q-dynamic",
        "benchmark_idx": 0,
        "query": "find graph retrieval methods",
        "query_date": "2002-01",
        "subquery_id": 1,
        "subquery": "graph method",
        "subquery_before_date": "2002-01",
        "subquery_target_k": 1,
        "retrieval_page_idx": 1,
        "retrieval_offset": 0,
        "selector_top_k": 1,
        "planner_checklist": "find methods",
        "retrieval_exclusion_arxiv_ids": [],
        "seed_papers": [
            {
                "paper_arxiv_id": "2001.00001",
                "observed_retrieval_score": 1.0,
                "observed_retrieval_rank": 1,
            }
        ],
        "baseline_selected_arxiv_ids": [],
    }
    events = [
        {**event_base, "retrieval_event_id": "q-dynamic:event-1"},
        {
            **event_base,
            "retrieval_event_id": "q-dynamic:event-2",
            "retrieval_page_idx": 2,
        },
    ]
    manager = OnePassPostprocessor(
        selector=FakeSelector(),
        paper_db=paper_db,
        writer=ArtifactWriter(str(tmp_path), "full"),
        s2_client=processor.s2,
        per_subquery_processor=processor,
        deep_retrieval_processor=None,
        scoring_backend="bm25",
        embedding_provider=None,
        run_per_subquery=True,
        event_workers=2,
    )

    result = manager.process_query(
        {"query": "find graph retrieval methods", "date": "2002-01"},
        [],
        events,
        set(),
    )

    assert len(calls) == 1
    assert result["per_subquery"]["dynamic_rerank_enabled"] is True
    policy_rows = _read_jsonl(tmp_path / "query_rerank_policies.jsonl")
    assert len(policy_rows) == 1
    assert policy_rows[0]["dynamic_rerank_enabled"] is True
    assert policy_rows[0]["formula_scope"] == (
        "one_policy_per_original_query_all_events"
    )
    pool_rows = _read_jsonl(tmp_path / "per_subquery/pool_records.jsonl")
    assert {row["rerank_policy_id"] for row in pool_rows} == {
        policy_rows[0]["rerank_policy_id"]
    }
    assert all(row["dynamic_rerank_enabled"] for row in pool_rows)


def test_dense_graph_concurrency_keeps_postprocess_embedding_calls_serial(tmp_path):
    class TrackingEmbeddingProvider:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def _enter(self):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)

        def _exit(self):
            with self.lock:
                self.active -= 1

        def embed_documents(self, texts):
            self._enter()
            try:
                time.sleep(0.02)
                return [[1.0, 0.0] for _ in texts]
            finally:
                self._exit()

        def embed_query(self, _text):
            self._enter()
            try:
                time.sleep(0.02)
                return [1.0, 0.0]
            finally:
                self._exit()

    paper_db = {
        f"2001.0000{index}": {
            "title": f"paper {index}",
            "abstract": "dense graph retrieval",
            "date": "2001-01",
        }
        for index in range(1, 6)
    }
    raw_provider = TrackingEmbeddingProvider()
    bounded_provider = BoundedEmbeddingProvider(raw_provider, max_concurrency=1)
    cached_provider = QueryScopedEmbeddingCache(bounded_provider)
    s2 = ConcurrentS2()
    processor = PerSubqueryProcessor(
        paper_db,
        s2,
        scoring_backend="embedding",
        embedding_provider=cached_provider,
    )
    events = [
        {
            "query_id": "q-dense-concurrent",
            "benchmark_idx": 0,
            "query": "dense graph retrieval",
            "query_date": "2002-01",
            "iteration_idx": index,
            "subquery_id": index,
            "subquery": f"dense graph retrieval {index}",
            "subquery_before_date": "2002-01",
            "subquery_target_k": 1,
            "retrieval_event_id": f"dense-event-{index}",
            "retrieval_page_idx": 1,
            "retrieval_offset": 0,
            "selector_top_k": 1,
            "planner_checklist": "check",
            "retrieval_exclusion_arxiv_ids": [],
            "seed_papers": [
                {
                    "paper_arxiv_id": f"2001.0000{index}",
                    "observed_retrieval_score": 1.0,
                    "observed_retrieval_rank": 1,
                }
            ],
            "baseline_selected_arxiv_ids": [],
        }
        for index in range(1, 5)
    ]
    manager = OnePassPostprocessor(
        selector=FakeSelector(),
        paper_db=paper_db,
        writer=ArtifactWriter(str(tmp_path), "full"),
        s2_client=s2,
        per_subquery_processor=processor,
        deep_retrieval_processor=None,
        scoring_backend="embedding",
        embedding_provider=cached_provider,
        run_per_subquery=True,
        event_workers=4,
    )

    result = manager.process_query(
        {"query": "dense graph retrieval", "date": "2002-01"}, [], events, set()
    )

    assert s2.max_active >= 2
    assert raw_provider.max_active == 1
    cache_stats = result["postprocess_embedding_cache_stats"]
    assert cache_stats["enabled"] is True
    assert cache_stats["saved_embedding_count"] > 0
    assert cache_stats["backend_embedding_count"] < cache_stats["total_request_count"]
    after_scope = cached_provider.snapshot_query_stats()
    assert after_scope["enabled"] is False
    assert after_scope["document_cache_entry_count"] == 0
    assert after_scope["query_cache_entry_count"] == 0
    pool_records = _read_jsonl(tmp_path / "per_subquery/pool_records.jsonl")
    assert [row["retrieval_event_id"] for row in pool_records] == [
        f"dense-event-{index}" for index in range(1, 5)
    ]


def test_graph_shadow_uses_frozen_baseline_exclusion_for_expanded_candidates():
    paper_db = {
        "2001.00001": {"title": "seed", "abstract": "graph", "date": "2001-01"},
        "2001.00003": {"title": "old selection", "abstract": "graph", "date": "2001-03"},
    }
    fake_s2 = FakeS2()
    graph_processor = PerSubqueryProcessor(
        paper_db, fake_s2, scoring_backend="bm25", embedding_provider=None
    )
    event = {
        "query_id": "q-exclusion",
        "benchmark_idx": 0,
        "query": "graph",
        "query_date": "2002-01",
        "iteration_idx": 2,
        "subquery_id": 1,
        "subquery": "graph",
        "subquery_before_date": "2002-01",
        "retrieval_event_id": "q-exclusion:retrieval:i2:s1:p2",
        "retrieval_page_idx": 2,
        "retrieval_offset": 10,
        "selector_top_k": 1,
        "planner_checklist": "check",
        "retrieval_exclusion_arxiv_ids": ["2001.00003"],
        "seed_papers": [
            {
                "paper_arxiv_id": "2001.00001",
                "observed_retrieval_score": 1.0,
                "observed_retrieval_rank": 1,
            }
        ],
        "baseline_selected_arxiv_ids": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        manager = OnePassPostprocessor(
            selector=FakeSelector(),
            paper_db=paper_db,
            writer=ArtifactWriter(tmp, "full"),
            s2_client=fake_s2,
            per_subquery_processor=graph_processor,
            deep_retrieval_processor=None,
            scoring_backend="bm25",
            embedding_provider=None,
            run_per_subquery=True,
        )
        manager.process_query({"query": "graph", "date": "2002-01"}, [], [event], set())

        pool = _read_jsonl(Path(tmp) / "per_subquery/pool_records.jsonl")[0]
        stats = _read_jsonl(Path(tmp) / "per_subquery/filter_stats.jsonl")[0]
        assert pool["local_pool_arxiv_ids"] == ["2001.00001"]
        assert stats["previously_selected_count"] == 1


def test_per_subquery_selector_failure_does_not_block_deep_merged_arm():
    paper_db = {
        "2001.00001": {"title": "seed", "abstract": "graph query", "date": "2001-01"},
        "2001.00003": {"title": "expanded", "abstract": "graph query", "date": "2001-03"},
    }
    fake_s2 = FakeS2()
    selector = FailFirstSelector()
    graph_processor = PerSubqueryProcessor(
        paper_db, fake_s2, scoring_backend="bm25", embedding_provider=None
    )
    deep_processor = DeepRetrievalProcessor(
        FakeRAG(paper_db), paper_db, scoring_backend="bm25", embedding_provider=None
    )
    event = {
        "query_id": "q-selector-failure",
        "benchmark_idx": 0,
        "query": "graph query",
        "query_date": "2002-01",
        "iteration_idx": 1,
        "subquery_id": 1,
        "subquery": "graph query",
        "subquery_before_date": "2002-01",
        "retrieval_event_id": "q-selector-failure:retrieval:i1:s1:p1",
        "retrieval_page_idx": 1,
        "retrieval_offset": 0,
        "selector_top_k": 1,
        "planner_checklist": "check",
        "retrieval_exclusion_arxiv_ids": [],
        "seed_papers": [
            {
                "paper_arxiv_id": "2001.00001",
                "observed_retrieval_score": 1.0,
                "observed_retrieval_rank": 1,
            }
        ],
        "baseline_selected_arxiv_ids": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        manager = OnePassPostprocessor(
            selector=selector,
            paper_db=paper_db,
            writer=ArtifactWriter(tmp, "full"),
            s2_client=fake_s2,
            per_subquery_processor=graph_processor,
            deep_retrieval_processor=deep_processor,
            scoring_backend="bm25",
            embedding_provider=None,
            run_per_subquery=True,
            run_deep_merged=True,
        )
        result = manager.process_query(
            {"query": "graph query", "date": "2002-01"}, [], [event], set()
        )

    assert result["per_subquery"]["selector_failed_events"] == 1
    assert result["per_subquery"]["error"]
    assert "error" not in result["deep_merged"]


def test_one_query_reuses_baseline_and_runs_graph_plus_deep_merged_control():
    paper_db = {
        "2001.00001": {"title": "seed one", "abstract": "graph query", "date": "2001-01"},
        "2001.00002": {"title": "seed two", "abstract": "retrieval", "date": "2001-02"},
        "2001.00003": {"title": "expanded", "abstract": "graph retrieval", "date": "2001-03"},
        "2001.00004": {"title": "deep four", "abstract": "graph query", "date": "2001-04"},
        "2001.00005": {"title": "deep five", "abstract": "retrieval query", "date": "2001-05"},
        "2001.00006": {"title": "deep six", "abstract": "retrieval graph", "date": "2001-06"},
    }
    fake_s2 = FakeS2()
    fake_selector = FakeSelector()
    processor = PerSubqueryProcessor(paper_db, fake_s2, scoring_backend="bm25", embedding_provider=None)
    deep_processor = ConcurrentDeepProcessor(
        FakeRAG(paper_db), paper_db, scoring_backend="bm25", embedding_provider=None
    )
    with tempfile.TemporaryDirectory() as tmp:
        writer = ArtifactWriter(tmp, "full")
        manager = OnePassPostprocessor(
            selector=fake_selector,
            paper_db=paper_db,
            writer=writer,
            s2_client=fake_s2,
            per_subquery_processor=processor,
            deep_retrieval_processor=deep_processor,
            scoring_backend="bm25",
            embedding_provider=None,
            run_per_subquery=True,
            run_deep_merged=True,
        )
        base = {
            "query_id": "q1",
            "benchmark_idx": 0,
            "query": "graph query",
            "query_date": "2002-01",
            "iteration_idx": 1,
            # The CLI page-size setting is deliberately smaller than the three
            # actual retrieval records across this complete baseline query.
            "results_per_query": 1,
            "planner_checklist": "check",
            "retrieval_backend": "bm25",
            "retrieval_exclusion_arxiv_ids": [],
        }
        events = [
            {
                **base,
                "subquery_id": 1,
                "subquery": "graph",
                "subquery_target_k": 7,
                "subquery_link_type": "derive",
                "subquery_before_date": "2002-01",
                "retrieval_page_idx": 1,
                "retrieval_event_id": "q1:retrieval:i1:s1:p1",
                "retrieval_offset": 0,
                "selector_top_k": 1,
                "seed_papers": [{"paper_arxiv_id": "2001.00001", "observed_retrieval_score": 2, "observed_retrieval_rank": 1}],
                "baseline_selected_arxiv_ids": ["2001.00001"],
            },
            {
                **base,
                "subquery_id": 2,
                "subquery": "retrieval",
                "subquery_target_k": 8,
                "subquery_link_type": "expand",
                "subquery_before_date": "2002-01",
                "retrieval_page_idx": 1,
                "retrieval_event_id": "q1:retrieval:i1:s2:p1",
                "retrieval_offset": 0,
                "selector_top_k": 2,
                "seed_papers": [
                    {"paper_arxiv_id": "2001.00001", "observed_retrieval_score": 3, "observed_retrieval_rank": 1},
                    {"paper_arxiv_id": "2001.00002", "observed_retrieval_score": 1, "observed_retrieval_rank": 2},
                ],
                "baseline_selected_arxiv_ids": [],
            },
        ]
        result = manager.process_query({"query": "graph query", "date": "2002-01"}, [], events, {"2001.00003"})
        assert set(result) >= {"baseline", "per_subquery", "deep_merged"}
        assert "global" not in result
        assert result["deep_merged"]["event_count"] == 2
        assert result["deep_merged"]["subquery_group_count"] == 2
        assert deep_processor.max_active >= 2
        assert [(sq.target_k, sq.link_type) for sq in fake_selector.subqueries[:2]] == [(7, "derive"), (8, "expand")]
        pool_records = _read_jsonl(Path(tmp) / "deep_merged/pool_records.jsonl")
        assert len(pool_records) == 2
        assert [row["subquery_id"] for row in pool_records] == [1, 2]
        for record in pool_records:
            assert record["deep_pool_rows"]
            assert {
                "paper_arxiv_id",
                "deep_retrieval_score_raw",
                "deep_retrieval_rank_in_local_pool",
                "rerank_score",
                "rerank_rank",
                "source_graph_event_features",
            } <= set(record["deep_pool_rows"][0])
        comparisons = _read_jsonl(Path(tmp) / "deep_merged/comparisons.jsonl")
        assert all(
            "pool_comparison" in row and "all_slices_topk_comparison" in row
            for row in comparisons
        )
        assert not (Path(tmp) / "global").exists()

    with tempfile.TemporaryDirectory() as tmp:
        minimal_manager = OnePassPostprocessor(
            selector=FakeSelector(),
            paper_db=paper_db,
            writer=ArtifactWriter(tmp, "minimal"),
            s2_client=fake_s2,
            per_subquery_processor=processor,
            deep_retrieval_processor=deep_processor,
            scoring_backend="bm25",
            embedding_provider=None,
            run_per_subquery=True,
            run_deep_merged=True,
        )
        minimal_manager.process_query(
            {"query": "graph query", "date": "2002-01"}, [], events, {"2001.00003"}
        )
        assert (Path(tmp) / "per_subquery/pool_records.jsonl").exists()
        assert (Path(tmp) / "deep_merged/pool_records.jsonl").exists()
        assert (Path(tmp) / "deep_merged/comparisons.jsonl").exists()
        assert not (Path(tmp) / "per_subquery/paper_rows.jsonl").exists()
        assert not (Path(tmp) / "deep_merged/paper_rows.jsonl").exists()


def test_merged_deep_control_sends_continue_topk_as_sequential_disjoint_slices():
    paper_db = {
        f"2001.0000{index}": {
            "title": f"paper {index} graph retrieval",
            "abstract": f"query method {index}",
            "date": f"2001-0{index}",
        }
        for index in range(1, 9)
    }
    fake_s2 = FakeS2()
    fake_selector = FakeSelector()
    graph_processor = PerSubqueryProcessor(
        paper_db, fake_s2, scoring_backend="bm25", embedding_provider=None
    )
    deep_processor = DeepRetrievalProcessor(
        FakeRAG(paper_db), paper_db, scoring_backend="bm25", embedding_provider=None
    )
    base = {
        "schema_version": "1.0",
        "query_id": "q-continue",
        "benchmark_idx": 3,
        "query": "graph retrieval query",
        "query_date": "2001-09",
        "subquery_id": 5,
        "subquery": "graph retrieval",
        "subquery_target_k": 1,
        "subquery_link_type": "continue",
        "subquery_before_date": "2001-09",
        "results_per_query": 1,
        "selector_top_k": 1,
        "retrieval_backend": "bm25",
    }
    events = [
        {
            **base,
            "iteration_idx": 1,
            "retrieval_page_idx": 1,
            "retrieval_event_id": "q-continue:retrieval:i1:s5:p1",
            "retrieval_offset": 0,
            "planner_checklist": "first checklist",
            "retrieval_exclusion_arxiv_ids": [],
            "seed_papers": [
                {"paper_arxiv_id": "2001.00001", "observed_retrieval_score": 8.0, "observed_retrieval_rank": 1}
            ],
            "baseline_selected_arxiv_ids": ["2001.00001"],
        },
        {
            **base,
            "iteration_idx": 2,
            "retrieval_page_idx": 2,
            "retrieval_event_id": "q-continue:retrieval:i2:s5:p2",
            "retrieval_offset": 1,
            "planner_checklist": "second checklist",
            "retrieval_exclusion_arxiv_ids": ["2001.00001"],
            "seed_papers": [
                {"paper_arxiv_id": "2001.00002", "observed_retrieval_score": 7.0, "observed_retrieval_rank": 1}
            ],
            "baseline_selected_arxiv_ids": [],
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        manager = OnePassPostprocessor(
            selector=fake_selector,
            paper_db=paper_db,
            writer=ArtifactWriter(tmp, "full"),
            s2_client=fake_s2,
            per_subquery_processor=graph_processor,
            deep_retrieval_processor=deep_processor,
            scoring_backend="bm25",
            embedding_provider=None,
            run_per_subquery=True,
            run_deep_merged=True,
        )
        result = manager.process_query(
            {"query": base["query"], "date": base["query_date"]},
            [],
            events,
            {"2001.00003"},
        )

        assert result["deep_merged"]["subquery_group_count"] == 1
        merged_calls = fake_selector.calls[-2:]
        assert merged_calls[0]["paper_ids"]
        assert merged_calls[1]["paper_ids"]
        assert set(merged_calls[0]["paper_ids"]).isdisjoint(merged_calls[1]["paper_ids"])
        assert [call["checklist"] for call in merged_calls] == [
            "first checklist",
            "second checklist",
        ]
        assert [call["iteration_index"] for call in merged_calls] == [1, 2]
        assert {call["old_overview"] for call in merged_calls} == {""}
        merged_record = _read_jsonl(Path(tmp) / "deep_merged/pool_records.jsonl")[0]
        assert merged_record["subquery_id"] == 5
        assert [slice_["selector_input_arxiv_ids"] for slice_ in merged_record["selector_slices"]] == [
            call["paper_ids"] for call in merged_calls
        ]
        assert len(merged_record["source_event_budgets"]) == 2
        merged_rows = _read_jsonl(Path(tmp) / "deep_merged/paper_rows.jsonl")
        assert any(row["source_graph_event_features"] for row in merged_rows)


def test_postprocess_metrics_aggregate_unique_queries_and_two_methods():
    stale = {
        "idx": 0,
        "postprocess_results": {
            "per_subquery": {
                "gt_count": 99,
                "candidate_count": 99,
                "selected_count": 99,
                "candidate_gt_ids": [],
                "selected_gt_ids": [],
            }
        },
    }
    query_zero = {
        "idx": 0,
        "postprocess_results": {
            "per_subquery": {
                "gt_count": 2,
                "candidate_count": 4,
                "selected_count": 2,
                "candidate_gt_ids": ["a"],
                "selected_gt_ids": ["a"],
                "candidate_recall": 0.5,
                "candidate_precision": 0.25,
                "selection_recall": 0.5,
                "selection_precision": 0.5,
            },
            "deep_merged": {"error": "selector failed"},
        },
    }
    query_one = {
        "idx": 1,
        "postprocess_results": {
            "per_subquery": {
                "gt_count": 1,
                "candidate_count": 2,
                "selected_count": 1,
                "candidate_gt_ids": ["b"],
                "selected_gt_ids": ["b"],
                "candidate_recall": 1.0,
                "candidate_precision": 0.5,
                "selection_recall": 1.0,
                "selection_precision": 1.0,
            },
            "deep_merged": {
                "gt_count": 1,
                "candidate_count": 3,
                "selected_count": 1,
                "candidate_gt_ids": ["b"],
                "selected_gt_ids": [],
                "candidate_recall": 1.0,
                "candidate_precision": 1.0 / 3.0,
                "selection_recall": 0.0,
                "selection_precision": 0.0,
            },
        },
    }

    metrics = aggregate_postprocess_metrics([stale, query_zero, query_one])

    assert metrics["source_query_count"] == 2
    per_subquery = metrics["per_subquery"]
    assert per_subquery["evaluated_query_count"] == 2
    assert per_subquery["total_gt_count"] == 3
    assert per_subquery["avg_candidate_recall"] == 0.75
    assert per_subquery["avg_candidate_precision"] == 0.375
    assert per_subquery["micro_candidate_recall"] == 2.0 / 3.0
    assert per_subquery["micro_selection_precision"] == 2.0 / 3.0
    assert metrics["deep_merged"]["evaluated_query_count"] == 1
    assert metrics["deep_merged"]["failed_query_count"] == 1


def test_stage_a_materializes_both_pools_without_legacy_rerank_or_selector(tmp_path):
    paper_db = {
        "2001.00001": {
            "title": "seed graph query",
            "abstract": "retrieval",
            "date": "2001-01",
        },
        "2001.00002": {
            "title": "deep query",
            "abstract": "retrieval",
            "date": "2001-02",
        },
        "2001.00003": {
            "title": "expanded methodology",
            "abstract": "graph query",
            "date": "2001-03",
        },
    }
    s2 = FakeS2()
    graph_processor = PerSubqueryProcessor(
        paper_db, s2, scoring_backend="bm25", embedding_provider=None
    )
    deep_processor = DeepRetrievalProcessor(
        FakeRAG(paper_db), paper_db, scoring_backend="bm25", embedding_provider=None
    )

    def legacy_rerank_must_not_run(*_args, **_kwargs):
        raise AssertionError("Stage A entered a legacy rerank method")

    graph_processor.process = legacy_rerank_must_not_run
    deep_processor.rerank_pool = legacy_rerank_must_not_run
    event = {
        "schema_version": "1.0",
        "query_id": "q-stage-a",
        "benchmark_idx": 0,
        "query": "graph query",
        "query_date": "2001-12",
        "iteration_idx": 1,
        "subquery_id": 1,
        "subquery": "graph retrieval",
        "subquery_target_k": 1,
        "subquery_link_type": "derive",
        "parent_subquery_id": 0,
        "subquery_before_date": "2001-12",
        "retrieval_event_id": "q-stage-a:retrieval:i1:s1:p1",
        "retrieval_page_idx": 1,
        "retrieval_offset": 0,
        "results_per_query": 1,
        "selector_top_k": 1,
        "planner_checklist": "retain methods",
        "retrieval_exclusion_arxiv_ids": [],
        "seed_papers": [
            {
                "paper_arxiv_id": "2001.00001",
                "observed_retrieval_score": 1.0,
                "observed_retrieval_rank": 1,
            }
        ],
        "baseline_selected_arxiv_ids": ["2001.00001"],
    }
    manager = OnePassPostprocessor(
        selector=NeverCallSelector(),
        paper_db=paper_db,
        writer=ArtifactWriter(str(tmp_path), "full"),
        s2_client=s2,
        per_subquery_processor=graph_processor,
        deep_retrieval_processor=deep_processor,
        scoring_backend="bm25",
        embedding_provider=None,
        run_per_subquery=True,
        run_deep_merged=True,
        postprocess_stage="materialize",
    )

    result = manager.process_query(
        {"query": "graph query", "date": "2001-12"},
        [],
        [event],
        {"2001.00003"},
    )

    assert result["postprocess_stage"] == "materialize"
    for method in ("per_subquery", "deep_merged"):
        assert result[method]["materialization_complete"] is True
        assert result[method]["legacy_rerank_applied"] is False
        assert result[method]["shadow_selector_applied"] is False
        assert result[method]["selector_call_count"] == 0

    graph_record = _read_jsonl(
        tmp_path / "per_subquery/pool_records.jsonl"
    )[0]
    graph_row = graph_record["local_pool_rows"][0]
    assert graph_record["local_pool_size"] == 2
    assert "selector_topk_arxiv_ids" not in graph_record
    assert "selected_arxiv_ids" not in graph_record
    assert {
        "query_score_raw",
        "query_score_normalized",
        "subquery_score_raw",
        "subquery_score_normalized",
        "intent_score",
        "path_count",
        "path_count_normalized",
        "source_seed_arxiv_ids",
        "intent_labels",
        "materialization_order_rank",
    } <= set(graph_row)
    assert not {"feature_weights", "rerank_score", "rerank_rank"} & set(graph_row)

    merged_record = _read_jsonl(
        tmp_path / "deep_merged/pool_records.jsonl"
    )[0]
    deep_row = merged_record["deep_pool_rows"][0]
    assert merged_record["deep_retrieval_order_arxiv_ids"] == [
        row["paper_arxiv_id"] for row in merged_record["deep_pool_rows"]
    ]
    assert "selector_slices" not in merged_record
    assert "deep_rerank_order_arxiv_ids" not in merged_record
    assert "feature_scorable" in deep_row
    assert not {"feature_weights", "rerank_score", "rerank_rank"} & set(deep_row)
    assert merged_record["source_event_budgets"][0]["planner_checklist"] == (
        "retain methods"
    )
    assert merged_record["source_event_budgets"][0]["selector_top_k"] == 1
    assert merged_record["source_graph_pool_occurrence_budget"] == 2

    for method in ("per_subquery", "deep_merged"):
        assert not (tmp_path / method / "selector_decisions.jsonl").exists()

    aggregate = aggregate_postprocess_metrics(
        [{"idx": 0, "postprocess_results": result}]
    )
    assert aggregate["postprocess_stage"] == "materialize"
    assert aggregate["per_subquery"]["materialized_query_count"] == 1
    assert aggregate["per_subquery"]["avg_pool_recall"] == 1.0
    assert aggregate["per_subquery"]["avg_candidate_recall"] is None
    assert aggregate["deep_merged"]["micro_selection_recall"] is None


def test_stage_a_pool_aggregates_exclude_zero_gt_queries_without_dropping_artifacts():
    def materialized_result(ground_truth_count, pool_count, pool_gt_count, recall):
        return {
            "postprocess_stage": "materialize",
            "materialization_complete": True,
            "ground_truth_count": ground_truth_count,
            "local_pool_count": pool_count,
            "local_pool_gt_count": pool_gt_count,
            "local_pool_recall": recall,
            "local_pool_precision": (
                pool_gt_count / pool_count if pool_count else 0.0
            ),
        }

    metrics = aggregate_postprocess_metrics(
        [
            {
                "idx": 0,
                "postprocess_results": {
                    "postprocess_stage": "materialize",
                    "per_subquery": materialized_result(2, 4, 1, 0.5),
                },
            },
            {
                "idx": 1,
                "postprocess_results": {
                    "postprocess_stage": "materialize",
                    "per_subquery": materialized_result(0, 7, 0, 0.0),
                },
            },
        ]
    )

    per_subquery = metrics["per_subquery"]
    assert per_subquery["materialized_query_count"] == 2
    assert per_subquery["pool_evaluated_query_count"] == 1
    assert per_subquery["queries_without_ground_truth_count"] == 1
    assert per_subquery["total_materialized_pool_count"] == 11
    assert per_subquery["total_pool_count"] == 4
    assert per_subquery["avg_pool_recall"] == 0.5


def test_stage_a_rolls_back_query_when_an_enabled_materializer_fails(tmp_path):
    class FailingS2(FakeS2):
        def expand(self, *_args, **_kwargs):
            raise RuntimeError("temporary graph failure")

    paper_db = {
        "2001.00001": {
            "title": "seed",
            "abstract": "paper",
            "date": "2001-01",
        }
    }
    s2 = FailingS2()
    manager = OnePassPostprocessor(
        selector=NeverCallSelector(),
        paper_db=paper_db,
        writer=ArtifactWriter(str(tmp_path), "full"),
        s2_client=s2,
        per_subquery_processor=PerSubqueryProcessor(
            paper_db, s2, scoring_backend="bm25", embedding_provider=None
        ),
        deep_retrieval_processor=None,
        scoring_backend="bm25",
        embedding_provider=None,
        run_per_subquery=True,
        postprocess_stage="materialize",
    )
    event = {
        "query_id": "q-stage-a-retry",
        "benchmark_idx": 4,
        "query": "query",
        "query_date": "2001-12",
        "iteration_idx": 1,
        "subquery_id": 1,
        "subquery": "query",
        "subquery_before_date": "2001-12",
        "retrieval_event_id": "q-stage-a-retry:event-1",
        "retrieval_page_idx": 1,
        "retrieval_offset": 0,
        "selector_top_k": 1,
        "retrieval_exclusion_arxiv_ids": [],
        "seed_papers": [
            {"paper_arxiv_id": "2001.00001", "observed_retrieval_rank": 1}
        ],
        "baseline_selected_arxiv_ids": [],
    }

    with pytest.raises(RuntimeError, match="not committed and can be retried"):
        manager.process_query(
            {"query": "query", "date": "2001-12"}, [], [event], set()
        )

    assert not (tmp_path / "per_subquery/pool_records.jsonl").exists()
    assert not (tmp_path / "per_subquery/errors.jsonl").exists()
    assert not (tmp_path / "query_results.jsonl").exists()
