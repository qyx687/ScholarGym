import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from deep_retrieval import DeepRetrievalProcessor


class FakeDocument:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeVectorStore:
    def __init__(self, rows):
        self.rows = rows
        self.requested_k = None

    def similarity_search_with_score(self, query, k):
        assert query == "stable subquery"
        self.requested_k = k
        return self.rows[:k]


class ProgressiveVectorStore:
    def __init__(self):
        self.requested_ks = []

    def similarity_search_with_score(self, query, k):
        assert query == "stable subquery"
        self.requested_ks.append(k)
        future = [
            (
                FakeDocument(
                    {
                        "arxiv_id": f"2002.{index:05d}",
                        "date": "2002-01",
                        "title": "future",
                        "abstract": "paper",
                    }
                ),
                1.0 - index / 10000.0,
            )
            for index in range(500)
        ]
        if k <= 500:
            return future
        valid = [
            (
                FakeDocument(
                    {
                        "arxiv_id": f"2001.9{index:04d}",
                        "date": "2001-01",
                        "title": "valid",
                        "abstract": "paper",
                    }
                ),
                0.4 - index / 1000.0,
            )
            for index in range(3)
        ]
        return future + valid


class FixedScores:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def get_scores(self, _tokens):
        return self.scores


def test_bm25_deep_retrieval_matches_baseline_exclusion_before_offset_order():
    ids = [f"2001.0000{index}" for index in range(1, 7)]
    rag = SimpleNamespace(
        bm25_index=FixedScores(range(len(ids), 0, -1)),
        bm25_index_to_id=dict(enumerate(ids)),
        paper_metadata={
            paper_id: {
                "arxiv_id": paper_id,
                "date": "2001-01",
                "title": paper_id,
                "abstract": "body",
            }
            for paper_id in ids
        },
        _preprocess_text_for_bm25=lambda _text: ["query"],
    )
    processor = DeepRetrievalProcessor(
        rag, {}, scoring_backend="bm25", embedding_provider=None
    )

    pools, _ = processor._retrieve_bm25_requests(
        subquery="query",
        before_date="2001-12",
        requests={
            "first": {"offset": 0, "count": 3, "exclude_arxiv_ids": []},
            "continue": {
                "offset": 2,
                "count": 2,
                "exclude_arxiv_ids": ["2001.00002"],
            },
        },
    )

    assert [row["paper_arxiv_id"] for row in pools["first"]] == ids[:3]
    assert [row["paper_arxiv_id"] for row in pools["continue"]] == ids[3:5]


def test_integrated_text_only_rerank_keeps_graph_weights_and_zero_graph_features():
    paper_db = {
        "2001.00001": {"title": "query method", "abstract": "query", "date": "2001-01"},
        "2001.00002": {"title": "subquery method", "abstract": "subquery", "date": "2001-02"},
    }
    processor = DeepRetrievalProcessor(
        SimpleNamespace(), paper_db, scoring_backend="bm25", embedding_provider=None
    )
    result = processor.rerank_pool(
        [
            {
                "paper_arxiv_id": paper_id,
                "deep_retrieval_score_raw": score,
                "deep_retrieval_rank_global_date_valid": rank,
                "deep_retrieval_rank_after_exclusion": rank,
                "deep_retrieval_rank_in_local_pool": rank,
                "_metadata": metadata,
            }
            for rank, (paper_id, metadata, score) in enumerate(
                [
                    ("2001.00001", paper_db["2001.00001"], 2.0),
                    ("2001.00002", paper_db["2001.00002"], 1.0),
                ],
                start=1,
            )
        ],
        query="query",
        subquery="subquery",
        cutoff="2001-12",
    )

    for row in result["rows"]:
        assert row["intent_score"] == 0.0
        assert row["path_count_normalized"] == 0.0
        assert row["rerank_score"] == (
            0.30 * row["query_score_normalized"]
            + 0.40 * row["subquery_score_normalized"]
        )
        assert 0.0 <= row["rerank_score"] <= 0.7


def test_vector_deep_retrieval_honors_date_exclusion_then_offset_and_records_ranks():
    store = FakeVectorStore(
        [
            (FakeDocument({"arxiv_id": "2001.00001", "date": "2001-01", "title": "one", "abstract": "a"}), 0.99),
            (FakeDocument({"arxiv_id": "2001.00002v2", "date": "2001-02", "title": "two", "abstract": "b"}), 0.98),
            # Canonical duplicate must not consume an offset position.
            (FakeDocument({"arxiv_id": "2001.00002", "date": "2001-02", "title": "two", "abstract": "b"}), 0.97),
            # Future paper must not enter artifacts or consume an offset position.
            (FakeDocument({"arxiv_id": "2002.00009", "date": "2002-01", "title": "future", "abstract": "x"}), 0.96),
            (FakeDocument({"arxiv_id": "2001.00003", "date": "2001-03", "title": "three", "abstract": "c"}), 0.95),
            (FakeDocument({"arxiv_id": "2001.00004", "date": "2001-04", "title": "four", "abstract": "d"}), 0.94),
        ]
    )
    rag = SimpleNamespace(qdrant_vector_store=store)
    processor = DeepRetrievalProcessor(
        rag, {}, scoring_backend="embedding", embedding_provider=None
    )

    pools, diagnostics = processor._retrieve_vector_requests(
        subquery="stable subquery",
        before_date="2001-12",
        requests={
            "page": {
                "offset": 1,
                "count": 2,
                "exclude_arxiv_ids": ["2001.00001"],
            }
        },
    )

    assert [row["paper_arxiv_id"] for row in pools["page"]] == [
        "2001.00003",
        "2001.00004",
    ]
    assert [row["deep_retrieval_score_raw"] for row in pools["page"]] == [0.95, 0.94]
    assert [row["deep_retrieval_rank_after_exclusion"] for row in pools["page"]] == [2, 3]
    assert [row["deep_retrieval_rank_in_local_pool"] for row in pools["page"]] == [1, 2]
    assert diagnostics["page"]["actual_count"] == 2
    assert diagnostics["page"]["canonical_arxiv_deduplication_before_exclusion_offset"] is True
    assert store.requested_k >= 6


def test_vector_deep_retrieval_expands_fetch_until_date_valid_budget_is_full():
    store = ProgressiveVectorStore()
    processor = DeepRetrievalProcessor(
        SimpleNamespace(qdrant_vector_store=store),
        {},
        scoring_backend="embedding",
        embedding_provider=None,
    )

    pools, diagnostics = processor._retrieve_vector_requests(
        subquery="stable subquery",
        before_date="2001-12",
        requests={"page": {"offset": 0, "count": 2, "exclude_arxiv_ids": []}},
    )

    assert [row["paper_arxiv_id"] for row in pools["page"]] == [
        "2001.90000",
        "2001.90001",
    ]
    assert store.requested_ks == [500, 1000]
    assert diagnostics["page"]["vector_fetch_attempts"] == 2
    assert diagnostics["page"]["actual_count"] == 2
