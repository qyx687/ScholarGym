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
    S2GraphClient,
)


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


def test_s2_cache_singleflight_avoids_duplicate_concurrent_api_calls(tmp_path):
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
        writer.append("deep_event/paper_rows.jsonl", {"query_id": "q-abort", "benchmark_idx": 4})
        writer.abort_query()

        assert not (Path(tmp) / "deep_event/paper_rows.jsonl").exists()
