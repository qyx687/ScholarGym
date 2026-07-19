import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from dimension_catalog import PAPER_TYPES
from online_paper_type import QwenPaperTypeResolver, S2PublicationTypeResolver
from paper_type import CLASSIFIER_VERSION, s2_publication_types_to_record


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


def test_s2_type_cache_is_resumable_and_offline_safe(tmp_path):
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


def test_s2_batch_failure_is_deferred_without_recursive_requests(tmp_path):
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


def test_s2_overlapping_resolves_are_single_flight(tmp_path):
    class SlowSession(FakeSession):
        def post(self, *args, **kwargs):
            time.sleep(0.05)
            return super().post(*args, **kwargs)

    session = SlowSession()
    resolver = S2PublicationTypeResolver(
        tmp_path / "types.jsonl",
        requests_per_second=0,
        session=session,
    )
    barrier = threading.Barrier(2)

    def resolve():
        barrier.wait(timeout=2)
        return resolver.resolve(["2001.00001", "2001.00002"])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: resolve(), range(2)))

    assert results[0] == results[1]
    assert len(session.calls) == 1


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


def test_qwen_type_cache_is_batched_resumable_and_full_taxonomy(tmp_path):
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
    assert set(first["2001.00001"]["negative_evidence_types"]) == set(PAPER_TYPES)

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


def test_qwen_overlapping_resolves_are_single_flight(tmp_path):
    class SlowClassifier(FakeQwenTypeClassifier):
        def classify_batch(self, papers):
            time.sleep(0.05)
            return super().classify_batch(papers)

    paper_db = {
        "2001.00001": {"title": "Dataset one", "abstract": "A benchmark."},
        "2001.00002": {"title": "Dataset two", "abstract": "A benchmark."},
    }
    classifier = SlowClassifier()
    resolver = QwenPaperTypeResolver(
        tmp_path / "qwen-types.jsonl",
        paper_db,
        "qwen-test",
        classifier=classifier,
    )
    barrier = threading.Barrier(2)

    def resolve():
        barrier.wait(timeout=2)
        return resolver.resolve(paper_db)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: resolve(), range(2)))

    assert results[0] == results[1]
    assert len(classifier.calls) == 1


def test_qwen_cache_is_bound_to_the_configured_model(tmp_path):
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


def test_type_caches_reject_records_from_the_other_backend(tmp_path):
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
