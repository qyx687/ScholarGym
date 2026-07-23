import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from online_paper_type import S2PublicationTypeResolver
from paper_type import S2_PUBLICATION_TYPES


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
    assert first["2001.00001"]["publication_types"] == ["Review"]
    assert set(resolver.supported_types) == set(S2_PUBLICATION_TYPES)

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


def test_s2_cache_rejects_non_s2_records(tmp_path):
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
