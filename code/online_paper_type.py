#!/usr/bin/env python3
"""Resumable paper-type backends for online reranking."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import requests

from dimension_catalog import PAPER_TYPES
from paper_type import (
    CLASSIFIER_VERSION,
    QWEN_EVIDENCE_SOURCE,
    S2_CLASSIFIER_VERSION,
    S2_EVIDENCE_SOURCE,
    S2_SUPPORTED_CANONICAL_TYPES,
    PaperTypeClassifier,
    normalize_paper_id,
    s2_publication_types_to_record,
    validate_type_record,
)


S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_MAX_IDS = 500


def qwen_model_classifier_version(model: str, base_version: str) -> str:
    """Bind append-only Qwen cache records to the exact configured model."""

    model_digest = hashlib.sha256(str(model).strip().encode("utf-8")).hexdigest()[:16]
    return f"{base_version}.model-{model_digest}"


class PaperTypeResolver(Protocol):
    """Runtime contract shared by S2 and Qwen paper-type providers."""

    backend: str
    evidence_source: str
    classifier_version: str
    supported_types: Sequence[str]
    model: Optional[str]

    def resolve(self, paper_ids: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        ...

    def snapshot_stats(self) -> Dict[str, int]:
        ...

    def snapshot_cache(self) -> Dict[str, Dict[str, Any]]:
        ...


class _RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = (
            1.0 / float(requests_per_second)
            if requests_per_second and requests_per_second > 0
            else 0.0
        )
        self.last_request_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            delay = self.interval - (time.time() - self.last_request_at)
            if delay > 0:
                time.sleep(delay)
            self.last_request_at = time.time()


class S2PublicationTypeResolver:
    """Resolve arXiv IDs to S2 ``publicationTypes`` with an append-only cache.

    Missing or failed metadata is intentionally treated as unknown.  In
    particular, an untyped paper is never hard-filtered merely because S2 did
    not return a type.
    """

    backend = "s2"
    evidence_source = S2_EVIDENCE_SOURCE
    classifier_version = S2_CLASSIFIER_VERSION
    supported_types = S2_SUPPORTED_CANONICAL_TYPES
    model = None

    def __init__(
        self,
        cache_path: str | Path,
        *,
        api_key: str = "",
        requests_per_second: float = 1.0,
        offline: bool = False,
        timeout: int = 60,
        max_retries: int = 3,
        session: Optional[Any] = None,
        batch_url: str = S2_BATCH_URL,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.offline = bool(offline)
        self.timeout = int(timeout)
        self.max_retries = max(1, int(max_retries))
        self.batch_url = str(batch_url)
        self.session = session or requests.Session()
        resolved_key = api_key or os.environ.get("S2_API_KEY", "")
        self.session.headers.update(
            {
                "User-Agent": "ScholarGym-Online-Dynamic-Rerank/1.0",
                **({"x-api-key": resolved_key} if resolved_key else {}),
            }
        )
        self.limiter = _RateLimiter(requests_per_second)
        self._lock = threading.Lock()
        self._resolve_lock = threading.Lock()
        self._stats: Dict[str, int] = defaultdict(int)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._deferred_ids = set()
        self._load_cache()

    def _inc(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._stats[key] += int(amount)

    def snapshot_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def snapshot_cache(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                paper_id: dict(record) for paper_id, record in self._cache.items()
            }

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    record = validate_type_record(value)
                except (json.JSONDecodeError, ValueError, TypeError):
                    self._stats["cache_invalid_lines"] += 1
                    continue
                if (
                    record.get("evidence_source") != self.evidence_source
                    or record.get("classifier_version") != self.classifier_version
                ):
                    self._stats["cache_backend_mismatch_lines"] += 1
                    continue
                self._cache[record["paper_arxiv_id"]] = record
        self._stats["cache_records_loaded"] = len(self._cache)

    def _append_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        validated = [validate_type_record(record) for record in records]
        with self._lock:
            with self.cache_path.open("a", encoding="utf-8") as handle:
                for record in validated:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                handle.flush()
            for record in validated:
                self._cache[record["paper_arxiv_id"]] = record
            self._stats["cache_records_written"] += len(validated)

    def _fetch_batch(self, paper_ids: Sequence[str]) -> List[Dict[str, Any]]:
        if not paper_ids:
            return []
        if len(paper_ids) > S2_BATCH_MAX_IDS:
            raise ValueError(f"S2 batch cannot exceed {S2_BATCH_MAX_IDS} IDs")
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                response = self.session.post(
                    self.batch_url,
                    params={"fields": "paperId,publicationTypes"},
                    json={"ids": [f"ARXIV:{paper_id}" for paper_id in paper_ids]},
                    timeout=self.timeout,
                )
                self._inc("api_calls")
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(f"retryable S2 status {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, list) or len(payload) != len(paper_ids):
                    raise ValueError("S2 batch response length does not match request")
                return [
                    s2_publication_types_to_record(
                        paper_id,
                        value.get("publicationTypes")
                        if isinstance(value, Mapping)
                        else [],
                        resolved=isinstance(value, Mapping),
                    )
                    for paper_id, value in zip(paper_ids, payload)
                ]
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    self._inc("retries")
                    time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error

    def _fetch_resilient(self, paper_ids: Sequence[str]) -> List[Dict[str, Any]]:
        try:
            return self._fetch_batch(paper_ids)
        except Exception:
            # Batch failures are normally provider/network-wide.  Recursive
            # bisection is appropriate for offline cache building but can turn
            # one online event into hundreds of blocking requests.  Defer these
            # IDs for the rest of this process and leave their type unknown.
            with self._lock:
                self._stats["failed_batches"] += 1
                self._stats["failed_papers"] += len(paper_ids)
                self._deferred_ids.update(paper_ids)
            return []

    def resolve(self, paper_ids: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        # OnePass may call the resolver concurrently from independent event
        # workers. Keep cache-miss discovery and provider I/O single-flight so
        # overlapping IDs cannot be fetched twice before the first write lands.
        with self._resolve_lock:
            return self._resolve_once(paper_ids)

    def _resolve_once(self, paper_ids: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        ids = list(
            dict.fromkeys(
                paper_id
                for value in paper_ids
                if (paper_id := normalize_paper_id(value))
            )
        )
        with self._lock:
            uncached = [paper_id for paper_id in ids if paper_id not in self._cache]
            pending = [
                paper_id for paper_id in uncached if paper_id not in self._deferred_ids
            ]
            deferred = len(uncached) - len(pending)
            self._stats["cache_hits"] += len(ids) - len(uncached)
            self._stats["cache_misses"] += len(pending)
            self._stats["deferred_misses"] += deferred
        if pending and self.offline:
            self._inc("offline_misses", len(pending))
        elif pending:
            for offset in range(0, len(pending), S2_BATCH_MAX_IDS):
                records = self._fetch_resilient(
                    pending[offset : offset + S2_BATCH_MAX_IDS]
                )
                self._append_records(records)
        with self._lock:
            return {
                paper_id: dict(self._cache[paper_id])
                for paper_id in ids
                if paper_id in self._cache
            }


class QwenPaperTypeResolver:
    """Classify candidate title/abstract batches with Qwen and cache results.

    The classifier is query-independent: neither the original query nor the
    active rerank policy is included in its prompt. Missing metadata, malformed
    output, and provider failures remain unknown instead of causing hard drops.
    """

    backend = "qwen"
    evidence_source = QWEN_EVIDENCE_SOURCE
    supported_types = PAPER_TYPES

    def __init__(
        self,
        cache_path: str | Path,
        paper_db: Mapping[str, Mapping[str, Any]],
        model: str,
        *,
        is_local: bool = False,
        batch_size: int = 16,
        offline: bool = False,
        classifier: Optional[Any] = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.paper_db = {
            normalize_paper_id(paper_id): dict(metadata)
            for paper_id, metadata in paper_db.items()
            if normalize_paper_id(paper_id) and isinstance(metadata, Mapping)
        }
        self.model = str(model)
        self.offline = bool(offline)
        self.batch_size = max(1, int(batch_size))
        self.classifier = classifier or PaperTypeClassifier(
            self.model,
            is_local=is_local,
        )
        self.classifier_version = qwen_model_classifier_version(
            self.model,
            str(getattr(self.classifier, "classifier_version", CLASSIFIER_VERSION)),
        )
        self._lock = threading.Lock()
        self._resolve_lock = threading.Lock()
        self._stats: Dict[str, int] = defaultdict(int)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._deferred_ids = set()
        self._load_cache()

    def _inc(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._stats[key] += int(amount)

    def snapshot_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def snapshot_cache(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                paper_id: dict(record) for paper_id, record in self._cache.items()
            }

    def _normalize_record(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = dict(value)
        normalized.update(
            {
                "classifier_version": self.classifier_version,
                "evidence_source": self.evidence_source,
                "publication_types": [],
                "supported_types": list(self.supported_types),
                "negative_evidence_types": list(self.supported_types),
            }
        )
        return validate_type_record(normalized)

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = validate_type_record(json.loads(line))
                except (json.JSONDecodeError, ValueError, TypeError):
                    self._stats["cache_invalid_lines"] += 1
                    continue
                if (
                    record.get("evidence_source") != self.evidence_source
                    or record.get("classifier_version") != self.classifier_version
                ):
                    self._stats["cache_backend_mismatch_lines"] += 1
                    continue
                self._cache[record["paper_arxiv_id"]] = record
        self._stats["cache_records_loaded"] = len(self._cache)

    def _append_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            return
        validated = [self._normalize_record(record) for record in records]
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.cache_path.open("a", encoding="utf-8") as handle:
                for record in validated:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                handle.flush()
            for record in validated:
                self._cache[record["paper_arxiv_id"]] = record
            self._stats["cache_records_written"] += len(validated)

    def resolve(self, paper_ids: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        # Serialize the miss-to-cache-write path across concurrent OnePass
        # events. The narrower cache lock below keeps its existing semantics.
        with self._resolve_lock:
            return self._resolve_once(paper_ids)

    def _resolve_once(self, paper_ids: Iterable[Any]) -> Dict[str, Dict[str, Any]]:
        ids = list(
            dict.fromkeys(
                paper_id
                for value in paper_ids
                if (paper_id := normalize_paper_id(value))
            )
        )
        with self._lock:
            uncached = [paper_id for paper_id in ids if paper_id not in self._cache]
            pending = [
                paper_id for paper_id in uncached if paper_id not in self._deferred_ids
            ]
            deferred = len(uncached) - len(pending)
            self._stats["cache_hits"] += len(ids) - len(uncached)
            self._stats["cache_misses"] += len(pending)
            self._stats["deferred_misses"] += deferred

        if pending and self.offline:
            self._inc("offline_misses", len(pending))
        elif pending:
            available = []
            missing = []
            for paper_id in pending:
                metadata = self.paper_db.get(paper_id)
                if metadata is None:
                    missing.append(paper_id)
                    continue
                available.append(
                    {
                        "paper_arxiv_id": paper_id,
                        "title": str(metadata.get("title") or ""),
                        "abstract": str(metadata.get("abstract") or ""),
                    }
                )
            if missing:
                with self._lock:
                    self._stats["missing_paper_metadata"] += len(missing)
                    self._deferred_ids.update(missing)
            for offset in range(0, len(available), self.batch_size):
                batch = available[offset : offset + self.batch_size]
                batch_ids = [paper["paper_arxiv_id"] for paper in batch]
                try:
                    self._inc("classifier_batches")
                    records = self.classifier.classify_batch(batch)
                    self._inc("classifier_papers", len(batch))
                    self._append_records(records)
                except Exception:
                    # A failed online batch remains unknown for this process.
                    # The classifier already performs one schema-repair call;
                    # recursive retries here would multiply latency and cost.
                    with self._lock:
                        self._stats["failed_batches"] += 1
                        self._stats["failed_papers"] += len(batch_ids)
                        self._deferred_ids.update(batch_ids)

        with self._lock:
            return {
                paper_id: dict(self._cache[paper_id])
                for paper_id in ids
                if paper_id in self._cache
            }
