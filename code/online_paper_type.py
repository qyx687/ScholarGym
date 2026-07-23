#!/usr/bin/env python3
"""Resumable native Semantic Scholar publication-type resolution."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence

import requests

from paper_type import (
    S2_CLASSIFIER_VERSION,
    S2_EVIDENCE_SOURCE,
    S2_PUBLICATION_TYPES,
    normalize_paper_id,
    s2_publication_types_to_record,
    validate_type_record,
)


S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_MAX_IDS = 500

class PaperTypeResolver(Protocol):
    """Runtime contract for native S2 publication-type metadata."""

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
    supported_types = S2_PUBLICATION_TYPES
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
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise TypeError("paper-type cache record must be an object")
                    raw_source = str(value.get("evidence_source") or "").strip()
                    raw_version = str(value.get("classifier_version") or "").strip()
                    if (
                        raw_source
                        and raw_source != self.evidence_source
                    ) or (
                        not raw_source
                        and raw_version
                        and not raw_version.startswith("s2_")
                    ):
                        self._stats["cache_backend_mismatch_lines"] += 1
                        raise ValueError(
                            "native S2 cache contains non-S2 record at line "
                            f"{line_number}: source={raw_source or 'missing'}, "
                            f"classifier_version={raw_version or 'missing'}"
                        )
                    record = validate_type_record(value)
                except ValueError as exc:
                    if "native S2 cache contains non-S2 record" in str(exc):
                        raise
                    self._stats["cache_invalid_lines"] += 1
                    continue
                except TypeError:
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
