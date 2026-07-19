#!/usr/bin/env python3
"""Shared graph expansion, local scoring, and analysis artifact helpers.

Paper identity in artifacts is intentionally limited to normalized arXiv IDs.
Titles and abstracts are used in memory for scoring and Selector calls but are
never emitted by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import quote, unquote

import numpy as np
import requests
from rank_bm25 import BM25Okapi

import config
from structures import Paper, SubQuery


EPSILON = 1e-12
RERANK_FORMULA_ID = "q030_sq040_intent015_path015_closed_pool_minmax_v1"
PAPER_EMBEDDING_SERIALIZATION_ID = (
    "scholargym_baseline_title_newline_space_abstract_v1"
)
DEFAULT_FEATURE_WEIGHTS = {
    "query_score_normalized": 0.30,
    "subquery_score_normalized": 0.40,
    "intent_score": 0.15,
    "path_count_normalized": 0.15,
}
INTENT_WEIGHTS = {"methodology": 1.0, "result": 0.75, "background": 0.35}
QUERY_SCOPED_EMBEDDING_CACHE_POLICY = (
    "query_scoped_exact_text_singleflight_float32_v1"
)


class BoundedEmbeddingProvider:
    """Limit concurrent local embedding calls without changing their backend.

    Baseline Qdrant retrieval keeps its original provider.  This wrapper is
    used only by postprocessing candidate-pool scorers, where many independent
    events may otherwise submit large Ollama batches at the same time.
    """

    def __init__(self, provider: Any, max_concurrency: int = 1) -> None:
        if provider is None:
            raise ValueError("provider is required")
        if int(max_concurrency) < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.provider = provider
        self.max_concurrency = int(max_concurrency)
        self._semaphore = threading.BoundedSemaphore(self.max_concurrency)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        with self._semaphore:
            return self.provider.embed_documents(list(texts))

    def embed_query(self, text: str) -> List[float]:
        with self._semaphore:
            return self.provider.embed_query(text)


class _PendingEmbedding:
    """One in-flight exact-text embedding shared by concurrent callers."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.vector: Optional[np.ndarray] = None
        self.error: Optional[BaseException] = None


class QueryScopedEmbeddingCache:
    """Share exact-text postprocess embeddings within one benchmark query.

    Graph, deep-event, and deep-merged pools repeatedly serialize the same
    papers and score the same query/subquery strings.  This wrapper stores raw
    float32 vectors for the lifetime of one benchmark query and coalesces
    concurrent misses, while leaving pool-local normalization and ranking in
    ``CandidateIndex`` unchanged.  Document and query caches are deliberately
    separate because embedding backends may apply different instructions to
    their two APIs.

    Calls made outside ``begin_query_scope``/``end_query_scope`` pass through
    unchanged.  The baseline retriever receives the unwrapped provider, so the
    cache affects only the three postprocess arms.
    """

    policy = QUERY_SCOPED_EMBEDDING_CACHE_POLICY

    def __init__(self, provider: Any) -> None:
        if provider is None:
            raise ValueError("provider is required")
        self.provider = provider
        self._lock = threading.Lock()
        self._active = False
        self._scope_id: Optional[str] = None
        self._document_cache: Dict[str, np.ndarray] = {}
        self._query_cache: Dict[str, np.ndarray] = {}
        self._document_pending: Dict[str, _PendingEmbedding] = {}
        self._query_pending: Dict[str, _PendingEmbedding] = {}
        self._stats: Dict[str, int] = self._empty_stats()

    @staticmethod
    def _empty_stats() -> Dict[str, int]:
        return {
            "document_request_count": 0,
            "document_duplicate_input_count": 0,
            "document_cache_hit_unique_text_count": 0,
            "document_inflight_wait_unique_text_count": 0,
            "document_backend_batch_count": 0,
            "document_backend_text_count": 0,
            "query_request_count": 0,
            "query_cache_hit_count": 0,
            "query_inflight_wait_count": 0,
            "query_backend_call_count": 0,
        }

    @staticmethod
    def _raw_vector(value: Any) -> np.ndarray:
        # CandidateIndex converts every backend result to float32 immediately;
        # storing that representation avoids retaining Python-float lists that
        # are several times larger without changing the downstream arithmetic.
        vector = np.asarray(value, dtype=np.float32)
        if vector.ndim != 1:
            raise RuntimeError("embedding provider returned an unexpected vector shape")
        return vector.copy()

    def begin_query_scope(self, scope_id: Any) -> None:
        with self._lock:
            if self._active:
                raise RuntimeError("an embedding-cache query scope is already active")
            if self._document_pending or self._query_pending:
                raise RuntimeError("cannot start an embedding-cache scope with pending calls")
            self._active = True
            self._scope_id = str(scope_id) if scope_id is not None else "unknown"
            self._document_cache.clear()
            self._query_cache.clear()
            self._stats = self._empty_stats()

    def snapshot_query_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats: Dict[str, Any] = dict(self._stats)
            stats.update(
                {
                    "enabled": self._active,
                    "policy": self.policy,
                    "scope_id": self._scope_id,
                    "document_cache_entry_count": len(self._document_cache),
                    "query_cache_entry_count": len(self._query_cache),
                    "cached_vector_bytes": sum(
                        int(vector.nbytes)
                        for vector in list(self._document_cache.values())
                        + list(self._query_cache.values())
                    ),
                }
            )
        total_requests = int(stats["document_request_count"]) + int(
            stats["query_request_count"]
        )
        backend_texts = int(stats["document_backend_text_count"]) + int(
            stats["query_backend_call_count"]
        )
        saved = max(0, total_requests - backend_texts)
        stats["total_request_count"] = total_requests
        stats["backend_embedding_count"] = backend_texts
        stats["saved_embedding_count"] = saved
        stats["reuse_ratio"] = (saved / total_requests) if total_requests else 0.0
        return stats

    def end_query_scope(self) -> Dict[str, Any]:
        stats = self.snapshot_query_stats()
        with self._lock:
            if self._document_pending or self._query_pending:
                raise RuntimeError("cannot end an embedding-cache scope with pending calls")
            self._active = False
            self._scope_id = None
            self._document_cache.clear()
            self._query_cache.clear()
        return stats

    @staticmethod
    def _publish_error(
        pending_map: Dict[str, _PendingEmbedding],
        owned: Mapping[str, _PendingEmbedding],
        error: BaseException,
        lock: threading.Lock,
    ) -> None:
        with lock:
            for text, pending in owned.items():
                pending.error = error
                if pending_map.get(text) is pending:
                    pending_map.pop(text, None)
                pending.event.set()

    def embed_documents(self, texts: Sequence[str]) -> List[Any]:
        items = list(texts)
        if not items:
            return []
        with self._lock:
            if not self._active:
                passthrough = True
                owned: Dict[str, _PendingEmbedding] = {}
                pending_for_text: Dict[str, _PendingEmbedding] = {}
                resolved: Dict[str, np.ndarray] = {}
            else:
                passthrough = False
                unique_texts = list(dict.fromkeys(items))
                self._stats["document_request_count"] += len(items)
                self._stats["document_duplicate_input_count"] += (
                    len(items) - len(unique_texts)
                )
                owned = {}
                pending_for_text = {}
                resolved = {}
                for text in unique_texts:
                    cached = self._document_cache.get(text)
                    if cached is not None:
                        resolved[text] = cached
                        self._stats["document_cache_hit_unique_text_count"] += 1
                        continue
                    pending = self._document_pending.get(text)
                    if pending is not None:
                        pending_for_text[text] = pending
                        self._stats["document_inflight_wait_unique_text_count"] += 1
                        continue
                    pending = _PendingEmbedding()
                    self._document_pending[text] = pending
                    pending_for_text[text] = pending
                    owned[text] = pending
                if owned:
                    self._stats["document_backend_batch_count"] += 1
                    self._stats["document_backend_text_count"] += len(owned)

        if passthrough:
            return self.provider.embed_documents(items)

        if owned:
            try:
                raw_vectors = self.provider.embed_documents(list(owned))
                if len(raw_vectors) != len(owned):
                    raise RuntimeError(
                        "embedding provider returned an unexpected document count"
                    )
                vectors = [self._raw_vector(vector) for vector in raw_vectors]
            except BaseException as exc:
                self._publish_error(
                    self._document_pending, owned, exc, self._lock
                )
                raise
            with self._lock:
                for (text, pending), vector in zip(owned.items(), vectors):
                    self._document_cache[text] = vector
                    pending.vector = vector
                    if self._document_pending.get(text) is pending:
                        self._document_pending.pop(text, None)
                    pending.event.set()

        for text, pending in pending_for_text.items():
            pending.event.wait()
            if pending.error is not None:
                raise pending.error
            if pending.vector is None:
                raise RuntimeError("embedding single-flight completed without a vector")
            resolved[text] = pending.vector
        return [resolved[text] for text in items]

    def embed_query(self, text: str) -> Any:
        with self._lock:
            if not self._active:
                passthrough = True
                pending = None
                owned = False
            else:
                passthrough = False
                self._stats["query_request_count"] += 1
                cached = self._query_cache.get(text)
                if cached is not None:
                    self._stats["query_cache_hit_count"] += 1
                    return cached
                pending = self._query_pending.get(text)
                if pending is not None:
                    owned = False
                    self._stats["query_inflight_wait_count"] += 1
                else:
                    pending = _PendingEmbedding()
                    self._query_pending[text] = pending
                    owned = True
                    self._stats["query_backend_call_count"] += 1

        if passthrough:
            return self.provider.embed_query(text)

        assert pending is not None
        if owned:
            try:
                vector = self._raw_vector(self.provider.embed_query(text))
            except BaseException as exc:
                self._publish_error(
                    self._query_pending, {text: pending}, exc, self._lock
                )
                raise
            with self._lock:
                self._query_cache[text] = vector
                pending.vector = vector
                if self._query_pending.get(text) is pending:
                    self._query_pending.pop(text, None)
                pending.event.set()

        pending.event.wait()
        if pending.error is not None:
            raise pending.error
        if pending.vector is None:
            raise RuntimeError("embedding single-flight completed without a vector")
        return pending.vector


def normalize_arxiv_id(value: Any) -> str:
    if value is None:
        return ""
    text = unquote(str(value)).strip().strip("[](){}<>.,;:'\"")
    if not text or text.lower() in {"n/a", "na", "none", "null", "nan", "unknown"}:
        return ""
    text = re.sub(r"(?i)^arxiv\s*:\s*", "", text)
    url_match = re.search(r"(?i)arxiv\.org/(?:abs|pdf|html)/([^?#\s]+)", text)
    if url_match:
        text = url_match.group(1)
    text = re.sub(r"(?i)\.pdf$", "", text)
    match = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", text)
    if match:
        return match.group(1)
    match = re.search(r"(?i)([a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)?/\d{7})(?:v\d+)?", text)
    if match:
        return re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE).lower()
    return re.sub(r"v\d+$", "", text, flags=re.IGNORECASE).lower()


def month_key(value: Any) -> str:
    text = str(value or "").strip()
    return text[:7] if len(text) >= 7 else ""


def tokenize(text: str) -> List[str]:
    # Match ScholarGym's baseline BM25 preprocessing exactly.
    return re.findall(r"\b[a-z]+\b", str(text or "").lower())


def baseline_paper_embedding_text(title: Any, abstract: Any) -> str:
    """Return the exact paper text used by ScholarGym build_vector_db.py."""

    title_text = str(title or "")
    abstract_text = str(abstract or "")
    if not title_text and not abstract_text:
        return ""
    return f"title: {title_text}\n abstract: {abstract_text}"


def minmax(values: Mapping[str, float], ids: Sequence[str]) -> Dict[str, float]:
    clean = [float(values.get(paper_id, 0.0) or 0.0) for paper_id in ids]
    if not clean:
        return {}
    low, high = min(clean), max(clean)
    if high - low <= EPSILON:
        return {paper_id: 0.0 for paper_id in ids}
    return {paper_id: (value - low) / (high - low) for paper_id, value in zip(ids, clean)}


def ranks_for(scores: Mapping[str, float], ids: Sequence[str]) -> Dict[str, int]:
    ordered = sorted(ids, key=lambda paper_id: (-float(scores.get(paper_id, 0.0)), paper_id))
    return {paper_id: rank for rank, paper_id in enumerate(ordered, start=1)}


class ArtifactWriter:
    """JSONL writer with optional query-scoped staging and reconciliation."""

    def __init__(self, output_dir: str, save_level: str = "minimal") -> None:
        if save_level not in {"minimal", "full"}:
            raise ValueError("save_level must be minimal or full")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_level = save_level
        self._lock = threading.Lock()
        self._transaction_dir: Optional[Path] = None

    def begin_query(self, query_id: Any, benchmark_idx: Any) -> None:
        """Stage subsequent JSONL appends outside the canonical artifacts."""
        with self._lock:
            if self._transaction_dir is not None:
                raise RuntimeError("an artifact query transaction is already active")
            identity = f"{benchmark_idx}:{query_id}"
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            index_text = str(benchmark_idx) if benchmark_idx is not None else "unknown"
            transaction_dir = self.output_dir / ".staging" / f"idx-{index_text}-{digest}"
            if transaction_dir.exists():
                shutil.rmtree(transaction_dir)
            transaction_dir.mkdir(parents=True, exist_ok=True)
            self._transaction_dir = transaction_dir

    def append(self, relative_path: str, record: Mapping[str, Any], *, full_only: bool = False) -> None:
        if full_only and self.save_level != "full":
            return
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
        with self._lock:
            base_dir = self._transaction_dir or self.output_dir
            path = base_dir / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()

    def commit_query(self) -> None:
        """Flush a completed query's staged files into the legacy flat JSONLs."""
        with self._lock:
            transaction_dir = self._transaction_dir
            if transaction_dir is None:
                raise RuntimeError("no artifact query transaction is active")
            for staged_path in sorted(transaction_dir.rglob("*.jsonl")):
                relative_path = staged_path.relative_to(transaction_dir)
                destination = self.output_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                with staged_path.open("rb") as source, destination.open("ab") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                    target.flush()
            shutil.rmtree(transaction_dir)
            self._transaction_dir = None

    def abort_query(self) -> None:
        """Discard the active staging directory after an interrupted query."""
        with self._lock:
            if self._transaction_dir is not None:
                shutil.rmtree(self._transaction_dir, ignore_errors=True)
                self._transaction_dir = None

    @staticmethod
    def _record_is_committed(
        record: Mapping[str, Any],
        committed_indices: Set[str],
        committed_query_ids: Set[str],
    ) -> bool:
        benchmark_idx = record.get("benchmark_idx")
        if benchmark_idx is not None:
            return str(benchmark_idx) in committed_indices
        query_id = record.get("query_id")
        if query_id is not None:
            return str(query_id) in committed_query_ids
        # Preserve non-query metadata conservatively.
        return True

    def reconcile_with_checkpoint(
        self,
        committed_indices: Iterable[Any],
        committed_query_ids: Iterable[Any],
    ) -> Dict[str, Any]:
        """Atomically remove canonical rows not committed by detailed_results."""
        index_tokens = {str(value) for value in committed_indices if value is not None}
        query_tokens = {str(value) for value in committed_query_ids if value is not None}
        stats: Dict[str, Any] = {
            "committed_index_count": len(index_tokens),
            "committed_query_id_count": len(query_tokens),
            "removed_rows": 0,
            "removed_malformed_rows": 0,
            "changed_files": {},
            "removed_staging_directories": 0,
        }
        with self._lock:
            if self._transaction_dir is not None:
                raise RuntimeError("cannot reconcile artifacts during an active query transaction")
            staging_root = self.output_dir / ".staging"
            if staging_root.exists():
                stats["removed_staging_directories"] = sum(
                    1 for path in staging_root.iterdir() if path.is_dir()
                )
                shutil.rmtree(staging_root)

            for path in sorted(self.output_dir.rglob("*.jsonl")):
                if ".staging" in path.parts:
                    continue
                requires_rewrite = False
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            requires_rewrite = True
                            break
                        if not isinstance(record, Mapping) or not self._record_is_committed(
                            record, index_tokens, query_tokens
                        ):
                            requires_rewrite = True
                            break
                if not requires_rewrite:
                    continue

                removed = 0
                malformed = 0
                tmp = path.with_suffix(path.suffix + ".reconcile.tmp")
                with path.open("r", encoding="utf-8") as source, tmp.open("w", encoding="utf-8") as target:
                    for line in source:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            malformed += 1
                            continue
                        if not isinstance(record, Mapping) or not self._record_is_committed(
                            record, index_tokens, query_tokens
                        ):
                            removed += 1
                            continue
                        target.write(line if line.endswith("\n") else line + "\n")
                    target.flush()
                os.replace(tmp, path)
                relative_path = str(path.relative_to(self.output_dir))
                stats["changed_files"][relative_path] = {
                    "removed_rows": removed,
                    "removed_malformed_rows": malformed,
                }
                stats["removed_rows"] += removed
                stats["removed_malformed_rows"] += malformed
        return stats

    def write_json(self, relative_path: str, value: Mapping[str, Any]) -> None:
        path = self.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)


def _normalized_embeddings(provider: Any, texts: Sequence[str]) -> np.ndarray:
    """Embed text with ScholarGym's LangChain/Ollama client and return cosine-ready rows.

    ``OllamaEmbeddings`` intentionally has no custom API/cache wrapper.  Supporting
    ``embed`` as a fallback keeps the local scorers easy to unit test without
    changing the production dense path.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    batch_size = max(1, int(getattr(config, "LOCAL_RERANK_EMBEDDING_BATCH_SIZE", 64)))
    batches: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        if hasattr(provider, "embed_documents"):
            vectors = provider.embed_documents(batch)
        elif hasattr(provider, "embed"):
            vectors = provider.embed(batch)
        else:
            raise TypeError("embedding provider must implement embed_documents or embed")
        batch_matrix = np.asarray(vectors, dtype=np.float32)
        if batch_matrix.ndim != 2 or batch_matrix.shape[0] != len(batch):
            raise RuntimeError("embedding provider returned an unexpected matrix shape")
        batches.append(batch_matrix)
    matrix = np.concatenate(batches, axis=0)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise RuntimeError("embedding provider returned an unexpected matrix shape")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, EPSILON)


def _normalized_query_embedding(provider: Any, text: str) -> np.ndarray:
    if hasattr(provider, "embed_query"):
        vector = np.asarray(provider.embed_query(text), dtype=np.float32)
        if vector.ndim != 1:
            raise RuntimeError("embedding provider returned an unexpected query shape")
        norm = float(np.linalg.norm(vector))
        return vector / max(norm, EPSILON)
    return _normalized_embeddings(provider, [text])[0]


class CandidateIndex:
    """Same-backend scoring over one closed seed+expanded candidate pool."""

    def __init__(
        self,
        candidate_ids: Sequence[str],
        metadata: Mapping[str, Mapping[str, Any]],
        backend: str,
        embedding_provider: Optional[Any] = None,
    ) -> None:
        self.backend = backend
        self.ids: List[str] = []
        self.texts: List[str] = []
        for paper_id in sorted(set(candidate_ids)):
            item = metadata.get(paper_id) or {}
            title = item.get("title") or ""
            abstract = item.get("abstract") or ""
            if backend == "embedding":
                # Match the paper serialization in ScholarGym's baseline Qdrant
                # builder so retrieval and closed-pool reranking encode papers
                # through the same model input format.
                text = baseline_paper_embedding_text(title, abstract)
            else:
                # Preserve the established graph-method BM25 local corpus.
                text = f"{title} {abstract}".strip()
            if text:
                self.ids.append(paper_id)
                self.texts.append(text)
        self.embedding_provider = embedding_provider
        self._bm25 = BM25Okapi([tokenize(text) for text in self.texts]) if backend == "bm25" and self.texts else None
        self._document_vectors = None
        if backend == "embedding" and self.texts:
            if embedding_provider is None:
                raise ValueError("embedding_provider is required for embedding rerank")
            self._document_vectors = _normalized_embeddings(embedding_provider, self.texts)

    def score(self, query: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        if not self.ids:
            return {}, {}, {}
        if self.backend == "bm25":
            values = self._bm25.get_scores(tokenize(query or "")) if self._bm25 is not None else np.zeros(len(self.ids))
        elif self.backend == "embedding":
            query_vector = _normalized_query_embedding(self.embedding_provider, query or "")
            if self._document_vectors.shape[1] != query_vector.shape[0]:
                raise RuntimeError(
                    "embedding dimension mismatch between local candidate papers and query"
                )
            values = self._document_vectors @ query_vector
        else:
            raise ValueError(f"unsupported scoring backend: {self.backend}")
        raw = {paper_id: float(values[index]) for index, paper_id in enumerate(self.ids)}
        normalized = minmax(raw, self.ids)
        return raw, normalized, ranks_for(raw, self.ids)


class _RateLimiter:
    def __init__(self, rps: float) -> None:
        self.interval = 1.0 / rps if rps and rps > 0 else 0.0
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            delay = self.interval - (time.time() - self.last)
            if delay > 0:
                time.sleep(delay)
            self.last = time.time()


class S2GraphClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    PAPER_FIELDS = (
        "paperId,externalIds,citationCount,referenceCount,publicationTypes"
    )
    EDGE_PAPER_FIELDS = (
        "paperId,externalIds,citationCount,referenceCount,publicationTypes"
    )

    def __init__(
        self,
        cache_dir: str,
        *,
        api_key: str = "",
        rate_limit_rps: float = 4.0,
        offline: bool = False,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("S2_API_KEY", "")
        self.offline = offline
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = _RateLimiter(rate_limit_rps)
        self._thread_local = threading.local()
        self._session_headers = {"User-Agent": "ScholarGym-Graph-Rerank/1.0"}
        if self.api_key:
            self._session_headers["x-api-key"] = self.api_key
        self.stats = defaultdict(int)
        self._lock = threading.Lock()
        self._cache_locks: Dict[str, threading.Lock] = {}

    def _session_for_thread(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self._session_headers)
            self._thread_local.session = session
        return session

    def _cache_lock(self, path: Path) -> threading.Lock:
        key = str(path)
        with self._lock:
            lock = self._cache_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._cache_locks[key] = lock
            return lock

    def snapshot_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self.stats)

    def _inc(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _cache_path(self, kind: str, key: str, suffix: str = "") -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", key)[:180]
        path = self.cache_dir / kind / f"{safe}{suffix}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _get(self, endpoint: str, params: Mapping[str, Any], path: Path) -> Tuple[Optional[dict], bool, int]:
        # Concurrent events often share seeds.  Serialize identical cache keys
        # so only one worker calls S2 and the others reuse its atomic result.
        with self._cache_lock(path):
            return self._get_locked(endpoint, params, path)

    def _get_locked(self, endpoint: str, params: Mapping[str, Any], path: Path) -> Tuple[Optional[dict], bool, int]:
        if path.exists():
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("endpoint") == endpoint and cached.get("params") == dict(params):
                    self._inc("cache_hits")
                    return cached.get("data"), True, 0
            except Exception:
                self._inc("cache_errors")
        self._inc("cache_misses")
        if self.offline:
            self._inc("offline_misses")
            return None, False, 0
        url = self.BASE_URL + endpoint
        last_status = 0
        for attempt in range(1, self.max_retries + 1):
            self.limiter.wait()
            try:
                response = self._session_for_thread().get(
                    url, params=dict(params), timeout=self.timeout
                )
                last_status = response.status_code
                self._inc("api_calls")
                if response.status_code == 404:
                    return None, False, attempt - 1
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                response.raise_for_status()
                data = response.json()
                tmp = path.with_suffix(
                    path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp"
                )
                tmp.write_text(json.dumps({"endpoint": endpoint, "params": dict(params), "data": data}), encoding="utf-8")
                os.replace(tmp, path)
                return data, False, attempt - 1
            except Exception:
                self._inc("api_errors")
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        self._inc(f"http_{last_status or 'exception'}")
        return None, False, self.max_retries - 1

    def resolve(self, arxiv_id: str) -> Tuple[Optional[dict], bool, int]:
        endpoint = f"/paper/{quote('ARXIV:' + arxiv_id, safe='')}"
        return self._get(endpoint, {"fields": self.PAPER_FIELDS}, self._cache_path("resolve", arxiv_id))

    def relations(self, s2_id: str, relation: str, limit: int) -> Tuple[List[dict], bool, int]:
        if relation == "citation":
            endpoint = f"/paper/{quote(s2_id, safe='')}/citations"
            fields = ",".join(f"citingPaper.{field}" for field in self.EDGE_PAPER_FIELDS.split(",")) + ",isInfluential,intents"
        else:
            endpoint = f"/paper/{quote(s2_id, safe='')}/references"
            fields = ",".join(f"citedPaper.{field}" for field in self.EDGE_PAPER_FIELDS.split(",")) + ",isInfluential,intents"
        data, hit, retries = self._get(
            endpoint,
            {"fields": fields, "offset": 0, "limit": int(limit)},
            self._cache_path(relation, s2_id, f".limit{limit}"),
        )
        rows = data.get("data") if isinstance(data, dict) else []
        return rows if isinstance(rows, list) else [], hit, retries

    @staticmethod
    def _external_arxiv(item: Mapping[str, Any]) -> str:
        for key, value in (item.get("externalIds") or {}).items():
            if str(key).lower() == "arxiv":
                return normalize_arxiv_id(value)
        return ""

    def expand(self, seed_ids: Sequence[str], method: str, limit: int) -> List[Dict[str, Any]]:
        relations = ["citation", "reference"] if method == "citations_references" else [method.rstrip("s")]
        edges: List[Dict[str, Any]] = []
        for seed_id in dict.fromkeys(seed_ids):
            seed, seed_hit, seed_retries = self.resolve(seed_id)
            if not seed or not seed.get("paperId"):
                continue
            for relation in relations:
                items, relation_hit, relation_retries = self.relations(seed["paperId"], relation, limit)
                paper_key = "citingPaper" if relation == "citation" else "citedPaper"
                for edge_rank, item in enumerate(items, start=1):
                    candidate = item.get(paper_key) if isinstance(item, dict) else None
                    if not isinstance(candidate, dict):
                        continue
                    expanded_id = self._external_arxiv(candidate)
                    if not expanded_id:
                        continue
                    edges.append(
                        {
                            "seed_arxiv_id": seed_id,
                            "expanded_arxiv_id": expanded_id,
                            "seed_s2_paper_id": seed.get("paperId", ""),
                            "expanded_s2_paper_id": candidate.get("paperId", ""),
                            "edge_type": relation,
                            "edge_rank": edge_rank,
                            "intents": item.get("intents") or [],
                            "is_influential": bool(item.get("isInfluential", False)),
                            "seed_citation_count": seed.get("citationCount"),
                            "seed_reference_count": seed.get("referenceCount"),
                            "expanded_citation_count": candidate.get("citationCount"),
                            "expanded_reference_count": candidate.get("referenceCount"),
                            "seed_publication_types": seed.get("publicationTypes") or [],
                            "expanded_publication_types": candidate.get(
                                "publicationTypes"
                            )
                            or [],
                            "s2_cache_hit": bool(seed_hit and relation_hit),
                            "s2_api_status": "cache" if seed_hit and relation_hit else "api",
                            "s2_retry_count": seed_retries + relation_retries,
                        }
                    )
        return edges


def paper_db_by_arxiv(paper_db: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for key, value in paper_db.items():
        if not isinstance(value, Mapping):
            continue
        paper_id = normalize_arxiv_id(value.get("arxiv_id") or key)
        if paper_id and paper_id not in output:
            output[paper_id] = dict(value)
    return output


def load_paper_db(path: str) -> Dict[str, Dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        return paper_db_by_arxiv(value)
    output: Dict[str, Dict[str, Any]] = {}
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict):
            paper_id = normalize_arxiv_id(item.get("arxiv_id") or item.get("id"))
            if paper_id:
                output.setdefault(paper_id, dict(item))
    return output


def _intent_features(candidate_ids: Sequence[str], seed_ids: Set[str], edge_map: Mapping[str, Sequence[Mapping[str, Any]]]) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    scores: Dict[str, float] = {}
    labels: Dict[str, List[str]] = {}
    for paper_id in candidate_ids:
        found: Set[str] = set()
        best = 0.0
        if paper_id not in seed_ids:
            for edge in edge_map.get(paper_id, []):
                for intent in edge.get("intents") or []:
                    label = str(intent).strip().lower()
                    if label:
                        found.add(label)
                        best = max(best, INTENT_WEIGHTS.get(label, 0.0))
        scores[paper_id] = best
        labels[paper_id] = sorted(found)
    return scores, labels


def _path_features(candidate_ids: Sequence[str], seed_ids: Set[str], edges: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, int], Dict[str, float]]:
    neighbors: Dict[str, Set[str]] = {paper_id: set() for paper_id in candidate_ids}
    seen: Set[Tuple[str, str, str]] = set()
    candidate_set = set(candidate_ids)
    for edge in edges:
        seed = normalize_arxiv_id(edge.get("seed_arxiv_id"))
        expanded = normalize_arxiv_id(edge.get("expanded_arxiv_id"))
        key = (seed, expanded, str(edge.get("edge_type") or ""))
        if key in seen or seed not in seed_ids or seed not in candidate_set or expanded not in candidate_set:
            continue
        seen.add(key)
        neighbors[seed].add(expanded)
        neighbors[expanded].add(seed)
    counts = {paper_id: len(values) for paper_id, values in neighbors.items()}
    return counts, minmax(counts, list(candidate_ids))


class PerSubqueryProcessor:
    """Expand and rerank exactly one newly retrieved subquery page."""

    def __init__(
        self,
        paper_db: Mapping[str, Mapping[str, Any]],
        s2_client: S2GraphClient,
        *,
        scoring_backend: str,
        embedding_provider: Optional[Any],
        expansion_method: str = "citations_references",
        expansion_limit: int = 100,
        weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.paper_db = dict(paper_db)
        self.s2 = s2_client
        self.backend = scoring_backend
        self.embedding_provider = embedding_provider
        self.method = expansion_method
        self.limit = expansion_limit
        requested_weights = dict(DEFAULT_FEATURE_WEIGHTS)
        requested_weights.update(weights or {})
        if requested_weights != DEFAULT_FEATURE_WEIGHTS:
            raise ValueError(
                f"{RERANK_FORMULA_ID} is fixed; custom rerank weights are unsupported"
            )
        self.weights = requested_weights

    def materialize(
        self,
        event: Mapping[str, Any],
        gt_ids: Optional[Set[str]] = None,
        exclude_arxiv_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Materialize one graph pool and its component features only.

        The returned rows are in canonical arXiv-ID order.  This method does
        not apply the runtime weighted formula, choose a Top-K, or construct
        Selector inputs.  Keeping this boundary explicit lets later offline
        formulas reuse the expensive graph/embedding work without inheriting
        an old ranking through row order.
        """
        seed_records = [row for row in event.get("seed_papers") or [] if normalize_arxiv_id(row.get("paper_arxiv_id"))]
        seed_ids = list(dict.fromkeys(normalize_arxiv_id(row["paper_arxiv_id"]) for row in seed_records))
        seed_set = set(seed_ids)
        observed = {normalize_arxiv_id(row["paper_arxiv_id"]): row for row in seed_records}
        raw_edges = self.s2.expand(seed_ids, self.method, self.limit)
        cutoff = month_key(event.get("subquery_before_date") or event.get("query_date"))
        kept_edges: List[Dict[str, Any]] = []
        filter_stats = {
            "raw_edge_count": len(raw_edges),
            "missing_from_paper_db_count": 0,
            "missing_date_count": 0,
            "after_cutoff_count": 0,
            "previously_selected_count": 0,
            "kept_edge_count": 0,
        }
        excluded = {normalize_arxiv_id(value) for value in (exclude_arxiv_ids or set()) if normalize_arxiv_id(value)}
        for edge in raw_edges:
            expanded = normalize_arxiv_id(edge.get("expanded_arxiv_id"))
            metadata = self.paper_db.get(expanded)
            paper_month = month_key((metadata or {}).get("date"))
            if not metadata:
                filter_stats["missing_from_paper_db_count"] += 1
                continue
            if not paper_month:
                filter_stats["missing_date_count"] += 1
                continue
            if cutoff and paper_month > cutoff:
                filter_stats["after_cutoff_count"] += 1
                continue
            if expanded in excluded and expanded not in seed_set:
                filter_stats["previously_selected_count"] += 1
                continue
            kept_edges.append(dict(edge))
        filter_stats["kept_edge_count"] = len(kept_edges)
        expanded_ids = list(dict.fromkeys(edge["expanded_arxiv_id"] for edge in kept_edges))
        candidate_ids = list(dict.fromkeys(seed_ids + expanded_ids))
        index = CandidateIndex(candidate_ids, self.paper_db, self.backend, self.embedding_provider)
        candidate_ids = index.ids
        query_raw, query_norm, query_rank = index.score(str(event.get("query") or ""))
        sub_raw, sub_norm, sub_rank = index.score(str(event.get("subquery") or ""))
        edge_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        s2_publication_type_map: Dict[str, Set[str]] = defaultdict(set)
        for edge in kept_edges:
            edge_map[edge["expanded_arxiv_id"]].append(edge)
            s2_publication_type_map[edge["seed_arxiv_id"]].update(
                str(value)
                for value in edge.get("seed_publication_types") or []
                if str(value)
            )
            s2_publication_type_map[edge["expanded_arxiv_id"]].update(
                str(value)
                for value in edge.get("expanded_publication_types") or []
                if str(value)
            )
        intent_scores, intent_labels = _intent_features(candidate_ids, seed_set, edge_map)
        path_count, path_norm = _path_features(candidate_ids, seed_set, kept_edges)
        rows: List[Dict[str, Any]] = []
        gt = set(gt_ids or set())
        for materialization_rank, paper_id in enumerate(candidate_ids, start=1):
            provenance = edge_map.get(paper_id, [])
            source_seeds = sorted({edge["seed_arxiv_id"] for edge in provenance})
            is_seed, is_expanded = paper_id in seed_set, bool(provenance)
            observed_rank = observed.get(paper_id, {}).get("observed_retrieval_rank")
            rank_after_exclusion = (
                int(event.get("retrieval_offset") or 0) + int(observed_rank)
                if observed_rank is not None
                else None
            )
            rows.append(
                {
                    **{key: event.get(key) for key in (
                        "schema_version", "run_id", "retrieval_event_id", "query_id", "benchmark_idx", "query", "query_source", "query_date",
                        "iteration_idx", "subquery_id", "subquery", "subquery_target_k", "subquery_link_type",
                        "parent_subquery_id", "subquery_before_date",
                        "retrieval_page_idx", "retrieval_offset", "raw_retrieval_page_count", "results_per_query", "selector_top_k", "planner_checklist",
                    )},
                    "method": event.get("method", "per_subquery"),
                    "paper_arxiv_id": paper_id,
                    "candidate_type": "seed_and_expanded" if is_seed and is_expanded else ("seed" if is_seed else "expanded"),
                    "is_seed": is_seed,
                    "is_expanded": is_expanded,
                    "source_seed_arxiv_ids": source_seeds,
                    "source_subquery_ids": [event.get("subquery_id")] if is_expanded else [],
                    "edge_types": sorted({edge["edge_type"] for edge in provenance}),
                    "expansion_path_count": len(provenance),
                    "passed_date_cutoff": True,
                    "date_cutoff_month": cutoff,
                    "retrieval_backend": self.backend,
                    "observed_retrieval_score": observed.get(paper_id, {}).get("observed_retrieval_score"),
                    "observed_retrieval_rank": observed_rank,
                    "observed_retrieval_rank_scope": "one_based_rank_in_returned_baseline_page",
                    "observed_retrieval_rank_after_exclusion": rank_after_exclusion,
                    # Backward-compatible alias; it is not a pre-exclusion global rank.
                    "observed_retrieval_absolute_rank": rank_after_exclusion,
                    "observed_retrieval_absolute_rank_scope": "one_based_rank_after_frozen_exclusion",
                    "retrieval_score_raw": sub_raw.get(paper_id, 0.0),
                    "retrieval_score_normalized": sub_norm.get(paper_id, 0.0),
                    "retrieval_rank": sub_rank.get(paper_id),
                    "retrieval_rank_scope": "closed_seed_expanded_pool",
                    "query_score_raw": query_raw.get(paper_id, 0.0),
                    "query_score_normalized": query_norm.get(paper_id, 0.0),
                    "query_component_rank": query_rank.get(paper_id),
                    "subquery_score_raw": sub_raw.get(paper_id, 0.0),
                    "subquery_score_normalized": sub_norm.get(paper_id, 0.0),
                    "subquery_component_rank": sub_rank.get(paper_id),
                    "component_rank_scope": "closed_seed_expanded_pool",
                    "normalization_scope": "closed_seed_expanded_pool_minmax",
                    "intent_labels": intent_labels.get(paper_id, []),
                    "intent_score": intent_scores.get(paper_id, 0.0),
                    "path_count": path_count.get(paper_id, 0),
                    "path_count_normalized": path_norm.get(paper_id, 0.0),
                    "s2_publication_types": sorted(
                        s2_publication_type_map.get(paper_id, set())
                    ),
                    "materialization_order_rank": materialization_rank,
                    "materialization_order_scope": "canonical_arxiv_id",
                    "is_ground_truth": paper_id in gt,
                }
            )
        edge_rows = []
        for edge in kept_edges:
            edge_rows.append(
                {
                    **{key: event.get(key) for key in (
                        "schema_version", "run_id", "retrieval_event_id", "query_id", "benchmark_idx", "query", "iteration_idx", "subquery_id",
                        "subquery", "retrieval_page_idx",
                    )},
                    "method": event.get("method", "per_subquery"),
                    **edge,
                    "date_cutoff_month": cutoff,
                    "passed_date_cutoff": True,
                }
            )
        return {
            "rows": rows,
            "edges": edge_rows,
            "filter_stats": filter_stats,
            "features_materialized": True,
            "legacy_rerank_applied": False,
        }

    def process(
        self,
        event: Mapping[str, Any],
        gt_ids: Optional[Set[str]] = None,
        exclude_arxiv_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Apply the fixed four-factor rerank to a materialized graph pool."""
        materialized = self.materialize(
            event,
            gt_ids,
            exclude_arxiv_ids=exclude_arxiv_ids,
        )
        rows = [dict(row) for row in materialized["rows"]]
        row_by_id = {row["paper_arxiv_id"]: row for row in rows}
        final_scores = {
            paper_id: (
                self.weights["query_score_normalized"]
                * float(row.get("query_score_normalized") or 0.0)
                + self.weights["subquery_score_normalized"]
                * float(row.get("subquery_score_normalized") or 0.0)
                + self.weights["intent_score"]
                * float(row.get("intent_score") or 0.0)
                + self.weights["path_count_normalized"]
                * float(row.get("path_count_normalized") or 0.0)
            )
            for paper_id, row in row_by_id.items()
        }

        def final_key(paper_id: str) -> Tuple[float, int, int, str]:
            row = row_by_id[paper_id]
            observed_rank = row.get("observed_retrieval_rank")
            return (
                -final_scores[paper_id],
                -int(bool(row.get("is_seed"))),
                int(observed_rank or 10**12),
                paper_id,
            )

        ordered = sorted(row_by_id, key=final_key)
        final_rank = {paper_id: rank for rank, paper_id in enumerate(ordered, start=1)}
        rows = [row_by_id[paper_id] for paper_id in ordered]
        for row in rows:
            paper_id = row["paper_arxiv_id"]
            row.pop("materialization_order_rank", None)
            row.pop("materialization_order_scope", None)
            row.update(
                {
                    "rerank_formula_id": RERANK_FORMULA_ID,
                    "feature_weights": dict(self.weights),
                    "rerank_score": final_scores[paper_id],
                    "rerank_rank": final_rank[paper_id],
                }
            )

        seed_ids = list(
            dict.fromkeys(
                normalize_arxiv_id(row.get("paper_arxiv_id"))
                for row in event.get("seed_papers") or []
                if normalize_arxiv_id(row.get("paper_arxiv_id"))
            )
        )
        requested_top_k = event.get("selector_top_k")
        top_k = int(requested_top_k if requested_top_k is not None else len(seed_ids))
        top_rows = rows[:top_k]
        papers = []
        for row in top_rows:
            metadata = self.paper_db.get(row["paper_arxiv_id"], {})
            papers.append(
                Paper(
                    id=row["paper_arxiv_id"],
                    arxiv_id=row["paper_arxiv_id"],
                    title=str(metadata.get("title") or "N/A"),
                    abstract=str(metadata.get("abstract") or "N/A"),
                    date=metadata.get("date") or "",
                    score=float(row["rerank_score"]),
                )
            )
        rank_dict = {
            row["paper_arxiv_id"]: {
                "rank": int(row["rerank_rank"]) - 1,
                "total": max(len(rows) - 1, 0),
                "score": row["rerank_score"],
                "raw_score": row["rerank_score"],
            }
            for row in rows
        }
        return {
            "rows": rows,
            "edges": materialized["edges"],
            "top_rows": top_rows,
            "papers": papers,
            "rank_dict": rank_dict,
            "filter_stats": materialized["filter_stats"],
            "features_materialized": True,
            "legacy_rerank_applied": True,
            "rerank_formula_id": RERANK_FORMULA_ID,
        }


def selector_decision_record(event: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], selected_ids: Iterable[str], overview: str, reasons: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    selected = {normalize_arxiv_id(value) for value in selected_ids if normalize_arxiv_id(value)}
    candidate_rows = []
    for row in rows:
        item = dict(row)
        paper_id = normalize_arxiv_id(item.get("paper_arxiv_id"))
        item["selector_selected"] = paper_id in selected
        item["selector_reason"] = (reasons or {}).get(paper_id, "")
        candidate_rows.append(item)
    return {
        "schema_version": event.get("schema_version", "1.0"),
        "retrieval_event_id": event.get("retrieval_event_id"),
        "query_id": event.get("query_id"),
        "benchmark_idx": event.get("benchmark_idx"),
        "iteration_idx": event.get("iteration_idx"),
        "subquery_id": event.get("subquery_id"),
        "subquery": event.get("subquery"),
        "planner_checklist": event.get("planner_checklist"),
        "candidate_rows": candidate_rows,
        "selected_arxiv_ids": sorted(selected),
        "selector_overview": overview or "",
    }
