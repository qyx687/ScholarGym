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

from structures import Paper, SubQuery

try:
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings
except Exception:  # Keep sparse-only installs importable.
    class LangChainEmbeddings:  # type: ignore
        pass


EPSILON = 1e-12
DEFAULT_FEATURE_WEIGHTS = {
    "query_score_normalized": 0.30,
    "subquery_score_normalized": 0.40,
    "intent_score": 0.15,
    "path_count_normalized": 0.15,
}
INTENT_WEIGHTS = {"methodology": 1.0, "result": 0.75, "background": 0.35}


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


class EmbeddingProvider(LangChainEmbeddings):
    """Ollama or OpenAI-compatible embedding client with an in-process cache."""

    def __init__(
        self,
        backend: str,
        model: str,
        *,
        base_url: str,
        api_key: str = "",
        batch_size: int = 64,
        timeout: int = 120,
    ) -> None:
        if backend not in {"ollama", "api"}:
            raise ValueError("embedding backend must be ollama or api")
        self.backend = backend
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = max(1, int(batch_size))
        self.timeout = timeout
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _key(self, text: str) -> str:
        payload = f"{self.backend}\0{self.model}\0{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        keys = [self._key(text) for text in texts]
        missing_texts: List[str] = []
        missing_keys: List[str] = []
        with self._lock:
            for key, text in zip(keys, texts):
                if key not in self._cache:
                    missing_keys.append(key)
                    missing_texts.append(text)
        for start in range(0, len(missing_texts), self.batch_size):
            batch = missing_texts[start : start + self.batch_size]
            vectors = None
            for attempt in range(3):
                try:
                    vectors = self._request(batch)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            if len(vectors) != len(batch):
                raise RuntimeError("embedding API returned an unexpected vector count")
            with self._lock:
                for key, vector in zip(missing_keys[start : start + self.batch_size], vectors):
                    self._cache[key] = np.asarray(vector, dtype=np.float32)
        with self._lock:
            matrix = np.stack([self._cache[key] for key in keys]).astype(np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, EPSILON)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0].tolist()

    def _request(self, texts: Sequence[str]) -> List[List[float]]:
        if self.backend == "ollama":
            url = self.base_url
            if not url.endswith("/api/embed"):
                url += "/api/embed"
            response = requests.post(url, json={"model": self.model, "input": list(texts)}, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return data.get("embeddings") or []
        url = self.base_url
        if not url.endswith("/embeddings"):
            url += "/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            url,
            headers=headers,
            json={"model": self.model, "input": list(texts), "encoding_format": "float"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        items = sorted(response.json().get("data") or [], key=lambda item: int(item.get("index", 0)))
        return [item["embedding"] for item in items]


class CandidateIndex:
    """Same-backend scoring over one closed seed+expanded candidate pool."""

    def __init__(
        self,
        candidate_ids: Sequence[str],
        metadata: Mapping[str, Mapping[str, Any]],
        backend: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.ids: List[str] = []
        self.texts: List[str] = []
        for paper_id in sorted(set(candidate_ids)):
            item = metadata.get(paper_id) or {}
            text = f"{item.get('title') or ''} {item.get('abstract') or ''}".strip()
            if text:
                self.ids.append(paper_id)
                self.texts.append(text)
        self.backend = backend
        self.embedding_provider = embedding_provider
        self._bm25 = BM25Okapi([tokenize(text) for text in self.texts]) if backend == "bm25" and self.texts else None
        self._document_vectors = None
        if backend == "embedding" and self.texts:
            if embedding_provider is None:
                raise ValueError("embedding_provider is required for embedding rerank")
            self._document_vectors = embedding_provider.embed(self.texts)

    def score(self, query: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        if not self.ids:
            return {}, {}, {}
        if self.backend == "bm25":
            values = self._bm25.get_scores(tokenize(query or "")) if self._bm25 is not None else np.zeros(len(self.ids))
        elif self.backend == "embedding":
            query_vector = self.embedding_provider.embed([query or ""])[0]
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
    PAPER_FIELDS = "paperId,externalIds,citationCount,referenceCount"
    EDGE_PAPER_FIELDS = "paperId,externalIds,citationCount,referenceCount"

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
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ScholarGym-Graph-Rerank/1.0"})
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
        self.stats = defaultdict(int)
        self._lock = threading.Lock()

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
                response = self.session.get(url, params=dict(params), timeout=self.timeout)
                last_status = response.status_code
                self._inc("api_calls")
                if response.status_code == 404:
                    return None, False, attempt - 1
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                response.raise_for_status()
                data = response.json()
                tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
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
        embedding_provider: Optional[EmbeddingProvider],
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
        self.weights = dict(DEFAULT_FEATURE_WEIGHTS)
        self.weights.update(weights or {})

    def process(
        self,
        event: Mapping[str, Any],
        gt_ids: Optional[Set[str]] = None,
        exclude_arxiv_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
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
        for edge in kept_edges:
            edge_map[edge["expanded_arxiv_id"]].append(edge)
        intent_scores, intent_labels = _intent_features(candidate_ids, seed_set, edge_map)
        path_count, path_norm = _path_features(candidate_ids, seed_set, kept_edges)
        final_scores = {
            paper_id: (
                self.weights["query_score_normalized"] * query_norm.get(paper_id, 0.0)
                + self.weights["subquery_score_normalized"] * sub_norm.get(paper_id, 0.0)
                + self.weights["intent_score"] * intent_scores.get(paper_id, 0.0)
                + self.weights["path_count_normalized"] * path_norm.get(paper_id, 0.0)
            )
            for paper_id in candidate_ids
        }

        def final_key(paper_id: str) -> Tuple[float, int, int, str]:
            observed_rank = observed.get(paper_id, {}).get("observed_retrieval_rank")
            return (-final_scores[paper_id], -int(paper_id in seed_set), int(observed_rank or 10**12), paper_id)

        ordered = sorted(candidate_ids, key=final_key)
        final_rank = {paper_id: rank for rank, paper_id in enumerate(ordered, start=1)}
        rows: List[Dict[str, Any]] = []
        gt = set(gt_ids or set())
        for paper_id in ordered:
            provenance = edge_map.get(paper_id, [])
            source_seeds = sorted({edge["seed_arxiv_id"] for edge in provenance})
            is_seed, is_expanded = paper_id in seed_set, bool(provenance)
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
                    "observed_retrieval_rank": observed.get(paper_id, {}).get("observed_retrieval_rank"),
                    "observed_retrieval_absolute_rank": (
                        int(event.get("retrieval_offset") or 0) + int(observed[paper_id]["observed_retrieval_rank"])
                        if paper_id in observed and observed[paper_id].get("observed_retrieval_rank") is not None
                        else None
                    ),
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
                    "intent_labels": intent_labels.get(paper_id, []),
                    "intent_score": intent_scores.get(paper_id, 0.0),
                    "path_count": path_count.get(paper_id, 0),
                    "path_count_normalized": path_norm.get(paper_id, 0.0),
                    "feature_weights": dict(self.weights),
                    "rerank_score": final_scores.get(paper_id, 0.0),
                    "rerank_rank": final_rank.get(paper_id),
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
        return {"rows": rows, "edges": edge_rows, "top_rows": top_rows, "papers": papers, "rank_dict": rank_dict, "filter_stats": filter_stats}


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
