#!/usr/bin/env python3
"""Per-subquery graph-augmented retrieval for ScholarGym DeepResearch.

Pipeline per subquery:

    retriever topK papers -> S2 citation/reference expansion -> local corpus
    date cutoff for expanded papers -> local BM25 rerank -> topK selector input

The original retriever topK are seeds. The selector sees the reranked topK
from the seed+expanded local pool.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import quote, unquote

import requests

from structures import Paper, SubQuery

try:
    from logger import get_logger
except Exception:  # pragma: no cover - keeps this module testable in minimal envs.
    import logging

    def get_logger(name: str, log_file: str = ""):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)


logger = get_logger(__name__, log_file="./log/per_subquery_graph.log")

DEFAULT_RERANK_MODE = "original_current_subquery_weighted"
SUPPORTED_RERANK_MODES = {
    "original_current_subquery_weighted",
    "original_plus_current_subquery_weighted",
    "original_current_subquery_max",
    "original_query",
    "original_query_only",
    "current_subquery",
    "current_subquery_only",
    "seed_order",
}
SUPPORTED_METHODS = {"citations", "references", "citations_references"}
EPSILON = 1e-12


def normalize_arxiv_id(x: Any) -> str:
    """Normalize common arXiv ID forms while preserving old-style slashes."""
    if x is None:
        return ""
    s = unquote(str(x)).strip()
    if not s:
        return ""
    if s.lower() in {"n/a", "na", "none", "null", "nan", "unknown"}:
        return ""
    s = s.strip().strip("[](){}<>.,;:'\"")
    s = re.sub(r"(?i)^arxiv\s*:\s*", "", s)
    s = re.sub(r"(?i)^arxiv\s+", "", s)
    url_match = re.search(r"(?i)arxiv\.org/(?:abs|pdf|html)/([^?#\s]+)", s)
    if url_match:
        s = url_match.group(1)
    s = re.sub(r"(?i)\.pdf$", "", s)
    new_style = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", s)
    if new_style:
        return new_style.group(1)
    old_style = re.search(r"(?i)([a-z][a-z0-9_.-]*(?:/[a-z0-9_.-]+)?/\d{7})(?:v\d+)?", s)
    if old_style:
        return re.sub(r"v\d+$", "", old_style.group(1), flags=re.IGNORECASE).lower()
    if "/" in s:
        return re.sub(r"v\d+$", "", s, flags=re.IGNORECASE).lower()
    return re.sub(r"v\d+$", "", s, flags=re.IGNORECASE)


def preprocess_for_bm25(text: str) -> List[str]:
    if not text:
        return []
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [tok for tok in text.split() if tok]


def build_paper_db_by_arxiv_id_from_metadata(
    paper_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw_key, raw_metadata in (paper_metadata or {}).items():
        if not isinstance(raw_metadata, Mapping):
            continue
        arxiv_id = normalize_arxiv_id(raw_metadata.get("arxiv_id") or raw_key)
        if not arxiv_id or arxiv_id in out:
            continue
        metadata = dict(raw_metadata)
        metadata.setdefault("arxiv_id", arxiv_id)
        out[arxiv_id] = metadata
    return out


def load_paper_db_by_arxiv_id(path_str: str) -> Dict[str, Dict[str, Any]]:
    with Path(path_str).open("r", encoding="utf-8") as f:
        value = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}

    def add(raw_key: Any, metadata: Any) -> None:
        if not isinstance(metadata, Mapping):
            return
        arxiv_id = normalize_arxiv_id(
            metadata.get("arxiv_id")
            or metadata.get("arxivId")
            or metadata.get("id")
            or metadata.get("paper_id")
            or raw_key
        )
        if not arxiv_id or arxiv_id in out:
            return
        item = dict(metadata)
        item.setdefault("arxiv_id", arxiv_id)
        out[arxiv_id] = item

    if isinstance(value, Mapping):
        for key, metadata in value.items():
            add(key, metadata)
    elif isinstance(value, list):
        for metadata in value:
            add(None, metadata)
    else:
        raise ValueError(f"Unsupported paper DB shape: {path_str}")
    return out


class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / rps if rps and rps > 0 else 0.0
        self.lock = threading.Lock()
        self.last_request_at = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_at
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_at = time.time()


class S2Client:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    PAPER_FIELDS = "paperId,corpusId,title,year,publicationDate,externalIds,citationCount,referenceCount,url"
    CITATION_FIELDS = (
        "citingPaper.paperId,citingPaper.corpusId,citingPaper.title,"
        "citingPaper.year,citingPaper.publicationDate,citingPaper.externalIds,"
        "citingPaper.citationCount,citingPaper.url,isInfluential,intents"
    )
    REFERENCE_FIELDS = (
        "citedPaper.paperId,citedPaper.corpusId,citedPaper.title,"
        "citedPaper.year,citedPaper.publicationDate,citedPaper.externalIds,"
        "citedPaper.citationCount,citedPaper.url,isInfluential,intents"
    )
    _global_limiters: Dict[float, RateLimiter] = {}
    _global_limiters_lock = threading.Lock()

    def __init__(
        self,
        cache_dir: str,
        api_key: Optional[str] = None,
        rate_limit_rps: float = 1.0,
        offline_cache_only: bool = False,
        timeout: int = 30,
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key if api_key is not None else os.environ.get("S2_API_KEY")
        self.offline_cache_only = offline_cache_only
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ScholarGym-PerSubqueryGraph/0.1"})
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
        self.rate_limiter = self._get_global_limiter(rate_limit_rps)
        self.stats_lock = threading.Lock()
        self.stats: Dict[str, Any] = {
            "api_call_count": 0,
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "cache_stale_count": 0,
            "offline_cache_miss_count": 0,
            "error_count": 0,
        }
        for dirname in ("resolve_arxiv", "citations", "references"):
            (self.cache_dir / dirname).mkdir(parents=True, exist_ok=True)

    @classmethod
    def _get_global_limiter(cls, rps: float) -> RateLimiter:
        key = float(rps or 0.0)
        with cls._global_limiters_lock:
            if key not in cls._global_limiters:
                cls._global_limiters[key] = RateLimiter(key)
            return cls._global_limiters[key]

    def get_stats(self) -> Dict[str, Any]:
        with self.stats_lock:
            return dict(self.stats)

    def _inc(self, key: str) -> None:
        with self.stats_lock:
            self.stats[key] = int(self.stats.get(key, 0)) + 1

    def resolve_arxiv(self, arxiv_id: str) -> Optional[dict]:
        normalized = normalize_arxiv_id(arxiv_id)
        if not normalized:
            return None
        path = self.cache_dir / "resolve_arxiv" / f"{self._safe_name(normalized)}.json"
        endpoint = f"/paper/{quote(f'ARXIV:{normalized}', safe='')}"
        data = self._get_json(endpoint, {"fields": self.PAPER_FIELDS}, path)
        return data if isinstance(data, dict) and data.get("paperId") else None

    def get_citations(self, paper_id: str, limit: int) -> List[dict]:
        if not paper_id or limit <= 0:
            return []
        path = self.cache_dir / "citations" / f"{self._safe_name(paper_id)}.limit{limit}.json"
        data = self._get_json(
            f"/paper/{quote(paper_id, safe='')}/citations",
            {"fields": self.CITATION_FIELDS, "offset": 0, "limit": int(limit)},
            path,
        )
        items = data.get("data", []) if isinstance(data, dict) else []
        return items if isinstance(items, list) else []

    def get_references(self, paper_id: str, limit: int) -> List[dict]:
        if not paper_id or limit <= 0:
            return []
        path = self.cache_dir / "references" / f"{self._safe_name(paper_id)}.limit{limit}.json"
        data = self._get_json(
            f"/paper/{quote(paper_id, safe='')}/references",
            {"fields": self.REFERENCE_FIELDS, "offset": 0, "limit": int(limit)},
            path,
        )
        items = data.get("data", []) if isinstance(data, dict) else []
        return items if isinstance(items, list) else []

    def _get_json(self, endpoint: str, params: Mapping[str, Any], cache_path: Path) -> Optional[dict]:
        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, dict) and "data" in cached and "endpoint" in cached:
                    cached_params = cached.get("params") or {}
                    if cached.get("endpoint") == endpoint and dict(cached_params) == dict(params):
                        self._inc("cache_hit_count")
                        return cached.get("data")
                    self._inc("cache_stale_count")
                else:
                    self._inc("cache_stale_count")
            except Exception:
                pass
        self._inc("cache_miss_count")
        if self.offline_cache_only:
            self._inc("offline_cache_miss_count")
            return None
        url = f"{self.BASE_URL}{endpoint}"
        for attempt in range(max(1, self.max_retries)):
            try:
                self.rate_limiter.wait()
                response = self.session.get(url, params=dict(params), timeout=self.timeout)
                self._inc("api_call_count")
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                data = response.json()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("w", encoding="utf-8") as f:
                    json.dump({"endpoint": endpoint, "params": dict(params), "data": data}, f, ensure_ascii=False)
                return data
            except Exception:
                self._inc("error_count")
                if attempt + 1 >= self.max_retries:
                    return None
                time.sleep(min(2 ** attempt, 8))
        return None

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))[:180]


class LocalBM25Index:
    def __init__(self):
        self.arxiv_ids: List[str] = []
        self.paper_metadata: Dict[str, Dict[str, Any]] = {}
        self.corpus: List[List[str]] = []
        self.doc_freq: Counter[str] = Counter()
        self.doc_lens: List[int] = []
        self.avgdl = 0.0

    def build(self, candidate_arxiv_ids: Sequence[str], paper_db_by_arxiv_id: Mapping[str, Mapping[str, Any]]) -> None:
        self.arxiv_ids = []
        self.paper_metadata = {}
        self.corpus = []
        self.doc_freq = Counter()
        self.doc_lens = []
        for arxiv_id in sorted({normalize_arxiv_id(x) for x in candidate_arxiv_ids if normalize_arxiv_id(x)}):
            metadata = paper_db_by_arxiv_id.get(arxiv_id)
            if not metadata:
                continue
            title = str(metadata.get("title") or "")
            abstract = str(metadata.get("abstract") or "")
            tokens = preprocess_for_bm25(f"{title} {abstract}".strip())
            if not tokens:
                continue
            self.arxiv_ids.append(arxiv_id)
            self.paper_metadata[arxiv_id] = dict(metadata)
            self.corpus.append(tokens)
            self.doc_lens.append(len(tokens))
            self.doc_freq.update(set(tokens))
        self.avgdl = float(sum(self.doc_lens)) / len(self.doc_lens) if self.doc_lens else 0.0

    def score(self, query: str) -> Dict[str, float]:
        if not self.arxiv_ids:
            return {}
        query_tokens = preprocess_for_bm25(query or "")
        if not query_tokens:
            return {arxiv_id: 0.0 for arxiv_id in self.arxiv_ids}
        n_docs = len(self.corpus)
        k1 = 1.5
        b = 0.75
        out: Dict[str, float] = {}
        for arxiv_id, tokens, doc_len in zip(self.arxiv_ids, self.corpus, self.doc_lens):
            tf = Counter(tokens)
            score = 0.0
            for tok in query_tokens:
                freq = tf.get(tok, 0)
                if freq <= 0:
                    continue
                df = self.doc_freq.get(tok, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = freq + k1 * (1.0 - b + b * doc_len / (self.avgdl or 1.0))
                score += idf * (freq * (k1 + 1.0)) / denom
            out[arxiv_id] = float(score)
        return out


def normalized_bm25_scores(index: LocalBM25Index, query: str) -> Dict[str, float]:
    raw = index.score(query)
    if not raw:
        return {}
    max_raw = max(raw.values()) if raw else 0.0
    if max_raw <= 0:
        return {arxiv_id: 0.0 for arxiv_id in raw}
    return {arxiv_id: float(score) / (max_raw + EPSILON) for arxiv_id, score in raw.items()}


def max_score_maps(score_maps: Sequence[Dict[str, float]], candidate_ids: Iterable[str]) -> Dict[str, float]:
    out = {arxiv_id: 0.0 for arxiv_id in candidate_ids}
    for scores in score_maps:
        for arxiv_id, score in scores.items():
            if score > out.get(arxiv_id, 0.0):
                out[arxiv_id] = float(score)
    return out


def rank_candidates(
    candidate_ids: Iterable[str],
    scores: Mapping[str, float],
    seed_rank: Mapping[str, int],
    paper_metadata: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    seed_ids = set(seed_rank)
    rows: List[Dict[str, Any]] = []
    for arxiv_id in sorted({normalize_arxiv_id(x) for x in candidate_ids if normalize_arxiv_id(x)}):
        metadata = paper_metadata.get(arxiv_id, {})
        rows.append(
            {
                "arxiv_id": arxiv_id,
                "score": float(scores.get(arxiv_id, 0.0)),
                "is_seed": arxiv_id in seed_ids,
                "seed_rank": seed_rank.get(arxiv_id),
                "title": metadata.get("title", ""),
                "abstract": metadata.get("abstract", ""),
                "date": metadata.get("date", ""),
            }
        )

    def sort_key(row: Dict[str, Any]) -> Tuple[float, int, int, str]:
        rank = row.get("seed_rank")
        best_seed_rank = int(rank) if isinstance(rank, int) else 10**12
        return (-float(row["score"]), -int(bool(row["is_seed"])), best_seed_rank, row["arxiv_id"])

    rows.sort(key=sort_key)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


@dataclass
class PerSubqueryGraphResult:
    papers: List[Paper]
    rank_dict: Dict[str, Dict[str, Any]]
    trace: Dict[str, Any]
    warnings: List[str]


class PerSubqueryGraphAugmenter:
    def __init__(
        self,
        s2_client: S2Client,
        paper_db_by_arxiv_id: Mapping[str, Mapping[str, Any]],
        *,
        method: str = "citations_references",
        expansion_limit: int = 100,
        rerank_mode: str = DEFAULT_RERANK_MODE,
        rerank_alpha: float = 0.5,
        fail_fast: bool = False,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported per-subquery graph method: {method}")
        if rerank_mode not in SUPPORTED_RERANK_MODES:
            raise ValueError(f"Unsupported per-subquery graph rerank mode: {rerank_mode}")
        self.s2_client = s2_client
        self.paper_db_by_arxiv_id = {
            normalize_arxiv_id(arxiv_id): dict(metadata)
            for arxiv_id, metadata in (paper_db_by_arxiv_id or {}).items()
            if normalize_arxiv_id(arxiv_id) and isinstance(metadata, Mapping)
        }
        self.corpus_arxiv_ids = set(self.paper_db_by_arxiv_id)
        self.method = method
        self.expansion_limit = max(0, int(expansion_limit or 0))
        self.rerank_mode = rerank_mode
        self.rerank_alpha = float(rerank_alpha)
        self.fail_fast = fail_fast

    def augment(
        self,
        *,
        original_query: str,
        subquery: SubQuery,
        seed_papers: Sequence[Paper],
        iter_idx: int,
        idx: int,
        qid: str = "",
        source: str = "",
        gt_arxiv_ids: Optional[Set[str]] = None,
        exclude_arxiv_ids: Optional[Set[str]] = None,
    ) -> PerSubqueryGraphResult:
        warnings: List[str] = []
        seed_ids, seed_metadata, seed_rank = self._seed_inputs(seed_papers)
        top_k = len(seed_ids)
        exclude_ids = {normalize_arxiv_id(x) for x in (exclude_arxiv_ids or set()) if normalize_arxiv_id(x)}
        if not seed_ids:
            trace = self._trace_record(original_query, subquery, iter_idx, idx, qid, source, [], [], [], {}, {"empty_seed_input": True}, ["empty_seed_input"], gt_arxiv_ids)
            return PerSubqueryGraphResult([], {}, trace, ["empty_seed_input"])
        try:
            expansion_ids, expansion_provenance, expansion_stats, expansion_warnings = self._expand_seed_ids(seed_ids)
            warnings.extend(expansion_warnings)
            metadata_by_arxiv_id: Dict[str, Dict[str, Any]] = {}
            for arxiv_id in seed_ids:
                metadata_by_arxiv_id[arxiv_id] = dict(self.paper_db_by_arxiv_id.get(arxiv_id) or seed_metadata.get(arxiv_id) or {})
                metadata_by_arxiv_id[arxiv_id].setdefault("arxiv_id", arxiv_id)
            for arxiv_id in expansion_ids:
                metadata = self.paper_db_by_arxiv_id.get(arxiv_id)
                if metadata:
                    metadata_by_arxiv_id[arxiv_id] = dict(metadata)
            candidate_ids, date_stats = self._candidate_ids_after_date_filter(
                seed_ids=seed_ids,
                expansion_ids=expansion_ids,
                metadata_by_arxiv_id=metadata_by_arxiv_id,
                before_date=subquery.before_date,
                exclude_arxiv_ids=exclude_ids,
            )
            ranked, rerank_info, rerank_warnings = self._rerank(candidate_ids, metadata_by_arxiv_id, original_query, subquery.text, seed_rank)
            warnings.extend(rerank_warnings)
            selector_input = ranked[:top_k]
            if not selector_input:
                selector_input = self._fallback_seed_rows(seed_ids, seed_metadata)
                ranked = selector_input
                warnings.append("rerank_empty_fallback_to_seed_order")
            papers = self._rows_to_papers(selector_input, metadata_by_arxiv_id)
            rank_dict = self._rank_dict(ranked)
            stats = {
                "method": self.method,
                "expansion_limit": self.expansion_limit,
                "rerank_mode": self.rerank_mode,
                "rerank_alpha": self.rerank_alpha,
                "seed_count": len(seed_ids),
                "requested_selector_top_k": top_k,
                "selector_input_count": len(papers),
                "local_candidate_count": len(candidate_ids),
                **expansion_stats,
                **date_stats,
                **rerank_info,
                "s2_client_stats": self.s2_client.get_stats(),
            }
            trace = self._trace_record(
                original_query,
                subquery,
                iter_idx,
                idx,
                qid,
                source,
                seed_ids,
                ranked,
                selector_input,
                rank_dict,
                stats,
                warnings,
                gt_arxiv_ids,
                expansion_provenance,
            )
            return PerSubqueryGraphResult(papers, rank_dict, trace, sorted(set(warnings)))
        except Exception as exc:
            if self.fail_fast:
                raise
            warnings.append(f"per_subquery_graph_failed:{type(exc).__name__}")
            logger.warning("[per_subquery_graph] failed for subquery %s: %s", getattr(subquery, "id", ""), exc)
            fallback_rows = self._fallback_seed_rows(seed_ids, seed_metadata)
            rank_dict = self._rank_dict(fallback_rows)
            trace = self._trace_record(original_query, subquery, iter_idx, idx, qid, source, seed_ids, fallback_rows, fallback_rows[:top_k], rank_dict, {"fallback_to_seed_order": True, "error": str(exc)}, warnings, gt_arxiv_ids)
            return PerSubqueryGraphResult(list(seed_papers), rank_dict, trace, sorted(set(warnings)))

    def _seed_inputs(self, seed_papers: Sequence[Paper]) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, int]]:
        seed_ids: List[str] = []
        seed_metadata: Dict[str, Dict[str, Any]] = {}
        seen: Set[str] = set()
        for paper in seed_papers or []:
            arxiv_id = normalize_arxiv_id(getattr(paper, "arxiv_id", "") or getattr(paper, "id", ""))
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            seed_ids.append(arxiv_id)
            seed_metadata[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "title": getattr(paper, "title", "") or "",
                "abstract": getattr(paper, "abstract", "") or "",
                "date": getattr(paper, "date", "") or "",
                "score": getattr(paper, "score", None),
            }
        return seed_ids, seed_metadata, {arxiv_id: rank for rank, arxiv_id in enumerate(seed_ids, start=1)}

    def _expand_seed_ids(self, seed_ids: Sequence[str]) -> Tuple[List[str], Dict[str, List[Dict[str, Any]]], Dict[str, Any], List[str]]:
        warnings: List[str] = []
        provenance: Dict[str, List[Dict[str, Any]]] = {}
        expansion_ids: List[str] = []
        seen_expansion_ids: Set[str] = set()
        stats = {
            "s2_resolve_success_count": 0,
            "s2_resolve_failure_count": 0,
            "expansion_edge_record_count": 0,
            "expansion_raw_arxiv_mappable_count": 0,
            "expansion_closed_corpus_count_before_date": 0,
            "expansion_not_in_corpus_count": 0,
        }
        for seed_id in seed_ids:
            seed_s2 = self.s2_client.resolve_arxiv(seed_id)
            seed_s2_id = (seed_s2 or {}).get("paperId", "")
            if not seed_s2_id:
                stats["s2_resolve_failure_count"] += 1
                warnings.append(f"s2_resolve_failed:{seed_id}")
                continue
            stats["s2_resolve_success_count"] += 1
            for edge_type, items in self._edge_items(seed_s2_id):
                for edge_rank, item in enumerate(items, start=1):
                    if not isinstance(item, Mapping):
                        continue
                    candidate = self._candidate_from_edge_item(item, edge_type)
                    stats["expansion_edge_record_count"] += 1
                    candidate_arxiv_id = self._external_arxiv_id(candidate.get("externalIds", {}))
                    if not candidate_arxiv_id:
                        continue
                    stats["expansion_raw_arxiv_mappable_count"] += 1
                    if candidate_arxiv_id not in self.corpus_arxiv_ids:
                        stats["expansion_not_in_corpus_count"] += 1
                        continue
                    row = {
                        "candidate_s2_paper_id": candidate.get("paperId", ""),
                        "candidate_arxiv_id": candidate_arxiv_id,
                        "candidate_title": candidate.get("title", ""),
                        "candidate_year": candidate.get("year"),
                        "candidate_publicationDate": candidate.get("publicationDate"),
                        "candidate_citationCount": candidate.get("citationCount"),
                        "edge_type": edge_type,
                        "edge_rank": edge_rank,
                        "source_seed_arxiv_id": seed_id,
                        "source_seed_s2_paper_id": seed_s2_id,
                        "isInfluential": item.get("isInfluential", False),
                        "intents": item.get("intents", []),
                    }
                    provenance.setdefault(candidate_arxiv_id, []).append(row)
                    if candidate_arxiv_id not in seen_expansion_ids:
                        seen_expansion_ids.add(candidate_arxiv_id)
                        expansion_ids.append(candidate_arxiv_id)
        stats["expansion_closed_corpus_count_before_date"] = len(expansion_ids)
        return expansion_ids, provenance, stats, sorted(set(warnings))

    def _edge_items(self, seed_s2_id: str) -> List[Tuple[str, List[dict]]]:
        if self.expansion_limit <= 0:
            return []
        items: List[Tuple[str, List[dict]]] = []
        if self.method in {"citations", "citations_references"}:
            items.append(("citation", self.s2_client.get_citations(seed_s2_id, self.expansion_limit)))
        if self.method in {"references", "citations_references"}:
            items.append(("reference", self.s2_client.get_references(seed_s2_id, self.expansion_limit)))
        return items

    @staticmethod
    def _candidate_from_edge_item(item: Mapping[str, Any], edge_type: str) -> Dict[str, Any]:
        key = "citingPaper" if edge_type == "citation" else "citedPaper"
        candidate = item.get(key, {}) if isinstance(item, Mapping) else {}
        return candidate if isinstance(candidate, dict) else {}

    @staticmethod
    def _external_arxiv_id(external_ids: Any) -> str:
        if not isinstance(external_ids, Mapping):
            return ""
        for key, value in external_ids.items():
            if str(key).lower() == "arxiv":
                return normalize_arxiv_id(value)
        return ""

    def _candidate_ids_after_date_filter(
        self,
        *,
        seed_ids: Sequence[str],
        expansion_ids: Sequence[str],
        metadata_by_arxiv_id: Mapping[str, Mapping[str, Any]],
        before_date: Optional[str],
        exclude_arxiv_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        candidate_ids: List[str] = []
        seen: Set[str] = set()
        exclude_arxiv_ids = exclude_arxiv_ids or set()
        excluded_previous_selected: List[str] = []
        for arxiv_id in seed_ids:
            if arxiv_id not in seen:
                seen.add(arxiv_id)
                candidate_ids.append(arxiv_id)
        before_month = self._date_month(before_date)
        cutoff_requested = before_date is not None
        invalid_cutoff = cutoff_requested and not before_month
        filtered: List[str] = []
        missing_date: List[str] = []
        invalid_cutoff_filtered: List[str] = []
        kept_expansions: List[str] = []
        for arxiv_id in expansion_ids:
            if arxiv_id in seen:
                continue
            if arxiv_id in exclude_arxiv_ids:
                excluded_previous_selected.append(arxiv_id)
                continue
            metadata = metadata_by_arxiv_id.get(arxiv_id, {})
            paper_month = self._date_month(metadata.get("date"))
            if not paper_month:
                filtered.append(arxiv_id)
                missing_date.append(arxiv_id)
                continue
            if invalid_cutoff:
                filtered.append(arxiv_id)
                invalid_cutoff_filtered.append(arxiv_id)
                continue
            if before_month and paper_month > before_month:
                filtered.append(arxiv_id)
                continue
            seen.add(arxiv_id)
            candidate_ids.append(arxiv_id)
            kept_expansions.append(arxiv_id)
        return candidate_ids, {
            "date_cutoff_before_date": before_date or "",
            "date_cutoff_before_month": before_month,
            "date_cutoff_requested": cutoff_requested,
            "date_cutoff_invalid": invalid_cutoff,
            "seed_date_cutoff_policy": "trusted_retriever_cutoff",
            "expansion_date_cutoff_policy": "drop_missing_date_and_keep_paper_db_date_lte_subquery_before_date",
            "expansion_closed_corpus_count_after_date": len(kept_expansions),
            "date_filtered_expansion_count": len(filtered),
            "date_filtered_expansion_ids_sample": filtered[:20],
            "date_missing_expansion_count": len(missing_date),
            "date_missing_expansion_ids_sample": missing_date[:20],
            "date_invalid_cutoff_filtered_expansion_count": len(invalid_cutoff_filtered),
            "date_invalid_cutoff_filtered_expansion_ids_sample": invalid_cutoff_filtered[:20],
            "excluded_previous_selected_count": len(set(excluded_previous_selected)),
            "excluded_previous_selected_ids_sample": sorted(set(excluded_previous_selected))[:20],
        }

    @staticmethod
    def _date_month(value: Any) -> str:
        text = str(value or "").strip()
        return text[:7] if len(text) >= 7 else ""

    def _rerank(
        self,
        candidate_ids: Sequence[str],
        metadata_by_arxiv_id: Mapping[str, Dict[str, Any]],
        original_query: str,
        subquery_text: str,
        seed_rank: Mapping[str, int],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
        warnings: List[str] = []
        index = LocalBM25Index()
        index.build(candidate_ids, metadata_by_arxiv_id)
        if not index.arxiv_ids:
            warnings.append("local_bm25_index_empty")
            return self._fallback_rank_candidates(candidate_ids, metadata_by_arxiv_id, seed_rank), {"local_rerank_document_count": 0, "rerank_query_strings": []}, warnings
        mode = self.rerank_mode
        indexed_ids = list(index.arxiv_ids)
        query_strings: List[str] = []
        if mode in {"original_current_subquery_weighted", "original_plus_current_subquery_weighted"}:
            original_scores = normalized_bm25_scores(index, original_query)
            subquery_scores = normalized_bm25_scores(index, subquery_text)
            scores = {
                arxiv_id: self.rerank_alpha * original_scores.get(arxiv_id, 0.0)
                + (1.0 - self.rerank_alpha) * subquery_scores.get(arxiv_id, 0.0)
                for arxiv_id in indexed_ids
            }
            query_strings = [original_query, subquery_text]
        elif mode in {"original_query", "original_query_only"}:
            scores = normalized_bm25_scores(index, original_query)
            query_strings = [original_query]
        elif mode in {"current_subquery", "current_subquery_only"}:
            scores = normalized_bm25_scores(index, subquery_text)
            query_strings = [subquery_text]
        elif mode == "original_current_subquery_max":
            scores = max_score_maps([normalized_bm25_scores(index, original_query), normalized_bm25_scores(index, subquery_text)], indexed_ids)
            query_strings = [original_query, subquery_text]
        elif mode == "seed_order":
            scores = {arxiv_id: 0.0 for arxiv_id in indexed_ids}
        else:
            raise ValueError(f"Unsupported rerank mode: {mode}")
        ranked = rank_candidates(indexed_ids, scores, seed_rank, index.paper_metadata)
        return ranked, {"local_rerank_document_count": len(ranked), "rerank_query_strings": query_strings}, warnings

    @staticmethod
    def _fallback_rank_candidates(
        candidate_ids: Sequence[str],
        metadata_by_arxiv_id: Mapping[str, Dict[str, Any]],
        seed_rank: Mapping[str, int],
    ) -> List[Dict[str, Any]]:
        scores = {normalize_arxiv_id(arxiv_id): 0.0 for arxiv_id in candidate_ids if normalize_arxiv_id(arxiv_id)}
        return rank_candidates(candidate_ids, scores, seed_rank, metadata_by_arxiv_id)

    @staticmethod
    def _fallback_seed_rows(seed_ids: Sequence[str], seed_metadata: Mapping[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for rank, arxiv_id in enumerate(seed_ids, start=1):
            metadata = seed_metadata.get(arxiv_id, {})
            rows.append(
                {
                    "arxiv_id": arxiv_id,
                    "score": float(metadata.get("score") or 0.0),
                    "is_seed": True,
                    "seed_rank": rank,
                    "title": metadata.get("title", ""),
                    "abstract": metadata.get("abstract", ""),
                    "date": metadata.get("date", ""),
                    "rank": rank,
                }
            )
        return rows

    @staticmethod
    def _rows_to_papers(rows: Sequence[Mapping[str, Any]], metadata_by_arxiv_id: Mapping[str, Mapping[str, Any]]) -> List[Paper]:
        papers: List[Paper] = []
        for row in rows:
            arxiv_id = normalize_arxiv_id(row.get("arxiv_id"))
            if not arxiv_id:
                continue
            metadata = metadata_by_arxiv_id.get(arxiv_id, {})
            papers.append(
                Paper(
                    id=arxiv_id,
                    title=str(metadata.get("title") or row.get("title") or "N/A"),
                    abstract=str(metadata.get("abstract") or row.get("abstract") or "N/A"),
                    arxiv_id=arxiv_id,
                    date=metadata.get("date") or row.get("date") or "",
                    score=float(row.get("score", 0.0) or 0.0),
                )
            )
        return papers

    @staticmethod
    def _rank_dict(ranked: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        total_rank = max(len(ranked) - 1, 0)
        out: Dict[str, Dict[str, Any]] = {}
        for zero_rank, row in enumerate(ranked):
            arxiv_id = normalize_arxiv_id(row.get("arxiv_id"))
            if arxiv_id:
                out[arxiv_id] = {
                    "rank": zero_rank,
                    "total": total_rank,
                    "score": float(row.get("score") or 0.0),
                    "raw_score": float(row.get("score") or 0.0),
                }
        return out

    def _trace_record(
        self,
        original_query: str,
        subquery: SubQuery,
        iter_idx: int,
        idx: int,
        qid: str,
        source: str,
        seed_ids: Sequence[str],
        ranked: Sequence[Mapping[str, Any]],
        selector_input: Sequence[Mapping[str, Any]],
        rank_dict: Mapping[str, Mapping[str, Any]],
        stats: Mapping[str, Any],
        warnings: Sequence[str],
        gt_arxiv_ids: Optional[Set[str]],
        expansion_provenance: Optional[Mapping[str, List[Mapping[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        gt = set(gt_arxiv_ids or set())
        selector_ids = [normalize_arxiv_id(row.get("arxiv_id")) for row in selector_input if normalize_arxiv_id(row.get("arxiv_id"))]
        candidate_info_by_arxiv = {}
        for row in ranked:
            arxiv_id = normalize_arxiv_id(row.get("arxiv_id"))
            if not arxiv_id:
                continue
            provenance_rows = list((expansion_provenance or {}).get(arxiv_id, []))
            is_seed = bool(row.get("is_seed"))
            source_type = "seed" if is_seed else "expanded"
            candidate_info_by_arxiv[arxiv_id] = {
                "rerank_rank": row.get("rank"),
                "rerank_score": row.get("score"),
                "source_type": source_type,
                "is_seed": is_seed,
                "seed_rank": row.get("seed_rank"),
                "date": row.get("date"),
                "source_seed_arxiv_ids": sorted({item.get("source_seed_arxiv_id") for item in provenance_rows if item.get("source_seed_arxiv_id")}),
                "edge_types": sorted({item.get("edge_type") for item in provenance_rows if item.get("edge_type")}),
            }
        selector_input_sources = [
            {
                "arxiv_id": arxiv_id,
                **candidate_info_by_arxiv.get(arxiv_id, {}),
            }
            for arxiv_id in selector_ids
        ]
        return {
            "idx": idx,
            "qid": qid,
            "source": source,
            "query": original_query,
            "iter_idx": iter_idx,
            "subquery_id": subquery.id,
            "subquery": subquery.text,
            "subquery_before_date": subquery.before_date,
            "seed_arxiv_ids": list(seed_ids),
            "selector_input_arxiv_ids": selector_ids,
            "selector_input_sources": selector_input_sources,
            "selector_input_gt_ids": sorted(gt & set(selector_ids)),
            "gt_arxiv_ids": sorted(gt),
            "candidate_info_by_arxiv": candidate_info_by_arxiv,
            "rank_dict": dict(rank_dict),
            "stats": dict(stats),
            "warnings": sorted(set(warnings)),
        }
