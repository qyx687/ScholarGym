#!/usr/bin/env python3
"""Build a resumable S2 publication-type cache for OnePass candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_paper_type_cache import (  # noqa: E402
    _append_failure,
    _append_records,
    compact_cache,
    iter_pool_candidate_ids,
    load_type_rule_query_ids,
)
from paper_type import (  # noqa: E402
    S2_CLASSIFIER_VERSION,
    load_paper_type_cache,
    s2_publication_types_to_record,
)
from runtime_env import load_env_file  # noqa: E402


S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_MAX_IDS = 500


class _RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = (
            1.0 / float(requests_per_second)
            if requests_per_second and requests_per_second > 0
            else 0.0
        )
        self.last_request_at = 0.0

    def wait(self) -> None:
        delay = self.interval - (time.time() - self.last_request_at)
        if delay > 0:
            time.sleep(delay)
        self.last_request_at = time.time()


class S2PublicationTypeClient:
    """Small batch client that never exposes the API key in artifacts."""

    def __init__(
        self,
        *,
        api_key: str = "",
        requests_per_second: float = 1.0,
        timeout: int = 60,
        max_retries: int = 3,
        session: Any = None,
        batch_url: str = S2_BATCH_URL,
    ) -> None:
        self.timeout = int(timeout)
        self.max_retries = max(1, int(max_retries))
        self.batch_url = str(batch_url)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "ScholarGym-Dynamic-Rerank/1.0",
                **({"x-api-key": api_key} if api_key else {}),
            }
        )
        self.limiter = _RateLimiter(requests_per_second)
        self.api_call_count = 0
        self.retry_count = 0

    def fetch_batch(self, paper_ids: Sequence[str]) -> List[Dict[str, Any]]:
        if not paper_ids:
            return []
        if len(paper_ids) > S2_BATCH_MAX_IDS:
            raise ValueError(
                f"S2 paper batch cannot exceed {S2_BATCH_MAX_IDS} IDs"
            )
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                response = self.session.post(
                    self.batch_url,
                    params={"fields": "paperId,publicationTypes"},
                    json={"ids": [f"ARXIV:{paper_id}" for paper_id in paper_ids]},
                    timeout=self.timeout,
                )
                self.api_call_count += 1
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
                    self.retry_count += 1
                    time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise last_error


def _fetch_resilient(
    client: S2PublicationTypeClient,
    paper_ids: Sequence[str],
    failure_path: Path,
) -> List[Dict[str, Any]]:
    try:
        return client.fetch_batch(paper_ids)
    except Exception as exc:
        if len(paper_ids) > 1:
            middle = len(paper_ids) // 2
            return _fetch_resilient(
                client, paper_ids[:middle], failure_path
            ) + _fetch_resilient(client, paper_ids[middle:], failure_path)
        _append_failure(
            failure_path,
            {
                "paper_arxiv_id": paper_ids[0],
                "error": f"{type(exc).__name__}: {exc}",
                "classifier_version": S2_CLASSIFIER_VERSION,
            },
        )
        return []


def build_cache(
    pool_records_path: Path,
    output_path: Path,
    *,
    query_policies_path: Path | None = None,
    exclude_cache_path: Path | None = None,
    batch_size: int = S2_BATCH_MAX_IDS,
    resume: bool = False,
    max_papers: int | None = None,
    client: S2PublicationTypeClient | None = None,
) -> Dict[str, Any]:
    pool_records_path = pool_records_path.resolve()
    output_path = output_path.resolve()
    if not pool_records_path.exists():
        raise FileNotFoundError(pool_records_path)
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"{output_path} already exists; pass --resume to append missing candidates"
        )
    batch_size = int(batch_size)
    if not 1 <= batch_size <= S2_BATCH_MAX_IDS:
        raise ValueError(f"batch_size must be between 1 and {S2_BATCH_MAX_IDS}")

    existing = load_paper_type_cache(output_path) if output_path.exists() else {}
    excluded: Dict[str, Dict[str, Any]] = {}
    if exclude_cache_path is not None:
        exclude_cache_path = exclude_cache_path.resolve()
        if not exclude_cache_path.exists():
            raise FileNotFoundError(exclude_cache_path)
        excluded = load_paper_type_cache(exclude_cache_path)
    allowed_query_ids = None
    if query_policies_path is not None:
        query_policies_path = query_policies_path.resolve()
        if not query_policies_path.exists():
            raise FileNotFoundError(query_policies_path)
        allowed_query_ids = load_type_rule_query_ids(query_policies_path)

    candidate_ids = sorted(
        set(iter_pool_candidate_ids(pool_records_path, allowed_query_ids))
    )
    pending = sorted(set(candidate_ids) - set(existing) - set(excluded))
    if max_papers is not None:
        pending = pending[: max(0, int(max_papers))]
    client = client or S2PublicationTypeClient(
        api_key=os.environ.get("S2_API_KEY", "")
    )
    failure_path = output_path.with_suffix(output_path.suffix + ".failures.jsonl")
    started = time.time()
    written = 0
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        records = _fetch_resilient(client, batch, failure_path)
        written += _append_records(output_path, records)
        print(
            f"fetched={min(offset + len(batch), len(pending))}/{len(pending)} "
            f"written={written}",
            flush=True,
        )

    duplicate_records_removed = compact_cache(output_path)
    final_cache = load_paper_type_cache(output_path) if output_path.exists() else {}
    final_candidate_records = [
        final_cache[paper_id]
        for paper_id in candidate_ids
        if paper_id in final_cache
    ]
    publication_type_counts = Counter(
        publication_type
        for record in final_candidate_records
        for publication_type in record.get("publication_types") or []
    )
    summary = {
        "pool_records": str(pool_records_path),
        "query_policies": str(query_policies_path) if query_policies_path else None,
        "exclude_cache": str(exclude_cache_path) if exclude_cache_path else None,
        "output": str(output_path),
        "classifier_version": S2_CLASSIFIER_VERSION,
        "type_rule_query_count": (
            len(allowed_query_ids) if allowed_query_ids is not None else None
        ),
        "unique_candidate_count": len(candidate_ids),
        "existing_cache_count": len(existing),
        "exclude_cache_count": len(excluded),
        "requested_pending_count": len(pending),
        "classified_and_written_count": written,
        "duplicate_records_removed": duplicate_records_removed,
        "final_cache_count": len(final_cache),
        "remaining_candidate_count": len(
            set(candidate_ids) - set(final_cache) - set(excluded)
        ),
        "s2_resolved_count": sum(
            float(record.get("confidence") or 0.0) > 0.0
            for record in final_candidate_records
        ),
        "publication_types_present_count": sum(
            bool(record.get("publication_types"))
            for record in final_candidate_records
        ),
        "mapped_canonical_type_count": sum(
            bool(record.get("type_probs")) for record in final_candidate_records
        ),
        "publication_type_counts": dict(publication_type_counts.most_common()),
        "api_call_count": client.api_call_count,
        "retry_count": client.retry_count,
        "failure_log": str(failure_path),
        "elapsed_seconds": time.time() - started,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool_records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query_policies", type=Path, default=None)
    parser.add_argument("--exclude_cache", type=Path, default=None)
    parser.add_argument("--env_file", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=S2_BATCH_MAX_IDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_papers", type=int, default=None)
    parser.add_argument("--rate_limit_rps", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    client = S2PublicationTypeClient(
        api_key=os.environ.get("S2_API_KEY", ""),
        requests_per_second=args.rate_limit_rps,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    summary = build_cache(
        args.pool_records,
        args.output,
        query_policies_path=args.query_policies,
        exclude_cache_path=args.exclude_cache,
        batch_size=args.batch_size,
        resume=args.resume,
        max_papers=args.max_papers,
        client=client,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
