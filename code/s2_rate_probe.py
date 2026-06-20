#!/usr/bin/env python3
"""Probe Semantic Scholar Graph API rate behavior for the current API key.

This script sends a small, configurable sequence of requests and reports
success/error counts, latency, 429s, and rate-related response headers.

Example:

    S2_API_KEY=... python code/s2_rate_probe.py --rps 1 2 4 --requests-per-step 20 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import quote

import requests


BASE_URL = "https://api.semanticscholar.org/graph/v1"
DEFAULT_PAPER_ID = "ARXIV:1706.03762"
DEFAULT_FIELDS = "paperId,title,year,externalIds"
RATE_HEADER_PREFIXES = (
    "x-ratelimit",
    "x-rate-limit",
    "retry-after",
)


@dataclass
class RequestRecord:
    step_rps: float
    request_index: int
    started_offset_s: float
    completed_offset_s: float
    status_code: int
    latency_s: float
    ok: bool
    retry_after: str
    rate_headers: Dict[str, str]
    error: str = ""


def rate_headers(headers: requests.structures.CaseInsensitiveDict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower.startswith(RATE_HEADER_PREFIXES):
            out[key] = value
    return out


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return values[idx]


def summarize(records: Iterable[RequestRecord], include_step_timing: bool = False) -> Dict[str, object]:
    rows = list(records)
    latencies = [r.latency_s for r in rows if r.latency_s >= 0]
    by_status: Dict[int, int] = {}
    for row in rows:
        by_status[row.status_code] = by_status.get(row.status_code, 0) + 1
    summary = {
        "requests": len(rows),
        "ok": sum(1 for r in rows if r.ok),
        "status_counts": dict(sorted(by_status.items())),
        "429_count": by_status.get(429, 0),
        "error_count": sum(1 for r in rows if r.error),
        "latency_mean_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "latency_p50_s": round(percentile(latencies, 50), 4),
        "latency_p95_s": round(percentile(latencies, 95), 4),
        "latency_max_s": round(max(latencies), 4) if latencies else 0.0,
    }
    if include_step_timing:
        elapsed_s = max((r.completed_offset_s for r in rows), default=0.0)
        summary["elapsed_s"] = round(elapsed_s, 4)
        summary["achieved_rps"] = round(len(rows) / elapsed_s, 4) if elapsed_s > 0 else 0.0
    return summary


def print_summary(step_rps: float, records: List[RequestRecord]) -> None:
    summary = summarize(records, include_step_timing=True)
    status_counts = ",".join(f"{k}:{v}" for k, v in summary["status_counts"].items())
    retry_after_values = sorted({r.retry_after for r in records if r.retry_after})
    print(
        f"rps={step_rps:g} requests={summary['requests']} ok={summary['ok']} "
        f"achieved_rps={summary['achieved_rps']} elapsed={summary['elapsed_s']}s "
        f"429={summary['429_count']} errors={summary['error_count']} "
        f"status={status_counts or '-'} "
        f"latency_mean={summary['latency_mean_s']}s p95={summary['latency_p95_s']}s "
        f"retry_after={','.join(retry_after_values) or '-'}"
    )


def build_url(endpoint: str, paper_id: str) -> str:
    encoded_paper_id = quote(paper_id, safe="")
    if endpoint == "paper":
        return f"{BASE_URL}/paper/{encoded_paper_id}"
    if endpoint == "citations":
        return f"{BASE_URL}/paper/{encoded_paper_id}/citations"
    if endpoint == "references":
        return f"{BASE_URL}/paper/{encoded_paper_id}/references"
    raise ValueError(f"unsupported endpoint: {endpoint}")


def build_params(endpoint: str, fields: str, limit: int) -> Dict[str, object]:
    if endpoint == "paper":
        return {"fields": fields}
    prefix = "citingPaper" if endpoint == "citations" else "citedPaper"
    relation_fields = ",".join(f"{prefix}.{field.strip()}" for field in fields.split(",") if field.strip())
    return {"fields": relation_fields, "offset": 0, "limit": limit}


def send_request(
    *,
    headers: Dict[str, str],
    url: str,
    params: Dict[str, object],
    step_rps: float,
    request_index: int,
    timeout: float,
    step_start: float,
) -> RequestRecord:
    start = time.monotonic()
    status_code = 0
    retry_after = ""
    response_rate_headers: Dict[str, str] = {}
    error = ""
    ok = False
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        status_code = response.status_code
        retry_after = response.headers.get("Retry-After", "")
        response_rate_headers = rate_headers(response.headers)
        ok = response.ok
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    completed = time.monotonic()
    latency = completed - start
    return RequestRecord(
        step_rps=step_rps,
        request_index=request_index,
        started_offset_s=round(start - step_start, 4),
        completed_offset_s=round(completed - step_start, 4),
        status_code=status_code,
        latency_s=round(latency, 4),
        ok=ok,
        retry_after=retry_after,
        rate_headers=response_rate_headers,
        error=error,
    )


def collect_done(futures, rows: List[RequestRecord], stop_on_429: bool) -> bool:
    stop = False
    done, _ = wait(futures, return_when=FIRST_COMPLETED)
    for future in done:
        futures.remove(future)
        row = future.result()
        rows.append(row)
        if stop_on_429 and row.status_code == 429:
            stop = True
    return stop


def probe_step(
    *,
    headers: Dict[str, str],
    url: str,
    params: Dict[str, object],
    step_rps: float,
    requests_per_step: int,
    timeout: float,
    stop_on_429: bool,
    concurrency: int,
) -> List[RequestRecord]:
    rows: List[RequestRecord] = []
    futures = []
    interval = 1.0 / step_rps if step_rps > 0 else 0.0
    next_at = time.monotonic()
    step_start = next_at
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        for idx in range(1, requests_per_step + 1):
            while len(futures) >= max(1, concurrency):
                if collect_done(futures, rows, stop_on_429):
                    break
            if stop_on_429 and any(row.status_code == 429 for row in rows):
                break
            now = time.monotonic()
            if now < next_at:
                time.sleep(next_at - now)
            futures.append(
                executor.submit(
                    send_request,
                    headers=headers,
                    url=url,
                    params=params,
                    step_rps=step_rps,
                    request_index=idx,
                    timeout=timeout,
                    step_start=step_start,
                )
            )
            next_at += interval
        while futures:
            collect_done(futures, rows, stop_on_429=False)
            if stop_on_429 and any(row.status_code == 429 for row in rows):
                # In-flight requests are still collected, but no more are scheduled.
                continue
    rows.sort(key=lambda row: row.request_index)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe S2 Graph API rate behavior.")
    parser.add_argument("--api-key", default=os.environ.get("S2_API_KEY"), help="S2 API key; defaults to S2_API_KEY env var")
    parser.add_argument("--paper-id", default=DEFAULT_PAPER_ID, help="S2 paper id to request, e.g. ARXIV:1706.03762")
    parser.add_argument("--endpoint", choices=["paper", "citations", "references"], default="paper")
    parser.add_argument("--fields", default=DEFAULT_FIELDS)
    parser.add_argument("--relation-limit", type=int, default=1, help="Limit for citations/references endpoint")
    parser.add_argument("--rps", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0], help="Target RPS values to test")
    parser.add_argument("--requests-per-step", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum in-flight requests for each RPS step")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds to sleep between RPS steps")
    parser.add_argument("--stop-on-429", action="store_true", help="Stop a step as soon as a 429 appears")
    parser.add_argument("--output-jsonl", default="", help="Optional path for per-request JSONL records")
    parser.add_argument("--no-key", action="store_true", help="Run without an API key to compare unauthenticated behavior")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.no_key and not args.api_key:
        print("Missing S2 API key. Set S2_API_KEY or pass --api-key, or use --no-key.", file=sys.stderr)
        return 2
    if args.requests_per_step <= 0:
        print("--requests-per-step must be positive", file=sys.stderr)
        return 2
    if any(rps <= 0 for rps in args.rps):
        print("--rps values must be positive", file=sys.stderr)
        return 2
    if args.concurrency <= 0:
        print("--concurrency must be positive", file=sys.stderr)
        return 2

    headers = {"User-Agent": "ScholarGym-S2RateProbe/0.1"}
    if args.api_key and not args.no_key:
        headers["x-api-key"] = args.api_key

    url = build_url(args.endpoint, args.paper_id)
    params = build_params(args.endpoint, args.fields, args.relation_limit)
    output_path: Optional[Path] = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    print(
        f"endpoint={args.endpoint} paper_id={args.paper_id} "
        f"requests_per_step={args.requests_per_step} concurrency={args.concurrency} "
        f"keyed={bool(args.api_key and not args.no_key)}"
    )
    all_records: List[RequestRecord] = []
    for step_num, step_rps in enumerate(args.rps, start=1):
        records = probe_step(
            headers=headers,
            url=url,
            params=params,
            step_rps=step_rps,
            requests_per_step=args.requests_per_step,
            timeout=args.timeout,
            stop_on_429=args.stop_on_429,
            concurrency=args.concurrency,
        )
        all_records.extend(records)
        print_summary(step_rps, records)
        if output_path:
            with output_path.open("a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        if step_num < len(args.rps) and args.cooldown > 0:
            time.sleep(args.cooldown)

    print("overall", json.dumps(summarize(all_records), ensure_ascii=False, sort_keys=True))
    if output_path:
        print(f"records={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
