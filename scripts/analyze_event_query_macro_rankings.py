#!/usr/bin/env python3
"""Evaluate retrieval-event rankings with query-balanced macro averaging.

The metric contract is intentionally strict:

1. compute R@K, nDCG@K, and AP@K independently for every retrieval event;
2. use the complete relevant-paper set of the event's original query;
3. average events with equal weight inside each original query;
4. average the query means with equal weight across the benchmark.

Both flat ``paper_rows.jsonl`` artifacts and nested
``ranked_candidates.jsonl`` replay artifacts are supported.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

try:
    import orjson
except ImportError:  # pragma: no cover - stdlib fallback for minimal installs
    orjson = None


ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class QueryTruth:
    benchmark_idx: int
    query_id: str
    query: str
    relevant_ids: frozenset[str]


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = (
                    orjson.loads(line)
                    if orjson is not None
                    else json.loads(line.decode("utf-8"))
                )
            except Exception as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def normalize_arxiv_id(value: Any) -> str:
    paper_id = str(value or "").strip()
    if not paper_id:
        return ""
    paper_id = paper_id.replace("https://arxiv.org/abs/", "")
    paper_id = paper_id.replace("http://arxiv.org/abs/", "")
    paper_id = paper_id.removeprefix("arXiv:").removeprefix("arxiv:")
    paper_id = paper_id.split("?", 1)[0].split("#", 1)[0].strip("/")
    return ARXIV_VERSION_RE.sub("", paper_id)


def load_benchmark(path: Path) -> tuple[
    Dict[int, QueryTruth],
    Dict[str, int],
    Dict[str, int],
]:
    truths: Dict[int, QueryTruth] = {}
    by_query_id: Dict[str, int] = {}
    by_query_text: Dict[str, int] = {}
    for benchmark_idx, row in enumerate(read_jsonl(path)):
        if row.get("valid") is False:
            continue
        cited = row.get("cited_paper") or []
        labels = row.get("gt_label") or []
        if len(cited) != len(labels):
            raise ValueError(
                f"benchmark row {benchmark_idx} has mismatched cited/label lengths"
            )
        relevant = frozenset(
            normalize_arxiv_id(paper.get("arxiv_id"))
            for paper, label in zip(cited, labels)
            if label and isinstance(paper, Mapping)
        )
        relevant = frozenset(paper_id for paper_id in relevant if paper_id)
        if not relevant:
            raise ValueError(f"benchmark row {benchmark_idx} has no relevant papers")
        query_id = str(row.get("qid") or f"RealScholarQuery_{benchmark_idx}")
        query = str(row.get("query") or "")
        truth = QueryTruth(
            benchmark_idx=benchmark_idx,
            query_id=query_id,
            query=query,
            relevant_ids=relevant,
        )
        truths[benchmark_idx] = truth
        if query_id in by_query_id:
            raise ValueError(f"duplicate benchmark query id: {query_id}")
        by_query_id[query_id] = benchmark_idx
        if query:
            if query in by_query_text:
                raise ValueError(f"duplicate benchmark query text: {query}")
            by_query_text[query] = benchmark_idx
    if not truths:
        raise ValueError(f"no valid benchmark queries found in {path}")
    return truths, by_query_id, by_query_text


def resolve_query_index(
    row: Mapping[str, Any],
    truths: Mapping[int, QueryTruth],
    by_query_id: Mapping[str, int],
    by_query_text: Mapping[str, int],
) -> int:
    raw_idx = row.get("benchmark_idx")
    try:
        benchmark_idx = int(raw_idx)
    except (TypeError, ValueError):
        benchmark_idx = -1
    if benchmark_idx in truths:
        truth = truths[benchmark_idx]
        row_query_id = str(row.get("query_id") or "")
        if row_query_id and row_query_id != truth.query_id:
            raise ValueError(
                f"query-id/index mismatch: {row_query_id} != {truth.query_id}"
            )
        return benchmark_idx
    query_id = str(row.get("query_id") or "")
    if query_id in by_query_id:
        return by_query_id[query_id]
    query = str(row.get("query") or "")
    if query in by_query_text:
        return by_query_text[query]
    raise ValueError(
        "cannot map artifact row to benchmark query: "
        f"benchmark_idx={raw_idx!r}, query_id={query_id!r}"
    )


def validate_ranking(
    rows: Sequence[tuple[int, str]],
    *,
    source: str,
) -> list[str]:
    ordered = sorted(rows, key=lambda item: (item[0], item[1]))
    ranks = [rank for rank, _ in ordered]
    expected = list(range(1, len(ordered) + 1))
    if ranks != expected:
        raise ValueError(
            f"non-consecutive ranks in {source}: "
            f"observed={ranks[:20]!r}, expected={expected[:20]!r}"
        )
    ranking = [paper_id for _, paper_id in ordered]
    if len(ranking) != len(set(ranking)):
        raise ValueError(f"duplicate paper ids in {source}")
    return ranking


def load_flat_events(
    path: Path,
    rank_field: str,
    truths: Mapping[int, QueryTruth],
    by_query_id: Mapping[str, int],
    by_query_text: Mapping[str, int],
) -> Dict[tuple[int, str], list[str]]:
    events: Dict[tuple[int, str], list[str]] = {}
    current_key: tuple[int, str] | None = None
    current_rows: list[tuple[int, str]] = []

    def finish() -> None:
        if current_key is None or not current_rows:
            return
        events[current_key] = validate_ranking(
            current_rows,
            source=f"{path}:{current_key[1]}",
        )

    for row in read_jsonl(path):
        if row.get("passed_date_cutoff") is False:
            raise ValueError(f"date-invalid row present in ranked artifact: {path}")
        benchmark_idx = resolve_query_index(
            row,
            truths,
            by_query_id,
            by_query_text,
        )
        event_id = str(row.get("retrieval_event_id") or "")
        if not event_id:
            raise ValueError(f"missing retrieval_event_id in {path}")
        key = (benchmark_idx, event_id)
        raw_rank = row.get(rank_field)
        if raw_rank is None:
            if not row.get("hard_filtered"):
                raise ValueError(
                    f"missing {rank_field} on non-filtered row for event "
                    f"{event_id} in {path}"
                )
            continue
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid {rank_field} for event {event_id} in {path}"
            ) from exc
        paper_id = normalize_arxiv_id(row.get("paper_arxiv_id"))
        if not paper_id:
            raise ValueError(f"missing paper_arxiv_id for event {event_id}")
        starts_repeated_occurrence = bool(
            key == current_key and current_rows and rank == 1
        )
        if current_key is not None and (
            key != current_key or starts_repeated_occurrence
        ):
            finish()
            current_rows = []
        current_key = key
        current_rows.append((rank, paper_id))
    finish()
    return events


def load_nested_events(
    path: Path,
    method_filter: str,
    truths: Mapping[int, QueryTruth],
    by_query_id: Mapping[str, int],
    by_query_text: Mapping[str, int],
) -> Dict[tuple[int, str], list[str]]:
    events: Dict[tuple[int, str], list[str]] = {}
    for row in read_jsonl(path):
        if str(row.get("method") or "") != method_filter:
            continue
        benchmark_idx = resolve_query_index(
            row,
            truths,
            by_query_id,
            by_query_text,
        )
        event_id = str(row.get("retrieval_event_id") or "")
        if not event_id:
            raise ValueError(f"missing retrieval_event_id in {path}")
        candidates = row.get("ranked_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"missing ranked_candidates for event {event_id}")
        ranked_rows: list[tuple[int, str]] = []
        for offset, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, Mapping):
                raise ValueError(f"invalid candidate for event {event_id}")
            if candidate.get("passed_date_cutoff") is False:
                raise ValueError(
                    f"date-invalid candidate in event {event_id} from {path}"
                )
            raw_rank = candidate.get("rerank_rank", offset)
            if raw_rank is None:
                if not candidate.get("hard_filtered"):
                    raise ValueError(
                        "candidate without rerank_rank is not marked "
                        f"hard-filtered in event {event_id}"
                    )
                continue
            try:
                rank = int(raw_rank)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid rerank_rank for event {event_id}"
                ) from exc
            paper_id = normalize_arxiv_id(candidate.get("paper_arxiv_id"))
            if not paper_id:
                raise ValueError(f"missing paper id for event {event_id}")
            ranked_rows.append((rank, paper_id))
        events[(benchmark_idx, event_id)] = validate_ranking(
            ranked_rows,
            source=f"{path}:{event_id}:{method_filter}",
        )
    return events


def event_metrics(
    ranking: Sequence[str],
    relevant: frozenset[str],
    cutoff: int,
) -> Dict[str, float]:
    top_k = ranking[:cutoff]
    hits = 0
    precision_sum = 0.0
    dcg = 0.0
    for rank, paper_id in enumerate(top_k, start=1):
        if paper_id not in relevant:
            continue
        hits += 1
        precision_sum += hits / rank
        dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), cutoff)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )
    return {
        "recall": hits / len(relevant),
        "ndcg": dcg / idcg if idcg else 0.0,
        "ap": precision_sum / ideal_hits if ideal_hits else 0.0,
    }


def aggregate_method(
    events: Mapping[tuple[int, str], Sequence[str]],
    truths: Mapping[int, QueryTruth],
    cutoffs: Sequence[int],
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    by_query: Dict[int, list[tuple[str, Sequence[str]]]] = defaultdict(list)
    for (benchmark_idx, event_id), ranking in events.items():
        if benchmark_idx not in truths:
            raise ValueError(f"unknown benchmark index: {benchmark_idx}")
        by_query[benchmark_idx].append((event_id, ranking))
    missing = sorted(set(truths) - set(by_query))
    if missing:
        raise ValueError(f"artifact has no retrieval events for queries: {missing}")

    query_rows: list[Dict[str, Any]] = []
    for benchmark_idx in sorted(truths):
        truth = truths[benchmark_idx]
        query_events = sorted(by_query[benchmark_idx])
        row: Dict[str, Any] = {
            "benchmark_idx": benchmark_idx,
            "query_id": truth.query_id,
            "event_count": len(query_events),
            "gt_count": len(truth.relevant_ids),
        }
        for cutoff in cutoffs:
            values = [
                event_metrics(ranking, truth.relevant_ids, cutoff)
                for _, ranking in query_events
            ]
            row[f"recall@{cutoff}"] = mean(
                value["recall"] for value in values
            )
            row[f"ndcg@{cutoff}"] = mean(
                value["ndcg"] for value in values
            )
            row[f"ap@{cutoff}"] = mean(value["ap"] for value in values)
        query_rows.append(row)

    event_counts = [int(row["event_count"]) for row in query_rows]
    ranking_depths = [len(ranking) for ranking in events.values()]
    summary: Dict[str, Any] = {
        "query_count": len(query_rows),
        "retrieval_event_count": sum(event_counts),
        "events_per_query": {
            "min": min(event_counts),
            "max": max(event_counts),
            "mean": mean(event_counts),
        },
        "ranking_depth": {
            "min": min(ranking_depths),
            "max": max(ranking_depths),
            "mean": mean(ranking_depths),
            "coverage_at_cutoff": {
                str(cutoff): (
                    sum(depth >= cutoff for depth in ranking_depths)
                    / len(ranking_depths)
                )
                for cutoff in cutoffs
            },
        },
        "metrics": {},
    }
    for cutoff in cutoffs:
        summary["metrics"][str(cutoff)] = {
            "recall": mean(row[f"recall@{cutoff}"] for row in query_rows),
            "ndcg": mean(row[f"ndcg@{cutoff}"] for row in query_rows),
            "map": mean(row[f"ap@{cutoff}"] for row in query_rows),
        }
    return summary, query_rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument(
        "--flat",
        action="append",
        nargs=3,
        default=[],
        metavar=("NAME", "PATH", "RANK_FIELD"),
        help="flat paper rows: NAME PATH RANK_FIELD",
    )
    parser.add_argument(
        "--nested",
        action="append",
        nargs=3,
        default=[],
        metavar=("NAME", "PATH", "METHOD"),
        help="nested ranked candidates: NAME PATH METHOD_FILTER",
    )
    parser.add_argument(
        "--cutoff",
        action="append",
        type=int,
        dest="cutoffs",
    )
    parser.add_argument("--expected_query_count", type=int, default=50)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    cutoffs = sorted(set(args.cutoffs or [10, 20]))
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        parser.error("cutoffs must be positive")
    truths, by_query_id, by_query_text = load_benchmark(
        args.benchmark.expanduser().resolve()
    )
    if len(truths) != args.expected_query_count:
        raise ValueError(
            f"expected {args.expected_query_count} valid queries, "
            f"found {len(truths)}"
        )
    if not args.flat and not args.nested:
        parser.error("at least one --flat or --nested input is required")

    methods: Dict[str, Dict[tuple[int, str], list[str]]] = {}
    sources: Dict[str, Dict[str, str]] = {}
    for name, raw_path, rank_field in args.flat:
        if name in methods:
            parser.error(f"duplicate method name: {name}")
        path = Path(raw_path).expanduser().resolve()
        methods[name] = load_flat_events(
            path,
            rank_field,
            truths,
            by_query_id,
            by_query_text,
        )
        sources[name] = {
            "format": "flat_paper_rows",
            "path": str(path),
            "rank_field": rank_field,
        }
    for name, raw_path, method_filter in args.nested:
        if name in methods:
            parser.error(f"duplicate method name: {name}")
        path = Path(raw_path).expanduser().resolve()
        methods[name] = load_nested_events(
            path,
            method_filter,
            truths,
            by_query_id,
            by_query_text,
        )
        sources[name] = {
            "format": "nested_ranked_candidates",
            "path": str(path),
            "method_filter": method_filter,
        }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "schema_version": "event_query_macro_ranking_v1",
        "aggregation": [
            "metric per retrieval event",
            "equal-weight mean over events within original query",
            "equal-weight mean over all valid original queries",
        ],
        "relevance": "complete GT set of the original query",
        "ap_denominator": "min(number_of_complete_query_gt, K)",
        "cutoffs": cutoffs,
        "benchmark": str(args.benchmark.expanduser().resolve()),
        "benchmark_query_count": len(truths),
        "benchmark_gt_count": sum(
            len(truth.relevant_ids) for truth in truths.values()
        ),
        "methods": {},
        "sources": sources,
    }
    all_query_rows: list[Dict[str, Any]] = []
    for name, events in methods.items():
        method_summary, query_rows = aggregate_method(
            events,
            truths,
            cutoffs,
        )
        summary["methods"][name] = method_summary
        all_query_rows.extend({"method": name, **row} for row in query_rows)

    write_json(output / "summary.json", summary)
    write_jsonl(output / "query_metrics.jsonl", all_query_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
