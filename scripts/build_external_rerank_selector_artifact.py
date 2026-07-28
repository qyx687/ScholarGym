#!/usr/bin/env python3
"""Build a strict OnePass Static-vs-external Selector replay artifact.

The static arm is copied byte-for-data from a completed canonical replay.  The
external arm ranks the exact same materialized per-subquery pool using a
row-level score file, then emits the same per-event Selector Top-K depth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def normalize_id(value: Any) -> str:
    return str(value or "").strip()


def load_pool_ids(path: Path) -> "OrderedDict[str, tuple[str, ...]]":
    output: "OrderedDict[str, tuple[str, ...]]" = OrderedDict()
    for row in iter_jsonl(path):
        event_id = str(row.get("retrieval_event_id") or "")
        raw_ids = row.get("local_pool_arxiv_ids")
        if not event_id or not isinstance(raw_ids, list):
            raise ValueError(
                "each pool record needs retrieval_event_id and "
                "local_pool_arxiv_ids"
            )
        ids = tuple(normalize_id(value) for value in raw_ids)
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"invalid candidate IDs in pool event {event_id}")
        if event_id in output:
            raise ValueError(f"duplicate pool event {event_id}")
        output[event_id] = ids
    if not output:
        raise ValueError(f"no pool records found in {path}")
    return output


def load_static_rows(path: Path) -> "OrderedDict[str, Dict[str, Any]]":
    output: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for row in iter_jsonl(path):
        if str(row.get("method") or "") != "legacy_static":
            continue
        event_id = str(row.get("retrieval_event_id") or "")
        if not event_id:
            raise ValueError("static ranked row is missing retrieval_event_id")
        if event_id in output:
            raise ValueError(f"duplicate legacy_static event {event_id}")
        candidates = row.get("ranked_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"static event {event_id} has no candidate list")
        output[event_id] = row
    if not output:
        raise ValueError(f"no legacy_static rows found in {path}")
    return output


def load_scores(
    path: Path,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    output: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in iter_jsonl(path):
        event_id = str(row.get("retrieval_event_id") or "")
        paper_id = normalize_id(row.get("paper_arxiv_id"))
        score = row.get("rerank_score")
        rank = row.get("rerank_rank")
        if not event_id or not paper_id:
            raise ValueError("score row is missing event or paper ID")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"invalid score for {event_id}:{paper_id}")
        if not math.isfinite(float(score)):
            raise ValueError(f"non-finite score for {event_id}:{paper_id}")
        event = output.setdefault(event_id, {})
        if paper_id in event:
            raise ValueError(f"duplicate score for {event_id}:{paper_id}")
        record = {"score": float(score)}
        if rank is not None:
            try:
                record["rank"] = float(rank)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid rank for {event_id}:{paper_id}"
                ) from exc
        event[paper_id] = record
    if not output:
        raise ValueError(f"no external scores found in {path}")
    return output


def ordered_external_ids(
    pool_ids: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
) -> list[str]:
    if all("rank" in scores[paper_id] for paper_id in pool_ids):
        ranks = [scores[paper_id]["rank"] for paper_id in pool_ids]
        if len(ranks) != len(set(ranks)):
            raise ValueError("external ranks are not unique within an event")
        return sorted(
            pool_ids,
            key=lambda paper_id: (
                scores[paper_id]["rank"],
                paper_id,
            ),
        )
    return sorted(
        pool_ids,
        key=lambda paper_id: (
            -scores[paper_id]["score"],
            paper_id,
        ),
    )


def selector_scores(
    pool_ids: Sequence[str],
    scores: Mapping[str, Mapping[str, float]],
    transform: str,
) -> tuple[Dict[str, float], bool]:
    raw = {paper_id: float(scores[paper_id]["score"]) for paper_id in pool_ids}
    if transform == "none":
        return raw, False
    if transform != "minmax":
        raise ValueError(f"unsupported score transform: {transform}")
    low = min(raw.values())
    high = max(raw.values())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return {paper_id: 0.5 for paper_id in pool_ids}, True
    span = high - low
    return {
        paper_id: (score - low) / span for paper_id, score in raw.items()
    }, False


def build_artifact(
    *,
    static_ranked_candidates: Path,
    pool_records: Path,
    external_scores: Path,
    external_policy_id: str,
    selector_score_transform: str,
    output: Path,
) -> Dict[str, Any]:
    pools = load_pool_ids(pool_records)
    static_rows = load_static_rows(static_ranked_candidates)
    scores = load_scores(external_scores)
    pool_events = set(pools)
    if set(static_rows) != pool_events:
        raise ValueError(
            "legacy_static events do not exactly match pool-record events"
        )
    if set(scores) != pool_events:
        missing = sorted(pool_events - set(scores))
        extra = sorted(set(scores) - pool_events)
        raise ValueError(
            f"external score events differ: missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    candidate_occurrences = 0
    selector_occurrences = 0
    constant_score_event_count = 0
    query_ids: set[str] = set()
    with temporary.open("w", encoding="utf-8") as handle:
        for event_id, pool_ids in pools.items():
            static = static_rows[event_id]
            event_scores = scores[event_id]
            if set(event_scores) != set(pool_ids):
                missing = sorted(set(pool_ids) - set(event_scores))
                extra = sorted(set(event_scores) - set(pool_ids))
                raise ValueError(
                    f"candidate mismatch in {event_id}: "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
            top_k = int(static.get("selector_top_k") or 0)
            if top_k <= 0:
                raise ValueError(f"invalid selector_top_k in {event_id}")
            ordered = ordered_external_ids(pool_ids, event_scores)
            calibrated_scores, constant_scores = selector_scores(
                pool_ids,
                event_scores,
                selector_score_transform,
            )
            constant_score_event_count += int(constant_scores)
            selected = ordered[:top_k]
            candidates = []
            for artifact_rank, paper_id in enumerate(selected, start=1):
                score = calibrated_scores[paper_id]
                candidates.append(
                    {
                        "artifact_rank": artifact_rank,
                        "hard_filtered": False,
                        "paper_arxiv_id": paper_id,
                        "rerank_policy_id": external_policy_id,
                        "rerank_rank": artifact_rank,
                        "rerank_score": score,
                        "selected_at_event_top_k": True,
                    }
                )
            external = {
                key: static.get(key)
                for key in (
                    "query_id",
                    "benchmark_idx",
                    "retrieval_event_id",
                    "iteration_idx",
                    "subquery_id",
                    "subquery",
                    "selector_top_k",
                )
            }
            external.update(
                {
                    "method": "dynamic_policy",
                    "rerank_policy_id": external_policy_id,
                    "ranked_candidates": candidates,
                }
            )
            handle.write(
                json.dumps(static, ensure_ascii=False, sort_keys=True) + "\n"
            )
            handle.write(
                json.dumps(external, ensure_ascii=False, sort_keys=True) + "\n"
            )
            candidate_occurrences += len(pool_ids)
            selector_occurrences += len(candidates)
            query_ids.add(str(static.get("query_id") or ""))
        handle.flush()
    os.replace(temporary, output)

    summary = {
        "static_ranked_candidates": str(
            static_ranked_candidates.expanduser().resolve()
        ),
        "pool_records": str(pool_records.expanduser().resolve()),
        "external_scores": str(external_scores.expanduser().resolve()),
        "external_policy_id": external_policy_id,
        "selector_score_transform": selector_score_transform,
        "constant_score_event_count": constant_score_event_count,
        "output": str(output),
        "query_count": len(query_ids - {""}),
        "event_count": len(pools),
        "candidate_occurrence_count": candidate_occurrences,
        "selector_occurrence_count_per_method": selector_occurrences,
        "candidate_pool_equality": True,
        "selector_top_k_equality": True,
        "static_rows_copied_without_rescoring": True,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-ranked-candidates", required=True, type=Path)
    parser.add_argument("--pool-records", required=True, type=Path)
    parser.add_argument("--external-scores", required=True, type=Path)
    parser.add_argument("--external-policy-id", required=True)
    parser.add_argument(
        "--selector-score-transform",
        choices=("none", "minmax"),
        default="none",
        help=(
            "Optional event-full-pool calibration applied only to scores shown "
            "to Selector; ordering and Top-K remain unchanged."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_artifact(
        static_ranked_candidates=args.static_ranked_candidates,
        pool_records=args.pool_records,
        external_scores=args.external_scores,
        external_policy_id=args.external_policy_id,
        selector_score_transform=args.selector_score_transform,
        output=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
