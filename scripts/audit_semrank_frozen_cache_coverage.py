#!/usr/bin/env python3
"""Read-only paper-profile coverage audit for frozen candidate pools."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator


def artifact_dir(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / "paper_rows.jsonl").is_file():
        return path
    nested = path / "online_artifacts"
    if (nested / "paper_rows.jsonl").is_file():
        return nested
    raise FileNotFoundError(f"paper_rows.jsonl not found below {path}")


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


def cache_paper_ids(value: str | Path) -> tuple[Path, set[str]]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "semrank.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        paper_ids = {
            str(row[0])
            for row in connection.execute(
                "SELECT paper_arxiv_id FROM paper_profiles"
            )
        }
    finally:
        connection.close()
    return path, paper_ids


def audit(
    candidate_run: str | Path,
    cache_values: list[str],
) -> Dict[str, Any]:
    artifacts = artifact_dir(candidate_run)
    candidate_ids: set[str] = set()
    query_ids: set[str] = set()
    subqueries: set[tuple[str, str]] = set()
    event_ids: set[str] = set()
    row_count = 0
    for row in iter_jsonl(artifacts / "paper_rows.jsonl"):
        row_count += 1
        candidate_ids.add(str(row.get("paper_arxiv_id") or ""))
        query_id = str(row.get("query_id") or "")
        query_ids.add(query_id)
        subqueries.add((query_id, str(row.get("subquery") or "")))
        event_ids.add(str(row.get("retrieval_event_id") or ""))
    candidate_ids.discard("")
    query_ids.discard("")
    event_ids.discard("")

    caches: Dict[str, Any] = {}
    for value in cache_values:
        path, cached_ids = cache_paper_ids(value)
        missing = sorted(candidate_ids - cached_ids)
        caches[str(path)] = {
            "paper_profile_count": len(cached_ids),
            "covered_candidate_id_count": len(candidate_ids & cached_ids),
            "missing_candidate_id_count": len(missing),
            "coverage_rate": (
                len(candidate_ids & cached_ids) / len(candidate_ids)
                if candidate_ids
                else 1.0
            ),
            "missing_candidate_id_preview": missing[:50],
        }
    return {
        "candidate_artifacts": str(artifacts),
        "row_count": row_count,
        "event_count": len(event_ids),
        "query_count": len(query_ids),
        "query_subquery_pair_count": len(subqueries),
        "unique_candidate_id_count": len(candidate_ids),
        "caches": caches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_run", required=True)
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--report_json")
    args = parser.parse_args()
    report = audit(args.candidate_run, args.cache)
    if args.report_json:
        output = Path(args.report_json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
