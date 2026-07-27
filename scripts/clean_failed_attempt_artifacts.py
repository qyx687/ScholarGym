#!/usr/bin/env python3
"""Back up and remove partial artifacts for uncheckpointed benchmark queries.

ScholarGym appends online artifacts before a query is checkpointed in
``detailed_results.jsonl``.  If a later Planner/Selector call fails, a normal
resume would otherwise mix the abandoned attempt with its retry.  This utility
derives unfinished indices from the checkpoint and is dry-run by default.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple


def artifact_dir(value: str | Path) -> tuple[Path, Path]:
    path = Path(value).expanduser().resolve()
    if (path / "online_artifacts").is_dir():
        return path, path / "online_artifacts"
    if path.name == "online_artifacts" and path.is_dir():
        return path.parent, path
    raise FileNotFoundError(f"online_artifacts not found below {path}")


def iter_jsonl(path: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield line, value


def completed_indices(detailed_results: Path) -> set[int]:
    output = set()
    if not detailed_results.is_file():
        return output
    for _, row in iter_jsonl(detailed_results):
        if row.get("idx") is not None:
            output.add(int(row["idx"]))
    return output


def record_index(
    row: Mapping[str, Any],
    *,
    query_indices: Mapping[str, int] | None = None,
    event_indices: Mapping[str, int] | None = None,
) -> int | None:
    if row.get("benchmark_idx") is not None:
        try:
            return int(row["benchmark_idx"])
        except (TypeError, ValueError):
            pass
    query_id = str(row.get("query_id") or "")
    if query_id and query_indices and query_id in query_indices:
        return int(query_indices[query_id])
    event_id = str(row.get("retrieval_event_id") or "")
    if event_id and event_indices and event_id in event_indices:
        return int(event_indices[event_id])
    return None


def _register_identifier(
    output: Dict[str, int],
    identifier: Any,
    benchmark_idx: int,
    *,
    kind: str,
) -> None:
    key = str(identifier or "")
    if not key:
        return
    previous = output.get(key)
    if previous is not None and previous != benchmark_idx:
        raise ValueError(
            f"conflicting {kind} mapping for {key!r}: "
            f"{previous} != {benchmark_idx}"
        )
    output[key] = benchmark_idx


def build_identifier_maps(
    files: Sequence[Path],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    query_indices: Dict[str, int] = {}
    event_indices: Dict[str, int] = {}
    for path in files:
        for _, row in iter_jsonl(path):
            if row.get("benchmark_idx") is None:
                continue
            try:
                benchmark_idx = int(row["benchmark_idx"])
            except (TypeError, ValueError):
                continue
            _register_identifier(
                query_indices,
                row.get("query_id"),
                benchmark_idx,
                kind="query_id",
            )
            _register_identifier(
                event_indices,
                row.get("retrieval_event_id"),
                benchmark_idx,
                kind="retrieval_event_id",
            )
    return query_indices, event_indices


def scan_file(
    path: Path,
    unfinished: set[int],
    *,
    query_indices: Mapping[str, int],
    event_indices: Mapping[str, int],
) -> Dict[str, int]:
    total = 0
    removed = 0
    for _, row in iter_jsonl(path):
        total += 1
        removed += int(
            record_index(
                row,
                query_indices=query_indices,
                event_indices=event_indices,
            )
            in unfinished
        )
    return {"total_rows": total, "removed_rows": removed}


def rewrite_file(
    path: Path,
    unfinished: set[int],
    backup_dir: Path,
    *,
    query_indices: Mapping[str, int],
    event_indices: Mapping[str, int],
) -> Dict[str, Any]:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / path.name
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite backup: {backup}")
    shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".resume-clean.tmp")
    total = 0
    removed = 0
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for line, row in iter_jsonl(path):
                total += 1
                if (
                    record_index(
                        row,
                        query_indices=query_indices,
                        event_indices=event_indices,
                    )
                    in unfinished
                ):
                    removed += 1
                    continue
                output.write(line if line.endswith("\n") else line + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "total_rows": total,
        "removed_rows": removed,
        "remaining_rows": total - removed,
        "backup": str(backup),
    }


def clean(
    run: str | Path,
    *,
    expected_count: int,
    apply: bool,
    backup_dir: str | Path | None = None,
) -> Dict[str, Any]:
    run_dir, artifacts = artifact_dir(run)
    detailed = run_dir / "detailed_results.jsonl"
    completed = completed_indices(detailed)
    unfinished = set(range(max(0, int(expected_count)))) - completed
    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup = run_dir / "resume_backups" / stamp
    else:
        backup = Path(backup_dir).expanduser().resolve()

    files = sorted(artifacts.glob("*.jsonl"))
    query_indices, event_indices = build_identifier_maps(files)
    scans = {
        path.name: scan_file(
            path,
            unfinished,
            query_indices=query_indices,
            event_indices=event_indices,
        )
        for path in files
    }
    affected = [
        path for path in files if scans[path.name]["removed_rows"] > 0
    ]
    rewrites: Dict[str, Dict[str, Any]] = {}
    if apply:
        for path in affected:
            rewrites[path.name] = rewrite_file(
                path,
                unfinished,
                backup,
                query_indices=query_indices,
                event_indices=event_indices,
            )
        backup.mkdir(parents=True, exist_ok=True)
        (backup / "cleanup_summary.json").write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "completed_indices": sorted(completed),
                    "unfinished_indices": sorted(unfinished),
                    "rewrites": rewrites,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "run_dir": str(run_dir),
        "detailed_results": str(detailed),
        "expected_count": int(expected_count),
        "completed_count": len(completed),
        "completed_indices": sorted(completed),
        "unfinished_indices": sorted(unfinished),
        "dry_run": not apply,
        "query_index_mapping_count": len(query_indices),
        "event_index_mapping_count": len(event_indices),
        "affected_files": [path.name for path in affected],
        "scan": scans,
        "backup_dir": str(backup) if apply else None,
        "rewrites": rewrites,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--expected_count", type=int, default=50)
    parser.add_argument("--backup_dir")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            clean(
                args.run,
                expected_count=args.expected_count,
                apply=args.apply,
                backup_dir=args.backup_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
