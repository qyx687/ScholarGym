#!/usr/bin/env python3
"""Remove an abandoned attempt for one completed, retried benchmark query.

Online artifacts are append-only.  If a process is interrupted after writing
artifacts but before checkpointing ``detailed_results.jsonl``, resuming the run
can leave both the abandoned attempt and the successful retry in JSONL files.

This utility is deliberately conservative:

* event/iteration-level rows are split at the final strict decrease in
  ``iteration_idx`` (for example, 4 -> 1);
* query-level rows without an iteration are deduplicated only when every
  occurrence is byte-identical;
* any other repeated rows are reported as ambiguous and make ``--apply`` fail;
* every changed JSONL file is copied in full before an atomic rewrite.

The default mode is a read-only dry run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence


def artifact_dir(value: str | Path) -> tuple[Path, Path]:
    path = Path(value).expanduser().resolve()
    if (path / "online_artifacts").is_dir():
        return path, path / "online_artifacts"
    if path.name == "online_artifacts" and path.is_dir():
        return path.parent, path
    raise FileNotFoundError(f"online_artifacts not found below {path}")


def iter_jsonl(
    path: Path,
) -> Iterator[tuple[int, str, Mapping[str, Any]]]:
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
            yield line_number, line, value


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def plan_file(path: Path, query_id: str) -> Dict[str, Any]:
    target: list[Dict[str, Any]] = []
    total_rows = 0
    for line_number, line, row in iter_jsonl(path):
        total_rows += 1
        if str(row.get("query_id") or "") != query_id:
            continue
        target.append(
            {
                "line_number": line_number,
                "raw": line.rstrip("\r\n"),
                "iteration_idx": _integer(row.get("iteration_idx")),
            }
        )

    result: Dict[str, Any] = {
        "total_rows": total_rows,
        "target_rows": len(target),
        "strategy": "none",
        "remove_line_numbers": [],
        "removed_rows": 0,
        "remaining_target_rows": len(target),
        "iteration_resets": [],
        "ambiguous": False,
    }
    if len(target) <= 1:
        return result

    resets: list[Dict[str, int]] = []
    previous_iteration: int | None = None
    for target_offset, item in enumerate(target):
        iteration = item["iteration_idx"]
        if iteration is None:
            continue
        if (
            previous_iteration is not None
            and iteration < previous_iteration
        ):
            resets.append(
                {
                    "target_offset": target_offset,
                    "previous_iteration_idx": previous_iteration,
                    "next_iteration_idx": iteration,
                    "line_number": item["line_number"],
                }
            )
        previous_iteration = iteration

    result["iteration_resets"] = resets
    if resets:
        boundary = resets[-1]["target_offset"]
        removed = target[:boundary]
        result.update(
            {
                "strategy": "last_iteration_reset",
                "remove_line_numbers": [
                    item["line_number"] for item in removed
                ],
                "removed_rows": len(removed),
                "remaining_target_rows": len(target) - len(removed),
                "kept_attempt_first_line": target[boundary]["line_number"],
            }
        )
        return result

    if all(item["iteration_idx"] is not None for item in target):
        result["strategy"] = "single_monotonic_attempt"
        return result

    raw_rows = [item["raw"] for item in target]
    if len(set(raw_rows)) == 1:
        removed = target[:-1]
        result.update(
            {
                "strategy": "identical_query_row_deduplication",
                "remove_line_numbers": [
                    item["line_number"] for item in removed
                ],
                "removed_rows": len(removed),
                "remaining_target_rows": 1,
                "kept_attempt_first_line": target[-1]["line_number"],
            }
        )
        return result

    result.update(
        {
            "strategy": "ambiguous_repeated_rows",
            "ambiguous": True,
        }
    )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_file(
    path: Path,
    *,
    remove_line_numbers: Sequence[int],
    backup_dir: Path,
) -> Dict[str, Any]:
    remove = set(int(value) for value in remove_line_numbers)
    if not remove:
        raise ValueError(f"no rows selected for removal from {path}")

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / path.name
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite backup: {backup}")
    shutil.copy2(path, backup)
    source_sha256 = _sha256(backup)

    temporary_handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".retry-clean.tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(temporary_handle.name)
    total_rows = 0
    removed_rows = 0
    try:
        with path.open("rb") as source, temporary_handle as output:
            for line_number, line in enumerate(source, start=1):
                total_rows += 1
                if line_number in remove:
                    removed_rows += 1
                    continue
                output.write(line)
            output.flush()
            os.fsync(output.fileno())
        if removed_rows != len(remove):
            raise RuntimeError(
                f"planned {len(remove)} removals from {path}, "
                f"but found {removed_rows}"
            )
        os.replace(temporary, path)
        shutil.copystat(backup, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "total_rows": total_rows,
        "removed_rows": removed_rows,
        "remaining_rows": total_rows - removed_rows,
        "backup": str(backup),
        "backup_sha256": source_sha256,
        "rewritten_sha256": _sha256(path),
    }


def clean(
    run: str | Path,
    *,
    query_id: str,
    apply: bool,
    backup_dir: str | Path | None = None,
) -> Dict[str, Any]:
    run_dir, artifacts = artifact_dir(run)
    if not query_id.strip():
        raise ValueError("query_id must be non-empty")

    files = sorted(artifacts.glob("*.jsonl"))
    plans = {path.name: plan_file(path, query_id) for path in files}
    affected = [
        path
        for path in files
        if plans[path.name]["removed_rows"] > 0
    ]
    ambiguous = [
        path.name
        for path in files
        if plans[path.name]["ambiguous"]
    ]
    if apply and ambiguous:
        raise ValueError(
            "refusing to modify artifacts with ambiguous repeated query rows: "
            + ", ".join(ambiguous)
        )

    if backup_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_query_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in query_id
        )
        backup = run_dir / "retry_backups" / f"{stamp}-{safe_query_id}"
    else:
        backup = Path(backup_dir).expanduser().resolve()

    rewrites: Dict[str, Dict[str, Any]] = {}
    if apply:
        if backup.exists() and any(backup.iterdir()):
            raise FileExistsError(
                f"refusing to use non-empty backup directory: {backup}"
            )
        for path in affected:
            rewrites[path.name] = rewrite_file(
                path,
                remove_line_numbers=plans[path.name][
                    "remove_line_numbers"
                ],
                backup_dir=backup,
            )
        backup.mkdir(parents=True, exist_ok=True)
        summary = {
            "run_dir": str(run_dir),
            "query_id": query_id,
            "plans": plans,
            "rewrites": rewrites,
        }
        (backup / "cleanup_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return {
        "run_dir": str(run_dir),
        "query_id": query_id,
        "dry_run": not apply,
        "affected_files": [path.name for path in affected],
        "ambiguous_files": ambiguous,
        "backup_dir": str(backup) if apply else None,
        "plans": plans,
        "rewrites": rewrites,
    }


def compact_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Make CLI output readable when a paper-level file has many removals."""
    output = {
        key: value
        for key, value in result.items()
        if key != "plans"
    }
    compact_plans: Dict[str, Dict[str, Any]] = {}
    for name, source_plan in result["plans"].items():
        plan = {
            key: value
            for key, value in source_plan.items()
            if key != "remove_line_numbers"
        }
        line_numbers = source_plan["remove_line_numbers"]
        plan["remove_line_number_count"] = len(line_numbers)
        plan["remove_line_number_range"] = (
            [min(line_numbers), max(line_numbers)]
            if line_numbers
            else None
        )
        compact_plans[name] = plan
    output["plans"] = compact_plans
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--query_id", required=True)
    parser.add_argument("--backup_dir")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = clean(
        args.run,
        query_id=args.query_id,
        apply=args.apply,
        backup_dir=args.backup_dir,
    )
    print(
        json.dumps(
            compact_result(result),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
