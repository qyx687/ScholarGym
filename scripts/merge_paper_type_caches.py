#!/usr/bin/env python3
"""Validate and atomically merge query-independent paper-type JSONL caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from paper_type import load_paper_type_cache  # noqa: E402


def merge_caches(inputs: list[Path], output: Path) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    merged = {}
    input_counts = {}
    for path in inputs:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        cache = load_paper_type_cache(resolved)
        input_counts[str(resolved)] = len(cache)
        merged.update(cache)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for paper_id in sorted(merged):
            record = dict(merged[paper_id])
            record["paper_arxiv_id"] = paper_id
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, output)
    return {
        "inputs": input_counts,
        "input_record_total": sum(input_counts.values()),
        "merged_unique_record_count": len(merged),
        "duplicate_id_count": sum(input_counts.values()) - len(merged),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge_caches(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
