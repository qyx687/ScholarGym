#!/usr/bin/env python3
"""Build a resumable Qwen paper-type cache for unique OnePass candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from paper_type import (  # noqa: E402
    CLASSIFIER_VERSION,
    PaperTypeClassifier,
    load_paper_type_cache,
    normalize_paper_id,
)
from runtime_env import load_env_file  # noqa: E402


def iter_pool_candidate_ids(
    path: Path,
    allowed_query_ids: Set[str] | None = None,
) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid pool JSONL at line {line_number}: {exc}") from exc
            query_id = str(record.get("query_id") or "")
            if allowed_query_ids is not None and query_id not in allowed_query_ids:
                continue
            rows = record.get("local_pool_rows")
            if not isinstance(rows, list):
                raise ValueError(f"pool line {line_number} has no local_pool_rows array")
            for row in rows:
                paper_id = normalize_paper_id(
                    row.get("paper_arxiv_id") if isinstance(row, Mapping) else None
                )
                if paper_id:
                    yield paper_id


def load_type_rule_query_ids(path: Path) -> Set[str]:
    output: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid query policy JSONL at line {line_number}: {exc}"
                ) from exc
            policy = record.get("validated_policy") or {}
            if policy.get("paper_type_rules"):
                query_id = str(record.get("query_id") or "")
                if query_id:
                    output.add(query_id)
    return output


class StreamingJSONObject:
    """Incrementally decode a large top-level JSON object without extra deps."""

    def __init__(self, path: Path, chunk_size: int = 1024 * 1024) -> None:
        self.path = path
        self.chunk_size = max(65536, int(chunk_size))
        self.decoder = json.JSONDecoder()

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        with self.path.open("r", encoding="utf-8") as handle:
            buffer = ""
            position = 0
            eof = False

            def read_more() -> bool:
                nonlocal buffer, position, eof
                if eof:
                    return False
                chunk = handle.read(self.chunk_size)
                if not chunk:
                    eof = True
                    return False
                if position:
                    buffer = buffer[position:] + chunk
                    position = 0
                else:
                    buffer += chunk
                return True

            def ensure_character() -> bool:
                while position >= len(buffer):
                    if not read_more():
                        return False
                return True

            def skip_space() -> bool:
                nonlocal position
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer):
                        return True
                    if not read_more():
                        return False

            def raw_decode() -> Tuple[Any, int]:
                nonlocal buffer, position
                while True:
                    try:
                        return self.decoder.raw_decode(buffer, position)
                    except json.JSONDecodeError:
                        if not read_more():
                            raise

            if not ensure_character() or not skip_space() or buffer[position] != "{":
                raise ValueError(f"paper DB must be a top-level JSON object: {self.path}")
            position += 1
            while True:
                if not skip_space():
                    raise ValueError("unexpected EOF in paper DB")
                if buffer[position] == "}":
                    return
                if buffer[position] == ",":
                    position += 1
                    if not skip_space():
                        raise ValueError("unexpected EOF after paper DB comma")
                key, end = raw_decode()
                if not isinstance(key, str):
                    raise ValueError("paper DB keys must be strings")
                position = end
                if not skip_space() or buffer[position] != ":":
                    raise ValueError(f"missing colon after paper DB key {key!r}")
                position += 1
                if not skip_space():
                    raise ValueError("unexpected EOF before paper DB value")
                value, end = raw_decode()
                position = end
                yield key, value
                if position > self.chunk_size * 2:
                    buffer = buffer[position:]
                    position = 0


def _append_records(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def compact_cache(path: Path) -> int:
    """Atomically rewrite a cache to one validated record per paper ID.

    The append-only format is crash-safe. Compaction also makes recovery from
    an accidentally overlapping resume deterministic: the last valid record
    for an ID wins, matching ``load_paper_type_cache`` semantics.
    """

    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        line_count = sum(1 for line in handle if line.strip())
    cache = load_paper_type_cache(path)
    tmp = path.with_suffix(path.suffix + ".compact.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for paper_id in sorted(cache):
            record = dict(cache[paper_id])
            record["paper_arxiv_id"] = paper_id
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return max(0, line_count - len(cache))


def _append_failure(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _classify_resilient(
    classifier: PaperTypeClassifier,
    papers: Sequence[Mapping[str, Any]],
    failure_path: Path,
) -> List[Dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            return classifier.classify_batch(papers)
        except Exception as exc:  # API and schema failures are both retryable here.
            last_error = exc
            if attempt == 0:
                time.sleep(2)
    if len(papers) > 1:
        middle = len(papers) // 2
        return _classify_resilient(classifier, papers[:middle], failure_path) + _classify_resilient(
            classifier, papers[middle:], failure_path
        )
    paper_id = normalize_paper_id(papers[0].get("paper_arxiv_id"))
    _append_failure(
        failure_path,
        {
            "paper_arxiv_id": paper_id,
            "error": f"{type(last_error).__name__}: {last_error}",
            "classifier_version": classifier.classifier_version,
        },
    )
    return []


def build_cache(
    pool_records_path: Path,
    paper_db_path: Path,
    output_path: Path,
    *,
    model: str,
    batch_size: int = 16,
    resume: bool = False,
    is_local: bool = False,
    max_papers: int | None = None,
    query_policies_path: Path | None = None,
    exclude_cache_path: Path | None = None,
    shard_count: int = 1,
    shard_index: int = 0,
    llm_call: Any = None,
) -> Dict[str, Any]:
    pool_records_path = pool_records_path.resolve()
    paper_db_path = paper_db_path.resolve()
    output_path = output_path.resolve()
    if not pool_records_path.exists():
        raise FileNotFoundError(pool_records_path)
    if not paper_db_path.exists():
        raise FileNotFoundError(paper_db_path)
    if output_path.exists() and not resume:
        raise FileExistsError(
            f"{output_path} already exists; pass --resume to append missing candidates"
        )
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
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
    all_candidate_ids = sorted(
        set(iter_pool_candidate_ids(pool_records_path, allowed_query_ids))
    )
    candidate_ids = set(all_candidate_ids[shard_index::shard_count])
    pending = sorted(candidate_ids - set(existing) - set(excluded))
    if max_papers is not None:
        pending = pending[: max(0, int(max_papers))]
    pending_set = set(pending)
    classifier = PaperTypeClassifier(
        model,
        is_local=is_local,
        llm_call=llm_call,
        classifier_version=CLASSIFIER_VERSION,
    )
    failure_path = output_path.with_suffix(output_path.suffix + ".failures.jsonl")
    found_ids: Set[str] = set()
    classified_count = 0
    batch: List[Dict[str, Any]] = []
    started = time.time()

    def flush() -> None:
        nonlocal classified_count, batch
        if not batch:
            return
        records = _classify_resilient(classifier, batch, failure_path)
        classified_count += _append_records(output_path, records)
        batch = []
        if classified_count and classified_count % max(batch_size * 10, 1) == 0:
            elapsed = time.time() - started
            print(
                f"classified={classified_count}/{len(pending)} "
                f"elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    if pending_set:
        for raw_key, raw_value in StreamingJSONObject(paper_db_path):
            if not pending_set:
                break
            paper_id = normalize_paper_id(
                ((raw_value or {}).get("arxiv_id") or raw_key)
                if isinstance(raw_value, Mapping)
                else raw_key
            )
            if paper_id not in pending_set:
                continue
            pending_set.remove(paper_id)
            found_ids.add(paper_id)
            metadata = raw_value if isinstance(raw_value, Mapping) else {}
            batch.append(
                {
                    "paper_arxiv_id": paper_id,
                    "title": str(metadata.get("title") or ""),
                    "abstract": str(metadata.get("abstract") or ""),
                }
            )
            if len(batch) >= max(1, int(batch_size)):
                flush()
        flush()

    missing_metadata_ids = sorted(pending_set)
    if missing_metadata_ids:
        _append_failure(
            failure_path,
            {
                "error": "candidate IDs missing from paper DB",
                "paper_arxiv_ids": missing_metadata_ids,
            },
        )
    duplicate_records_removed = compact_cache(output_path)
    final_cache = load_paper_type_cache(output_path) if output_path.exists() else {}
    summary = {
        "pool_records": str(pool_records_path),
        "query_policies": str(query_policies_path) if query_policies_path else None,
        "exclude_cache": str(exclude_cache_path) if exclude_cache_path else None,
        "exclude_cache_count": len(excluded),
        "shard_count": shard_count,
        "shard_index": shard_index,
        "type_rule_query_count": (
            len(allowed_query_ids) if allowed_query_ids is not None else None
        ),
        "paper_db": str(paper_db_path),
        "output": str(output_path),
        "model": model,
        "classifier_version": CLASSIFIER_VERSION,
        "all_unique_candidate_count": len(all_candidate_ids),
        "unique_candidate_count": len(candidate_ids),
        "existing_cache_count": len(existing),
        "requested_pending_count": len(pending),
        "metadata_found_count": len(found_ids),
        "classified_and_written_count": classified_count,
        "duplicate_records_removed": duplicate_records_removed,
        "missing_metadata_count": len(missing_metadata_ids),
        "final_cache_count": len(final_cache),
        "remaining_candidate_count": len(
            candidate_ids - set(final_cache) - set(excluded)
        ),
        "elapsed_seconds": time.time() - started,
        "failure_log": str(failure_path),
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
    parser.add_argument("--paper_db", type=Path, required=True)
    parser.add_argument(
        "--query_policies",
        type=Path,
        default=None,
        help="Optionally classify only pools for queries with non-empty type rules.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument(
        "--env_file",
        type=Path,
        default=None,
        help="Optional local KEY=value file loaded without logging secret values.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--is_local", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_papers", type=int, default=None)
    parser.add_argument(
        "--exclude_cache",
        type=Path,
        default=None,
        help="Skip IDs already present in this validated cache (useful for shards).",
    )
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    summary = build_cache(
        args.pool_records,
        args.paper_db,
        args.output,
        model=args.model,
        batch_size=args.batch_size,
        resume=args.resume,
        is_local=args.is_local,
        max_papers=args.max_papers,
        query_policies_path=args.query_policies,
        exclude_cache_path=args.exclude_cache,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
