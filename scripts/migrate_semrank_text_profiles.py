#!/usr/bin/env python3
"""Copy SemRank text profiles without copying concept-vector rows."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from semrank import (  # noqa: E402
    PaperConceptProfile,
    QueryConceptProfile,
    SemRankCache,
    paper_text_profile_identity,
    query_text_profile_identity,
    stable_hash,
)


VECTOR_BINDING_AT_USE = "vector_encoder_bound_at_use_v1"


def readonly_connection(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro",
        uri=True,
    )


def source_summary(path: Path) -> Dict[str, Any]:
    with readonly_connection(path) as connection:
        schema = connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        counts = {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table}"  # noqa: S608
                ).fetchone()[0]
            )
            for table in (
                "paper_profiles",
                "query_profiles",
                "concept_embeddings",
            )
        }
        namespaces = [
            {"encoder_id": str(encoder_id), "rows": int(rows)}
            for encoder_id, rows in connection.execute(
                """
                SELECT encoder_id, COUNT(*)
                FROM concept_embeddings
                GROUP BY encoder_id
                ORDER BY encoder_id
                """
            )
        ]
    return {
        "path": str(path),
        "schema_version": str(schema[0]) if schema else None,
        "counts": counts,
        "vector_namespaces_not_copied": namespaces,
    }


def profile_rows(
    path: Path,
    table: str,
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if table not in {"paper_profiles", "query_profiles"}:
        raise ValueError(f"unsupported profile table: {table}")
    with readonly_connection(path) as connection:
        cursor = connection.execute(
            f"""  -- noqa: S608
            SELECT identity_json, profile_json
            FROM {table}
            ORDER BY created_at, cache_key
            """
        )
        for identity_json, profile_json in cursor:
            yield json.loads(identity_json), json.loads(profile_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_cache",
        action="append",
        required=True,
        help="Archived SemRank SQLite cache; may be repeated.",
    )
    parser.add_argument("--target_cache", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument(
        "--allow_existing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--collision_policy",
        choices=["replace-later", "keep-target"],
        default="replace-later",
        help=(
            "Whether a source text profile replaces an existing target key."
        ),
    )
    args = parser.parse_args()

    sources = [
        Path(value).expanduser().resolve() for value in args.source_cache
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    target = Path(args.target_cache).expanduser().resolve()
    if target.exists() and not args.allow_existing:
        raise FileExistsError(
            f"target already exists; refusing implicit merge: {target}"
        )

    migration_counts: Counter[str] = Counter()
    paper_statuses: Counter[str] = Counter()
    seen_paper_keys = set()
    seen_query_keys = set()
    with SemRankCache(target) as cache:
        initial_target_counts = cache.table_counts()
        for source in sources:
            for old_identity, raw_profile in profile_rows(
                source,
                "paper_profiles",
            ):
                identity = paper_text_profile_identity(old_identity)
                cache_key = stable_hash(identity)
                profile = PaperConceptProfile.from_dict(raw_profile)
                migrated = replace(
                    profile,
                    profile_id=cache_key,
                    concept_encoder=VECTOR_BINDING_AT_USE,
                    cache_hit=False,
                )
                migration_counts["paper_rows_read"] += 1
                paper_statuses[migrated.status] += 1
                if cache_key in seen_paper_keys:
                    migration_counts["paper_key_collisions"] += 1
                seen_paper_keys.add(cache_key)
                if (
                    args.collision_policy == "keep-target"
                    and cache.get_paper_profile(cache_key) is not None
                ):
                    migration_counts["paper_existing_kept"] += 1
                    continue
                cache.put_paper_profile(cache_key, identity, migrated)
                migration_counts["paper_rows_written"] += 1

            for old_identity, raw_profile in profile_rows(
                source,
                "query_profiles",
            ):
                identity = query_text_profile_identity(old_identity)
                cache_key = stable_hash(identity)
                profile = QueryConceptProfile.from_dict(raw_profile)
                migrated = replace(
                    profile,
                    query_profile_id=cache_key,
                    concept_encoder=VECTOR_BINDING_AT_USE,
                    cache_hit=False,
                )
                migration_counts["query_rows_read"] += 1
                if cache_key in seen_query_keys:
                    migration_counts["query_key_collisions"] += 1
                seen_query_keys.add(cache_key)
                if (
                    args.collision_policy == "keep-target"
                    and cache.get_query_profile(cache_key) is not None
                ):
                    migration_counts["query_existing_kept"] += 1
                    continue
                cache.put_query_profile(cache_key, identity, migrated)
                migration_counts["query_rows_written"] += 1

        target_counts = cache.table_counts()
        target_status_counts = cache.paper_profile_status_counts()
        if (
            target_counts["concept_embeddings"]
            != initial_target_counts["concept_embeddings"]
        ):
            raise AssertionError(
                "text-profile migration changed concept-embedding row count"
            )

    report = {
        "migration": "semrank_text_profiles_only_v1",
        "sources": [source_summary(source) for source in sources],
        "target": str(target),
        "initial_target_counts": initial_target_counts,
        "target_counts": target_counts,
        "target_paper_status_counts": target_status_counts,
        "unique_paper_profile_keys": len(seen_paper_keys),
        "unique_query_profile_keys": len(seen_query_keys),
        "paper_status_counts_from_source_rows": dict(
            sorted(paper_statuses.items())
        ),
        "migration_counts": dict(sorted(migration_counts.items())),
        "concept_vectors_copied": 0,
        "classifier_calls": 0,
        "llm_calls": 0,
        "collision_policy": args.collision_policy,
    }
    report_path = Path(args.report_json).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
