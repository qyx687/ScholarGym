#!/usr/bin/env python3
"""Derive classifier-only paper profiles from cached classifier candidates.

The source cache is read-only.  Full-SemRank selected topics, keyphrases, raw
LLM output, and query profiles are intentionally ignored.  The target contains
only classifier candidate topics as paper concepts and reuses Qwen vectors
when the exact encoder-bound cache key already exists.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from semrank import (  # noqa: E402
    PaperConceptProfile,
    SEMRANK_CLASSIFIER_ONLY_PAPER_PROMPT_VERSION,
    SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION,
    SemRankCache,
    stable_hash,
    stable_unique,
)
from semrank.cache import SCHEMA_VERSION  # noqa: E402
from semrank.models import canonical_json  # noqa: E402


def batched(values: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def table_counts(connection: sqlite3.Connection) -> Dict[str, int]:
    return {
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


def namespace_counts(connection: sqlite3.Connection) -> Dict[str, int]:
    return {
        str(encoder_id): int(count)
        for encoder_id, count in connection.execute(
            """
            SELECT encoder_id, COUNT(*)
            FROM concept_embeddings
            GROUP BY encoder_id
            ORDER BY encoder_id
            """
        )
    }


def classifier_only_identity(
    source_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    identity = dict(source_identity)
    identity["paper_prompt_version"] = (
        SEMRANK_CLASSIFIER_ONLY_PAPER_PROMPT_VERSION
    )
    identity["extraction_llm"] = "none"
    identity["pipeline_version"] = (
        SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION
    )
    return identity


def candidate_topic_concepts(
    candidates: Iterable[Mapping[str, Any]],
) -> List[str]:
    return stable_unique(item.get("concept") for item in candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_cache", required=True)
    parser.add_argument("--target_cache", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--write_batch_size", type=int, default=1000)
    args = parser.parse_args()

    source_path = Path(args.source_cache).expanduser().resolve()
    target_path = Path(args.target_cache).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target_path.exists():
        raise FileExistsError(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Create the target with the production cache schema.
    with SemRankCache(target_path):
        pass

    source = sqlite3.connect(
        f"file:{source_path.as_posix()}?mode=ro",
        uri=True,
    )
    target = sqlite3.connect(str(target_path))
    try:
        source_schema = source.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if not source_schema or str(source_schema[0]) != SCHEMA_VERSION:
            raise ValueError(
                f"source schema must be {SCHEMA_VERSION!r}, got "
                f"{source_schema!r}"
            )
        source_namespaces = namespace_counts(source)
        if len(source_namespaces) != 1:
            raise AssertionError(
                "source cache must contain exactly one active vector "
                f"namespace, got {source_namespaces}"
            )
        encoder_id = next(iter(source_namespaces))
        if "model=qwen3-embedding:0.6b" not in encoder_id:
            raise AssertionError(
                "source vector namespace is not Qwen3-Embedding-0.6B: "
                f"{encoder_id}"
            )

        profile_rows = []
        unique_topics = set()
        candidate_count_distribution: Counter[int] = Counter()
        empty_profiles = 0
        full_llm_payloads_ignored = 0
        created_at = time.time()
        cursor = source.execute(
            "SELECT identity_json, profile_json FROM paper_profiles"
        )
        for source_identity_json, source_profile_json in cursor:
            source_identity = json.loads(source_identity_json)
            source_profile = json.loads(source_profile_json)
            candidates = [
                dict(item)
                for item in source_profile.get("candidate_topics") or []
            ]
            topics = candidate_topic_concepts(candidates)
            candidate_count_distribution[len(topics)] += 1
            empty_profiles += int(not topics)
            full_llm_payloads_ignored += int(
                bool(source_profile.get("selected_topics"))
                or bool(source_profile.get("keyphrases"))
                or bool(source_profile.get("raw_llm_output"))
            )
            unique_topics.update(topics)
            identity = classifier_only_identity(source_identity)
            cache_key = stable_hash(identity)
            profile = PaperConceptProfile(
                paper_arxiv_id=str(
                    source_profile.get("paper_arxiv_id") or ""
                ),
                profile_id=cache_key,
                title_abstract_hash=str(
                    source_profile.get("title_abstract_hash") or ""
                ),
                candidate_topics=candidates,
                selected_topics=topics,
                keyphrases=[],
                concepts=topics,
                status="ok" if topics else "empty",
                fallback_reason=(
                    None if topics else "classifier_topics_empty"
                ),
                pipeline_version=(
                    SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION
                ),
                classifier_id=str(
                    source_profile.get("classifier_id") or ""
                ),
                label_space_id=str(
                    source_profile.get("label_space_id") or ""
                ),
                llm_model="none",
                prompt_version=(
                    SEMRANK_CLASSIFIER_ONLY_PAPER_PROMPT_VERSION
                ),
                concept_encoder="vector_encoder_bound_at_use_v1",
                cache_hit=False,
                raw_llm_output=None,
            )
            profile_rows.append(
                (
                    cache_key,
                    profile.paper_arxiv_id,
                    canonical_json(identity),
                    canonical_json(profile.to_dict(include_raw=True)),
                    created_at,
                )
            )

        for chunk in batched(
            profile_rows,
            max(1, args.write_batch_size),
        ):
            with target:
                target.executemany(
                    """
                    INSERT INTO paper_profiles(
                        cache_key, paper_arxiv_id, identity_json,
                        profile_json, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    chunk,
                )

        vector_rows = []
        copied_topics = set()
        for concept in sorted(unique_topics):
            cache_key = stable_hash(
                {
                    "encoder_id": encoder_id,
                    "concept": concept,
                    "normalization": "nfkc_lower_space_v1",
                }
            )
            row = source.execute(
                """
                SELECT cache_key, encoder_id, concept_normalized, dimension,
                       vector_f32, created_at
                FROM concept_embeddings
                WHERE cache_key=?
                """,
                (cache_key,),
            ).fetchone()
            if row is not None:
                vector_rows.append(row)
                copied_topics.add(concept)
        for chunk in batched(
            vector_rows,
            max(1, args.write_batch_size),
        ):
            with target:
                target.executemany(
                    """
                    INSERT INTO concept_embeddings(
                        cache_key, encoder_id, concept_normalized, dimension,
                        vector_f32, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    chunk,
                )

        target_counts = table_counts(target)
        target_namespaces = namespace_counts(target)
        missing_topics = sorted(unique_topics - copied_topics)
        if target_counts["paper_profiles"] != len(profile_rows):
            raise AssertionError("classifier-only profile count mismatch")
        if target_counts["query_profiles"] != 0:
            raise AssertionError(
                "full query profiles must not enter classifier-only cache"
            )
        if any(
            json.loads(payload).get("keyphrases")
            for (payload,) in target.execute(
                "SELECT profile_json FROM paper_profiles"
            )
        ):
            raise AssertionError(
                "classifier-only cache contains paper keyphrases"
            )

        report = {
            "operation": "derive_semrank_classifier_only_cache_v1",
            "source_cache": str(source_path),
            "target_cache": str(target_path),
            "schema_version": SCHEMA_VERSION,
            "paper_concept_mode": "classifier_only",
            "paper_concept_contract": "all_classifier_candidate_topics",
            "classifier_topic_k": max(
                candidate_count_distribution,
                default=0,
            ),
            "source_counts": table_counts(source),
            "target_counts": target_counts,
            "source_vector_namespaces": source_namespaces,
            "target_vector_namespaces": target_namespaces,
            "derived_paper_profiles": len(profile_rows),
            "empty_paper_profiles": empty_profiles,
            "candidate_topic_count_distribution": {
                str(key): int(value)
                for key, value in sorted(
                    candidate_count_distribution.items()
                )
            },
            "unique_classifier_topics": len(unique_topics),
            "qwen_vectors_copied": len(copied_topics),
            "qwen_vectors_missing_for_reencode": len(missing_topics),
            "missing_topic_preview": missing_topics[:25],
            "full_llm_payloads_ignored": full_llm_payloads_ignored,
            "paper_level_llm_calls": 0,
            "query_level_llm_calls": 0,
            "classifier_calls": 0,
            "query_profiles_copied": 0,
            "paper_keyphrases_in_target": 0,
            "paper_raw_llm_outputs_in_target": 0,
        }
    finally:
        target.close()
        source.close()

    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
