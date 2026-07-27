#!/usr/bin/env python3
"""Encode cached SemRank concept strings without classifier or LLM calls."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from graph_methods import EmbeddingProvider  # noqa: E402
from semrank import (  # noqa: E402
    CachedConceptEncoder,
    EmbeddingProviderConceptEncoder,
    SemRankCache,
    stable_unique,
)


def profile_payloads(
    cache_path: Path,
    table: str,
) -> Iterator[Dict[str, Any]]:
    if table not in {"paper_profiles", "query_profiles"}:
        raise ValueError(f"unsupported profile table: {table}")
    connection = sqlite3.connect(
        f"file:{cache_path.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        cursor = connection.execute(
            f"SELECT profile_json FROM {table}"  # noqa: S608
        )
        for (payload,) in cursor:
            yield json.loads(payload)
    finally:
        connection.close()


def collect_text_concepts(cache_path: Path) -> tuple[List[str], Dict[str, int]]:
    concepts: Set[str] = set()
    counts = {
        "paper_profiles": 0,
        "query_profiles": 0,
        "paper_union_mismatches": 0,
        "selected_paper_topics": 0,
        "paper_keyphrases": 0,
        "selected_query_concepts": 0,
    }
    for profile in profile_payloads(cache_path, "paper_profiles"):
        counts["paper_profiles"] += 1
        selected_topics = stable_unique(profile.get("selected_topics") or [])
        keyphrases = stable_unique(profile.get("keyphrases") or [])
        declared = stable_unique(profile.get("concepts") or [])
        combined = stable_unique(selected_topics + keyphrases)
        if declared != combined:
            counts["paper_union_mismatches"] += 1
        concepts.update(combined)
        counts["selected_paper_topics"] += len(selected_topics)
        counts["paper_keyphrases"] += len(keyphrases)
    for profile in profile_payloads(cache_path, "query_profiles"):
        counts["query_profiles"] += 1
        selected = stable_unique(profile.get("selected_concepts") or [])
        concepts.update(selected)
        counts["selected_query_concepts"] += len(selected)
    if counts["paper_union_mismatches"]:
        raise AssertionError(
            "cached paper concepts are not exactly selected_topics union "
            "keyphrases"
        )
    return sorted(concepts), counts


def namespace_counts(cache_path: Path) -> Dict[str, int]:
    connection = sqlite3.connect(str(cache_path))
    try:
        return {
            str(encoder_id): int(rows)
            for encoder_id, rows in connection.execute(
                """
                SELECT encoder_id, COUNT(*)
                FROM concept_embeddings
                GROUP BY encoder_id
                ORDER BY encoder_id
                """
            )
        }
    finally:
        connection.close()


def batched(values: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", required=True)
    parser.add_argument(
        "--embedding_backend",
        choices=["ollama", "api"],
        default="ollama",
    )
    parser.add_argument(
        "--embedding_model",
        default="qwen3-embedding:0.6b",
    )
    parser.add_argument(
        "--embedding_base_url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--embedding_api_key_env",
        default="EMBEDDING_API_KEY",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--concept_chunk_size", type=int, default=4096)
    parser.add_argument("--report_json", required=True)
    parser.add_argument(
        "--require_exclusive_namespace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    cache_path = Path(args.cache_path).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    concepts, text_counts = collect_text_concepts(cache_path)
    provider = EmbeddingProvider(
        backend=args.embedding_backend,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        api_key=os.environ.get(args.embedding_api_key_env, ""),
        batch_size=args.embedding_batch_size,
    )
    backend = EmbeddingProviderConceptEncoder(provider)
    with SemRankCache(cache_path) as cache:
        encoder = CachedConceptEncoder(backend, cache)
        for chunk in batched(concepts, max(1, args.concept_chunk_size)):
            encoder.encode(chunk)
            # The persistent concept cache is authoritative. Keeping every
            # unique vector in the provider's process-local cache would
            # duplicate several GiB on a full benchmark warm-up.
            provider.clear_memory_cache()
        encoder_stats = encoder.snapshot_stats()
        cache_counts = cache.table_counts()
        cache_stats = cache.snapshot_stats()

    namespaces = namespace_counts(cache_path)
    current_rows = namespaces.get(backend.encoder_id, 0)
    if current_rows != len(concepts):
        raise AssertionError(
            "Qwen concept namespace does not contain exactly one vector per "
            f"selected concept: {current_rows} != {len(concepts)}"
        )
    other_namespaces = {
        key: value
        for key, value in namespaces.items()
        if key != backend.encoder_id
    }
    if args.require_exclusive_namespace and other_namespaces:
        raise AssertionError(
            "active cache contains a non-Qwen concept-vector namespace: "
            f"{other_namespaces}"
        )

    report = {
        "operation": "reencode_cached_semrank_concept_strings_v1",
        "cache_path": str(cache_path),
        "input_contract": [
            "selected_query_concepts",
            "selected_paper_topics",
            "extracted_paper_keyphrases",
        ],
        "whole_paper_title_abstract_embeddings": 0,
        "classifier_calls": 0,
        "llm_calls": 0,
        "unique_normalized_concepts": len(concepts),
        "text_profile_counts": text_counts,
        "encoder_id": backend.encoder_id,
        "encoder_stats": encoder_stats,
        "provider_stats": provider.snapshot_stats(),
        "cache_stats": cache_stats,
        "cache_counts": cache_counts,
        "vector_namespace_counts": namespaces,
        "other_vector_namespaces": other_namespaces,
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
