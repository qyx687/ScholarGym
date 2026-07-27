#!/usr/bin/env python3
"""Clone a SemRank cache only when every vector is in the Qwen namespace."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from graph_methods import EmbeddingProvider  # noqa: E402
from semrank import EmbeddingProviderConceptEncoder  # noqa: E402
from semrank.cache import SCHEMA_VERSION  # noqa: E402


def counts(connection: sqlite3.Connection) -> Dict[str, int]:
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


def namespaces(connection: sqlite3.Connection) -> Dict[str, int]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_cache", required=True)
    parser.add_argument("--target_cache", required=True)
    parser.add_argument("--report_json", required=True)
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
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    args = parser.parse_args()

    source_path = Path(args.source_cache).expanduser().resolve()
    target_path = Path(args.target_cache).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target_path.exists():
        raise FileExistsError(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    provider = EmbeddingProvider(
        backend=args.embedding_backend,
        model=args.embedding_model,
        base_url=args.embedding_base_url,
        batch_size=args.embedding_batch_size,
    )
    expected_encoder = EmbeddingProviderConceptEncoder(provider).encoder_id
    source = sqlite3.connect(
        f"file:{source_path.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        source_schema = source.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if not source_schema or str(source_schema[0]) != SCHEMA_VERSION:
            raise ValueError(
                f"source schema must be {SCHEMA_VERSION!r}, got "
                f"{source_schema!r}"
            )
        source_counts = counts(source)
        source_namespaces = namespaces(source)
        unexpected = {
            key: value
            for key, value in source_namespaces.items()
            if key != expected_encoder
        }
        if unexpected:
            raise AssertionError(
                "refusing cache clone with a non-Qwen vector namespace: "
                f"{unexpected}"
            )
        target = sqlite3.connect(str(target_path))
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()

    verification = sqlite3.connect(str(target_path))
    try:
        target_counts = counts(verification)
        target_namespaces = namespaces(verification)
    finally:
        verification.close()
    if target_counts != source_counts:
        raise AssertionError(
            f"cache clone count mismatch: {target_counts} != {source_counts}"
        )
    if target_namespaces != source_namespaces:
        raise AssertionError(
            "cache clone vector namespace mismatch: "
            f"{target_namespaces} != {source_namespaces}"
        )

    report = {
        "operation": "clone_qwen_only_semrank_cache_v1",
        "source_cache": str(source_path),
        "target_cache": str(target_path),
        "schema_version": SCHEMA_VERSION,
        "expected_qwen_encoder_id": expected_encoder,
        "source_counts": source_counts,
        "target_counts": target_counts,
        "source_vector_namespaces": source_namespaces,
        "target_vector_namespaces": target_namespaces,
        "non_qwen_vector_rows": 0,
        "classifier_calls": 0,
        "llm_calls": 0,
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
