#!/usr/bin/env python3
"""Read-only scientific audit for a SemRank classifier-only SQLite cache."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from semrank import (  # noqa: E402
    SEMRANK_CLASSIFIER_ONLY_PAPER_PROMPT_VERSION,
    SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION,
    SEMRANK_QUERY_PROMPT_VERSION,
    normalize_concept,
    stable_hash,
    stable_unique,
)
from semrank.cache import SCHEMA_VERSION  # noqa: E402


def _count(
    connection: sqlite3.Connection,
    table: str,
) -> int:
    if table not in {
        "paper_profiles",
        "query_profiles",
        "concept_embeddings",
    }:
        raise ValueError(f"unsupported table: {table}")
    return int(
        connection.execute(
            f"SELECT COUNT(*) FROM {table}"  # noqa: S608
        ).fetchone()[0]
    )


def _increment_if(
    violations: Counter[str],
    condition: bool,
    key: str,
) -> None:
    if condition:
        violations[key] += 1


def audit_cache(cache_path: str | Path) -> Dict[str, Any]:
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        metadata = {
            str(key): str(value)
            for key, value in connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            )
        }
        table_counts = {
            table: _count(connection, table)
            for table in (
                "paper_profiles",
                "query_profiles",
                "concept_embeddings",
            )
        }

        paper_violations: Counter[str] = Counter()
        paper_statuses: Counter[str] = Counter()
        paper_concept_counts: Counter[int] = Counter()
        referenced_concepts: set[str] = set()
        paper_ids: set[str] = set()
        for (
            cache_key,
            paper_arxiv_id,
            identity_json,
            profile_json,
        ) in connection.execute(
            """
            SELECT cache_key, paper_arxiv_id, identity_json, profile_json
            FROM paper_profiles
            """
        ):
            identity = json.loads(identity_json)
            profile = json.loads(profile_json)
            candidates = [
                dict(item)
                for item in profile.get("candidate_topics") or []
            ]
            expected_topics = stable_unique(
                item.get("concept") for item in candidates
            )
            selected_topics = list(profile.get("selected_topics") or [])
            concepts = list(profile.get("concepts") or [])
            paper_ids.add(str(paper_arxiv_id))
            referenced_concepts.update(concepts)
            paper_statuses[str(profile.get("status") or "unknown")] += 1
            paper_concept_counts[len(concepts)] += 1

            _increment_if(
                paper_violations,
                stable_hash(identity) != str(cache_key),
                "identity_signature_mismatch",
            )
            _increment_if(
                paper_violations,
                str(profile.get("profile_id") or "") != str(cache_key),
                "profile_id_mismatch",
            )
            _increment_if(
                paper_violations,
                str(profile.get("paper_arxiv_id") or "")
                != str(paper_arxiv_id),
                "paper_id_mismatch",
            )
            _increment_if(
                paper_violations,
                expected_topics != selected_topics,
                "selected_topics_not_all_classifier_candidates",
            )
            _increment_if(
                paper_violations,
                expected_topics != concepts,
                "concepts_not_all_classifier_candidates",
            )
            _increment_if(
                paper_violations,
                bool(profile.get("keyphrases")),
                "paper_keyphrases_present",
            )
            _increment_if(
                paper_violations,
                bool(profile.get("raw_llm_output")),
                "paper_raw_llm_output_present",
            )
            _increment_if(
                paper_violations,
                str(profile.get("llm_model") or "") != "none",
                "paper_llm_model_not_none",
            )
            _increment_if(
                paper_violations,
                str(profile.get("pipeline_version") or "")
                != SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION,
                "paper_pipeline_version_mismatch",
            )
            _increment_if(
                paper_violations,
                str(profile.get("prompt_version") or "")
                != SEMRANK_CLASSIFIER_ONLY_PAPER_PROMPT_VERSION,
                "paper_prompt_version_mismatch",
            )
            _increment_if(
                paper_violations,
                str(identity.get("extraction_llm") or "") != "none",
                "paper_identity_extraction_llm_not_none",
            )
            _increment_if(
                paper_violations,
                str(identity.get("pipeline_version") or "")
                != SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION,
                "paper_identity_pipeline_version_mismatch",
            )
            _increment_if(
                paper_violations,
                "concept_encoder" in identity,
                "text_profile_identity_contains_encoder",
            )

        query_violations: Counter[str] = Counter()
        query_statuses: Counter[str] = Counter()
        query_ids: Counter[str] = Counter()
        query_concept_counts: Counter[int] = Counter()
        for (
            cache_key,
            query_id,
            identity_json,
            profile_json,
        ) in connection.execute(
            """
            SELECT cache_key, query_id, identity_json, profile_json
            FROM query_profiles
            """
        ):
            identity = json.loads(identity_json)
            profile = json.loads(profile_json)
            selected = [
                normalize_concept(value)
                for value in profile.get("selected_concepts") or []
                if normalize_concept(value)
            ]
            referenced_concepts.update(selected)
            query_ids[str(query_id)] += 1
            query_statuses[
                str(profile.get("selection_status") or "unknown")
            ] += 1
            query_concept_counts[len(selected)] += 1

            _increment_if(
                query_violations,
                stable_hash(identity) != str(cache_key),
                "identity_signature_mismatch",
            )
            _increment_if(
                query_violations,
                str(profile.get("query_profile_id") or "")
                != str(cache_key),
                "profile_id_mismatch",
            )
            _increment_if(
                query_violations,
                str(profile.get("query_id") or "") != str(query_id),
                "query_id_mismatch",
            )
            _increment_if(
                query_violations,
                bool(profile.get("candidate_keyphrases")),
                "classifier_only_query_keyphrases_present",
            )
            _increment_if(
                query_violations,
                str(profile.get("prompt_version") or "")
                != SEMRANK_QUERY_PROMPT_VERSION,
                "query_prompt_version_mismatch",
            )
            _increment_if(
                query_violations,
                str(profile.get("topic_pipeline_version") or "")
                != SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION,
                "query_topic_pipeline_version_mismatch",
            )
            _increment_if(
                query_violations,
                str(identity.get("paper_extraction_llm") or "") != "none",
                "query_identity_paper_extraction_llm_not_none",
            )
            _increment_if(
                query_violations,
                str(identity.get("paper_concept_pipeline_version") or "")
                != SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION,
                "query_identity_pipeline_version_mismatch",
            )
            _increment_if(
                query_violations,
                "concept_encoder" in identity
                or "paper_concept_encoder" in identity,
                "text_profile_identity_contains_encoder",
            )

        vector_violations: Counter[str] = Counter()
        vector_namespaces: Counter[str] = Counter()
        vector_dimensions: Counter[int] = Counter()
        vector_concepts: set[str] = set()
        for (
            cache_key,
            encoder_id,
            concept,
            dimension,
            blob_bytes,
        ) in connection.execute(
            """
            SELECT cache_key, encoder_id, concept_normalized, dimension,
                   length(vector_f32)
            FROM concept_embeddings
            """
        ):
            encoder = str(encoder_id)
            normalized = normalize_concept(concept)
            dimension_value = int(dimension)
            vector_namespaces[encoder] += 1
            vector_dimensions[dimension_value] += 1
            vector_concepts.add(str(concept))
            expected_key = stable_hash(
                {
                    "encoder_id": encoder,
                    "concept": normalized,
                    "normalization": "nfkc_lower_space_v1",
                }
            )
            _increment_if(
                vector_violations,
                expected_key != str(cache_key),
                "encoder_bound_signature_mismatch",
            )
            _increment_if(
                vector_violations,
                normalized != str(concept),
                "concept_not_normalized",
            )
            _increment_if(
                vector_violations,
                int(blob_bytes or 0) != dimension_value * 4,
                "f32_blob_size_mismatch",
            )

        duplicate_query_ids = {
            key: value
            for key, value in query_ids.items()
            if value != 1
        }
        missing_referenced_vectors = sorted(
            referenced_concepts - vector_concepts
        )
        namespace_ids = sorted(vector_namespaces)
        qwen_namespace_only = (
            len(namespace_ids) == 1
            and "model=qwen3-embedding:0.6b" in namespace_ids[0].lower()
            and "specter" not in namespace_ids[0].lower()
        )
        checks = {
            "schema_version": metadata.get("schema_version")
            == SCHEMA_VERSION,
            "paper_profiles_nonempty": table_counts["paper_profiles"] > 0,
            "exactly_50_query_profiles": (
                table_counts["query_profiles"] == 50
            ),
            "one_profile_per_query_id": (
                len(query_ids) == table_counts["query_profiles"]
                and not duplicate_query_ids
            ),
            "paper_classifier_only_contract": not paper_violations,
            "query_classifier_only_contract": not query_violations,
            "qwen_vector_namespace_exclusive": qwen_namespace_only,
            "encoder_bound_vector_signatures": not vector_violations,
            "all_referenced_concepts_have_vectors": (
                not missing_referenced_vectors
            ),
            "one_vector_row_per_concept": (
                len(vector_concepts)
                == table_counts["concept_embeddings"]
            ),
        }
        return {
            "cache_path": str(path),
            "read_only": True,
            "metadata": metadata,
            "table_counts": table_counts,
            "paper_profile_count": sum(paper_statuses.values()),
            "unique_paper_ids": len(paper_ids),
            "paper_status_counts": dict(sorted(paper_statuses.items())),
            "paper_concept_count_distribution": {
                str(key): int(value)
                for key, value in sorted(paper_concept_counts.items())
            },
            "paper_contract_violations": dict(
                sorted(paper_violations.items())
            ),
            "query_status_counts": dict(sorted(query_statuses.items())),
            "query_concept_count_distribution": {
                str(key): int(value)
                for key, value in sorted(query_concept_counts.items())
            },
            "duplicate_query_ids": duplicate_query_ids,
            "vector_namespace_counts": dict(
                sorted(vector_namespaces.items())
            ),
            "vector_dimension_counts": {
                str(key): int(value)
                for key, value in sorted(vector_dimensions.items())
            },
            "vector_contract_violations": dict(
                sorted(vector_violations.items())
            ),
            "referenced_concept_count": len(referenced_concepts),
            "vector_concept_count": len(vector_concepts),
            "missing_referenced_vector_count": len(
                missing_referenced_vectors
            ),
            "missing_referenced_vector_preview": (
                missing_referenced_vectors[:25]
            ),
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--report_json", required=True)
    args = parser.parse_args()
    report = audit_cache(args.cache_path)
    report_path = Path(args.report_json).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
