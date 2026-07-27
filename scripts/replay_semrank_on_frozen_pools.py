#!/usr/bin/env python3
"""Score saved candidate pools with fixed original-query SemRank profiles."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from graph_methods import EmbeddingProvider, load_paper_db  # noqa: E402
from semrank import (  # noqa: E402
    CachedConceptEncoder,
    EmbeddingProviderConceptEncoder,
    OfficialSemRankTopicClassifier,
    PaperConceptService,
    QueryConceptProfile,
    SemRankCache,
    SemRankConfig,
    SemRankLLMClient,
    SemRankQSQReranker,
    Specter2MeanPoolEncoder,
)


def artifact_dir(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / "paper_rows.jsonl").is_file():
        return path
    if (path / "online_artifacts" / "paper_rows.jsonl").is_file():
        return path / "online_artifacts"
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


def iter_events(path: Path) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    current = ""
    rows: List[Dict[str, Any]] = []
    for row in iter_jsonl(path):
        event_id = str(row.get("retrieval_event_id") or "")
        if current and event_id != current:
            yield current, rows
            rows = []
        current = event_id
        rows.append(row)
    if current:
        yield current, rows


def load_query_profiles(path: Path) -> Dict[str, QueryConceptProfile]:
    output = {}
    for value in iter_jsonl(path):
        profile = QueryConceptProfile.from_dict(value)
        previous = output.get(profile.query_id)
        if (
            previous is not None
            and previous.query_profile_id != profile.query_profile_id
        ):
            raise ValueError(
                f"multiple SemRank profiles found for {profile.query_id}"
            )
        output[profile.query_id] = profile
    if not output:
        raise ValueError(f"no SemRank query profiles found at {path}")
    return output


def load_query_profiles_from_cache(
    value: str | Path,
) -> Tuple[Dict[str, QueryConceptProfile], Path]:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "semrank.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(f"SemRank cache not found: {path}")
    connection = sqlite3.connect(
        f"file:{path.as_posix()}?mode=ro",
        uri=True,
    )
    try:
        rows = connection.execute(
            "SELECT profile_json FROM query_profiles"
        ).fetchall()
    finally:
        connection.close()
    output: Dict[str, QueryConceptProfile] = {}
    for (profile_json,) in rows:
        profile = QueryConceptProfile.from_dict(json.loads(profile_json))
        previous = output.get(profile.query_id)
        if (
            previous is not None
            and previous.query_profile_id != profile.query_profile_id
        ):
            raise ValueError(
                f"multiple SemRank profiles found for {profile.query_id}"
            )
        output[profile.query_id] = profile
    if not output:
        raise ValueError(f"no SemRank query profiles found in {path}")
    return output, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_run", required=True)
    profile_source = parser.add_mutually_exclusive_group(required=True)
    profile_source.add_argument("--semrank_run")
    profile_source.add_argument("--query_profile_cache")
    parser.add_argument("--paper_db", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument(
        "--checkpoint",
        default=(
            "../third_party/SemRank/classifier/"
            "topic_classifier_specter2.pt"
        ),
    )
    parser.add_argument(
        "--labels",
        default="../third_party/SemRank/classifier/labels.txt",
    )
    parser.add_argument(
        "--concept_encoder_backend",
        choices=["ollama", "api", "specter2"],
        default="ollama",
    )
    parser.add_argument(
        "--concept_encoder",
        default="qwen3-embedding:0.6b",
    )
    parser.add_argument(
        "--concept_encoder_base_url",
        default="http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--concept_encoder_api_key_env",
        default="EMBEDDING_API_KEY",
    )
    parser.add_argument(
        "--concept_encoder_revision",
        default="",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classifier_batch_size", type=int, default=4)
    parser.add_argument("--encoder_batch_size", type=int, default=64)
    parser.add_argument(
        "--paper_concept_mode",
        choices=["full", "classifier_only"],
        default="full",
    )
    parser.add_argument(
        "--topic_classifier_encoder",
        default="allenai/specter2_base",
    )
    parser.add_argument(
        "--topic_classifier_encoder_revision",
        default="3447645e1def9117997203454fa4495937bfbd83",
    )
    parser.add_argument(
        "--llm_model",
        default="qwen3-30b-a3b-instruct-2507",
    )
    parser.add_argument("--llm_workers", type=int, default=8)
    parser.add_argument("--cache_dir", default="cache/semrank_pasa_full")
    parser.add_argument("--cache_only", action="store_true")
    args = parser.parse_args()

    candidates = artifact_dir(args.candidate_run)
    semrank_artifacts = None
    query_profile_cache = None
    if args.semrank_run:
        semrank_artifacts = artifact_dir(args.semrank_run)
        query_profiles = load_query_profiles(
            semrank_artifacts / "semrank_query_profiles.jsonl"
        )
    else:
        query_profiles, query_profile_cache = (
            load_query_profiles_from_cache(args.query_profile_cache)
        )
    paper_db = load_paper_db(args.paper_db)
    cache_path = Path(args.cache_dir).expanduser().resolve() / "semrank.sqlite3"
    config = SemRankConfig(
        concept_encoder_backend=args.concept_encoder_backend,
        concept_encoder=args.concept_encoder,
        concept_encoder_base_url=args.concept_encoder_base_url,
        concept_encoder_revision=args.concept_encoder_revision,
        concept_encoder_device=args.device,
        concept_encoder_batch_size=args.encoder_batch_size,
        paper_concept_mode=args.paper_concept_mode,
        llm_model=args.llm_model,
        llm_is_local=False,
        llm_enable_thinking=False,
        llm_temperature=0.0,
        llm_top_p=1.0,
        llm_workers=args.llm_workers,
        topic_classifier_checkpoint=str(
            Path(args.checkpoint).expanduser().resolve()
        ),
        topic_labels_path=str(Path(args.labels).expanduser().resolve()),
        topic_classifier_encoder=args.topic_classifier_encoder,
        topic_classifier_encoder_revision=(
            args.topic_classifier_encoder_revision
        ),
        topic_classifier_device=args.device,
        topic_classifier_batch_size=args.classifier_batch_size,
        cache_path=str(cache_path),
        allow_lazy_paper_concepts=not args.cache_only,
        cache_only=args.cache_only,
    )
    output = Path(args.output_jsonl).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    event_count = 0
    candidate_count = 0

    with SemRankCache(cache_path) as cache:
        classifier = OfficialSemRankTopicClassifier(
            config.topic_classifier_checkpoint,
            config.topic_labels_path,
            encoder_name=config.topic_classifier_encoder,
            encoder_revision=config.topic_classifier_encoder_revision,
            device=config.topic_classifier_device,
            batch_size=config.topic_classifier_batch_size,
        )
        if config.concept_encoder_backend == "specter2":
            encoder_backend = Specter2MeanPoolEncoder(
                config.concept_encoder,
                revision=config.concept_encoder_revision,
                device=config.concept_encoder_device,
                batch_size=config.concept_encoder_batch_size,
                max_length=config.concept_encoder_max_length,
            )
        else:
            provider = EmbeddingProvider(
                backend=config.concept_encoder_backend,
                model=config.concept_encoder,
                base_url=config.concept_encoder_base_url,
                api_key=os.environ.get(
                    args.concept_encoder_api_key_env,
                    "",
                ),
                batch_size=config.concept_encoder_batch_size,
            )
            encoder_backend = EmbeddingProviderConceptEncoder(provider)
        encoder = CachedConceptEncoder(
            encoder_backend,
            cache,
        )
        llm = SemRankLLMClient(
            config.llm_model,
            is_local=False,
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            workers=config.llm_workers,
        )
        paper_service = PaperConceptService(
            config,
            cache,
            classifier,
            llm,
            encoder,
        )
        reranker = SemRankQSQReranker(
            config,
            query_profiles=None,
            paper_concepts=paper_service,
            encoder=encoder,
        )
        with temp_output.open("w", encoding="utf-8") as handle:
            for event_id, rows in iter_events(
                candidates / "paper_rows.jsonl"
            ):
                query_id = str(rows[0].get("query_id") or "")
                if query_id not in query_profiles:
                    raise KeyError(
                        f"no SemRank original-query profile for {query_id}"
                    )
                reranker.active_query_profile = query_profiles[query_id]
                ranked, event_profile = reranker.rerank(
                    rows,
                    paper_db,
                    retrieval_event_id=event_id,
                )
                event_count += 1
                candidate_count += len(ranked)
                for row in ranked:
                    handle.write(
                        json.dumps(
                            {
                                "retrieval_event_id": event_id,
                                "query_id": query_id,
                                "paper_arxiv_id": row["paper_arxiv_id"],
                                "rerank_score": row["rerank_score"],
                                "rerank_rank": row["rerank_rank"],
                                "query_profile_id": row["query_profile_id"],
                                "candidate_pool_signature": (
                                    event_profile.candidate_pool_signature
                                ),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                handle.flush()
        temp_output.replace(output)
        summary = {
            "candidate_run": str(candidates),
            "semrank_run": (
                str(semrank_artifacts)
                if semrank_artifacts is not None
                else None
            ),
            "query_profile_cache": (
                str(query_profile_cache)
                if query_profile_cache is not None
                else None
            ),
            "output_jsonl": str(output),
            "event_count": event_count,
            "candidate_count": candidate_count,
            "query_profile_count": len(query_profiles),
            "paper_profile_stats": paper_service.snapshot_stats(),
            "classifier_stats": classifier.snapshot_stats(),
            "llm_stats": llm.snapshot_stats(),
            "encoder_stats": encoder.snapshot_stats(),
            "cache_counts": cache.table_counts(),
            "cache_only": args.cache_only,
        }
        llm.close()
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
