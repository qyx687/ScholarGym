#!/usr/bin/env python3
"""Prebuild official-pipeline SemRank paper concept profiles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Set

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from graph_methods import (  # noqa: E402
    EmbeddingProvider,
    load_paper_db,
    normalize_arxiv_id,
)
from semrank import (  # noqa: E402
    CachedConceptEncoder,
    EmbeddingProviderConceptEncoder,
    OfficialSemRankTopicClassifier,
    PaperConceptService,
    SemRankCache,
    SemRankConfig,
    SemRankLLMClient,
    Specter2MeanPoolEncoder,
)


class DeferredConceptEncoder:
    """Preserve Qwen identity while a separate worker fills vector rows."""

    def __init__(self, encoder_id: str) -> None:
        self.encoder_id = str(encoder_id)
        self._stats: Counter[str] = Counter()

    def encode(self, concepts):
        self._stats["deferred_calls"] += 1
        self._stats["deferred_concepts"] += len(concepts)
        return np.zeros((len(concepts), 0), dtype=np.float32)

    def snapshot_stats(self):
        return {key: int(value) for key, value in self._stats.items()}


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def artifact_paper_rows(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    candidates = [
        path,
        path / "paper_rows.jsonl",
        path / "online_artifacts" / "paper_rows.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"paper_rows.jsonl not found below {path}")


def collect_ids(
    paper_id_files: Iterable[str],
    artifacts: Iterable[str],
) -> Set[str]:
    output: Set[str] = set()
    for value in paper_id_files:
        path = Path(value).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                parsed = line
            if isinstance(parsed, dict):
                parsed = (
                    parsed.get("paper_arxiv_id")
                    or parsed.get("arxiv_id")
                    or parsed.get("id")
                )
            paper_id = normalize_arxiv_id(parsed)
            if paper_id:
                output.add(paper_id)
    for value in artifacts:
        for row in iter_jsonl(artifact_paper_rows(value)):
            paper_id = normalize_arxiv_id(row.get("paper_arxiv_id"))
            if paper_id:
                output.add(paper_id)
    return output


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper_db", required=True)
    parser.add_argument("--paper_id", action="append", default=[])
    parser.add_argument("--paper_id_file", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--all_corpus", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--defer_concept_encoding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Persist text profiles without blocking on Qwen; run the "
            "dedicated re-encoder before evaluation."
        ),
    )
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
    parser.add_argument(
        "--cache_dir",
        default="cache/semrank_pasa_full",
    )
    parser.add_argument(
        "--audit_jsonl",
        default="cache/semrank/paper_concept_build_audit.jsonl",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--retry_failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry only cached profiles whose audited status is failed.",
    )
    parser.add_argument(
        "--require_no_failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit nonzero if any cached profile still has status=failed.",
    )
    args = parser.parse_args()

    paper_db = load_paper_db(args.paper_db)
    requested = collect_ids(args.paper_id_file, args.artifact)
    requested.update(
        paper_id
        for paper_id in (
            normalize_arxiv_id(value) for value in args.paper_id
        )
        if paper_id
    )
    if args.all_corpus:
        requested.update(paper_db)
    if not requested:
        parser.error(
            "provide --paper_id, --paper_id_file, --artifact, or explicit "
            "--all_corpus"
        )
    ordered = sorted(paper_id for paper_id in requested if paper_id in paper_db)
    if args.limit is not None:
        ordered = ordered[: max(0, args.limit)]
    if args.shard_count <= 0:
        parser.error("--shard_count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard_index must satisfy 0 <= index < count")
    eligible_before_sharding = len(ordered)
    ordered = ordered[args.shard_index :: args.shard_count]
    missing_db = len(requested) - len(
        {paper_id for paper_id in requested if paper_id in paper_db}
    )
    cache_path = Path(args.cache_dir).expanduser().resolve() / "semrank.sqlite3"
    config = SemRankConfig(
        concept_encoder_backend=args.concept_encoder_backend,
        concept_encoder=args.concept_encoder,
        concept_encoder_base_url=args.concept_encoder_base_url,
        concept_encoder_revision=args.concept_encoder_revision,
        concept_encoder_device=args.device,
        concept_encoder_batch_size=args.encoder_batch_size,
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
        rebuild_paper_concepts=args.rebuild,
        retry_failed_paper_concepts=args.retry_failed,
    )

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
        encoder = (
            DeferredConceptEncoder(encoder_backend.encoder_id)
            if args.defer_concept_encoding
            else CachedConceptEncoder(encoder_backend, cache)
        )
        llm = SemRankLLMClient(
            config.llm_model,
            is_local=False,
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            workers=config.llm_workers,
        )
        service = PaperConceptService(
            config,
            cache,
            classifier,
            llm,
            encoder,
        )
        for start in range(0, len(ordered), max(1, args.batch_size)):
            paper_ids = ordered[start : start + max(1, args.batch_size)]
            service.get_or_build(
                {paper_id: paper_db[paper_id] for paper_id in paper_ids}
            )
            audit = service.drain_audit_records()
            append_jsonl(Path(args.audit_jsonl), audit)
            print(
                json.dumps(
                    {
                        "completed": min(start + len(paper_ids), len(ordered)),
                        "total": len(ordered),
                        "paper_stats": service.snapshot_stats(),
                        "classifier_stats": classifier.snapshot_stats(),
                        "llm_stats": llm.snapshot_stats(),
                        "encoder_stats": encoder.snapshot_stats(),
                        "cache_counts": cache.table_counts(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        final = {
            "requested_ids": len(requested),
            "eligible_ids": len(ordered),
            "eligible_ids_before_sharding": eligible_before_sharding,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "defer_concept_encoding": args.defer_concept_encoding,
            "missing_from_paper_db": missing_db,
            "paper_stats": service.snapshot_stats(),
            "classifier_stats": classifier.snapshot_stats(),
            "llm_stats": llm.snapshot_stats(),
            "encoder_stats": encoder.snapshot_stats(),
            "cache_stats": cache.snapshot_stats(),
            "cache_counts": cache.table_counts(),
            "paper_profile_status_counts": (
                cache.paper_profile_status_counts()
            ),
            "cache_path": str(cache_path),
        }
        llm.close()
    print(json.dumps(final, indent=2, sort_keys=True))
    if (
        args.require_no_failed
        and final["paper_profile_status_counts"].get("failed", 0)
    ):
        raise RuntimeError(
            "SemRank warm-up retains failed paper profiles; rerun the same "
            "command to retry only those records."
        )


if __name__ == "__main__":
    main()
