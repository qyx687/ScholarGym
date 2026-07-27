#!/usr/bin/env python3
"""Build classifier-only query profiles from saved date-valid retrieval IDs.

Only the original query, date cutoff, and ordered auxiliary retrieval paper IDs
are reused from a full SemRank run.  Full query concepts, candidate concepts,
keyphrases, and raw LLM outputs are ignored.  Classifier-only paper topics are
read from the target cache and the query-level LLM is called again.  A target
Top-M smaller than the saved source Top-M uses the exact ordered prefix, which
preserves the source retriever and strict date-filter contract while giving
the target profile its own Top-M-bound cache identity.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from graph_methods import load_paper_db  # noqa: E402
from semrank import (  # noqa: E402
    AuxiliaryPaper,
    QueryConceptProfile,
    SEMRANK_CLASSIFIER_ONLY_PAPER_PROMPT_VERSION,
    SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION,
    SEMRANK_CONCEPT_NORMALIZATION_VERSION,
    SEMRANK_QUERY_PROMPT_VERSION,
    SemRankCache,
    SemRankLLMClient,
    normalize_concept,
    query_text_profile_identity,
    stable_hash,
)


def artifact_dir(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if (path / "semrank_query_profiles.jsonl").is_file():
        return path
    nested = path / "online_artifacts"
    if (nested / "semrank_query_profiles.jsonl").is_file():
        return nested
    raise FileNotFoundError(
        f"semrank_query_profiles.jsonl not found below {path}"
    )


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"expected JSON object at {path}:{line_number}"
                )
            yield value


def load_source_profiles(path: Path) -> List[Dict[str, Any]]:
    profiles: Dict[str, Dict[str, Any]] = {}
    for value in iter_jsonl(path):
        query_id = str(value.get("query_id") or "")
        if not query_id:
            raise ValueError("source query profile has no query_id")
        previous = profiles.get(query_id)
        if previous is not None and (
            previous.get("query_profile_id")
            != value.get("query_profile_id")
        ):
            raise ValueError(
                f"multiple full query profiles found for {query_id}"
            )
        profiles[query_id] = value
    return [profiles[key] for key in sorted(profiles)]


def rank_frequencies(
    counter: Counter[str],
    limit: int,
) -> List[Dict[str, Any]]:
    return [
        {"concept": concept, "frequency": int(frequency)}
        for concept, frequency in sorted(
            counter.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[:limit]
    ]


class PaperProfileLookup:
    def __init__(self, cache_path: Path) -> None:
        self.connection = sqlite3.connect(
            f"file:{cache_path.as_posix()}?mode=ro",
            uri=True,
        )

    def get(self, paper_id: str) -> Dict[str, Any]:
        rows = self.connection.execute(
            """
            SELECT profile_json
            FROM paper_profiles
            WHERE paper_arxiv_id=?
            """,
            (str(paper_id),),
        ).fetchall()
        if len(rows) != 1:
            raise KeyError(
                "classifier-only cache must contain exactly one profile for "
                f"{paper_id}; found {len(rows)}"
            )
        return json.loads(rows[0][0])

    def classifier_identity(self) -> tuple[str, str]:
        rows = self.connection.execute(
            """
            SELECT profile_json
            FROM paper_profiles
            LIMIT 1
            """
        ).fetchall()
        if len(rows) != 1:
            raise RuntimeError("classifier-only paper cache is empty")
        value = json.loads(rows[0][0])
        return (
            str(value.get("classifier_id") or ""),
            str(value.get("label_space_id") or ""),
        )

    def encoder_id(self) -> str:
        rows = self.connection.execute(
            """
            SELECT encoder_id, COUNT(*)
            FROM concept_embeddings
            GROUP BY encoder_id
            """
        ).fetchall()
        if len(rows) != 1:
            raise RuntimeError(
                "classifier-only cache must have exactly one vector namespace"
            )
        return str(rows[0][0])

    def close(self) -> None:
        self.connection.close()


def auxiliary_papers(
    profile: Mapping[str, Any],
    paper_db: Mapping[str, Mapping[str, Any]],
) -> List[AuxiliaryPaper]:
    output = []
    for rank, paper_id in enumerate(
        profile.get("initial_retrieval_paper_ids") or [],
        start=1,
    ):
        metadata = paper_db.get(str(paper_id)) or {}
        if not metadata:
            raise KeyError(
                f"paper DB is missing auxiliary paper {paper_id}"
            )
        output.append(
            AuxiliaryPaper(
                paper_arxiv_id=str(paper_id),
                score=0.0,
                title=str(metadata.get("title") or ""),
                abstract=str(metadata.get("abstract") or ""),
                date=str(metadata.get("date") or ""),
                rank=rank,
            )
        )
    return output


def query_identity(
    source: Mapping[str, Any],
    *,
    classifier_id: str,
    label_space_id: str,
    query_llm_id: str,
    initial_top_m: int,
    feedback_top_n: int,
    prompt_top_papers: int,
    candidate_topic_k: int,
    candidate_phrase_k: int,
    classifier_topic_k: int,
) -> Dict[str, Any]:
    return query_text_profile_identity(
        {
            "query_id": str(source.get("query_id") or ""),
            "original_query_hash": stable_hash(
                str(source.get("query") or "")
            ),
            "date_cutoff": str(source.get("date_cutoff") or "")[:7],
            "initial_retriever_identity": str(
                source.get("retriever_identity") or ""
            ),
            "initial_top_m": initial_top_m,
            "feedback_top_n": feedback_top_n,
            "prompt_top_papers": prompt_top_papers,
            "candidate_topic_k": candidate_topic_k,
            "candidate_phrase_k": candidate_phrase_k,
            "paper_concept_pipeline_version": (
                SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION
            ),
            "paper_classifier_topic_k": classifier_topic_k,
            "paper_prompt_version": (
                SEMRANK_CLASSIFIER_ONLY_PAPER_PROMPT_VERSION
            ),
            "paper_topic_classifier": classifier_id,
            "paper_topic_label_space": label_space_id,
            "paper_extraction_llm": "none",
            "concept_normalization": (
                SEMRANK_CONCEPT_NORMALIZATION_VERSION
            ),
            "query_prompt_version": SEMRANK_QUERY_PROMPT_VERSION,
            "query_llm": query_llm_id,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_semrank_run", required=True)
    parser.add_argument("--paper_db", required=True)
    parser.add_argument("--target_cache", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument(
        "--llm_model",
        default="qwen3-30b-a3b-instruct-2507",
    )
    parser.add_argument(
        "--llm_is_local",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--initial_top_m", type=int, default=1000)
    parser.add_argument("--feedback_top_n", type=int, default=100)
    parser.add_argument("--prompt_top_papers", type=int, default=50)
    parser.add_argument("--candidate_topic_k", type=int, default=50)
    parser.add_argument("--candidate_phrase_k", type=int, default=50)
    parser.add_argument("--classifier_topic_k", type=int, default=100)
    args = parser.parse_args()
    if args.initial_top_m <= 0:
        raise ValueError("--initial_top_m must be positive")
    if args.feedback_top_n <= 0:
        raise ValueError("--feedback_top_n must be positive")
    if args.feedback_top_n > args.initial_top_m:
        raise ValueError(
            "--feedback_top_n cannot exceed --initial_top_m"
        )
    if args.prompt_top_papers > args.feedback_top_n:
        raise ValueError(
            "--prompt_top_papers cannot exceed --feedback_top_n"
        )

    report_path = Path(args.report_json).expanduser().resolve()
    prior_report: Dict[str, Any] = {}
    if report_path.is_file():
        prior_report = json.loads(report_path.read_text(encoding="utf-8"))
    prior_empty_selections = {
        str(query_id)
        for query_id, reason in (
            prior_report.get("query_profile_failures") or {}
        ).items()
        if reason == "empty_selection"
    }
    prior_cumulative_llm_stats = (
        prior_report.get("cumulative_query_level_llm_stats")
        or prior_report.get("query_level_llm_stats")
        or {}
    )

    source_artifacts = artifact_dir(args.source_semrank_run)
    source_profiles = load_source_profiles(
        source_artifacts / "semrank_query_profiles.jsonl"
    )
    if len(source_profiles) != 50:
        raise AssertionError(
            f"expected 50 source query profiles, found {len(source_profiles)}"
        )
    target_cache = Path(args.target_cache).expanduser().resolve()
    if not target_cache.is_file():
        raise FileNotFoundError(target_cache)
    paper_db = load_paper_db(args.paper_db)
    lookup = PaperProfileLookup(target_cache)
    classifier_id, label_space_id = lookup.classifier_identity()
    encoder_id = lookup.encoder_id()
    llm = SemRankLLMClient(
        args.llm_model,
        is_local=args.llm_is_local,
        enable_thinking=False,
        temperature=0.0,
        top_p=1.0,
        workers=max(1, args.workers),
    )

    tasks = []
    already_cached = 0
    paper_profile_reads = 0
    source_query_llm_payloads_ignored = 0
    source_initial_retrieval_counts: Counter[int] = Counter()
    truncated_source_query_count = 0
    with SemRankCache(target_cache) as cache:
        for source in source_profiles:
            identity = query_identity(
                source,
                classifier_id=classifier_id,
                label_space_id=label_space_id,
                query_llm_id=llm.llm_id,
                initial_top_m=args.initial_top_m,
                feedback_top_n=args.feedback_top_n,
                prompt_top_papers=args.prompt_top_papers,
                candidate_topic_k=args.candidate_topic_k,
                candidate_phrase_k=args.candidate_phrase_k,
                classifier_topic_k=args.classifier_topic_k,
            )
            cache_key = stable_hash(identity)
            if cache.get_query_profile(cache_key) is not None:
                already_cached += 1
                continue
            source_query_llm_payloads_ignored += int(
                bool(source.get("candidate_topics"))
                or bool(source.get("candidate_keyphrases"))
                or bool(source.get("selected_concepts"))
                or bool(source.get("raw_llm_output"))
            )
            source_retrieved = auxiliary_papers(source, paper_db)
            source_initial_retrieval_counts[len(source_retrieved)] += 1
            if len(source_retrieved) < args.initial_top_m:
                raise AssertionError(
                    f"{source['query_id']} has only "
                    f"{len(source_retrieved)} saved auxiliary papers; "
                    f"at least {args.initial_top_m} are required"
                )
            truncated_source_query_count += int(
                len(source_retrieved) > args.initial_top_m
            )
            retrieved = source_retrieved[: args.initial_top_m]
            source_feedback_ids = [
                str(value)
                for value in source.get("feedback_paper_ids") or []
            ]
            expected_feedback_ids = [
                item.paper_arxiv_id
                for item in retrieved[: args.feedback_top_n]
            ]
            if (
                len(source_feedback_ids) < args.feedback_top_n
                or source_feedback_ids[: args.feedback_top_n]
                != expected_feedback_ids
            ):
                raise AssertionError(
                    f"{source['query_id']} feedback IDs do not match the "
                    "saved initial-retrieval prefix"
                )
            feedback_ids = expected_feedback_ids
            topic_frequency: Counter[str] = Counter()
            for paper_id in feedback_ids:
                paper_profile = lookup.get(paper_id)
                paper_profile_reads += 1
                if paper_profile.get("keyphrases"):
                    raise AssertionError(
                        "classifier-only paper profile contains keyphrases"
                    )
                topic_frequency.update(
                    {
                        normalize_concept(value)
                        for value in (
                            paper_profile.get("selected_topics") or []
                        )
                        if normalize_concept(value)
                    }
                )
            candidate_topics = rank_frequencies(
                topic_frequency,
                args.candidate_topic_k,
            )
            if not candidate_topics:
                raise RuntimeError(
                    f"{source['query_id']} has no classifier topics"
                )
            tasks.append(
                {
                    "source": source,
                    "identity": identity,
                    "cache_key": cache_key,
                    "retrieved": retrieved,
                    "candidate_topics": candidate_topics,
                }
            )

        def select(task: Mapping[str, Any]) -> tuple[List[str], str]:
            return llm.select_query_concepts(
                str(task["source"].get("query") or ""),
                task["retrieved"][: args.prompt_top_papers],
                task["candidate_topics"],
                [],
            )

        def store_profile(
            task: Mapping[str, Any],
            *,
            selected: Sequence[str],
            raw_output: str | None,
            status: str,
            fallback_reason: str | None,
        ) -> None:
            source = task["source"]
            profile = QueryConceptProfile(
                query_id=str(source.get("query_id") or ""),
                query_profile_id=str(task["cache_key"]),
                query=str(source.get("query") or ""),
                date_cutoff=str(source.get("date_cutoff") or "")[:7],
                initial_retrieval_paper_ids=[
                    item.paper_arxiv_id for item in task["retrieved"]
                ],
                feedback_paper_ids=[
                    item.paper_arxiv_id
                    for item in task["retrieved"][: args.feedback_top_n]
                ],
                candidate_topics=task["candidate_topics"],
                candidate_keyphrases=[],
                selected_concepts=list(selected),
                selection_status=status,
                fallback_reason=fallback_reason,
                prompt_version=SEMRANK_QUERY_PROMPT_VERSION,
                llm_model=llm.model,
                concept_encoder=encoder_id,
                topic_pipeline_version=(
                    SEMRANK_CLASSIFIER_ONLY_TOPIC_PIPELINE_VERSION
                ),
                retriever_identity=str(
                    source.get("retriever_identity") or ""
                ),
                initial_retrieval_target=args.initial_top_m,
                initial_retrieval_count=len(task["retrieved"]),
                initial_retrieval_complete=True,
                cache_hit=False,
                raw_llm_output=raw_output,
            )
            cache.put_query_profile(
                str(task["cache_key"]),
                task["identity"],
                profile,
            )

        completed = 0
        completed_ok = 0
        completed_fallback = 0
        restored_empty_query_ids: List[str] = []
        failures: Dict[str, str] = {}
        pending_tasks = []
        for task in tasks:
            query_id = str(task["source"].get("query_id") or "")
            if query_id not in prior_empty_selections:
                pending_tasks.append(task)
                continue
            # The live QueryProfileBuilder caches an empty model selection as
            # a legitimate base-score fallback.  The prior interrupted
            # prebuild recorded that first result but had not persisted it.
            # Restore that exact semantic outcome without a second LLM call.
            store_profile(
                task,
                selected=[],
                raw_output=None,
                status="fallback",
                fallback_reason="query_concept_selection_empty",
            )
            completed += 1
            completed_fallback += 1
            restored_empty_query_ids.append(query_id)

        with ThreadPoolExecutor(
            max_workers=max(1, args.workers)
        ) as executor:
            futures = {
                executor.submit(select, task): task
                for task in pending_tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                source = task["source"]
                query_id = str(source.get("query_id") or "")
                try:
                    selected, raw_output = future.result()
                except Exception as exc:
                    failures[query_id] = type(exc).__name__
                    continue
                if not selected:
                    store_profile(
                        task,
                        selected=[],
                        raw_output=raw_output,
                        status="fallback",
                        fallback_reason="query_concept_selection_empty",
                    )
                    completed += 1
                    completed_fallback += 1
                    continue
                store_profile(
                    task,
                    selected=selected,
                    raw_output=raw_output,
                    status="ok",
                    fallback_reason=None,
                )
                completed += 1
                completed_ok += 1

        counts = cache.table_counts()
        cache_stats = cache.snapshot_stats()

    lookup.close()
    llm_stats = llm.snapshot_stats()
    llm.close()
    cumulative_llm_stats = {
        key: int(prior_cumulative_llm_stats.get(key, 0))
        + int(llm_stats.get(key, 0))
        for key in set(prior_cumulative_llm_stats) | set(llm_stats)
    }
    report = {
        "operation": "build_classifier_only_query_profiles_v2",
        "source_semrank_run": str(source_artifacts),
        "target_cache": str(target_cache),
        "source_query_profile_count": len(source_profiles),
        "source_query_concepts_reused": 0,
        "source_query_llm_payloads_ignored": (
            source_query_llm_payloads_ignored
        ),
        "paper_profile_reads": paper_profile_reads,
        "paper_keyphrases_consumed": 0,
        "paper_level_llm_calls": 0,
        "initial_top_m": args.initial_top_m,
        "feedback_top_n": args.feedback_top_n,
        "prompt_top_papers": args.prompt_top_papers,
        "candidate_topic_k": args.candidate_topic_k,
        "candidate_phrase_k": args.candidate_phrase_k,
        "classifier_topic_k": args.classifier_topic_k,
        "source_initial_retrieval_count_distribution": {
            str(key): int(value)
            for key, value in sorted(
                source_initial_retrieval_counts.items()
            )
        },
        "source_queries_truncated_to_target_top_m": (
            truncated_source_query_count
        ),
        "query_profiles_already_cached": already_cached,
        "query_profiles_built": completed,
        "query_profiles_ok_built": completed_ok,
        "query_profiles_fallback_built": completed_fallback,
        "empty_selection_fallback_query_ids": sorted(
            restored_empty_query_ids
        ),
        "query_profile_failures": failures,
        "query_level_llm_stats": llm_stats,
        "cumulative_query_level_llm_stats": cumulative_llm_stats,
        "cache_stats": cache_stats,
        "cache_counts": counts,
        "paper_concept_mode": "classifier_only",
        "query_candidate_keyphrases": 0,
        "workers": max(1, args.workers),
        "all_checks_passed": (
            not failures
            and counts["query_profiles"] == len(source_profiles)
            and llm_stats.get("paper_concept_llm_calls", 0) == 0
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(
            f"classifier-only query profile failures: {failures}"
        )


if __name__ == "__main__":
    main()
