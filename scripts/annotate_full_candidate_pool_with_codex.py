#!/usr/bin/env python3
"""Blindly annotate the full post-process candidate pool with Codex CLI.

The pipeline deliberately separates semantic annotation from retrieval provenance:

1. ``prepare`` unions and deduplicates query-paper candidates from the saved
   baseline, graph, deep-event, and deep-merged full artifacts; joins paper
   metadata; writes blinded Codex inputs; and preserves normalized occurrence
   rows for later Top-K analysis.
2. ``annotate-rubrics`` creates one stable query rubric per benchmark query.
3. ``annotate-candidates`` labels blinded paper batches with ``codex exec``.
4. ``aggregate`` reconnects labels to source provenance and produces overall
   graph-vs-deep comparison tables. It does not apply a rerank or Top-K cutoff.

Every Codex response is schema constrained, validated, written atomically, and
safe to resume. The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


SCHEMA_VERSION = "candidate_annotation_v1"
PROMPT_VERSION = "candidate_annotation_prompt_v2"

SOURCE_PATHS = {
    "baseline": Path("onepass_artifacts/baseline/paper_rows.jsonl"),
    "graph": Path("onepass_artifacts/per_subquery/paper_rows.jsonl"),
    "deep_event": Path("onepass_artifacts/deep_event/paper_rows.jsonl"),
    "deep_merged": Path("onepass_artifacts/deep_merged/paper_rows.jsonl"),
}
DEFAULT_SOURCES = tuple(SOURCE_PATHS)

RELEVANCE_GRADES = (
    "direct",
    "partial",
    "contextual",
    "unrelated",
    "insufficient_evidence",
)
SEMANTIC_DISTANCES = ("exact", "near", "adjacent", "distant", "unknown")
PAPER_TYPES = (
    "primary_empirical",
    "method_or_theory",
    "dataset_or_benchmark",
    "survey_or_review",
    "application_or_case_study",
    "position_or_commentary",
    "other",
    "unknown",
)
SCHOLARLY_ROLES = (
    "direct_target",
    "method_component",
    "task_or_application",
    "dataset_or_domain",
    "evaluation_or_metric",
    "empirical_evidence",
    "comparison_or_baseline",
    "background_or_foundation",
    "survey_or_taxonomy",
    "limitation_or_negative_result",
    "none",
)
INFORMATION_ADDED_VALUES = (
    "method_variant",
    "application_domain",
    "dataset_or_population",
    "metric_or_benchmark",
    "mechanism_or_theory",
    "empirical_evidence",
    "comparison_or_baseline",
    "limitation_or_failure",
    "taxonomy_or_synthesis",
    "historical_context",
    "none",
)
EXCLUSION_REASONS = (
    "wrong_method",
    "wrong_task",
    "wrong_domain_or_population",
    "wrong_outcome_or_evidence",
    "outside_date_scope",
    "secondary_not_primary",
    "mentions_without_using",
    "insufficient_evidence",
    "other",
    "none",
)
CONFIDENCE_VALUES = (0.25, 0.5, 0.75, 1.0)

RELEVANCE_SCORES = {
    "direct": 1.0,
    "partial": 2.0 / 3.0,
    "contextual": 1.0 / 3.0,
    "unrelated": 0.0,
}

# Canonical public field names follow the original annotation design.  The
# legacy aliases remain readable so completed runs can be aggregated without
# spending model budget on a field-name-only migration.
ANNOTATION_FIELD_ALIASES = {
    "relevance_grade": "relationship",
    "scholarly_roles": "relation_roles",
    "information_added": "contribution_types",
    "information_summary": "semantic_contribution",
}


RUBRIC_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "prompt_version": {"type": "string"},
        "query_id": {"type": "string"},
        "input_sha256": {"type": "string"},
        "scope_summary": {"type": "string"},
        "inclusion_criteria": {"type": "array", "items": {"type": "string"}},
        "exclusion_criteria": {"type": "array", "items": {"type": "string"}},
        "aspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect_id": {"type": "string"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "required_for_direct": {"type": "boolean"},
                },
                "required": ["aspect_id", "label", "description", "required_for_direct"],
                "additionalProperties": False,
            },
        },
        "direct_rule": {"type": "string"},
        "partial_rule": {"type": "string"},
        "contextual_rule": {"type": "string"},
    },
    "required": [
        "schema_version",
        "prompt_version",
        "query_id",
        "input_sha256",
        "scope_summary",
        "inclusion_criteria",
        "exclusion_criteria",
        "aspects",
        "direct_rule",
        "partial_rule",
        "contextual_rule",
    ],
    "additionalProperties": False,
}


CANDIDATE_BATCH_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "prompt_version": {"type": "string"},
        "batch_id": {"type": "string"},
        "input_sha256": {"type": "string"},
        "annotations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "relevance_grade": {
                        "type": "string",
                        "enum": list(RELEVANCE_GRADES),
                    },
                    "semantic_distance": {
                        "type": "string",
                        "enum": list(SEMANTIC_DISTANCES),
                    },
                    "paper_type": {"type": "string", "enum": list(PAPER_TYPES)},
                    "matched_aspect_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "scholarly_roles": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SCHOLARLY_ROLES)},
                    },
                    "information_added": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(INFORMATION_ADDED_VALUES),
                        },
                    },
                    "information_summary": {"type": "string"},
                    "key_concepts": {"type": "array", "items": {"type": "string"}},
                    "exclusion_reasons": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(EXCLUSION_REASONS)},
                    },
                    "evidence_phrases": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "needs_full_text": {"type": "boolean"},
                    "confidence": {"type": "number", "enum": list(CONFIDENCE_VALUES)},
                },
                "required": [
                    "candidate_id",
                    "relevance_grade",
                    "semantic_distance",
                    "paper_type",
                    "matched_aspect_ids",
                    "scholarly_roles",
                    "information_added",
                    "information_summary",
                    "key_concepts",
                    "exclusion_reasons",
                    "evidence_phrases",
                    "needs_full_text",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "schema_version",
        "prompt_version",
        "batch_id",
        "input_sha256",
        "annotations",
    ],
    "additionalProperties": False,
}


RUBRIC_INSTRUCTIONS = """You are creating a stable rubric for scientific-paper relevance annotation.

Use only the supplied query context. Do not inspect files, call tools, browse, or use retrieval-source assumptions.

Create 2-8 atomic aspects named A1, A2, ... . Mark an aspect required_for_direct only when a paper must satisfy it to directly answer the query. Convert the planner checklist into concise inclusion/exclusion rules, but resolve conflicts in favor of the original query. A direct paper must explicitly satisfy the required method/task/domain relationship; keyword mention alone is not enough. Partial papers satisfy a meaningful subset but miss or leave ambiguous at least one required condition. Contextual papers must explicitly connect to at least one query-specific aspect and provide information that could materially help answer the query. Generic tools, broad same-field background, or papers sharing only vocabulary are unrelated, not contextual.

Echo schema_version, prompt_version, query_id, and input_sha256 exactly. Return JSON only under the provided schema.
"""


CANDIDATE_INSTRUCTIONS = """You are blindly annotating scientific papers against one fixed query rubric.

Use only the supplied query, rubric, title, abstract, date, and categories. Do not inspect files, call tools, browse, infer retrieval provenance, or assume that a candidate is relevant because it was retrieved. You are not told whether a paper came from graph expansion, semantic retrieval, or ground truth.

Label rules:
- direct: the title/abstract explicitly supports all required conditions in the rubric.
- partial: it supports a substantive subset but misses or leaves ambiguous a required condition.
- contextual: it explicitly connects to at least one query-specific aspect and contributes a component, dataset, metric, survey, comparison, mechanism, or neighboring result that could materially help answer the query, but is not itself a qualifying answer.
- unrelated: no meaningful query-specific contribution.
- insufficient_evidence: title/abstract is too incomplete or ambiguous to decide; set needs_full_text=true.

Calibration boundary: being in the same broad field is not enough for contextual. Generic software/toolkits, general ML/NLP methods, generic datasets, or papers on a different task are unrelated unless the abstract makes a concrete connection to a rubric aspect. A paper labeled contextual should normally match at least one aspect. Use partial only for a substantive portion of the target relationship, not for loose topical similarity.

information_summary must state, in one concise English sentence, what query-relevant information this paper adds. It must not claim novelty relative to other candidates. For unrelated papers write "No query-relevant semantic contribution identified." For insufficient evidence explain the missing evidence briefly.

Use only valid rubric aspect IDs. evidence_phrases must contain 0-2 short phrases copied or tightly excerpted from the supplied title/abstract. key_concepts must contain at most 5 concise terms. Use "none" alone when no relation role, contribution type, or exclusion reason applies. Return exactly one annotation for every candidate_id, without duplicates. Echo schema_version, prompt_version, batch_id, and input_sha256 exactly. Return JSON only under the provided schema.
"""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_token(prefix: str, *parts: Any, length: int = 20) -> str:
    raw = "\0".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:length]}"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def _iter_top_level_json_object(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[Tuple[str, Any]]:
    """Stream key/value pairs from a large top-level JSON object with stdlib only."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def read_more(preserve_from: int) -> Tuple[str, int, bool]:
            nonlocal buffer, position, eof
            prefix = buffer[preserve_from:]
            chunk = handle.read(chunk_size)
            eof = chunk == ""
            buffer = prefix + chunk
            position = 0
            return buffer, position, eof

        read_more(0)

        def skip_ws() -> None:
            nonlocal buffer, position, eof
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                read_more(position)

        def expect(character: str) -> None:
            nonlocal buffer, position, eof
            skip_ws()
            if position >= len(buffer) and not eof:
                read_more(position)
                skip_ws()
            if position >= len(buffer) or buffer[position] != character:
                nearby = buffer[position : position + 40]
                raise ValueError(f"Expected {character!r} in {path}, found {nearby!r}")
            position += 1

        def decode_value() -> Any:
            nonlocal buffer, position, eof
            skip_ws()
            start = position
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, start)
                    position = end
                    return value
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"Invalid JSON object in {path}: {exc}") from exc
                    prefix = buffer[start:]
                    chunk = handle.read(chunk_size)
                    eof = chunk == ""
                    buffer = prefix + chunk
                    position = 0
                    start = 0

        skip_ws()
        expect("{")
        skip_ws()
        first = True
        while True:
            skip_ws()
            if position < len(buffer) and buffer[position] == "}":
                position += 1
                break
            if not first:
                expect(",")
            key = decode_value()
            if not isinstance(key, str):
                raise ValueError(f"Non-string key in {path}: {key!r}")
            expect(":")
            value = decode_value()
            yield key, value
            first = False
            if position > chunk_size:
                read_more(position)

        skip_ws()
        if not eof:
            trailing = buffer[position:] + handle.read()
        else:
            trailing = buffer[position:]
        if trailing.strip():
            raise ValueError(f"Unexpected trailing data in {path}")


def _normalize_arxiv_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^arxiv:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\.pdf$", "", text, flags=re.IGNORECASE)
    return re.sub(r"v\d+$", "", text, flags=re.IGNORECASE)


def _latest_detailed_records(path: Path) -> Dict[int, Dict[str, Any]]:
    output: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return output
    for row in _iter_jsonl(path):
        if row.get("idx") is None:
            continue
        output[int(row["idx"])] = row
    return output


def _query_id_from_detailed(row: Mapping[str, Any]) -> Optional[str]:
    postprocess = row.get("postprocess_results")
    if not isinstance(postprocess, Mapping):
        return None
    for method in ("per_subquery", "deep_event", "deep_merged", "baseline"):
        value = postprocess.get(method)
        if isinstance(value, Mapping) and value.get("query_id"):
            return str(value["query_id"])
    return None


def _set_min(target: MutableMapping[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(number):
        return
    previous = target.get(key)
    if previous is None or number < previous:
        target[key] = number


def _set_max(target: MutableMapping[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(number):
        return
    previous = target.get(key)
    if previous is None or number > previous:
        target[key] = number


def _new_source_stats() -> Dict[str, Any]:
    return {
        "occurrence_count": 0,
        "event_ids": set(),
        "subquery_ids": set(),
        "iteration_indices": set(),
        "edge_types": set(),
        "source_seed_ids": set(),
        "is_seed": False,
        "is_expanded": False,
        "ever_in_selector_topk": False,
        "ever_selector_selected": False,
    }


def _new_candidate(row: Mapping[str, Any], source: str, paper_id: str, query_id: str) -> Dict[str, Any]:
    benchmark_idx = int(row.get("benchmark_idx", -1))
    return {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": _stable_token("c", query_id, paper_id),
        "benchmark_idx": benchmark_idx,
        "query_id": query_id,
        "paper_id": paper_id,
        "sources": set(),
        "source_stats": {},
        "row_marked_ground_truth": False,
    }


def _update_candidate(candidate: MutableMapping[str, Any], row: Mapping[str, Any], source: str) -> None:
    candidate["sources"].add(source)
    stats = candidate["source_stats"].setdefault(source, _new_source_stats())
    stats["occurrence_count"] += 1
    for field, destination in (
        ("retrieval_event_id", "event_ids"),
        ("subquery_id", "subquery_ids"),
        ("iteration_idx", "iteration_indices"),
    ):
        value = row.get(field)
        if value is not None:
            stats[destination].add(value)
    for value in row.get("edge_types") or []:
        stats["edge_types"].add(str(value))
    for value in row.get("source_seed_arxiv_ids") or []:
        normalized = _normalize_arxiv_id(value)
        if normalized:
            stats["source_seed_ids"].add(normalized)
    stats["is_seed"] = stats["is_seed"] or bool(row.get("is_seed"))
    stats["is_expanded"] = stats["is_expanded"] or bool(row.get("is_expanded"))
    stats["ever_in_selector_topk"] = stats["ever_in_selector_topk"] or bool(
        row.get("in_selector_topk")
    )
    stats["ever_selector_selected"] = stats["ever_selector_selected"] or bool(
        row.get("selector_selected")
    )
    candidate["row_marked_ground_truth"] = candidate["row_marked_ground_truth"] or bool(
        row.get("is_ground_truth")
    )

    for field in (
        "rerank_rank",
        "selector_input_rank",
        "observed_retrieval_rank",
        "deep_retrieval_rank_after_exclusion",
        "deep_retrieval_rank_in_local_pool",
        "deep_retrieval_rank_global_date_valid",
    ):
        _set_min(stats, f"best_{field}", row.get(field))
    for field in (
        "rerank_score",
        "query_score_normalized",
        "subquery_score_normalized",
        "intent_score",
        "path_count",
        "path_count_normalized",
        "deep_retrieval_score_raw",
        "observed_retrieval_score",
    ):
        _set_max(stats, f"max_{field}", row.get(field))


def _finalize_source_stats(stats: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, set):
            output[key] = sorted(value, key=lambda item: str(item))
        else:
            output[key] = value
    output["event_count"] = len(output.get("event_ids", []))
    output["source_seed_count"] = len(output.get("source_seed_ids", []))
    return output


def _source_partition(sources: Set[str]) -> str:
    in_graph = "graph" in sources
    in_deep = "deep_event" in sources or "deep_merged" in sources
    if in_graph and in_deep:
        return "graph_and_deep"
    if in_graph:
        return "graph_only"
    if in_deep:
        return "deep_only"
    if "baseline" in sources:
        return "baseline_only"
    return "other"


def _normalized_occurrence(row: Mapping[str, Any], source: str, candidate_id: str, paper_id: str) -> Dict[str, Any]:
    fields = (
        "benchmark_idx",
        "query_id",
        "retrieval_event_id",
        "iteration_idx",
        "subquery_id",
        "subquery",
        "retrieval_page_idx",
        "selector_top_k",
        "in_selector_topk",
        "selector_input_rank",
        "selector_selected",
        "is_seed",
        "is_expanded",
        "edge_types",
        "source_seed_arxiv_ids",
        "path_count",
        "path_count_normalized",
        "query_score_normalized",
        "subquery_score_normalized",
        "intent_score",
        "rerank_rank",
        "rerank_score",
        "observed_retrieval_rank",
        "observed_retrieval_score",
        "deep_retrieval_rank_after_exclusion",
        "deep_retrieval_rank_in_local_pool",
        "deep_retrieval_rank_global_date_valid",
        "deep_retrieval_score_raw",
        "is_ground_truth",
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "paper_id": paper_id,
        "source": source,
    }
    for field in fields:
        if field in row:
            output[field] = row[field]
    return output


def _update_query_context(context: MutableMapping[str, Any], row: Mapping[str, Any]) -> None:
    for field in ("query", "query_date", "query_source"):
        if not context.get(field) and row.get(field):
            context[field] = row[field]
    checklist = str(row.get("planner_checklist") or "").strip()
    if checklist:
        context["planner_checklists"].add(checklist)
    subquery = str(row.get("subquery") or "").strip()
    if subquery:
        key = (
            row.get("subquery_id"),
            subquery,
            row.get("subquery_before_date"),
            row.get("subquery_link_type"),
        )
        context["subqueries"][key] = {
            "subquery_id": row.get("subquery_id"),
            "subquery": subquery,
            "before_date": row.get("subquery_before_date"),
            "link_type": row.get("subquery_link_type"),
        }


def _metadata_lookup_keys(paper_ids: Iterable[str]) -> Dict[str, Set[str]]:
    output: Dict[str, Set[str]] = defaultdict(set)
    for paper_id in paper_ids:
        output[_normalize_arxiv_id(paper_id)].add(paper_id)
    return output


def _load_needed_paper_metadata(paper_db: Path, paper_ids: Set[str]) -> Dict[str, Dict[str, Any]]:
    lookup = _metadata_lookup_keys(paper_ids)
    remaining = set(lookup)
    output: Dict[str, Dict[str, Any]] = {}
    for key, value in _iter_top_level_json_object(paper_db):
        normalized_key = _normalize_arxiv_id(key)
        if normalized_key not in remaining:
            continue
        if not isinstance(value, Mapping):
            value = {}
        metadata = {
            "title": str(value.get("title") or "").strip(),
            "abstract": str(value.get("abstract") or "").strip(),
            "date": value.get("date"),
            "categories": list(value.get("category") or value.get("categories") or []),
        }
        for original_id in lookup[normalized_key]:
            output[original_id] = dict(metadata)
        remaining.remove(normalized_key)
        if not remaining:
            break
    return output


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    paper_db = Path(args.paper_db).resolve()
    work_dir = Path(args.work_dir).resolve()
    sources = tuple(args.sources)
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not paper_db.is_file():
        raise FileNotFoundError(paper_db)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    work_dir.mkdir(parents=True, exist_ok=True)

    detailed = _latest_detailed_records(run_dir / "detailed_results.jsonl")
    gt_by_idx: Dict[int, Set[str]] = {}
    detailed_query_by_idx: Dict[int, str] = {}
    detailed_query_id_by_idx: Dict[int, str] = {}
    for benchmark_idx, row in detailed.items():
        gt_by_idx[benchmark_idx] = {
            _normalize_arxiv_id(value) for value in row.get("ground_truth_arxiv_ids") or []
        }
        if row.get("query"):
            detailed_query_by_idx[benchmark_idx] = str(row["query"])
        query_id = _query_id_from_detailed(row)
        if query_id:
            detailed_query_id_by_idx[benchmark_idx] = query_id

    candidates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    query_contexts: Dict[str, Dict[str, Any]] = {}
    source_row_counts: Counter[str] = Counter()
    occurrence_path = work_dir / "manifest" / "occurrences.jsonl"
    occurrence_path.parent.mkdir(parents=True, exist_ok=True)
    occurrence_tmp = occurrence_path.with_name(f".{occurrence_path.name}.tmp-{os.getpid()}")

    with occurrence_tmp.open("w", encoding="utf-8") as occurrence_handle:
        for source in sources:
            source_path = run_dir / SOURCE_PATHS[source]
            if not source_path.exists():
                if args.allow_missing_source:
                    continue
                raise FileNotFoundError(source_path)
            for row in _iter_jsonl(source_path):
                paper_id = _normalize_arxiv_id(row.get("paper_arxiv_id"))
                query_id = str(row.get("query_id") or "").strip()
                if not paper_id or not query_id:
                    continue
                benchmark_idx = int(row.get("benchmark_idx", -1))
                expected_query_id = detailed_query_id_by_idx.get(benchmark_idx)
                if expected_query_id and query_id != expected_query_id:
                    raise ValueError(
                        f"Query id mismatch for idx={benchmark_idx}: {query_id} != {expected_query_id}"
                    )
                key = (query_id, paper_id)
                candidate = candidates.get(key)
                if candidate is None:
                    candidate = _new_candidate(row, source, paper_id, query_id)
                    candidates[key] = candidate
                _update_candidate(candidate, row, source)
                source_row_counts[source] += 1

                context = query_contexts.setdefault(
                    query_id,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "query_token": _stable_token("q", query_id, length=16),
                        "benchmark_idx": benchmark_idx,
                        "query_id": query_id,
                        "query": detailed_query_by_idx.get(benchmark_idx, ""),
                        "query_date": None,
                        "query_source": None,
                        "planner_checklists": set(),
                        "subqueries": {},
                        "ground_truth_ids": set(gt_by_idx.get(benchmark_idx, set())),
                    },
                )
                _update_query_context(context, row)
                occurrence_handle.write(
                    json.dumps(
                        _normalized_occurrence(row, source, candidate["candidate_id"], paper_id),
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    os.replace(occurrence_tmp, occurrence_path)

    needed_paper_ids = {paper_id for _, paper_id in candidates}
    metadata = _load_needed_paper_metadata(paper_db, needed_paper_ids)

    finalized_queries: List[Dict[str, Any]] = []
    for context in query_contexts.values():
        context["planner_checklists"] = sorted(context["planner_checklists"])
        context["subqueries"] = sorted(
            context["subqueries"].values(),
            key=lambda value: (
                value.get("subquery_id") is None,
                value.get("subquery_id") or 0,
                value.get("subquery") or "",
            ),
        )
        context["ground_truth_ids"] = sorted(context["ground_truth_ids"])
        finalized_queries.append(context)
    finalized_queries.sort(key=lambda value: (value["benchmark_idx"], value["query_id"]))

    finalized_candidates: List[Dict[str, Any]] = []
    partition_counts: Counter[str] = Counter()
    metadata_missing_count = 0
    for candidate in candidates.values():
        sources_set = set(candidate["sources"])
        paper_metadata = metadata.get(candidate["paper_id"], {})
        title = str(paper_metadata.get("title") or "").strip()
        abstract = str(paper_metadata.get("abstract") or "").strip()
        labelable = bool(title or abstract)
        if not labelable:
            metadata_missing_count += 1
        benchmark_idx = int(candidate["benchmark_idx"])
        is_ground_truth = candidate["paper_id"] in gt_by_idx.get(benchmark_idx, set())
        if bool(candidate["row_marked_ground_truth"]) != is_ground_truth and benchmark_idx in gt_by_idx:
            raise ValueError(
                f"Ground-truth mismatch for idx={benchmark_idx}, paper={candidate['paper_id']}"
            )
        partition = _source_partition(sources_set)
        partition_counts[partition] += 1
        finalized_candidates.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": candidate["candidate_id"],
                "benchmark_idx": benchmark_idx,
                "query_id": candidate["query_id"],
                "paper_id": candidate["paper_id"],
                "sources": sorted(sources_set),
                "source_partition": partition,
                "source_stats": {
                    source: _finalize_source_stats(stats)
                    for source, stats in sorted(candidate["source_stats"].items())
                },
                "is_ground_truth": is_ground_truth,
                "title": title,
                "abstract": abstract,
                "paper_date": paper_metadata.get("date"),
                "categories": list(paper_metadata.get("categories") or []),
                "metadata_available": labelable,
            }
        )
    finalized_candidates.sort(
        key=lambda value: (value["benchmark_idx"], value["query_id"], value["paper_id"])
    )

    manifest_dir = work_dir / "manifest"
    input_rubric_dir = work_dir / "inputs" / "rubrics"
    input_candidate_dir = work_dir / "inputs" / "candidates"
    schema_dir = work_dir / "schemas"
    prompt_dir = work_dir / "prompts"
    for directory in (manifest_dir, input_rubric_dir, input_candidate_dir, schema_dir, prompt_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_jsonl(manifest_dir / "queries.jsonl", finalized_queries)
    _write_jsonl(manifest_dir / "candidates.jsonl", finalized_candidates)
    _atomic_write_json(schema_dir / "rubric.schema.json", RUBRIC_SCHEMA)
    _atomic_write_json(schema_dir / "candidate_batch.schema.json", CANDIDATE_BATCH_SCHEMA)
    _atomic_write_text(prompt_dir / "rubric_instructions.txt", RUBRIC_INSTRUCTIONS)
    _atomic_write_text(prompt_dir / "candidate_instructions.txt", CANDIDATE_INSTRUCTIONS)

    query_by_id = {query["query_id"]: query for query in finalized_queries}
    rubric_manifest: List[Dict[str, Any]] = []
    for query in finalized_queries:
        blinded = {
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
            "query_id": query["query_id"],
            "query": query["query"],
            "query_date": query.get("query_date"),
            "planner_checklists": query["planner_checklists"],
            "subqueries": query["subqueries"],
        }
        input_sha = _sha256_json(blinded)
        blinded["input_sha256"] = input_sha
        input_path = input_rubric_dir / f"{query['query_token']}.json"
        _atomic_write_json(input_path, blinded)
        rubric_manifest.append(
            {
                "query_id": query["query_id"],
                "query_token": query["query_token"],
                "input_path": _relative(input_path, work_dir),
                "input_sha256": input_sha,
            }
        )
    _write_jsonl(manifest_dir / "rubric_jobs.jsonl", rubric_manifest)

    candidates_by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in finalized_candidates:
        if candidate["metadata_available"]:
            candidates_by_query[candidate["query_id"]].append(candidate)

    batch_manifest: List[Dict[str, Any]] = []
    for query in finalized_queries:
        query_candidates = candidates_by_query.get(query["query_id"], [])
        for batch_index, start in enumerate(range(0, len(query_candidates), args.batch_size), 1):
            chunk = query_candidates[start : start + args.batch_size]
            batch_id = f"{query['query_token']}-b{batch_index:05d}"
            blinded_candidates = []
            for candidate in chunk:
                abstract = candidate["abstract"]
                truncated = len(abstract) > args.max_abstract_chars
                if truncated:
                    abstract = abstract[: args.max_abstract_chars].rstrip() + " …"
                blinded_candidates.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "title": candidate["title"],
                        "abstract": abstract,
                        "abstract_truncated": truncated,
                        "paper_date": candidate.get("paper_date"),
                        "categories": candidate.get("categories") or [],
                    }
                )
            blinded_batch = {
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "batch_id": batch_id,
                "query_id": query["query_id"],
                "query": query["query"],
                "query_date": query.get("query_date"),
                "candidates": blinded_candidates,
            }
            input_sha = _sha256_json(blinded_batch)
            blinded_batch["input_sha256"] = input_sha
            input_path = input_candidate_dir / f"{batch_id}.json"
            _atomic_write_json(input_path, blinded_batch)
            batch_manifest.append(
                {
                    "batch_id": batch_id,
                    "query_id": query["query_id"],
                    "query_token": query["query_token"],
                    "input_path": _relative(input_path, work_dir),
                    "input_sha256": input_sha,
                    "candidate_count": len(chunk),
                    "candidate_ids": [candidate["candidate_id"] for candidate in chunk],
                }
            )
    _write_jsonl(manifest_dir / "candidate_jobs.jsonl", batch_manifest)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "run_dir": str(run_dir),
        "paper_db": str(paper_db),
        "work_dir": str(work_dir),
        "sources": list(sources),
        "source_row_counts": dict(source_row_counts),
        "query_count": len(finalized_queries),
        "candidate_count": len(finalized_candidates),
        "unique_paper_count": len(needed_paper_ids),
        "partition_counts": dict(partition_counts),
        "metadata_missing_count": metadata_missing_count,
        "codex_candidate_count": len(finalized_candidates) - metadata_missing_count,
        "rubric_job_count": len(rubric_manifest),
        "candidate_batch_count": len(batch_manifest),
        "batch_size": args.batch_size,
        "max_abstract_chars": args.max_abstract_chars,
        "candidate_scope": "query-paper union of all saved full source artifacts; no rerank or Top-K cutoff",
        "annotation_blinding": "Codex inputs exclude retrieval source, rank, graph path, selector output, and ground truth",
    }
    _atomic_write_json(manifest_dir / "prepare_summary.json", summary)
    _atomic_write_json(
        work_dir / "run_config.json",
        {
            **summary,
            "rubric_schema_sha256": _sha256_json(RUBRIC_SCHEMA),
            "candidate_schema_sha256": _sha256_json(CANDIDATE_BATCH_SCHEMA),
            "rubric_prompt_sha256": hashlib.sha256(RUBRIC_INSTRUCTIONS.encode("utf-8")).hexdigest(),
            "candidate_prompt_sha256": hashlib.sha256(
                CANDIDATE_INSTRUCTIONS.encode("utf-8")
            ).hexdigest(),
        },
    )
    return summary


def refresh_prompts(args: argparse.Namespace) -> Dict[str, Any]:
    """Refresh prompt snapshots and input signatures without rescanning artifacts."""

    work_dir = Path(args.work_dir).resolve()
    manifest_dir = work_dir / "manifest"
    rubric_jobs = list(_iter_jsonl(manifest_dir / "rubric_jobs.jsonl"))
    candidate_jobs = list(_iter_jsonl(manifest_dir / "candidate_jobs.jsonl"))

    for jobs in (rubric_jobs, candidate_jobs):
        for job in jobs:
            input_path = work_dir / job["input_path"]
            payload = _load_json(input_path)
            payload["prompt_version"] = PROMPT_VERSION
            payload.pop("input_sha256", None)
            input_sha = _sha256_json(payload)
            payload["input_sha256"] = input_sha
            job["input_sha256"] = input_sha
            _atomic_write_json(input_path, payload)

    _write_jsonl(manifest_dir / "rubric_jobs.jsonl", rubric_jobs)
    _write_jsonl(manifest_dir / "candidate_jobs.jsonl", candidate_jobs)
    _atomic_write_text(work_dir / "prompts" / "rubric_instructions.txt", RUBRIC_INSTRUCTIONS)
    _atomic_write_text(work_dir / "prompts" / "candidate_instructions.txt", CANDIDATE_INSTRUCTIONS)

    prepare_summary = _load_json(manifest_dir / "prepare_summary.json")
    prepare_summary["prompt_version"] = PROMPT_VERSION
    _atomic_write_json(manifest_dir / "prepare_summary.json", prepare_summary)
    run_config = _load_json(work_dir / "run_config.json")
    run_config.update(
        {
            "prompt_version": PROMPT_VERSION,
            "rubric_prompt_sha256": hashlib.sha256(RUBRIC_INSTRUCTIONS.encode("utf-8")).hexdigest(),
            "candidate_prompt_sha256": hashlib.sha256(
                CANDIDATE_INSTRUCTIONS.encode("utf-8")
            ).hexdigest(),
        }
    )
    _atomic_write_json(work_dir / "run_config.json", run_config)
    return {
        "prompt_version": PROMPT_VERSION,
        "rubric_job_count": len(rubric_jobs),
        "candidate_job_count": len(candidate_jobs),
        "existing_outputs_are_reused_only_if_their_updated_signatures_validate": True,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _exclusive_none(values: Sequence[str]) -> bool:
    return "none" not in values or len(values) == 1


def canonicalize_annotation_fields(annotation: Mapping[str, Any]) -> Dict[str, Any]:
    """Return one annotation with canonical public field names.

    Completed v1 batches used four implementation-oriented field names.  Read
    those aliases for backward compatibility, but always remove them from the
    canonical annotation written to analysis outputs.
    """

    canonical = dict(annotation)
    for public_name, legacy_name in ANNOTATION_FIELD_ALIASES.items():
        if public_name not in canonical and legacy_name in canonical:
            canonical[public_name] = canonical[legacy_name]
        canonical.pop(legacy_name, None)
    return canonical


def _validate_rubric(value: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("rubric schema_version mismatch")
    if value.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("rubric prompt_version mismatch")
    if value.get("query_id") != job["query_id"]:
        raise ValueError("rubric query_id mismatch")
    if value.get("input_sha256") != job["input_sha256"]:
        raise ValueError("rubric input_sha256 mismatch")
    aspects = value.get("aspects")
    if not isinstance(aspects, list) or not 2 <= len(aspects) <= 8:
        raise ValueError("rubric must contain 2-8 aspects")
    expected_ids = [f"A{index}" for index in range(1, len(aspects) + 1)]
    actual_ids = [aspect.get("aspect_id") for aspect in aspects if isinstance(aspect, Mapping)]
    if actual_ids != expected_ids:
        raise ValueError(f"rubric aspect IDs must be contiguous: {expected_ids}")


def _validate_candidate_batch(
    value: Mapping[str, Any],
    job: Mapping[str, Any],
    valid_aspect_ids: Set[str],
) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("candidate schema_version mismatch")
    if value.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("candidate prompt_version mismatch")
    if value.get("batch_id") != job["batch_id"]:
        raise ValueError("candidate batch_id mismatch")
    if value.get("input_sha256") != job["input_sha256"]:
        raise ValueError("candidate input_sha256 mismatch")
    annotations = value.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("annotations must be an array")
    ids = [annotation.get("candidate_id") for annotation in annotations if isinstance(annotation, Mapping)]
    expected_ids = list(job["candidate_ids"])
    if len(ids) != len(annotations) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("candidate IDs are missing, duplicated, or unexpected")
    for raw_annotation in annotations:
        annotation = canonicalize_annotation_fields(raw_annotation)
        if annotation.get("relevance_grade") not in RELEVANCE_GRADES:
            raise ValueError("invalid relevance_grade")
        if annotation.get("semantic_distance") not in SEMANTIC_DISTANCES:
            raise ValueError("invalid semantic_distance")
        if annotation.get("paper_type") not in PAPER_TYPES:
            raise ValueError("invalid paper_type")
        if annotation.get("confidence") not in CONFIDENCE_VALUES:
            raise ValueError("invalid confidence")
        aspect_ids = annotation.get("matched_aspect_ids") or []
        if not isinstance(aspect_ids, list) or not set(aspect_ids).issubset(valid_aspect_ids):
            raise ValueError("invalid matched_aspect_ids")
        if len(annotation.get("key_concepts") or []) > 5:
            raise ValueError("too many key_concepts")
        if len(annotation.get("evidence_phrases") or []) > 2:
            raise ValueError("too many evidence_phrases")
        for key, allowed in (
            ("scholarly_roles", SCHOLARLY_ROLES),
            ("information_added", INFORMATION_ADDED_VALUES),
            ("exclusion_reasons", EXCLUSION_REASONS),
        ):
            values = annotation.get(key)
            if not isinstance(values, list) or not values or len(values) != len(set(values)):
                raise ValueError(f"{key} must be a non-empty unique array")
            if not set(values).issubset(set(allowed)) or not _exclusive_none(values):
                raise ValueError(f"invalid {key}")


def _parse_usage(stdout: str) -> Dict[str, Any]:
    usage: Dict[str, Any] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), Mapping):
            usage = dict(event["usage"])
    return usage


def _codex_command(args: argparse.Namespace, schema_path: Path, output_path: Path) -> List[str]:
    command = [
        args.codex_bin,
        "exec",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ignore-rules",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "--json",
        "--color",
        "never",
    ]
    if not args.load_user_config:
        command.append("--ignore-user-config")
    if args.model:
        command.extend(["--model", args.model])
    if args.reasoning_effort:
        command.extend(["--config", f'model_reasoning_effort="{args.reasoning_effort}"'])
    command.append("-")
    return command


def _resolve_codex_bin(value: str) -> str:
    """Resolve Codex in non-login shells, including common NVM installations."""

    explicit = Path(value).expanduser()
    if explicit.parent != Path(".") and explicit.is_file():
        return str(explicit.resolve())
    resolved = shutil.which(value)
    if resolved:
        return resolved
    candidates = sorted(
        (Path.home() / ".nvm" / "versions" / "node").glob(f"*/bin/{value}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0].resolve())
    raise FileNotFoundError(
        f"Could not resolve Codex executable {value!r}; pass --codex-bin with an absolute path"
    )


def _run_codex_job(
    *,
    args: argparse.Namespace,
    work_dir: Path,
    kind: str,
    job_id: str,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    validator: Any,
) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = work_dir / "logs" / kind
    meta_dir = work_dir / "outputs" / "meta" / kind
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    isolated_cwd = Path(tempfile.gettempdir()) / "scholargym-candidate-annotation-codex"
    isolated_cwd.mkdir(parents=True, exist_ok=True)

    last_error = "unknown error"
    for attempt in range(1, args.max_attempts + 1):
        temporary_output = output_path.with_name(
            f".{output_path.name}.tmp-{os.getpid()}-{threading.get_ident()}-{attempt}"
        )
        if temporary_output.exists():
            temporary_output.unlink()
        command = _codex_command(args, schema_path, temporary_output)
        started = time.time()
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=isolated_cwd,
                timeout=args.timeout_seconds,
                check=False,
            )
            duration = time.time() - started
            log = {
                "job_id": job_id,
                "kind": kind,
                "attempt": attempt,
                "returncode": completed.returncode,
                "duration_seconds": duration,
                "command": command[:-1] + ["<stdin>"],
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "usage": _parse_usage(completed.stdout),
            }
            _atomic_write_json(log_dir / f"{job_id}.attempt-{attempt}.json", log)
            if completed.returncode != 0:
                last_error = f"codex exited {completed.returncode}"
            elif not temporary_output.exists():
                last_error = "codex did not write the final output"
            else:
                try:
                    value = _load_json(temporary_output)
                    validator(value)
                except Exception as exc:  # validation failure is retryable
                    last_error = f"invalid output: {exc}"
                else:
                    os.replace(temporary_output, output_path)
                    _atomic_write_json(
                        meta_dir / f"{job_id}.json",
                        {
                            "job_id": job_id,
                            "kind": kind,
                            "attempt": attempt,
                            "duration_seconds": duration,
                            "usage": log["usage"],
                            "model": args.model,
                            "reasoning_effort": args.reasoning_effort,
                        },
                    )
                    return {"job_id": job_id, "status": "completed", "attempt": attempt}
        except subprocess.TimeoutExpired as exc:
            duration = time.time() - started
            last_error = f"timeout after {args.timeout_seconds}s"
            _atomic_write_json(
                log_dir / f"{job_id}.attempt-{attempt}.json",
                {
                    "job_id": job_id,
                    "kind": kind,
                    "attempt": attempt,
                    "duration_seconds": duration,
                    "error": last_error,
                    "stdout": exc.stdout or "",
                    "stderr": exc.stderr or "",
                },
            )
        finally:
            if temporary_output.exists():
                temporary_output.unlink()
        if attempt < args.max_attempts:
            time.sleep(min(20, 2 ** attempt))
    return {"job_id": job_id, "status": "failed", "error": last_error}


def _valid_existing_rubric(path: Path, job: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        _validate_rubric(_load_json(path), job)
    except Exception:
        return False
    return True


def _valid_existing_candidate(
    path: Path, job: Mapping[str, Any], valid_aspect_ids: Set[str]
) -> bool:
    if not path.exists():
        return False
    try:
        _validate_candidate_batch(_load_json(path), job, valid_aspect_ids)
    except Exception:
        return False
    return True


def _run_jobs_concurrently(
    jobs: Sequence[Mapping[str, Any]], worker: Any, workers: int
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if workers <= 0:
        raise ValueError("--workers must be positive")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(worker, job): job for job in jobs}
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "job_id": job.get("batch_id") or job.get("query_id"),
                    "status": "failed",
                    "error": str(exc),
                }
            results.append(result)
            completed_count += 1
            print(
                f"[{completed_count}/{len(jobs)}] {result.get('job_id')}: {result.get('status')}",
                file=sys.stderr,
                flush=True,
            )
    return results


def annotate_rubrics(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    args.codex_bin = _resolve_codex_bin(args.codex_bin)
    jobs = list(_iter_jsonl(work_dir / "manifest" / "rubric_jobs.jsonl"))
    if args.query_id:
        jobs = [job for job in jobs if job["query_id"] == args.query_id]
    output_dir = work_dir / "outputs" / "rubrics"
    pending = [
        job
        for job in jobs
        if not _valid_existing_rubric(output_dir / f"{job['query_token']}.json", job)
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    def worker(job: Mapping[str, Any]) -> Dict[str, Any]:
        payload = _load_json(work_dir / job["input_path"])
        prompt = RUBRIC_INSTRUCTIONS + "\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        output_path = output_dir / f"{job['query_token']}.json"
        return _run_codex_job(
            args=args,
            work_dir=work_dir,
            kind="rubrics",
            job_id=job["query_token"],
            prompt=prompt,
            schema_path=work_dir / "schemas" / "rubric.schema.json",
            output_path=output_path,
            validator=lambda value: _validate_rubric(value, job),
        )

    results = _run_jobs_concurrently(pending, worker, args.workers) if pending else []
    summary = {
        "kind": "rubrics",
        "job_count": len(jobs),
        "pending_selected_count": len(pending),
        "completed_count": sum(result["status"] == "completed" for result in results),
        "failed_count": sum(result["status"] == "failed" for result in results),
        "results": results,
    }
    _atomic_write_json(work_dir / "run_summaries" / "annotate_rubrics_latest.json", summary)
    return summary


def _load_rubrics(work_dir: Path) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for job in _iter_jsonl(work_dir / "manifest" / "rubric_jobs.jsonl"):
        path = work_dir / "outputs" / "rubrics" / f"{job['query_token']}.json"
        if not _valid_existing_rubric(path, job):
            continue
        output[job["query_id"]] = _load_json(path)
    return output


def annotate_candidates(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    args.codex_bin = _resolve_codex_bin(args.codex_bin)
    rubrics = _load_rubrics(work_dir)
    jobs = list(_iter_jsonl(work_dir / "manifest" / "candidate_jobs.jsonl"))
    if args.query_id:
        jobs = [job for job in jobs if job["query_id"] == args.query_id]
    missing_rubric_queries = sorted({job["query_id"] for job in jobs if job["query_id"] not in rubrics})
    if missing_rubric_queries:
        raise RuntimeError(
            "Missing valid rubrics for queries: " + ", ".join(missing_rubric_queries[:10])
        )
    output_dir = work_dir / "outputs" / "candidates"
    pending: List[Dict[str, Any]] = []
    for job in jobs:
        aspect_ids = {aspect["aspect_id"] for aspect in rubrics[job["query_id"]]["aspects"]}
        output_path = output_dir / f"{job['batch_id']}.json"
        if not _valid_existing_candidate(output_path, job, aspect_ids):
            pending.append(job)
    if args.limit is not None:
        pending = pending[: args.limit]

    def worker(job: Mapping[str, Any]) -> Dict[str, Any]:
        payload = _load_json(work_dir / job["input_path"])
        rubric = rubrics[job["query_id"]]
        prompt_payload = {"rubric": rubric, "batch": payload}
        prompt = CANDIDATE_INSTRUCTIONS + "\nINPUT:\n" + json.dumps(
            prompt_payload, ensure_ascii=False, indent=2
        )
        aspect_ids = {aspect["aspect_id"] for aspect in rubric["aspects"]}
        output_path = output_dir / f"{job['batch_id']}.json"
        return _run_codex_job(
            args=args,
            work_dir=work_dir,
            kind="candidates",
            job_id=job["batch_id"],
            prompt=prompt,
            schema_path=work_dir / "schemas" / "candidate_batch.schema.json",
            output_path=output_path,
            validator=lambda value: _validate_candidate_batch(value, job, aspect_ids),
        )

    results = _run_jobs_concurrently(pending, worker, args.workers) if pending else []
    summary = {
        "kind": "candidates",
        "job_count": len(jobs),
        "pending_selected_count": len(pending),
        "completed_count": sum(result["status"] == "completed" for result in results),
        "failed_count": sum(result["status"] == "failed" for result in results),
        "results": results,
    }
    _atomic_write_json(work_dir / "run_summaries" / "annotate_candidates_latest.json", summary)
    return summary


def _automatic_missing_metadata_annotation(candidate_id: str) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "relevance_grade": "insufficient_evidence",
        "semantic_distance": "unknown",
        "paper_type": "unknown",
        "matched_aspect_ids": [],
        "scholarly_roles": ["none"],
        "information_added": ["none"],
        "information_summary": "No title or abstract is available for semantic annotation.",
        "key_concepts": [],
        "exclusion_reasons": ["insufficient_evidence"],
        "evidence_phrases": [],
        "needs_full_text": True,
        "confidence": 0.25,
    }


def _analysis_groups(candidate: Mapping[str, Any]) -> List[str]:
    sources = set(candidate.get("sources") or [])
    groups = ["all", f"partition:{candidate['source_partition']}"]
    for source in DEFAULT_SOURCES:
        if source in sources:
            groups.append(f"source:{source}")
    if "deep_event" in sources or "deep_merged" in sources:
        groups.append("source:deep_any")
    return groups


def _safe_rate(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def aggregate(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    rubrics = _load_rubrics(work_dir)
    jobs = list(_iter_jsonl(work_dir / "manifest" / "candidate_jobs.jsonl"))
    annotations_by_id: Dict[str, Dict[str, Any]] = {}
    invalid_or_missing_batches: List[str] = []
    for job in jobs:
        rubric = rubrics.get(job["query_id"])
        if rubric is None:
            invalid_or_missing_batches.append(job["batch_id"])
            continue
        aspect_ids = {aspect["aspect_id"] for aspect in rubric["aspects"]}
        path = work_dir / "outputs" / "candidates" / f"{job['batch_id']}.json"
        if not _valid_existing_candidate(path, job, aspect_ids):
            invalid_or_missing_batches.append(job["batch_id"])
            continue
        for raw_annotation in _load_json(path)["annotations"]:
            annotation = canonicalize_annotation_fields(raw_annotation)
            candidate_id = annotation["candidate_id"]
            if candidate_id in annotations_by_id:
                raise ValueError(f"Duplicate candidate annotation: {candidate_id}")
            annotations_by_id[candidate_id] = annotation
    if invalid_or_missing_batches and not args.allow_incomplete:
        raise RuntimeError(
            f"{len(invalid_or_missing_batches)} candidate batches are missing or invalid; "
            "rerun annotate-candidates or pass --allow-incomplete for a progress snapshot"
        )

    analysis_dir = work_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = analysis_dir / "annotations.jsonl"
    annotation_tmp = annotation_path.with_name(f".{annotation_path.name}.tmp-{os.getpid()}")

    group_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    query_group_counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    label_counts: Dict[Tuple[str, str, str], int] = Counter()
    candidate_count = 0
    codex_annotated_count = 0
    automatic_count = 0
    pending_count = 0

    with annotation_tmp.open("w", encoding="utf-8") as output_handle:
        for candidate in _iter_jsonl(work_dir / "manifest" / "candidates.jsonl"):
            candidate_count += 1
            annotation = annotations_by_id.get(candidate["candidate_id"])
            if annotation is not None:
                annotation_status = "codex"
                codex_annotated_count += 1
            elif not candidate.get("metadata_available"):
                annotation = _automatic_missing_metadata_annotation(candidate["candidate_id"])
                annotation_status = "automatic_insufficient_metadata"
                automatic_count += 1
            else:
                annotation_status = "pending"
                pending_count += 1

            output_row = {
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "candidate_id": candidate["candidate_id"],
                "benchmark_idx": candidate["benchmark_idx"],
                "query_id": candidate["query_id"],
                "paper_id": candidate["paper_id"],
                "sources": candidate["sources"],
                "source_partition": candidate["source_partition"],
                "is_ground_truth": candidate["is_ground_truth"],
                "metadata_available": candidate["metadata_available"],
                "annotation_status": annotation_status,
                "annotation": annotation,
            }
            output_handle.write(json.dumps(output_row, ensure_ascii=False, sort_keys=True) + "\n")
            if annotation is None:
                continue

            for group in _analysis_groups(candidate):
                counts = group_counts[group]
                query_counts = query_group_counts[(group, candidate["query_id"])]
                for target in (counts, query_counts):
                    target["candidate_count"] += 1
                    target[f"relevance_grade:{annotation['relevance_grade']}"] += 1
                    target[f"distance:{annotation['semantic_distance']}"] += 1
                    target[f"paper_type:{annotation['paper_type']}"] += 1
                    target["ground_truth_count"] += int(candidate["is_ground_truth"])
                    target["confidence_sum"] += float(annotation["confidence"])
                    if annotation["relevance_grade"] in ("direct", "partial"):
                        target["direct_or_partial_count"] += 1
                    if annotation["relevance_grade"] in (
                        "direct",
                        "partial",
                        "contextual",
                    ):
                        target["information_bearing_count"] += 1
                    if annotation["relevance_grade"] in RELEVANCE_SCORES:
                        target["scored_count"] += 1
                        target["relevance_score_sum"] += RELEVANCE_SCORES[
                            annotation["relevance_grade"]
                        ]
                for dimension, values in (
                    ("scholarly_role", annotation["scholarly_roles"]),
                    ("information_added", annotation["information_added"]),
                    ("exclusion_reason", annotation["exclusion_reasons"]),
                    ("matched_aspect", annotation["matched_aspect_ids"]),
                ):
                    for value in values:
                        label_counts[(group, dimension, value)] += 1
    os.replace(annotation_tmp, annotation_path)

    source_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []
    for group, counts in sorted(group_counts.items()):
        query_keys = [key for key in query_group_counts if key[0] == group]
        macro_fields = {
            "macro_direct_rate": [],
            "macro_direct_or_partial_rate": [],
            "macro_information_bearing_rate": [],
            "macro_relevance_score": [],
        }
        for _, query_id in query_keys:
            q = query_group_counts[(group, query_id)]
            total = q["candidate_count"]
            scored = q["scored_count"]
            query_row = {
                "group": group,
                "query_id": query_id,
                "candidate_count": total,
                "direct_rate": _safe_rate(q["relevance_grade:direct"], total),
                "direct_or_partial_rate": _safe_rate(q["direct_or_partial_count"], total),
                "information_bearing_rate": _safe_rate(q["information_bearing_count"], total),
                "mean_relevance_score": _safe_rate(q["relevance_score_sum"], scored),
            }
            query_rows.append(query_row)
            for target, source in (
                ("macro_direct_rate", "direct_rate"),
                ("macro_direct_or_partial_rate", "direct_or_partial_rate"),
                ("macro_information_bearing_rate", "information_bearing_rate"),
                ("macro_relevance_score", "mean_relevance_score"),
            ):
                if query_row[source] is not None:
                    macro_fields[target].append(query_row[source])

        total = counts["candidate_count"]
        scored = counts["scored_count"]
        row: Dict[str, Any] = {
            "group": group,
            "query_count": len(query_keys),
            "candidate_count": total,
            "ground_truth_count": counts["ground_truth_count"],
            "ground_truth_rate": _safe_rate(counts["ground_truth_count"], total),
            "direct_count": counts["relevance_grade:direct"],
            "partial_count": counts["relevance_grade:partial"],
            "contextual_count": counts["relevance_grade:contextual"],
            "unrelated_count": counts["relevance_grade:unrelated"],
            "insufficient_evidence_count": counts[
                "relevance_grade:insufficient_evidence"
            ],
            "direct_rate": _safe_rate(counts["relevance_grade:direct"], total),
            "direct_or_partial_rate": _safe_rate(counts["direct_or_partial_count"], total),
            "information_bearing_rate": _safe_rate(counts["information_bearing_count"], total),
            "mean_relevance_score": _safe_rate(counts["relevance_score_sum"], scored),
            "mean_confidence": _safe_rate(counts["confidence_sum"], total),
        }
        for field, values in macro_fields.items():
            row[field] = sum(values) / len(values) if values else None
        source_rows.append(row)

    label_rows: List[Dict[str, Any]] = []
    for (group, dimension, label), count in sorted(label_counts.items()):
        denominator = group_counts[group]["candidate_count"]
        label_rows.append(
            {
                "group": group,
                "dimension": dimension,
                "label": label,
                "paper_count": count,
                "paper_rate": _safe_rate(count, denominator),
            }
        )

    _write_jsonl(analysis_dir / "query_summary.jsonl", query_rows)
    _write_jsonl(analysis_dir / "source_comparison.jsonl", source_rows)
    _write_jsonl(analysis_dir / "label_distribution.jsonl", label_rows)
    _write_csv(analysis_dir / "source_comparison.csv", source_rows)
    _write_csv(analysis_dir / "label_distribution.csv", label_rows)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "candidate_scope": "complete saved artifact union before rerank/Top-K",
        "candidate_count": candidate_count,
        "codex_annotated_count": codex_annotated_count,
        "automatic_insufficient_metadata_count": automatic_count,
        "pending_candidate_count": pending_count,
        "invalid_or_missing_batch_count": len(invalid_or_missing_batches),
        "complete": pending_count == 0 and not invalid_or_missing_batches,
        "source_comparison": source_rows,
    }
    _atomic_write_json(analysis_dir / "summary.json", summary)
    return summary


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def status(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    rubric_jobs = list(_iter_jsonl(work_dir / "manifest" / "rubric_jobs.jsonl"))
    candidate_jobs = list(_iter_jsonl(work_dir / "manifest" / "candidate_jobs.jsonl"))
    rubrics = _load_rubrics(work_dir)
    valid_candidate_jobs = 0
    annotated_candidate_count = 0
    for job in candidate_jobs:
        rubric = rubrics.get(job["query_id"])
        if rubric is None:
            continue
        aspect_ids = {aspect["aspect_id"] for aspect in rubric["aspects"]}
        path = work_dir / "outputs" / "candidates" / f"{job['batch_id']}.json"
        if _valid_existing_candidate(path, job, aspect_ids):
            valid_candidate_jobs += 1
            annotated_candidate_count += int(job["candidate_count"])
    prepare_summary = _load_json(work_dir / "manifest" / "prepare_summary.json")
    output = {
        "query_count": prepare_summary["query_count"],
        "candidate_count": prepare_summary["candidate_count"],
        "rubric_jobs_total": len(rubric_jobs),
        "rubric_jobs_valid": len(rubrics),
        "candidate_jobs_total": len(candidate_jobs),
        "candidate_jobs_valid": valid_candidate_jobs,
        "codex_candidates_total": prepare_summary["codex_candidate_count"],
        "codex_candidates_annotated": annotated_candidate_count,
        "metadata_missing_count": prepare_summary["metadata_missing_count"],
    }
    output["complete"] = (
        output["rubric_jobs_valid"] == output["rubric_jobs_total"]
        and output["candidate_jobs_valid"] == output["candidate_jobs_total"]
    )
    return output


def wait_and_aggregate(args: argparse.Namespace) -> Dict[str, Any]:
    """Wait for a local Linux annotation PID, then run strict aggregation."""

    if args.pid <= 0:
        raise ValueError("--pid must be positive")
    poll_seconds = min(max(args.poll_seconds, 1), 60)
    process_path = Path("/proc") / str(args.pid)
    started = time.time()
    while process_path.exists():
        time.sleep(poll_seconds)
    result = aggregate(SimpleNamespace(work_dir=args.work_dir, allow_incomplete=False))
    result["waited_for_pid"] = args.pid
    result["wait_duration_seconds"] = time.time() - started
    return result


def _add_codex_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh"),
        default="medium",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-id", default=None)
    parser.add_argument(
        "--load-user-config",
        action="store_true",
        help="Load ~/.codex/config.toml; by default only saved CLI auth is reused.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Build blinded manifests and batches")
    prepare_parser.add_argument("--run-dir", required=True)
    prepare_parser.add_argument("--paper-db", required=True)
    prepare_parser.add_argument("--work-dir", required=True)
    prepare_parser.add_argument(
        "--sources",
        nargs="+",
        choices=DEFAULT_SOURCES,
        default=list(DEFAULT_SOURCES),
    )
    prepare_parser.add_argument("--batch-size", type=int, default=20)
    prepare_parser.add_argument("--max-abstract-chars", type=int, default=6000)
    prepare_parser.add_argument("--allow-missing-source", action="store_true")

    refresh_parser = subparsers.add_parser(
        "refresh-prompts", help="Refresh prompt/input versions without rescanning the paper DB"
    )
    refresh_parser.add_argument("--work-dir", required=True)

    rubric_parser = subparsers.add_parser("annotate-rubrics", help="Run Codex query-rubric jobs")
    _add_codex_arguments(rubric_parser)

    candidate_parser = subparsers.add_parser(
        "annotate-candidates", help="Run Codex blinded candidate-batch jobs"
    )
    _add_codex_arguments(candidate_parser)

    aggregate_parser = subparsers.add_parser("aggregate", help="Aggregate labels by source")
    aggregate_parser.add_argument("--work-dir", required=True)
    aggregate_parser.add_argument("--allow-incomplete", action="store_true")

    status_parser = subparsers.add_parser("status", help="Report resumable job progress")
    status_parser.add_argument("--work-dir", required=True)

    watcher_parser = subparsers.add_parser(
        "wait-and-aggregate",
        help="Wait for an annotation PID and then run strict aggregation",
    )
    watcher_parser.add_argument("--work-dir", required=True)
    watcher_parser.add_argument("--pid", required=True, type=int)
    watcher_parser.add_argument("--poll-seconds", type=int, default=60)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "refresh-prompts":
        result = refresh_prompts(args)
    elif args.command == "annotate-rubrics":
        result = annotate_rubrics(args)
    elif args.command == "annotate-candidates":
        result = annotate_candidates(args)
    elif args.command == "aggregate":
        result = aggregate(args)
    elif args.command == "status":
        result = status(args)
    elif args.command == "wait-and-aggregate":
        result = wait_and_aggregate(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
