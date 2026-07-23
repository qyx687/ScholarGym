#!/usr/bin/env python3
"""Refine three coarse semantic-complement labels with blind Codex annotation.

The parent first-stage annotation remains immutable.  This second stage selects
information-bearing candidates from configurable source partitions
(``graph_only``, ``deep_merged_only``, and/or ``graph_and_deep_merged``) that
carry at least one of:

* ``historical_context``
* ``mechanism_or_theory``
* ``application_domain``

Retrieval provenance and ground truth are excluded from Codex inputs.  The
fixed output axes distinguish predecessor/foundation/background, explicit or
implicit mechanisms, and same/adjacent/cross-domain relations.  Every
applicable axis permits ``insufficient_evidence``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import annotate_full_candidate_pool_with_codex as base  # noqa: E402


SCHEMA_VERSION = "semantic_relation_refinement_v1"
PROMPT_VERSION = "semantic_relation_refinement_prompt_v1"
INFORMATION_RELATIONSHIPS = {"direct", "partial", "contextual"}
CONFIDENCE_VALUES = (0.25, 0.5, 0.75, 1.0)
PRIMARY_PARTITIONS = (
    "graph_only",
    "deep_merged_only",
    "graph_and_deep_merged",
)
DEFAULT_PARTITIONS = ("graph_only", "deep_merged_only")

AXIS_SPECS: Mapping[str, Mapping[str, Any]] = {
    "historical_relation": {
        "trigger": ("information_added", "historical_context"),
        "labels": (
            "direct_predecessor",
            "enabling_foundation",
            "historical_background",
            "retrospective_or_survey",
            "insufficient_evidence",
            "not_applicable",
        ),
    },
    "mechanism_relation": {
        "trigger": ("information_added", "mechanism_or_theory"),
        "labels": (
            "explicit_target_mechanism",
            "implicit_explanatory_mechanism",
            "generic_theory",
            "insufficient_evidence",
            "not_applicable",
        ),
    },
    "domain_relation": {
        "trigger": ("information_added", "application_domain"),
        "labels": (
            "same_domain",
            "adjacent_domain",
            "cross_domain_transfer",
            "unrelated_domain_drift",
            "insufficient_evidence",
            "not_applicable",
        ),
    },
}


def _axis_schema(labels: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": list(labels)},
            "rationale": {"type": "string"},
            "evidence_phrases": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 2,
            },
            "needs_full_text": {"type": "boolean"},
            "confidence": {"type": "number", "enum": list(CONFIDENCE_VALUES)},
        },
        "required": [
            "label",
            "rationale",
            "evidence_phrases",
            "needs_full_text",
            "confidence",
        ],
        "additionalProperties": False,
    }


RELATION_BATCH_SCHEMA: Dict[str, Any] = {
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
                    **{
                        axis: _axis_schema(spec["labels"])
                        for axis, spec in AXIS_SPECS.items()
                    },
                },
                "required": ["candidate_id", *AXIS_SPECS.keys()],
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


RELATION_INSTRUCTIONS = """You are performing a blinded second-stage annotation of scientific papers.

The first stage identified one or more coarse semantic contribution types. Your task is to refine only the axes listed in each candidate's applicable_axes. You are not told whether a paper came from graph expansion or semantic retrieval. Do not inspect files, call tools, browse, or infer retrieval provenance. Use only the supplied query, rubric, title, abstract, paper date/categories, and first-stage semantic annotation.

General rules:
- For an axis listed in applicable_axes, choose one substantive label or insufficient_evidence. Never output not_applicable for an applicable axis.
- For an axis absent from applicable_axes, output not_applicable, confidence 1.0, needs_full_text false, an empty evidence_phrases array, and a short rationale saying the coarse label did not trigger this axis.
- Use insufficient_evidence whenever title/abstract evidence cannot distinguish the substantive labels. For insufficient_evidence set needs_full_text true and confidence to 0.25 or 0.5.
- For a substantive label set needs_full_text false. Do not force a fine relation merely because the first-stage coarse label was present.
- Evidence phrases must contain 0-2 short phrases copied or tightly excerpted from the supplied title/abstract. Rationale must be one concise English sentence.

Historical relation (triggered by historical_context):
- direct_predecessor: the paper explicitly introduces an earlier method, concept, dataset, or result in the direct lineage of the query target. Being old, cited, or topically related alone is insufficient.
- enabling_foundation: the paper supplies foundational theory, infrastructure, representation, dataset, or capability that enables the query target but is not itself its direct predecessor.
- historical_background: the paper supplies earlier context, observations, comparisons, or ordinary related background without a demonstrated direct lineage or enabling role.
- retrospective_or_survey: the paper's relevant role is chiefly reviewing, surveying, taxonomizing, or retrospectively recounting the field.
- Publication date alone never proves direct_predecessor.

Mechanism relation (triggered by mechanism_or_theory):
- explicit_target_mechanism: the mechanism or theory is explicitly part of the phenomenon, method, or causal/explanatory relationship requested by the query.
- implicit_explanatory_mechanism: the paper is not a direct answer to the query, but its stated mechanism/theory explicitly explains, enables, constrains, or diagnoses why/how the queried phenomenon or method works or fails.
- generic_theory: the paper contributes general or adjacent theory that is relevant background but lacks an explicit explanatory link to the query target.
- Do not call a mechanism implicit merely because the paper is partial/contextual; the explanatory link must be explicit in the supplied evidence.

Domain relation (triggered by application_domain):
- same_domain: the paper and query operate in the same substantive application domain or population.
- adjacent_domain: the domains are neighboring or analogous, but the paper does not explicitly transfer a method/concept across distinct domains.
- cross_domain_transfer: the abstract explicitly shows a method, representation, dataset, mechanism, or finding originating in one distinct domain being adapted, transferred, or evaluated in another target domain relevant to the query.
- unrelated_domain_drift: the apparent domain connection is actually outside the query scope and does not provide a defensible bridge.
- Different domain vocabulary alone does not prove cross_domain_transfer; both the cross-domain bridge and transfer/application must be evidenced.

Echo schema_version, prompt_version, batch_id, input_sha256, and every candidate_id exactly. Return exactly one annotation per candidate, without duplicates. Return JSON only under the supplied schema.
"""


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    return base._iter_jsonl(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    base._write_jsonl(path, rows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _safe_rate(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _primary_partition(sources: Iterable[str]) -> Optional[str]:
    source_set = set(sources)
    graph = "graph" in source_set
    deep = "deep_merged" in source_set
    if graph and not deep:
        return "graph_only"
    if deep and not graph:
        return "deep_merged_only"
    if graph and deep:
        return "graph_and_deep_merged"
    return None


def _applicable_axes(annotation: Mapping[str, Any]) -> List[str]:
    axes = []
    for axis, spec in AXIS_SPECS.items():
        field, label = spec["trigger"]
        if label in set(annotation.get(field) or []):
            axes.append(axis)
    return axes


def _load_parent_queries(parent_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for query in _iter_jsonl(parent_dir / "manifest" / "queries.jsonl"):
        rubric_path = parent_dir / "outputs" / "rubrics" / f"{query['query_token']}.json"
        rubric = base._load_json(rubric_path)
        if rubric.get("query_id") != query["query_id"]:
            raise ValueError(f"Query/rubric mismatch at {rubric_path}")
        rows.append(
            {
                "query_id": query["query_id"],
                "query_token": query["query_token"],
                "query": query.get("query", ""),
                "query_date": query.get("query_date"),
                "rubric": rubric,
            }
        )
    return rows


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    parent_dir = Path(args.parent_work_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_abstract_chars <= 0:
        raise ValueError("--max-abstract-chars must be positive")
    requested_partitions = tuple(
        getattr(args, "partitions", None) or DEFAULT_PARTITIONS
    )
    invalid_partitions = set(requested_partitions) - set(PRIMARY_PARTITIONS)
    if invalid_partitions:
        raise ValueError(f"Unknown source partitions: {sorted(invalid_partitions)}")
    parent_summary = base._load_json(parent_dir / "analysis" / "summary.json")
    if not parent_summary.get("complete"):
        raise RuntimeError("Second-stage preparation requires a complete first-stage aggregation")
    work_dir.mkdir(parents=True, exist_ok=True)

    query_rows = _load_parent_queries(parent_dir)
    query_by_id = {row["query_id"]: row for row in query_rows}
    selected: List[Dict[str, Any]] = []
    partition_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()
    candidate_iter = _iter_jsonl(parent_dir / "manifest" / "candidates.jsonl")
    annotation_iter = _iter_jsonl(parent_dir / "analysis" / "annotations.jsonl")
    for candidate, annotation_row in itertools.zip_longest(candidate_iter, annotation_iter):
        if candidate is None or annotation_row is None:
            raise ValueError("Parent candidate and annotation files have different lengths")
        if candidate["candidate_id"] != annotation_row["candidate_id"]:
            raise ValueError("Parent candidate and annotation order/IDs do not match")
        partition = _primary_partition(candidate.get("sources") or [])
        if partition is None or partition not in requested_partitions:
            continue
        raw_annotation = annotation_row.get("annotation")
        annotation = (
            base.canonicalize_annotation_fields(raw_annotation)
            if raw_annotation
            else None
        )
        if not annotation or annotation.get("relevance_grade") not in INFORMATION_RELATIONSHIPS:
            continue
        axes = _applicable_axes(annotation)
        if not axes:
            continue
        if candidate["query_id"] not in query_by_id:
            raise ValueError(f"Missing query context for {candidate['query_id']}")
        row = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate["candidate_id"],
            "benchmark_idx": candidate.get("benchmark_idx"),
            "query_id": candidate["query_id"],
            "paper_id": candidate["paper_id"],
            "primary_partition": partition,
            "sources": candidate.get("sources") or [],
            "is_ground_truth": bool(candidate.get("is_ground_truth")),
            "applicable_axes": axes,
            "title": candidate.get("title") or "",
            "abstract": candidate.get("abstract") or "",
            "paper_date": candidate.get("paper_date"),
            "categories": candidate.get("categories") or [],
            "first_stage_annotation": annotation,
        }
        selected.append(row)
        partition_counts[partition] += 1
        for axis in axes:
            axis_counts[axis] += 1

    selected.sort(
        key=lambda row: (
            row.get("benchmark_idx") is None,
            row.get("benchmark_idx") or -1,
            row["query_id"],
            row["paper_id"],
        )
    )
    manifest_dir = work_dir / "manifest"
    input_dir = work_dir / "inputs" / "relations"
    schema_dir = work_dir / "schemas"
    prompt_dir = work_dir / "prompts"
    for directory in (manifest_dir, input_dir, schema_dir, prompt_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest_dir / "queries.jsonl", query_rows)
    _write_jsonl(manifest_dir / "selected_candidates.jsonl", selected)
    base._atomic_write_json(schema_dir / "relation_batch.schema.json", RELATION_BATCH_SCHEMA)
    base._atomic_write_text(prompt_dir / "relation_instructions.txt", RELATION_INSTRUCTIONS)

    selected_by_query: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in selected:
        selected_by_query[row["query_id"]].append(row)
    jobs: List[Dict[str, Any]] = []
    for query in query_rows:
        query_candidates = selected_by_query.get(query["query_id"], [])
        for batch_index, start in enumerate(range(0, len(query_candidates), args.batch_size), 1):
            chunk = query_candidates[start : start + args.batch_size]
            batch_id = f"{query['query_token']}-r{batch_index:05d}"
            blinded_candidates = []
            applicable_by_id = {}
            for candidate in chunk:
                abstract = candidate["abstract"]
                truncated = len(abstract) > args.max_abstract_chars
                if truncated:
                    abstract = abstract[: args.max_abstract_chars].rstrip() + " …"
                first = candidate["first_stage_annotation"]
                blinded_candidates.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "title": candidate["title"],
                        "abstract": abstract,
                        "abstract_truncated": truncated,
                        "paper_date": candidate.get("paper_date"),
                        "categories": candidate.get("categories") or [],
                        "applicable_axes": candidate["applicable_axes"],
                        "first_stage_annotation": {
                            key: first.get(key)
                            for key in (
                                "relevance_grade",
                                "semantic_distance",
                                "paper_type",
                                "matched_aspect_ids",
                                "scholarly_roles",
                                "information_added",
                                "information_summary",
                                "key_concepts",
                                "evidence_phrases",
                            )
                        },
                    }
                )
                applicable_by_id[candidate["candidate_id"]] = candidate["applicable_axes"]
            payload = {
                "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION,
                "batch_id": batch_id,
                "query_id": query["query_id"],
                "query": query["query"],
                "query_date": query.get("query_date"),
                "rubric": query["rubric"],
                "candidates": blinded_candidates,
            }
            input_sha = base._sha256_json(payload)
            payload["input_sha256"] = input_sha
            input_path = input_dir / f"{batch_id}.json"
            base._atomic_write_json(input_path, payload)
            jobs.append(
                {
                    "batch_id": batch_id,
                    "query_id": query["query_id"],
                    "query_token": query["query_token"],
                    "input_path": base._relative(input_path, work_dir),
                    "input_sha256": input_sha,
                    "candidate_count": len(chunk),
                    "candidate_ids": [row["candidate_id"] for row in chunk],
                    "applicable_axes_by_candidate": applicable_by_id,
                }
            )
    _write_jsonl(manifest_dir / "relation_jobs.jsonl", jobs)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "parent_work_dir": str(parent_dir),
        "work_dir": str(work_dir),
        "candidate_universe": (
            "graph union deep_merged; selected source partitions; "
            "information-bearing coarse-label positives only"
        ),
        "selected_partitions": list(requested_partitions),
        "query_count": len({row["query_id"] for row in selected}),
        "selected_candidate_count": len(selected),
        "partition_counts": dict(partition_counts),
        "axis_applicable_counts": dict(axis_counts),
        "batch_size": args.batch_size,
        "relation_job_count": len(jobs),
        "max_abstract_chars": args.max_abstract_chars,
        "annotation_blinding": (
            "Codex inputs exclude retrieval source, primary partition, graph structure, rank, "
            "ground truth, and paper ID"
        ),
    }
    base._atomic_write_json(manifest_dir / "prepare_summary.json", summary)
    base._atomic_write_json(
        work_dir / "run_config.json",
        {
            **summary,
            "schema_sha256": base._sha256_json(RELATION_BATCH_SCHEMA),
            "prompt_sha256": hashlib.sha256(RELATION_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        },
    )
    return summary


def _validate_axis(value: Any, axis: str, applicable: bool) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{axis} must be an object")
    required = {"label", "rationale", "evidence_phrases", "needs_full_text", "confidence"}
    if set(value) != required:
        raise ValueError(f"{axis} fields do not match the schema")
    label = value.get("label")
    if label not in AXIS_SPECS[axis]["labels"]:
        raise ValueError(f"invalid {axis} label")
    rationale = value.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError(f"{axis} rationale must be non-empty")
    evidence = value.get("evidence_phrases")
    if (
        not isinstance(evidence, list)
        or len(evidence) > 2
        or len(evidence) != len(set(evidence))
        or not all(isinstance(item, str) and item.strip() for item in evidence)
    ):
        raise ValueError(f"invalid {axis} evidence_phrases")
    confidence = value.get("confidence")
    if confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"invalid {axis} confidence")
    if not isinstance(value.get("needs_full_text"), bool):
        raise ValueError(f"invalid {axis} needs_full_text")
    if not applicable:
        if label != "not_applicable" or evidence or value["needs_full_text"] or confidence != 1.0:
            raise ValueError(f"non-applicable {axis} must use the fixed not_applicable form")
        return
    if label == "not_applicable":
        raise ValueError(f"applicable {axis} cannot be not_applicable")
    if label == "insufficient_evidence":
        if not value["needs_full_text"] or confidence not in (0.25, 0.5):
            raise ValueError(
                f"insufficient_evidence {axis} requires needs_full_text and low confidence"
            )
    elif value["needs_full_text"]:
        raise ValueError(f"substantive {axis} cannot set needs_full_text")


def _validate_batch(value: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("prompt_version", PROMPT_VERSION),
        ("batch_id", job["batch_id"]),
        ("input_sha256", job["input_sha256"]),
    ):
        if value.get(key) != expected:
            raise ValueError(f"{key} mismatch")
    annotations = value.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("annotations must be an array")
    ids = [row.get("candidate_id") for row in annotations if isinstance(row, Mapping)]
    expected_ids = list(job["candidate_ids"])
    if len(ids) != len(annotations) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise ValueError("candidate IDs are missing, duplicated, or unexpected")
    applicable_by_id = job["applicable_axes_by_candidate"]
    for row in annotations:
        candidate_id = row["candidate_id"]
        if set(row) != {"candidate_id", *AXIS_SPECS.keys()}:
            raise ValueError("annotation fields do not match the schema")
        applicable_axes = set(applicable_by_id[candidate_id])
        for axis in AXIS_SPECS:
            _validate_axis(row[axis], axis, axis in applicable_axes)


def _valid_existing(path: Path, job: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        _validate_batch(base._load_json(path), job)
    except Exception:
        return False
    return True


def _agent_message_json_from_log(path: Path) -> Optional[Dict[str, Any]]:
    log = base._load_json(path)
    final_text = None
    for line in str(log.get("stdout") or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            event.get("type") == "item.completed"
            and isinstance(event.get("item"), Mapping)
            and event["item"].get("type") == "agent_message"
        ):
            final_text = event["item"].get("text")
    if not final_text:
        return None
    try:
        value = json.loads(final_text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _validate_partial_batch(value: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("prompt_version", PROMPT_VERSION),
        ("batch_id", job["batch_id"]),
        ("input_sha256", job["input_sha256"]),
    ):
        if value.get(key) != expected:
            raise ValueError(f"partial {key} mismatch")
    annotations = value.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("partial annotations must be a non-empty array")
    ids = [row.get("candidate_id") for row in annotations if isinstance(row, Mapping)]
    expected_ids = set(job["candidate_ids"])
    if (
        len(ids) != len(annotations)
        or len(ids) != len(set(ids))
        or not set(ids).issubset(expected_ids)
    ):
        raise ValueError("partial candidate IDs are duplicated or unexpected")
    for row in annotations:
        candidate_id = row["candidate_id"]
        if set(row) != {"candidate_id", *AXIS_SPECS.keys()}:
            raise ValueError("partial annotation fields do not match the schema")
        applicable_axes = set(job["applicable_axes_by_candidate"][candidate_id])
        for axis in AXIS_SPECS:
            _validate_axis(row[axis], axis, axis in applicable_axes)


def _fixed_not_applicable_axis() -> Dict[str, Any]:
    return {
        "label": "not_applicable",
        "rationale": "The coarse label did not trigger this axis.",
        "evidence_phrases": [],
        "needs_full_text": False,
        "confidence": 1.0,
    }


def _normalize_non_applicable_axes(
    value: Mapping[str, Any], job: Mapping[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    normalized = json.loads(json.dumps(value))
    corrections = []
    for row in normalized.get("annotations") or []:
        candidate_id = row.get("candidate_id")
        if candidate_id not in job["applicable_axes_by_candidate"]:
            continue
        applicable = set(job["applicable_axes_by_candidate"][candidate_id])
        for axis in AXIS_SPECS:
            if axis in applicable:
                continue
            if row.get(axis) != _fixed_not_applicable_axis():
                corrections.append(
                    {
                        "candidate_id": candidate_id,
                        "axis": axis,
                        "discarded_label": str((row.get(axis) or {}).get("label")),
                    }
                )
                row[axis] = _fixed_not_applicable_axis()
    return normalized, corrections


def _best_valid_partial(
    work_dir: Path, job: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Path, List[Dict[str, str]]]:
    best: Optional[Tuple[int, Dict[str, Any], Path, List[Dict[str, str]]]] = None
    paths = sorted((work_dir / "logs" / "relations").glob(f"{job['batch_id']}.attempt-*.json"))
    for path in paths:
        value = _agent_message_json_from_log(path)
        if value is None:
            continue
        value, corrections = _normalize_non_applicable_axes(value, job)
        try:
            _validate_partial_batch(value, job)
        except Exception:
            continue
        size = len(value["annotations"])
        if best is None or size >= best[0]:
            best = (size, value, path, corrections)
    if best is None:
        raise RuntimeError(f"No structurally valid partial output exists for {job['batch_id']}")
    return best[1], best[2], best[3]


def recover_missing(args: argparse.Namespace) -> Dict[str, Any]:
    """Recover an omitted candidate through a separate, schema-validated Codex job."""

    work_dir = Path(args.work_dir).resolve()
    args.codex_bin = base._resolve_codex_bin(args.codex_bin)
    jobs = {
        job["batch_id"]: job
        for job in _iter_jsonl(work_dir / "manifest" / "relation_jobs.jsonl")
    }
    if args.batch_id not in jobs:
        raise ValueError(f"Unknown batch ID: {args.batch_id}")
    job = jobs[args.batch_id]
    normal_output = work_dir / "outputs" / "relations" / f"{job['batch_id']}.json"
    if _valid_existing(normal_output, job):
        return {"batch_id": job["batch_id"], "status": "already_valid"}
    partial, partial_log, deterministic_corrections = _best_valid_partial(work_dir, job)
    partial_by_id = {row["candidate_id"]: row for row in partial["annotations"]}
    missing_ids = [candidate_id for candidate_id in job["candidate_ids"] if candidate_id not in partial_by_id]
    if not missing_ids:
        raise RuntimeError("Partial output is complete but fails validation for another reason")

    original_payload = base._load_json(work_dir / job["input_path"])
    recovery_batch_id = f"{job['batch_id']}-missing-recovery"
    recovery_payload = {
        **original_payload,
        "batch_id": recovery_batch_id,
        "candidates": [
            row for row in original_payload["candidates"] if row["candidate_id"] in set(missing_ids)
        ],
    }
    recovery_payload.pop("input_sha256", None)
    recovery_sha = base._sha256_json(recovery_payload)
    recovery_payload["input_sha256"] = recovery_sha
    recovery_input = work_dir / "inputs" / "recovery" / f"{recovery_batch_id}.json"
    base._atomic_write_json(recovery_input, recovery_payload)
    recovery_job = {
        "batch_id": recovery_batch_id,
        "input_sha256": recovery_sha,
        "candidate_ids": missing_ids,
        "applicable_axes_by_candidate": {
            candidate_id: job["applicable_axes_by_candidate"][candidate_id]
            for candidate_id in missing_ids
        },
    }
    recovery_output = work_dir / "outputs" / "recovery" / f"{recovery_batch_id}.json"
    if not _valid_existing(recovery_output, recovery_job):
        prompt = RELATION_INSTRUCTIONS + "\nINPUT:\n" + json.dumps(
            recovery_payload, ensure_ascii=False, indent=2
        )
        result = base._run_codex_job(
            args=args,
            work_dir=work_dir,
            kind="recovery",
            job_id=recovery_batch_id,
            prompt=prompt,
            schema_path=work_dir / "schemas" / "relation_batch.schema.json",
            output_path=recovery_output,
            validator=lambda value: _validate_batch(value, recovery_job),
        )
        if result["status"] != "completed":
            return {
                "batch_id": job["batch_id"],
                "status": "recovery_failed",
                "missing_candidate_ids": missing_ids,
                "result": result,
            }
    recovered = base._load_json(recovery_output)
    combined_by_id = {
        **partial_by_id,
        **{row["candidate_id"]: row for row in recovered["annotations"]},
    }
    combined = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "batch_id": job["batch_id"],
        "input_sha256": job["input_sha256"],
        "annotations": [combined_by_id[candidate_id] for candidate_id in job["candidate_ids"]],
    }
    _validate_batch(combined, job)
    base._atomic_write_json(normal_output, combined)
    audit = {
        "batch_id": job["batch_id"],
        "status": "recovered",
        "policy": (
            "Kept a structurally and semantically validated partial output unchanged; "
            "annotated only omitted candidate IDs in a separate Codex job; then validated "
            "the combined original batch."
        ),
        "partial_log": str(partial_log),
        "partial_candidate_count": len(partial_by_id),
        "deterministic_non_applicable_corrections": deterministic_corrections,
        "missing_candidate_ids": missing_ids,
        "recovery_input": str(recovery_input),
        "recovery_output": str(recovery_output),
        "combined_output": str(normal_output),
    }
    base._atomic_write_json(
        work_dir / "run_summaries" / f"recover_missing_{job['batch_id']}.json", audit
    )
    return audit


def annotate(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    args.codex_bin = base._resolve_codex_bin(args.codex_bin)
    jobs = list(_iter_jsonl(work_dir / "manifest" / "relation_jobs.jsonl"))
    if args.query_id:
        jobs = [job for job in jobs if job["query_id"] == args.query_id]
    output_dir = work_dir / "outputs" / "relations"
    pending = [
        job for job in jobs if not _valid_existing(output_dir / f"{job['batch_id']}.json", job)
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    def worker(job: Mapping[str, Any]) -> Dict[str, Any]:
        payload = base._load_json(work_dir / job["input_path"])
        prompt = RELATION_INSTRUCTIONS + "\nINPUT:\n" + json.dumps(
            payload, ensure_ascii=False, indent=2
        )
        output_path = output_dir / f"{job['batch_id']}.json"
        return base._run_codex_job(
            args=args,
            work_dir=work_dir,
            kind="relations",
            job_id=job["batch_id"],
            prompt=prompt,
            schema_path=work_dir / "schemas" / "relation_batch.schema.json",
            output_path=output_path,
            validator=lambda value: _validate_batch(value, job),
        )

    results = base._run_jobs_concurrently(pending, worker, args.workers) if pending else []
    summary = {
        "kind": "relations",
        "job_count": len(jobs),
        "pending_selected_count": len(pending),
        "completed_count": sum(result["status"] == "completed" for result in results),
        "failed_count": sum(result["status"] == "failed" for result in results),
        "results": results,
    }
    base._atomic_write_json(work_dir / "run_summaries" / "annotate_relations_latest.json", summary)
    return summary


def _agent_message_json_from_log(path: Path) -> Dict[str, Any]:
    log = base._load_json(path)
    latest: Optional[Dict[str, Any]] = None
    for line in str(log.get("stdout") or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and isinstance(item, Mapping)
            and item.get("type") == "agent_message"
        ):
            try:
                value = json.loads(str(item.get("text") or ""))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                latest = value
    if latest is None:
        raise ValueError(f"No JSON agent message in {path}")
    return latest


def recover_batch(args: argparse.Namespace) -> Dict[str, Any]:
    """Recover a batch whose logged output omitted otherwise valid candidates."""

    work_dir = Path(args.work_dir).resolve()
    jobs = {
        job["batch_id"]: job
        for job in _iter_jsonl(work_dir / "manifest" / "relation_jobs.jsonl")
    }
    if args.batch_id not in jobs:
        raise ValueError(f"Unknown batch ID: {args.batch_id}")
    job = jobs[args.batch_id]
    expected_ids = list(job["candidate_ids"])
    expected_set = set(expected_ids)
    candidates = []
    rejected_logs: List[str] = []
    log_dir = work_dir / "logs" / "relations"
    for path in sorted(log_dir.glob(f"{args.batch_id}.attempt-*.json")):
        try:
            value = _agent_message_json_from_log(path)
        except Exception as exc:
            rejected_logs.append(f"{path.name}: parse error: {exc}")
            continue
        annotations = value.get("annotations")
        if not isinstance(annotations, list):
            continue
        ids = [row.get("candidate_id") for row in annotations if isinstance(row, Mapping)]
        if (
            len(ids) != len(annotations)
            or len(ids) != len(set(ids))
            or not set(ids).issubset(expected_set)
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get("prompt_version") != PROMPT_VERSION
            or value.get("batch_id") != job["batch_id"]
            or value.get("input_sha256") != job["input_sha256"]
        ):
            rejected_logs.append(
                f"{path.name}: signature/ID subset check failed; "
                f"expected={len(expected_ids)}, returned={len(ids)}, "
                f"unexpected={sorted(set(ids) - expected_set)}"
            )
            continue
        valid_annotations = []
        invalid_annotation_errors = []
        for annotation in annotations:
            candidate_id = annotation["candidate_id"]
            single_value = dict(value)
            single_value["annotations"] = [annotation]
            single_job = dict(job)
            single_job["candidate_ids"] = [candidate_id]
            single_job["applicable_axes_by_candidate"] = {
                candidate_id: job["applicable_axes_by_candidate"][candidate_id]
            }
            try:
                _validate_batch(single_value, single_job)
            except Exception as exc:
                invalid_annotation_errors.append(f"{candidate_id}: {exc}")
            else:
                valid_annotations.append(annotation)
        if invalid_annotation_errors:
            rejected_logs.append(
                f"{path.name}: rejected individual annotations: "
                + "; ".join(invalid_annotation_errors)
            )
        if valid_annotations:
            sanitized = dict(value)
            sanitized["annotations"] = valid_annotations
            candidates.append((len(valid_annotations), path, sanitized))
    if not candidates:
        raise RuntimeError(
            "No strictly valid partial output found in attempt logs; "
            + " | ".join(rejected_logs)
        )
    _, source_log, partial = max(candidates, key=lambda row: (row[0], row[1].name))
    partial_by_id = {row["candidate_id"]: row for row in partial["annotations"]}
    missing_ids = [candidate_id for candidate_id in expected_ids if candidate_id not in partial_by_id]
    if not missing_ids:
        raise RuntimeError("Best logged output is already complete; recovery is unnecessary")

    original_payload = base._load_json(work_dir / job["input_path"])
    args.codex_bin = base._resolve_codex_bin(args.codex_bin)
    combined_by_id = dict(partial_by_id)
    recovery_records = []
    original_candidate_by_id = {
        row["candidate_id"]: row for row in original_payload["candidates"]
    }
    for recovery_index, candidate_id in enumerate(missing_ids, 1):
        if candidate_id not in original_candidate_by_id:
            raise ValueError(f"Missing candidate {candidate_id} from the original input")
        recovery_id = (
            f"{args.batch_id}-missing-{recovery_index:02d}-"
            f"{candidate_id.removeprefix('c_')[:8]}"
        )
        recovery_payload = {
            key: value
            for key, value in original_payload.items()
            if key not in {"input_sha256", "batch_id", "candidates"}
        }
        recovery_payload["batch_id"] = recovery_id
        recovery_payload["candidates"] = [original_candidate_by_id[candidate_id]]
        recovery_sha = base._sha256_json(recovery_payload)
        recovery_payload["input_sha256"] = recovery_sha
        recovery_input = work_dir / "inputs" / "recovery" / f"{recovery_id}.json"
        base._atomic_write_json(recovery_input, recovery_payload)
        recovery_job = {
            "batch_id": recovery_id,
            "input_sha256": recovery_sha,
            "candidate_ids": [candidate_id],
            "applicable_axes_by_candidate": {
                candidate_id: job["applicable_axes_by_candidate"][candidate_id]
            },
        }
        recovery_output = work_dir / "outputs" / "recovery" / f"{recovery_id}.json"
        prompt = RELATION_INSTRUCTIONS + "\nINPUT:\n" + json.dumps(
            recovery_payload, ensure_ascii=False, indent=2
        )
        result = base._run_codex_job(
            args=args,
            work_dir=work_dir,
            kind="relation_recovery",
            job_id=recovery_id,
            prompt=prompt,
            schema_path=work_dir / "schemas" / "relation_batch.schema.json",
            output_path=recovery_output,
            validator=lambda value, recovery_job=recovery_job: _validate_batch(
                value, recovery_job
            ),
        )
        if result["status"] != "completed":
            return {
                "batch_id": args.batch_id,
                "status": "failed",
                "missing_candidate_ids": missing_ids,
                "failed_candidate_id": candidate_id,
                "recovery_result": result,
            }
        recovered = base._load_json(recovery_output)
        combined_by_id[candidate_id] = recovered["annotations"][0]
        recovery_records.append(
            {
                "candidate_id": candidate_id,
                "recovery_input": str(recovery_input.relative_to(work_dir)),
                "recovery_output": str(recovery_output.relative_to(work_dir)),
            }
        )
    combined = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "batch_id": job["batch_id"],
        "input_sha256": job["input_sha256"],
        "annotations": [combined_by_id[candidate_id] for candidate_id in expected_ids],
    }
    _validate_batch(combined, job)
    output_path = work_dir / "outputs" / "relations" / f"{job['batch_id']}.json"
    base._atomic_write_json(output_path, combined)
    audit = {
        "batch_id": args.batch_id,
        "status": "completed",
        "source_partial_log": str(source_log.relative_to(work_dir)),
        "source_partial_candidate_count": len(partial_by_id),
        "recovered_candidate_ids": missing_ids,
        "recovery_records": recovery_records,
        "final_output": str(output_path.relative_to(work_dir)),
        "final_output_strictly_valid": True,
    }
    base._atomic_write_json(
        work_dir / "run_summaries" / f"recover_{args.batch_id}.json", audit
    )
    return audit


def status(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    jobs = list(_iter_jsonl(work_dir / "manifest" / "relation_jobs.jsonl"))
    valid_jobs = 0
    annotated_candidates = 0
    for job in jobs:
        path = work_dir / "outputs" / "relations" / f"{job['batch_id']}.json"
        if _valid_existing(path, job):
            valid_jobs += 1
            annotated_candidates += int(job["candidate_count"])
    prepared = base._load_json(work_dir / "manifest" / "prepare_summary.json")
    return {
        "selected_candidate_count": prepared["selected_candidate_count"],
        "relation_jobs_total": len(jobs),
        "relation_jobs_valid": valid_jobs,
        "candidates_annotated": annotated_candidates,
        "candidates_remaining": prepared["selected_candidate_count"] - annotated_candidates,
        "complete": valid_jobs == len(jobs),
    }


def _bootstrap_mean_ci(values: Sequence[float], samples: int, seed: int) -> Optional[List[float]]:
    if not values or samples <= 0:
        return None
    rng = random.Random(seed)
    size = len(values)
    means = [sum(rng.choice(values) for _ in range(size)) / size for _ in range(samples)]
    means.sort()
    return [
        means[max(0, int(samples * 0.025))],
        means[min(samples - 1, int(samples * 0.975) - 1)],
    ]


def _distribution_rows(
    counts: Mapping[Tuple[str, str], Counter[str]],
) -> List[Dict[str, Any]]:
    rows = []
    for (group, axis), axis_counts in sorted(counts.items()):
        applicable = axis_counts["applicable_count"]
        conclusive = axis_counts["conclusive_count"]
        for label in AXIS_SPECS[axis]["labels"]:
            if label == "not_applicable":
                continue
            count = axis_counts[f"label:{label}"]
            rows.append(
                {
                    "group": group,
                    "axis": axis,
                    "label": label,
                    "applicable_count": applicable,
                    "conclusive_count": conclusive,
                    "paper_count": count,
                    "rate_among_applicable": _safe_rate(count, applicable),
                    "rate_among_conclusive": (
                        _safe_rate(count, conclusive)
                        if label != "insufficient_evidence"
                        else None
                    ),
                }
            )
    return rows


def _comparison_rows(
    counts: Mapping[Tuple[str, str], Counter[str]],
    query_counts: Mapping[Tuple[str, str, str], Counter[str]],
    *,
    bootstrap_samples: int,
) -> List[Dict[str, Any]]:
    rows = []
    for axis_index, axis in enumerate(AXIS_SPECS):
        graph = counts[("graph_only", axis)]
        deep = counts[("deep_merged_only", axis)]
        for label_index, label in enumerate(AXIS_SPECS[axis]["labels"]):
            if label == "not_applicable":
                continue
            graph_count = graph[f"label:{label}"]
            deep_count = deep[f"label:{label}"]
            graph_rate = _safe_rate(graph_count, graph["applicable_count"])
            deep_rate = _safe_rate(deep_count, deep["applicable_count"])
            paired = []
            query_ids = sorted(
                {
                    query_id
                    for group, query_id, query_axis in query_counts
                    if group == "graph_only" and query_axis == axis
                }
                & {
                    query_id
                    for group, query_id, query_axis in query_counts
                    if group == "deep_merged_only" and query_axis == axis
                }
            )
            for query_id in query_ids:
                graph_query = query_counts[("graph_only", query_id, axis)]
                deep_query = query_counts[("deep_merged_only", query_id, axis)]
                if not graph_query["applicable_count"] or not deep_query["applicable_count"]:
                    continue
                paired.append(
                    (
                        graph_query[f"label:{label}"] / graph_query["applicable_count"],
                        deep_query[f"label:{label}"] / deep_query["applicable_count"],
                    )
                )
            differences = [left - right for left, right in paired]
            ci = _bootstrap_mean_ci(
                differences,
                bootstrap_samples,
                20261000 + axis_index * 100 + label_index,
            )
            rows.append(
                {
                    "axis": axis,
                    "label": label,
                    "graph_only_applicable_count": graph["applicable_count"],
                    "deep_merged_only_applicable_count": deep["applicable_count"],
                    "graph_only_count": graph_count,
                    "deep_merged_only_count": deep_count,
                    "graph_only_rate": graph_rate,
                    "deep_merged_only_rate": deep_rate,
                    "graph_minus_deep_rate": (
                        graph_rate - deep_rate
                        if graph_rate is not None and deep_rate is not None
                        else None
                    ),
                    "graph_over_deep_rate_ratio": (
                        graph_rate / deep_rate
                        if graph_rate is not None and deep_rate not in (None, 0)
                        else None
                    ),
                    "paired_query_count": len(paired),
                    "graph_macro_rate": (
                        sum(left for left, _ in paired) / len(paired) if paired else None
                    ),
                    "deep_merged_macro_rate": (
                        sum(right for _, right in paired) / len(paired) if paired else None
                    ),
                    "macro_graph_minus_deep": (
                        sum(differences) / len(differences) if differences else None
                    ),
                    "macro_bootstrap_95_ci": ci,
                    "graph_wins": sum(value > 0 for value in differences),
                    "deep_merged_wins": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                }
            )
    return rows


def _report_markdown(
    prepared: Mapping[str, Any],
    distribution: Sequence[Mapping[str, Any]],
    complete: bool,
) -> str:
    by_key = {(row["group"], row["axis"], row["label"]): row for row in distribution}
    report_groups = [
        group
        for group in PRIMARY_PARTITIONS
        if int((prepared.get("partition_counts") or {}).get(group, 0)) > 0
    ]
    display_names = {
        "graph_only": "Graph-only",
        "deep_merged_only": "Deep-merged-only",
        "graph_and_deep_merged": "Graph ∩ Deep merged",
    }
    lines = [
        "# Semantic relation refinement — Graph vs Deep merged",
        "",
        f"Complete: **{complete}**. Selected candidates: **{prepared['selected_candidate_count']:,}**.",
        "The annotation was blinded to retrieval source and ground truth.",
    ]
    for axis in AXIS_SPECS:
        lines.extend(
            [
                "",
                f"## {axis}",
                "",
                "| Label | "
                + " | ".join(
                    f"{display_names[group]} count | {display_names[group]} rate"
                    for group in report_groups
                )
                + " |",
                "|---|" + "---:|---:|" * len(report_groups),
            ]
        )
        for label in AXIS_SPECS[axis]["labels"]:
            if label == "not_applicable":
                continue
            rows = [by_key.get((group, axis, label)) for group in report_groups]
            if any(row is None for row in rows):
                continue
            cells = []
            for row in rows:
                cells.extend(
                    (
                        f"{row['paper_count']:,}",
                        f"{100 * row['rate_among_applicable']:.2f}%",
                    )
                )
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "`insufficient_evidence` is retained in every applicable-axis denominator. "
            "Substantive-label rates conditional on conclusive cases are available in the CSV/JSONL outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(args: argparse.Namespace) -> Dict[str, Any]:
    work_dir = Path(args.work_dir).resolve()
    jobs = list(_iter_jsonl(work_dir / "manifest" / "relation_jobs.jsonl"))
    annotations_by_id: Dict[str, Dict[str, Any]] = {}
    invalid_jobs = []
    for job in jobs:
        path = work_dir / "outputs" / "relations" / f"{job['batch_id']}.json"
        if not _valid_existing(path, job):
            invalid_jobs.append(job["batch_id"])
            continue
        for annotation in base._load_json(path)["annotations"]:
            if annotation["candidate_id"] in annotations_by_id:
                raise ValueError(f"Duplicate relation annotation: {annotation['candidate_id']}")
            annotations_by_id[annotation["candidate_id"]] = annotation
    if invalid_jobs and not args.allow_incomplete:
        raise RuntimeError(
            f"{len(invalid_jobs)} relation batches are missing or invalid; rerun annotate"
        )

    analysis_dir = work_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = analysis_dir / "annotations.jsonl"
    temporary = annotation_path.with_name(f".{annotation_path.name}.tmp-{os.getpid()}")
    counts: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
    query_counts: Dict[Tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    pending_count = 0
    annotated_count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for candidate in _iter_jsonl(work_dir / "manifest" / "selected_candidates.jsonl"):
            annotation = annotations_by_id.get(candidate["candidate_id"])
            if annotation is None:
                pending_count += 1
            else:
                annotated_count += 1
            handle.write(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "prompt_version": PROMPT_VERSION,
                        "candidate_id": candidate["candidate_id"],
                        "query_id": candidate["query_id"],
                        "paper_id": candidate["paper_id"],
                        "primary_partition": candidate["primary_partition"],
                        "sources": candidate["sources"],
                        "is_ground_truth": candidate["is_ground_truth"],
                        "applicable_axes": candidate["applicable_axes"],
                        "first_stage_annotation": base.canonicalize_annotation_fields(
                            candidate["first_stage_annotation"]
                        ),
                        "annotation_status": "codex" if annotation is not None else "pending",
                        "annotation": annotation,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            if annotation is None:
                continue
            for axis in candidate["applicable_axes"]:
                label = annotation[axis]["label"]
                for group in ("union", candidate["primary_partition"]):
                    axis_counts = counts[(group, axis)]
                    query_axis_counts = query_counts[(group, candidate["query_id"], axis)]
                    for target in (axis_counts, query_axis_counts):
                        target["applicable_count"] += 1
                        target[f"label:{label}"] += 1
                        target["confidence_sum"] += annotation[axis]["confidence"]
                        if label != "insufficient_evidence":
                            target["conclusive_count"] += 1
    os.replace(temporary, annotation_path)

    distribution = _distribution_rows(counts)
    query_distribution = []
    for (group, query_id, axis), axis_counts in sorted(query_counts.items()):
        for label in AXIS_SPECS[axis]["labels"]:
            if label == "not_applicable":
                continue
            query_distribution.append(
                {
                    "group": group,
                    "query_id": query_id,
                    "axis": axis,
                    "label": label,
                    "applicable_count": axis_counts["applicable_count"],
                    "paper_count": axis_counts[f"label:{label}"],
                    "rate_among_applicable": _safe_rate(
                        axis_counts[f"label:{label}"], axis_counts["applicable_count"]
                    ),
                }
            )
    comparison = _comparison_rows(
        counts, query_counts, bootstrap_samples=args.bootstrap_samples
    )
    _write_jsonl(analysis_dir / "relation_distribution.jsonl", distribution)
    _write_csv(analysis_dir / "relation_distribution.csv", distribution)
    _write_jsonl(analysis_dir / "query_relation_distribution.jsonl", query_distribution)
    _write_jsonl(analysis_dir / "graph_vs_deep_comparison.jsonl", comparison)
    _write_csv(analysis_dir / "graph_vs_deep_comparison.csv", comparison)
    prepared = base._load_json(work_dir / "manifest" / "prepare_summary.json")
    complete = pending_count == 0 and not invalid_jobs
    summary = {
        "schema_version": SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "complete": complete,
        "selected_candidate_count": prepared["selected_candidate_count"],
        "annotated_candidate_count": annotated_count,
        "pending_candidate_count": pending_count,
        "invalid_or_missing_batch_count": len(invalid_jobs),
        "partition_counts": prepared["partition_counts"],
        "axis_applicable_counts": prepared["axis_applicable_counts"],
        "relation_distribution": distribution,
    }
    base._atomic_write_json(analysis_dir / "summary.json", summary)
    base._atomic_write_json(
        analysis_dir / "label_definitions.json",
        {
            "schema_version": SCHEMA_VERSION,
            "axes": AXIS_SPECS,
            "insufficient_evidence_policy": (
                "Allowed for every triggered axis; requires needs_full_text=true and confidence 0.25 or 0.5."
            ),
        },
    )
    base._atomic_write_text(
        analysis_dir / "report.md",
        _report_markdown(prepared, distribution, complete),
    )
    return summary


def wait_and_aggregate(args: argparse.Namespace) -> Dict[str, Any]:
    if args.pid <= 0:
        raise ValueError("--pid must be positive")
    poll_seconds = min(max(args.poll_seconds, 1), 60)
    process_path = Path("/proc") / str(args.pid)
    started = time.time()
    while process_path.exists():
        time.sleep(poll_seconds)
    result = aggregate(
        SimpleNamespace(
            work_dir=args.work_dir,
            allow_incomplete=False,
            bootstrap_samples=args.bootstrap_samples,
        )
    )
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
    parser.add_argument("--load-user-config", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--parent-work-dir", required=True)
    prepare_parser.add_argument("--work-dir", required=True)
    prepare_parser.add_argument("--batch-size", type=int, default=10)
    prepare_parser.add_argument("--max-abstract-chars", type=int, default=6000)
    prepare_parser.add_argument(
        "--partitions",
        nargs="+",
        choices=PRIMARY_PARTITIONS,
        default=list(DEFAULT_PARTITIONS),
        help=(
            "Source partitions to refine. Defaults to the two exclusive arms; "
            "use graph_and_deep_merged to annotate the intersection separately."
        ),
    )

    annotate_parser = subparsers.add_parser("annotate")
    _add_codex_arguments(annotate_parser)

    recovery_parser = subparsers.add_parser("recover-batch")
    _add_codex_arguments(recovery_parser)
    recovery_parser.add_argument("--batch-id", required=True)

    recovery_parser = subparsers.add_parser(
        "recover-missing",
        help="Annotate only candidates omitted from one otherwise valid batch output",
    )
    _add_codex_arguments(recovery_parser)
    recovery_parser.add_argument("--batch-id", required=True)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--work-dir", required=True)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--work-dir", required=True)
    aggregate_parser.add_argument("--allow-incomplete", action="store_true")
    aggregate_parser.add_argument("--bootstrap-samples", type=int, default=10000)

    watcher_parser = subparsers.add_parser("wait-and-aggregate")
    watcher_parser.add_argument("--work-dir", required=True)
    watcher_parser.add_argument("--pid", type=int, required=True)
    watcher_parser.add_argument("--poll-seconds", type=int, default=60)
    watcher_parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "annotate":
        result = annotate(args)
    elif args.command == "recover-batch":
        result = recover_batch(args)
    elif args.command == "recover-missing":
        result = recover_missing(args)
    elif args.command == "status":
        result = status(args)
    elif args.command == "aggregate":
        result = aggregate(args)
    elif args.command == "wait-and-aggregate":
        result = wait_and_aggregate(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
