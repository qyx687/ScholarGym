import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "annotate_semantic_relation_refinement_with_codex.py"
)
SPEC = importlib.util.spec_from_file_location(
    "annotate_semantic_relation_refinement_with_codex", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _candidate(candidate_id, paper_id, sources):
    return {
        "candidate_id": candidate_id,
        "benchmark_idx": 0,
        "query_id": "q0",
        "paper_id": paper_id,
        "sources": sources,
        "is_ground_truth": False,
        "title": f"Title {candidate_id}",
        "abstract": f"Abstract evidence for {candidate_id}.",
        "paper_date": "2020-01-01",
        "categories": ["cs.AI"],
    }


def _first_stage(candidate_id, relationship, contributions):
    annotation = {
        "candidate_id": candidate_id,
        "relationship": relationship,
        "semantic_distance": "near",
        "paper_type": "method_or_theory",
        "matched_aspect_ids": ["A1"],
        "relation_roles": ["background_or_foundation"],
        "contribution_types": contributions,
        "semantic_contribution": "Adds query-relevant context.",
        "key_concepts": ["concept"],
        "evidence_phrases": ["Abstract evidence"],
    }
    return {"candidate_id": candidate_id, "annotation": annotation}


def _axis(label, *, confidence=0.75, needs_full_text=False, evidence=None):
    return {
        "label": label,
        "rationale": "The supplied evidence supports this relation.",
        "evidence_phrases": evidence if evidence is not None else ["Abstract evidence"],
        "needs_full_text": needs_full_text,
        "confidence": confidence,
    }


def _not_applicable():
    return {
        "label": "not_applicable",
        "rationale": "The coarse label did not trigger this axis.",
        "evidence_phrases": [],
        "needs_full_text": False,
        "confidence": 1.0,
    }


def _prepare_parent(tmp_path):
    parent = tmp_path / "parent"
    (parent / "analysis").mkdir(parents=True)
    (parent / "analysis" / "summary.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    _write_jsonl(
        parent / "manifest" / "queries.jsonl",
        [
            {
                "query_id": "q0",
                "query_token": "q_token",
                "query": "Find method M for task T.",
                "query_date": "2024-01",
            }
        ],
    )
    rubric_path = parent / "outputs" / "rubrics" / "q_token.json"
    rubric_path.parent.mkdir(parents=True)
    rubric_path.write_text(
        json.dumps(
            {
                "query_id": "q0",
                "scope_summary": "Method M for task T.",
                "inclusion_criteria": ["Uses M"],
                "exclusion_criteria": ["Wrong task"],
                "aspects": [
                    {
                        "aspect_id": "A1",
                        "label": "Method",
                        "description": "Uses M",
                        "required_for_direct": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = [
        _candidate("g1", "paper-g", ["graph"]),
        _candidate("d1", "paper-d", ["deep_merged"]),
        _candidate("both", "paper-both", ["graph", "deep_merged"]),
        _candidate("unrelated", "paper-u", ["graph"]),
        _candidate("no-axis", "paper-n", ["graph"]),
        _candidate("other-source", "paper-o", ["baseline"]),
    ]
    annotations = [
        _first_stage(
            "g1", "contextual", ["historical_context", "mechanism_or_theory"]
        ),
        _first_stage("d1", "partial", ["application_domain"]),
        _first_stage("both", "contextual", ["historical_context"]),
        _first_stage("unrelated", "unrelated", ["historical_context"]),
        _first_stage("no-axis", "contextual", ["metric_or_benchmark"]),
        _first_stage("other-source", "contextual", ["historical_context"]),
    ]
    _write_jsonl(parent / "manifest" / "candidates.jsonl", candidates)
    _write_jsonl(parent / "analysis" / "annotations.jsonl", annotations)
    return parent


def test_prepare_blinds_sources_and_selects_only_triggered_exclusive_candidates(tmp_path):
    parent = _prepare_parent(tmp_path)
    work = tmp_path / "refinement"

    summary = MODULE.prepare(
        SimpleNamespace(
            parent_work_dir=str(parent),
            work_dir=str(work),
            batch_size=10,
            max_abstract_chars=6000,
        )
    )

    assert summary["selected_candidate_count"] == 2
    assert summary["partition_counts"] == {"graph_only": 1, "deep_merged_only": 1}
    assert summary["axis_applicable_counts"] == {
        "historical_relation": 1,
        "mechanism_relation": 1,
        "domain_relation": 1,
    }
    job = next(MODULE._iter_jsonl(work / "manifest" / "relation_jobs.jsonl"))
    payload = MODULE.base._load_json(work / job["input_path"])
    serialized = json.dumps(payload)
    assert "sources" not in serialized
    assert "primary_partition" not in serialized
    assert "is_ground_truth" not in serialized
    assert "paper-g" not in serialized
    assert "paper-d" not in serialized
    assert {row["candidate_id"] for row in payload["candidates"]} == {"g1", "d1"}
    first_stage = payload["candidates"][0]["first_stage_annotation"]
    assert "relevance_grade" in first_stage
    assert "scholarly_roles" in first_stage
    assert "information_added" in first_stage
    assert "information_summary" in first_stage
    assert "relationship" not in first_stage
    assert "relation_roles" not in first_stage
    assert "contribution_types" not in first_stage
    assert "semantic_contribution" not in first_stage


def test_prepare_can_select_graph_deep_intersection_partition(tmp_path):
    parent = _prepare_parent(tmp_path)
    work = tmp_path / "intersection-refinement"

    summary = MODULE.prepare(
        SimpleNamespace(
            parent_work_dir=str(parent),
            work_dir=str(work),
            batch_size=10,
            max_abstract_chars=6000,
            partitions=["graph_and_deep_merged"],
        )
    )

    assert summary["selected_partitions"] == ["graph_and_deep_merged"]
    assert summary["selected_candidate_count"] == 1
    assert summary["partition_counts"] == {"graph_and_deep_merged": 1}
    assert summary["axis_applicable_counts"] == {"historical_relation": 1}
    selected = list(
        MODULE._iter_jsonl(work / "manifest" / "selected_candidates.jsonl")
    )
    assert selected[0]["candidate_id"] == "both"
    assert selected[0]["primary_partition"] == "graph_and_deep_merged"
    job = next(MODULE._iter_jsonl(work / "manifest" / "relation_jobs.jsonl"))
    payload = MODULE.base._load_json(work / job["input_path"])
    serialized = json.dumps(payload)
    assert "graph_and_deep_merged" not in serialized
    assert "sources" not in serialized


def test_insufficient_evidence_validates_and_aggregate_refines_each_axis(tmp_path):
    parent = _prepare_parent(tmp_path)
    work = tmp_path / "refinement"
    MODULE.prepare(
        SimpleNamespace(
            parent_work_dir=str(parent),
            work_dir=str(work),
            batch_size=10,
            max_abstract_chars=6000,
        )
    )
    job = next(MODULE._iter_jsonl(work / "manifest" / "relation_jobs.jsonl"))
    output = {
        "schema_version": MODULE.SCHEMA_VERSION,
        "prompt_version": MODULE.PROMPT_VERSION,
        "batch_id": job["batch_id"],
        "input_sha256": job["input_sha256"],
        "annotations": [
            {
                "candidate_id": "g1",
                "historical_relation": _axis("direct_predecessor"),
                "mechanism_relation": _axis(
                    "insufficient_evidence",
                    confidence=0.5,
                    needs_full_text=True,
                    evidence=[],
                ),
                "domain_relation": _not_applicable(),
            },
            {
                "candidate_id": "d1",
                "historical_relation": _not_applicable(),
                "mechanism_relation": _not_applicable(),
                "domain_relation": _axis("cross_domain_transfer"),
            },
        ],
    }
    MODULE._validate_batch(output, job)
    MODULE.base._atomic_write_json(
        work / "outputs" / "relations" / f"{job['batch_id']}.json", output
    )

    summary = MODULE.aggregate(
        SimpleNamespace(
            work_dir=str(work), allow_incomplete=False, bootstrap_samples=20
        )
    )

    assert summary["complete"] is True
    rows = {
        (row["group"], row["axis"], row["label"]): row
        for row in MODULE._iter_jsonl(work / "analysis" / "relation_distribution.jsonl")
    }
    assert rows[("graph_only", "historical_relation", "direct_predecessor")][
        "paper_count"
    ] == 1
    assert rows[("graph_only", "mechanism_relation", "insufficient_evidence")][
        "paper_count"
    ] == 1
    assert rows[("deep_merged_only", "domain_relation", "cross_domain_transfer")][
        "paper_count"
    ] == 1
    aggregate_row = next(MODULE._iter_jsonl(work / "analysis" / "annotations.jsonl"))
    first_stage = aggregate_row["first_stage_annotation"]
    assert "relevance_grade" in first_stage
    assert "scholarly_roles" in first_stage
    assert "information_added" in first_stage
    assert "information_summary" in first_stage
    assert "relationship" not in first_stage

    invalid = json.loads(json.dumps(output))
    invalid["annotations"][0]["historical_relation"] = _not_applicable()
    with pytest.raises(ValueError, match="applicable historical_relation"):
        MODULE._validate_batch(invalid, job)


def test_insufficient_evidence_requires_full_text_and_low_confidence():
    invalid = _axis(
        "insufficient_evidence",
        confidence=0.75,
        needs_full_text=False,
        evidence=[],
    )
    with pytest.raises(ValueError, match="requires needs_full_text and low confidence"):
        MODULE._validate_axis(invalid, "mechanism_relation", True)
