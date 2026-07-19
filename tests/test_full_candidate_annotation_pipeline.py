import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "annotate_full_candidate_pool_with_codex.py"
)
SPEC = importlib.util.spec_from_file_location("annotate_full_candidate_pool_with_codex", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _base_row(paper_id, *, source_fields=None, is_gt=False):
    row = {
        "benchmark_idx": 0,
        "query_id": "q0",
        "query": "Find papers that use method M for task T.",
        "query_date": "2024-01",
        "query_source": "synthetic",
        "paper_arxiv_id": paper_id,
        "retrieval_event_id": "q0:e1",
        "iteration_idx": 1,
        "subquery_id": 1,
        "subquery": "method M for task T",
        "planner_checklist": "Include papers that explicitly use M for T.",
        "rerank_rank": 1,
        "rerank_score": 0.9,
        "in_selector_topk": True,
        "is_ground_truth": is_gt,
    }
    row.update(source_fields or {})
    return row


def _prepare_fixture(tmp_path):
    run_dir = tmp_path / "run"
    work_dir = tmp_path / "annotations"
    paper_db = tmp_path / "paper_db.json"
    detailed = {
        "idx": 0,
        "query": "Find papers that use method M for task T.",
        "ground_truth_arxiv_ids": ["p-graph"],
        "postprocess_results": {"per_subquery": {"query_id": "q0"}},
    }
    _write_jsonl(run_dir / "detailed_results.jsonl", [detailed])
    _write_jsonl(
        run_dir / MODULE.SOURCE_PATHS["baseline"],
        [_base_row("p-overlap")],
    )
    _write_jsonl(
        run_dir / MODULE.SOURCE_PATHS["graph"],
        [
            _base_row(
                "p-graph",
                is_gt=True,
                source_fields={
                    "is_expanded": True,
                    "edge_types": ["reference"],
                    "source_seed_arxiv_ids": ["seed-1"],
                    "path_count": 2,
                },
            ),
            _base_row("p-overlap", source_fields={"is_seed": True}),
            _base_row("p-missing", source_fields={"is_expanded": True}),
        ],
    )
    _write_jsonl(
        run_dir / MODULE.SOURCE_PATHS["deep_event"],
        [
            _base_row("p-deep", source_fields={"deep_retrieval_rank_after_exclusion": 1}),
            _base_row("p-overlap", source_fields={"deep_retrieval_rank_after_exclusion": 2}),
        ],
    )
    _write_jsonl(
        run_dir / MODULE.SOURCE_PATHS["deep_merged"],
        [_base_row("p-deep", source_fields={"deep_retrieval_rank_after_exclusion": 1})],
    )
    paper_db.write_text(
        json.dumps(
            {
                "p-graph": {
                    "title": "Method M for Task T",
                    "abstract": "We use method M to solve task T.",
                    "date": "2023-01-01",
                    "category": ["cs.AI"],
                },
                "p-deep": {
                    "title": "Method M components",
                    "abstract": "We study a component of method M.",
                    "date": "2022-01-01",
                    "category": ["cs.LG"],
                },
                "p-overlap": {
                    "title": "A survey of T",
                    "abstract": "This survey covers task T.",
                    "date": "2021-01-01",
                    "category": ["cs.AI"],
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        run_dir=str(run_dir),
        paper_db=str(paper_db),
        work_dir=str(work_dir),
        sources=list(MODULE.DEFAULT_SOURCES),
        batch_size=2,
        max_abstract_chars=6000,
        allow_missing_source=False,
    )
    summary = MODULE.prepare(args)
    return work_dir, summary


def test_streams_large_top_level_json_object_with_small_chunks(tmp_path):
    path = tmp_path / "object.json"
    expected = {
        "a": {"text": "α" * 20},
        "b": {"nested": [1, 2, 3]},
        "c": "done",
    }
    path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")

    actual = dict(MODULE._iter_top_level_json_object(path, chunk_size=7))

    assert actual == expected


def test_legacy_annotation_field_names_remain_readable():
    job = {"batch_id": "b1", "input_sha256": "sha", "candidate_ids": ["c1"]}
    value = {
        "schema_version": MODULE.SCHEMA_VERSION,
        "prompt_version": MODULE.PROMPT_VERSION,
        "batch_id": "b1",
        "input_sha256": "sha",
        "annotations": [
            {
                "candidate_id": "c1",
                "relationship": "partial",
                "semantic_distance": "near",
                "paper_type": "method_or_theory",
                "matched_aspect_ids": ["A1"],
                "relation_roles": ["method_component"],
                "contribution_types": ["method_variant"],
                "semantic_contribution": "Adds a method component.",
                "key_concepts": ["method"],
                "exclusion_reasons": ["none"],
                "evidence_phrases": [],
                "needs_full_text": False,
                "confidence": 0.75,
            }
        ],
    }

    MODULE._validate_candidate_batch(value, job, {"A1"})
    canonical = MODULE.canonicalize_annotation_fields(value["annotations"][0])

    assert canonical["relevance_grade"] == "partial"
    assert canonical["scholarly_roles"] == ["method_component"]
    assert canonical["information_added"] == ["method_variant"]
    assert canonical["information_summary"] == "Adds a method component."
    assert not set(MODULE.ANNOTATION_FIELD_ALIASES.values()).intersection(canonical)


def test_prepare_deduplicates_sources_and_blinds_codex_inputs(tmp_path):
    work_dir, summary = _prepare_fixture(tmp_path)

    assert summary["candidate_count"] == 4
    assert summary["partition_counts"] == {
        "graph_only": 2,
        "graph_and_deep": 1,
        "deep_only": 1,
    }
    assert summary["metadata_missing_count"] == 1
    assert summary["codex_candidate_count"] == 3
    assert summary["candidate_batch_count"] == 2

    candidates = list(MODULE._iter_jsonl(work_dir / "manifest" / "candidates.jsonl"))
    overlap = next(row for row in candidates if row["paper_id"] == "p-overlap")
    assert overlap["sources"] == ["baseline", "deep_event", "graph"]
    assert overlap["source_partition"] == "graph_and_deep"

    jobs = list(MODULE._iter_jsonl(work_dir / "manifest" / "candidate_jobs.jsonl"))
    blinded = MODULE._load_json(work_dir / jobs[0]["input_path"])
    serialized = json.dumps(blinded)
    assert "source_partition" not in serialized
    assert "is_ground_truth" not in serialized
    assert "rerank_rank" not in serialized
    assert "p-graph" not in serialized

    occurrences = list(MODULE._iter_jsonl(work_dir / "manifest" / "occurrences.jsonl"))
    assert len(occurrences) == 7
    assert {row["source"] for row in occurrences} == set(MODULE.DEFAULT_SOURCES)


def test_aggregate_reconnects_blind_labels_to_source_groups(tmp_path):
    work_dir, _ = _prepare_fixture(tmp_path)
    rubric_job = next(MODULE._iter_jsonl(work_dir / "manifest" / "rubric_jobs.jsonl"))
    rubric = {
        "schema_version": MODULE.SCHEMA_VERSION,
        "prompt_version": MODULE.PROMPT_VERSION,
        "query_id": "q0",
        "input_sha256": rubric_job["input_sha256"],
        "scope_summary": "Papers applying M to T.",
        "inclusion_criteria": ["Uses M", "Targets T"],
        "exclusion_criteria": ["Mentions M without use"],
        "aspects": [
            {
                "aspect_id": "A1",
                "label": "Method",
                "description": "Uses M",
                "required_for_direct": True,
            },
            {
                "aspect_id": "A2",
                "label": "Task",
                "description": "Targets T",
                "required_for_direct": True,
            },
        ],
        "direct_rule": "Both aspects are explicit.",
        "partial_rule": "One aspect is explicit.",
        "contextual_rule": "Useful context only.",
    }
    rubric_path = work_dir / "outputs" / "rubrics" / f"{rubric_job['query_token']}.json"
    MODULE._atomic_write_json(rubric_path, rubric)

    relevance_grade_by_title = {
        "Method M for Task T": "direct",
        "Method M components": "partial",
        "A survey of T": "contextual",
    }
    for job in MODULE._iter_jsonl(work_dir / "manifest" / "candidate_jobs.jsonl"):
        payload = MODULE._load_json(work_dir / job["input_path"])
        annotations = []
        for paper in payload["candidates"]:
            relevance_grade = relevance_grade_by_title[paper["title"]]
            annotations.append(
                {
                    "candidate_id": paper["candidate_id"],
                    "relevance_grade": relevance_grade,
                    "semantic_distance": "exact" if relevance_grade == "direct" else "near",
                    "paper_type": "primary_empirical",
                    "matched_aspect_ids": ["A1", "A2"]
                    if relevance_grade == "direct"
                    else ["A1"],
                    "scholarly_roles": ["direct_target"]
                    if relevance_grade == "direct"
                    else ["method_component"],
                    "information_added": ["method_variant"],
                    "information_summary": "Adds query-relevant information.",
                    "key_concepts": ["M", "T"],
                    "exclusion_reasons": ["none"],
                    "evidence_phrases": [],
                    "needs_full_text": False,
                    "confidence": 0.75,
                }
            )
        output = {
            "schema_version": MODULE.SCHEMA_VERSION,
            "prompt_version": MODULE.PROMPT_VERSION,
            "batch_id": job["batch_id"],
            "input_sha256": job["input_sha256"],
            "annotations": annotations,
        }
        MODULE._atomic_write_json(
            work_dir / "outputs" / "candidates" / f"{job['batch_id']}.json",
            output,
        )

    result = MODULE.aggregate(SimpleNamespace(work_dir=str(work_dir), allow_incomplete=False))

    assert result["complete"] is True
    assert result["candidate_count"] == 4
    assert result["codex_annotated_count"] == 3
    assert result["automatic_insufficient_metadata_count"] == 1
    by_group = {row["group"]: row for row in result["source_comparison"]}
    assert by_group["source:graph"]["candidate_count"] == 3
    assert by_group["source:deep_any"]["candidate_count"] == 2
    assert by_group["partition:graph_only"]["ground_truth_count"] == 1
    assert (work_dir / "analysis" / "annotations.jsonl").exists()
    annotations = list(MODULE._iter_jsonl(work_dir / "analysis" / "annotations.jsonl"))
    direct = next(
        row["annotation"]
        for row in annotations
        if row["annotation"] and row["annotation"].get("relevance_grade") == "direct"
    )
    assert direct["scholarly_roles"] == ["direct_target"]
    assert direct["information_added"] == ["method_variant"]
    assert direct["information_summary"] == "Adds query-relevant information."
    assert "relationship" not in direct
    assert "relation_roles" not in direct
    assert "contribution_types" not in direct
    assert "semantic_contribution" not in direct
