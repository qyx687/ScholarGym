import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_graph_vs_deep_merged.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_graph_vs_deep_merged", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _candidate(candidate_id, sources, *, graph_stats=None, ground_truth=False):
    source_stats = {}
    if graph_stats is not None:
        source_stats["graph"] = graph_stats
    return {
        "candidate_id": candidate_id,
        "query_id": "q0",
        "sources": sources,
        "source_stats": source_stats,
        "is_ground_truth": ground_truth,
    }


def _annotation(
    candidate_id,
    relationship,
    aspects,
    *,
    roles=None,
    contributions=None,
):
    return {
        "candidate_id": candidate_id,
        "annotation": {
            "candidate_id": candidate_id,
            "relationship": relationship,
            "semantic_distance": "near" if relationship != "unrelated" else "distant",
            "paper_type": "empirical",
            "confidence": 3,
            "needs_full_text": False,
            "matched_aspect_ids": aspects,
            "relation_roles": roles or ["none"],
            "contribution_types": contributions or ["none"],
            "exclusion_reasons": [] if relationship != "unrelated" else ["wrong_task"],
        },
    }


def _prepare_fixture(tmp_path):
    work_dir = tmp_path / "annotations"
    (work_dir / "analysis").mkdir(parents=True)
    (work_dir / "analysis" / "summary.json").write_text(
        json.dumps({"complete": True}), encoding="utf-8"
    )
    _write_jsonl(
        work_dir / "manifest" / "queries.jsonl",
        [{"query_id": "q0", "query": "method M for task T", "query_token": "q0token"}],
    )
    rubric_path = work_dir / "outputs" / "rubrics" / "q0token.json"
    rubric_path.parent.mkdir(parents=True)
    rubric_path.write_text(
        json.dumps(
            {
                "query_id": "q0",
                "aspects": [
                    {"aspect_id": f"A{index}", "label": f"Aspect {index}"}
                    for index in range(1, 5)
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = [
        _candidate(
            "g1",
            ["graph"],
            graph_stats={
                "is_expanded": True,
                "edge_types": ["citation", "reference"],
                "max_path_count": 2,
                "source_seed_count": 2,
                "event_count": 2,
                "occurrence_count": 7,
            },
        ),
        _candidate(
            "g2",
            ["graph"],
            graph_stats={
                "is_expanded": True,
                "edge_types": ["citation"],
                "max_path_count": 4,
                "source_seed_count": 3,
                "event_count": 4,
            },
        ),
        _candidate(
            "g3",
            ["graph"],
            graph_stats={
                "is_expanded": True,
                "edge_types": ["reference"],
                "max_path_count": 1,
                "source_seed_count": 1,
                "event_count": 1,
            },
        ),
        _candidate("d1", ["deep_merged"], ground_truth=True),
        _candidate(
            "both",
            ["graph", "deep_merged"],
            graph_stats={"is_expanded": True, "edge_types": ["citation"]},
        ),
        _candidate("ignored", ["baseline"]),
    ]
    annotations = [
        _annotation(
            "g1",
            "contextual",
            ["A2"],
            roles=["background_or_foundation"],
            contributions=["historical_context"],
        ),
        _annotation(
            "g2",
            "contextual",
            ["A3"],
            roles=["dataset_or_domain"],
            contributions=["dataset_or_population"],
        ),
        _annotation("g3", "unrelated", ["A4"]),
        _annotation(
            "d1",
            "direct",
            ["A1"],
            roles=["direct_target", "method_component"],
            contributions=["method_or_algorithm"],
        ),
        _annotation(
            "both",
            "partial",
            ["A2"],
            roles=["task_or_application"],
            contributions=["application_domain"],
        ),
        _annotation("ignored", "direct", ["A4"], roles=["direct_target"]),
    ]
    _write_jsonl(work_dir / "manifest" / "candidates.jsonl", candidates)
    _write_jsonl(work_dir / "analysis" / "annotations.jsonl", annotations)
    return work_dir


def test_deep_merged_primary_analysis_excludes_other_sources_and_tracks_novelty(tmp_path):
    work_dir = _prepare_fixture(tmp_path)

    summary = MODULE.analyze(
        work_dir, bootstrap_samples=20, rarefaction_samples=20
    )

    groups = {row["group"]: row for row in summary["group_summary"]}
    assert groups["graph_only"]["candidate_count"] == 3
    assert groups["deep_merged_only"]["candidate_count"] == 1
    assert groups["graph_and_deep_merged"]["candidate_count"] == 1
    assert groups["union"]["candidate_count"] == 5
    assert groups["deep_merged"]["candidate_count"] == 2
    assert groups["deep_merged_only"]["direct_task_method_match_count"] == 1
    assert groups["graph_only"]["direct_task_method_match_count"] == 0

    marginal = summary["marginal_coverage"]
    assert marginal["candidate_pool_growth_rate"] == pytest.approx(3 / 2)
    information = next(
        row for row in marginal["metrics"] if row["metric"] == "information_bearing"
    )
    assert information["deep_merged_base_count"] == 2
    assert information["graph_only_added_count"] == 2
    assert information["union_count"] == 4
    assert information["relative_gain_over_deep_merged"] == pytest.approx(1.0)
    assert information["gain_efficiency_vs_pool_growth"] == pytest.approx(2 / 3)

    aspect = summary["aspect_coverage_summary"]
    assert aspect["rubric_query_aspect_count"] == 4
    assert aspect["deep_merged_information_query_aspect_count"] == 2
    assert aspect["novel_graph_information_query_aspect_count"] == 1
    assert aspect["union_information_query_aspect_count"] == 3
    assert aspect["micro_aspect_coverage_gain"] == pytest.approx(0.25)
    aspect_row = json.loads(
        (work_dir / "analysis" / "deep_merged_primary" / "aspect_novelty.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert aspect_row["novel_graph_information_aspect_ids"] == ["A3"]

    structure_rows = list(
        MODULE._iter_jsonl(
            work_dir
            / "analysis"
            / "deep_merged_primary"
            / "graph_structure_quality.jsonl"
        )
    )
    structure = {(row["dimension"], row["bucket"]): row for row in structure_rows}
    assert structure[("all", "all")]["candidate_count"] == 3
    assert structure[("edge_type", "citation_and_reference")]["candidate_count"] == 1
    assert structure[("edge_type", "citation_only")]["candidate_count"] == 1
    assert structure[("edge_type", "reference_only")]["candidate_count"] == 1
    assert sum(
        row["candidate_count"]
        for row in structure_rows
        if row["dimension"] == "edge_type"
    ) == 3
    assert structure[("path_count", "1")]["candidate_count"] == 1
    assert structure[("path_count", "2")]["candidate_count"] == 1
    assert structure[("path_count", "3-4")]["candidate_count"] == 1

    output_dir = work_dir / "analysis" / "deep_merged_primary"
    definitions = json.loads((output_dir / "metric_definitions.json").read_text())
    assert "implicit_mechanism" in definitions["secondary_annotation_needed_for_strict_claims"]
    semantic_rows = list(MODULE._iter_jsonl(output_dir / "semantic_complement.jsonl"))
    assert {row["field"] for row in semantic_rows} == {
        "derived",
        "information_added",
        "scholarly_roles",
    }
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "Deep event" not in report
    assert not {"�", "鈥", "鈭"}.intersection(report)


def test_primary_partition_uses_only_graph_and_deep_merged():
    assert MODULE.primary_partition(["graph"]) == "graph_only"
    assert MODULE.primary_partition(["deep_merged"]) == "deep_merged_only"
    assert (
        MODULE.primary_partition(["graph", "deep_merged"])
        == "graph_and_deep_merged"
    )
    assert MODULE.primary_partition(["baseline"]) is None
