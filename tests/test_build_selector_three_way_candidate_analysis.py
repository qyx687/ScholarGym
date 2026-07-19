import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_selector_three_way_candidate_analysis.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_selector_three_way_candidate_analysis", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_load_selector_sets_and_build_three_partitions(tmp_path):
    graph_path = tmp_path / "graph.jsonl"
    deep_path = tmp_path / "deep.jsonl"
    _write_jsonl(
        graph_path,
        [{"query_id": "q", "selected_count": 2, "selected_arxiv_ids": ["a", "b"]}],
    )
    _write_jsonl(
        deep_path,
        [{"query_id": "q", "selected_count": 2, "selected_arxiv_ids": ["b", "c"]}],
    )

    graph = MODULE.load_selector_query_sets(graph_path)
    deep = MODULE.load_selector_query_sets(deep_path)
    partitions = MODULE.build_partition_sets(graph, deep)

    assert partitions["graph_only"] == {("q", "a")}
    assert partitions["graph_and_deep_merged"] == {("q", "b")}
    assert partitions["deep_merged_only"] == {("q", "c")}
    assert partitions["graph"] == {("q", "a"), ("q", "b")}
    assert partitions["deep_merged"] == {("q", "b"), ("q", "c")}


def test_validate_formula_rejects_a_different_saved_weight(tmp_path):
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "feature_weights": {
                    "query_score_normalized": 0.30,
                    "subquery_score_normalized": 0.40,
                    "intent_score": 0.15,
                    "path_count_normalized": 0.15,
                }
            }
        ),
        encoding="utf-8",
    )

    assert MODULE.validate_formula(
        path,
        query_weight=0.30,
        subquery_weight=0.40,
        intent_weight=0.15,
        path_weight=0.15,
    )["query_score_normalized"] == pytest.approx(0.30)

    with pytest.raises(ValueError, match="query_score_normalized"):
        MODULE.validate_formula(
            path,
            query_weight=0.55,
            subquery_weight=0.15,
            intent_weight=0.15,
            path_weight=0.15,
        )
