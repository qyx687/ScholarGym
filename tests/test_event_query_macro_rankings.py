import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_event_query_macro_rankings.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_event_query_macro_rankings",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _truths():
    return {
        0: MODULE.QueryTruth(
            benchmark_idx=0,
            query_id="q0",
            query="zero",
            relevant_ids=frozenset({"a", "b"}),
        ),
        1: MODULE.QueryTruth(
            benchmark_idx=1,
            query_id="q1",
            query="one",
            relevant_ids=frozenset({"c"}),
        ),
    }


def test_event_metrics_use_complete_query_gt():
    metric = MODULE.event_metrics(["a", "x"], frozenset({"a", "b"}), 2)

    assert metric["recall"] == 0.5
    assert metric["ap"] == 0.5
    assert metric["ndcg"] == pytest.approx(1.0 / (1.0 + 1.0 / MODULE.math.log2(3)))


def test_aggregation_is_event_then_query_macro():
    events = {
        (0, "q0:e1"): ["a"],
        (0, "q0:e2"): ["x"],
        (1, "q1:e1"): ["c"],
    }

    summary, rows = MODULE.aggregate_method(events, _truths(), [1])

    assert rows[0]["recall@1"] == 0.25
    assert rows[1]["recall@1"] == 1.0
    assert summary["metrics"]["1"]["recall"] == 0.625
    assert summary["retrieval_event_count"] == 3
    assert summary["ranking_depth"]["coverage_at_cutoff"]["1"] == 1.0


def test_flat_loader_keeps_latest_retried_event(tmp_path):
    path = tmp_path / "paper_rows.jsonl"
    _write_jsonl(
        path,
        [
            {
                "benchmark_idx": 0,
                "query_id": "q0",
                "retrieval_event_id": "e1",
                "rerank_rank": 1,
                "paper_arxiv_id": "old",
            },
            {
                "benchmark_idx": 0,
                "query_id": "q0",
                "retrieval_event_id": "e1",
                "rerank_rank": 1,
                "paper_arxiv_id": "a",
            },
        ],
    )

    events = MODULE.load_flat_events(
        path,
        "rerank_rank",
        _truths(),
        {"q0": 0, "q1": 1},
        {"zero": 0, "one": 1},
    )

    assert events[(0, "e1")] == ["a"]


def test_nested_loader_selects_requested_method(tmp_path):
    path = tmp_path / "ranked.jsonl"
    _write_jsonl(
        path,
        [
            {
                "benchmark_idx": 0,
                "query_id": "q0",
                "retrieval_event_id": "e1",
                "method": "legacy_static",
                "ranked_candidates": [
                    {"rerank_rank": 1, "paper_arxiv_id": "x"}
                ],
            },
            {
                "benchmark_idx": 0,
                "query_id": "q0",
                "retrieval_event_id": "e1",
                "method": "dynamic_policy",
                "ranked_candidates": [
                    {"rerank_rank": 1, "paper_arxiv_id": "a"}
                ],
            },
        ],
    )

    events = MODULE.load_nested_events(
        path,
        "dynamic_policy",
        _truths(),
        {"q0": 0, "q1": 1},
        {"zero": 0, "one": 1},
    )

    assert events[(0, "e1")] == ["a"]
