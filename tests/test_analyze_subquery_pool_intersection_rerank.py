import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_subquery_pool_intersection_rerank.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_subquery_pool_intersection_rerank", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _budget(event_id, k):
    return MODULE.rerank.BaselineBudget(
        query_id="q",
        benchmark_idx=0,
        retrieval_event_id=event_id,
        iteration_idx=1,
        subquery_id=1,
        subquery="topic",
        k=k,
        selector_input_ids=tuple(f"baseline-{event_id}-{i}" for i in range(k)),
    )


def _graph_row(paper_id, event_id, score, *, path=0.0):
    return {
        "paper_arxiv_id": paper_id,
        "retrieval_event_id": event_id,
        "candidate_type": "expanded",
        "is_seed": False,
        "is_expanded": True,
        "query_score_normalized": score,
        "subquery_score_normalized": score,
        "intent_score": 0.0,
        "path_count_normalized": path,
        "observed_retrieval_rank": None,
    }


def test_stable_subquery_intersection_ranks_once_then_slices_with_budget_shortfall():
    budgets = {"e1": _budget("e1", 2), "e2": _budget("e2", 2)}
    graph = {
        "e1": {
            "a": _graph_row("a", "e1", 0.9),
            "b": _graph_row("b", "e1", 0.6),
            "graph-only": _graph_row("graph-only", "e1", 1.0),
        },
        "e2": {
            # The second occurrence makes b outrank a for the stable group.
            "b": _graph_row("b", "e2", 0.95),
            "c": _graph_row("c", "e2", 0.7),
        },
    }
    record = {
        "query_id": "q",
        "benchmark_idx": 0,
        "retrieval_event_id": "group",
        "subquery_id": 1,
        "subquery": "topic",
        "selector_slices": [
            {"selector_slice_idx": 1, "retrieval_event_id": "e1"},
            {"selector_slice_idx": 2, "retrieval_event_id": "e2"},
        ],
        "source_graph_pool_union_arxiv_ids": ["a", "b", "c", "graph-only"],
        "deep_pool_rows": [
            {"paper_arxiv_id": "a"},
            {"paper_arxiv_id": "b"},
            {"paper_arxiv_id": "c"},
            {"paper_arxiv_id": "deep-only"},
        ],
    }

    group, events, pool_ids, selected_ids = MODULE.rerank_intersection_group(
        record=record,
        graph_event_rows=graph,
        budgets=budgets,
        query_weight=0.30,
        subquery_weight=0.40,
        intent_weight=0.15,
        path_weight=0.15,
    )

    assert pool_ids == {"a", "b", "c"}
    assert group["selected_ids"] == ["b", "a", "c"]
    assert selected_ids == {"a", "b", "c"}
    assert [event["selected_count"] for event in events] == [2, 1]
    assert [event["budget_shortfall"] for event in events] == [0, 1]
    assert group["budget_shortfall"] == 1
    assert events[0]["selected"][0]["winning_graph_event_id"] == "e2"
