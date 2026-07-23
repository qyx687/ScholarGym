import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_baseline_budget_ret_rerank.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_baseline_budget_ret_rerank", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_baseline_budget_comes_from_actual_selector_input_rows(tmp_path):
    path = tmp_path / "selector_decisions.jsonl"
    candidates = [
        {
            "query_id": "q0",
            "retrieval_event_id": "q0:e1",
            "paper_arxiv_id": paper_id,
        }
        for paper_id in ("a", "b", "c")
    ]
    _write_jsonl(
        path,
        [
            {
                "query_id": "q0",
                "benchmark_idx": 0,
                "iteration_idx": 1,
                "subquery_id": 2,
                "subquery": "example",
                "selector_top_k": 99,
                "candidate_rows": candidates,
            }
        ],
    )

    budgets = MODULE.load_baseline_budgets(path)

    assert budgets["q0:e1"].k == 3
    assert budgets["q0:e1"].selector_input_ids == ("a", "b", "c")


def test_exact_semantic_formula_and_graph_tie_breaks():
    rows = [
        {
            "paper_arxiv_id": "expanded",
            "candidate_type": "expanded",
            "query_score_normalized": 0.25,
            "subquery_score_normalized": 0.75,
            "observed_retrieval_rank": None,
        },
        {
            "paper_arxiv_id": "seed",
            "candidate_type": "seed",
            "query_score_normalized": 0.25,
            "subquery_score_normalized": 0.75,
            "observed_retrieval_rank": 10,
        },
        {
            "paper_arxiv_id": "lower",
            "candidate_type": "seed",
            "query_score_normalized": 0.20,
            "subquery_score_normalized": 0.70,
            "observed_retrieval_rank": 1,
        },
    ]

    assert MODULE.semantic_score(rows[0]) == pytest.approx(
        0.30 * 0.25 + 0.40 * 0.75
    )
    assert [row["paper_arxiv_id"] for row in MODULE.rank_graph_rows(rows)] == [
        "seed",
        "expanded",
        "lower",
    ]


def test_configurable_four_factor_formula_uses_intent_and_normalized_path():
    row = {
        "paper_arxiv_id": "candidate",
        "query_score_normalized": 0.2,
        "subquery_score_normalized": 0.4,
        "intent_score": 0.6,
        "path_count_normalized": 0.8,
    }

    score = MODULE.semantic_score(
        row,
        query_weight=0.55,
        subquery_weight=0.30,
        intent_weight=0.10,
        path_weight=0.05,
    )

    assert score == pytest.approx(0.55 * 0.2 + 0.30 * 0.4 + 0.10 * 0.6 + 0.05 * 0.8)


def test_expanded_endpoint_path_definition_ignores_seed_endpoint_and_renormalizes(tmp_path):
    path = tmp_path / "paper_rows.jsonl"
    _write_jsonl(
        path,
        [
            {
                "retrieval_event_id": "event-1",
                "paper_arxiv_id": "seed",
                "candidate_type": "seed",
                "is_seed": True,
                "is_expanded": False,
                "expansion_path_count": 0,
            },
            {
                "retrieval_event_id": "event-1",
                "paper_arxiv_id": "expanded-one-edge",
                "candidate_type": "expanded",
                "is_seed": False,
                "is_expanded": True,
                "expansion_path_count": 1,
            },
            {
                "retrieval_event_id": "event-1",
                "paper_arxiv_id": "expanded-two-edges",
                "candidate_type": "expanded",
                "is_seed": False,
                "is_expanded": True,
                "expansion_path_count": 2,
            },
            {
                "retrieval_event_id": "event-1",
                "paper_arxiv_id": "seed-and-expanded",
                "candidate_type": "seed_and_expanded",
                "is_seed": True,
                "is_expanded": True,
                "expansion_path_count": 1,
            },
        ],
    )

    overrides, diagnostics = MODULE.load_expanded_endpoint_path_overrides(path)

    assert overrides["event-1"] == {
        "seed": (0, 0.0),
        "expanded-one-edge": (1, 0.5),
        "expanded-two-edges": (2, 1.0),
        "seed-and-expanded": (1, 0.5),
    }
    assert diagnostics["nonzero_normalized_path_count"] == 3
    assert diagnostics["raw_path_count_distribution"] == {
        "0": 1,
        "1": 2,
        "2": 1,
    }
    assert diagnostics["edge_contribution"] == "seed endpoint +0; expanded endpoint +1"


def test_event_record_infers_graph_roles_from_candidate_type():
    budget = MODULE.BaselineBudget(
        query_id="q0",
        benchmark_idx=0,
        retrieval_event_id="q0:e1",
        iteration_idx=1,
        subquery_id=1,
        subquery="topic",
        k=1,
        selector_input_ids=("paper",),
    )
    event = MODULE._event_record(
        formula=MODULE.NEW_FORMULA,
        method="graph",
        budget=budget,
        pool_size=1,
        selected_rows=[
            {
                "paper_arxiv_id": "paper",
                "candidate_type": "seed_and_expanded",
            }
        ],
        start_rank=1,
        query_weight=0.40,
        subquery_weight=0.60,
    )

    assert event["selected"][0]["is_seed"] is True
    assert event["selected"][0]["is_expanded"] is True


def test_deep_merged_slices_use_baseline_k_not_saved_requested_top_k(tmp_path):
    budgets = {
        "q0:e1": MODULE.BaselineBudget(
            query_id="q0",
            benchmark_idx=0,
            retrieval_event_id="q0:e1",
            iteration_idx=1,
            subquery_id=1,
            subquery="topic",
            k=1,
            selector_input_ids=("d",),
        ),
        "q0:e2": MODULE.BaselineBudget(
            query_id="q0",
            benchmark_idx=0,
            retrieval_event_id="q0:e2",
            iteration_idx=2,
            subquery_id=1,
            subquery="topic",
            k=2,
            selector_input_ids=("c", "b"),
        ),
    }
    deep_rows = []
    for rank, (paper_id, score) in enumerate(
        (("a", 1.0), ("b", 0.8), ("c", 0.6), ("d", 0.4)), start=1
    ):
        deep_rows.append(
            {
                "paper_arxiv_id": paper_id,
                "query_score_normalized": score,
                "subquery_score_normalized": score,
                "deep_retrieval_rank_in_local_pool": rank,
                "rerankable": True,
                "rerank_score": score,
            }
        )
    path = tmp_path / "pool_records.jsonl"
    _write_jsonl(
        path,
        [
            {
                "query_id": "q0",
                "deep_pool_rows": deep_rows,
                "selector_slices": [
                    {
                        "selector_slice_idx": 1,
                        "retrieval_event_id": "q0:e1",
                        "selector_requested_top_k": 99,
                        "selector_input_arxiv_ids": ["d"],
                    },
                    {
                        "selector_slice_idx": 2,
                        "retrieval_event_id": "q0:e2",
                        "selector_requested_top_k": 99,
                        "selector_input_arxiv_ids": ["c", "b"],
                    },
                ],
            }
        ],
    )
    selections = MODULE._selection_container()
    event_rows = []
    occurrence_counts = Counter()

    seen = MODULE.replay_deep_merged(
        path,
        budgets,
        selections,
        event_rows,
        occurrence_counts,
        query_weight=0.40,
        subquery_weight=0.60,
    )

    assert seen == {"q0:e1", "q0:e2"}
    assert occurrence_counts[(MODULE.NEW_FORMULA, "deep_merged")] == 3
    assert selections[MODULE.NEW_FORMULA]["deep_merged"]["q0"] == {"a", "b", "c"}
    new_events = [row for row in event_rows if row["formula"] == MODULE.NEW_FORMULA]
    assert [row["baseline_selector_input_k"] for row in new_events] == [1, 2]
    assert [row["selection_start_rank"] for row in new_events] == [1, 2]
