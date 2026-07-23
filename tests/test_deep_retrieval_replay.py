import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_per_subquery_deep_retrieval as replay


class FixedScores:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def get_scores(self, _tokens):
        return self.scores


class RecordingSelector:
    def __init__(self):
        self.calls = []

    async def decide_for_subquery(self, papers, planner_checklist, **kwargs):
        ids = [paper.id for paper in papers]
        self.calls.append((ids, planner_checklist, kwargs["iteration_index"], kwargs["old_overview"]))
        selected = papers[:1]
        reasons = {selected[0].id: "first in chronological slice"} if selected else {}
        return selected, "unused overview", {}, {"reasons": reasons}


def _fake_rag(count=30):
    return SimpleNamespace(
        bm25_index=FixedScores(range(count, 0, -1)),
        bm25_index_to_id={index: f"doc-{index + 1}" for index in range(count)},
        paper_metadata={
            f"doc-{index + 1}": {
                "arxiv_id": f"p{index + 1}",
                "title": f"paper {index + 1}",
                "abstract": "body",
                "date": "2020-01",
            }
            for index in range(count)
        },
        _preprocess_text_for_bm25=lambda _text: ["query"],
    )


def _planner(checklist):
    return {
        "benchmark_idx": 36,
        "iteration_idx": 1,
        "planner_checklist": checklist,
        "planner_input_state": {"retrieval_exclusion_arxiv_ids": []},
        "subqueries": [
            {
                "subquery_id": 1,
                "subquery": "stable text",
                "target_k": 2,
                "link_type": "derive",
                "parent_subquery_id": 0,
                "subquery_before_date": "2024-09",
            }
        ],
    }


def _paper_row(checklist, paper_id, *, local=False, rerank_rank=1):
    row = {
        "benchmark_idx": 36,
        "query_id": "q36",
        "query": "query",
        "query_date": "2024-09",
        "iteration_idx": 1,
        "subquery_id": 1,
        "subquery": "stable text",
        "subquery_target_k": 2,
        "subquery_before_date": "2024-09",
        "subquery_link_type": "derive",
        "parent_subquery_id": 0,
        "planner_checklist": checklist,
        "retrieval_event_id": "q36:retrieval:i1:s1:p1",
        "retrieval_page_idx": 1,
        "retrieval_offset": 0,
        "selector_top_k": 2,
        "paper_arxiv_id": paper_id,
    }
    if local:
        row["rerank_rank"] = rerank_rank
    else:
        row["observed_retrieval_rank"] = rerank_rank
    return row


def test_only_merged_subquery_method_is_exposed():
    assert replay.METHODS == (replay.MERGED_METHOD,)


def test_baseline_compatible_exclude_then_offset_can_skip_unseen_papers():
    pools, diagnostics = replay.retrieve_bm25_requests(
        _fake_rag(),
        subquery="query",
        before_date="2024-09",
        requests={
            "page1": {"offset": 0, "count": 10, "exclude_arxiv_ids": []},
            "page2": {"offset": 10, "count": 10, "exclude_arxiv_ids": ["p3", "p7"]},
            "merged": {"offset": 0, "count": 20, "exclude_arxiv_ids": []},
        },
    )

    assert [row["paper_arxiv_id"] for row in pools["page1"]] == [f"p{i}" for i in range(1, 11)]
    assert [row["paper_arxiv_id"] for row in pools["page2"]] == [f"p{i}" for i in range(13, 23)]
    assert [row["paper_arxiv_id"] for row in pools["merged"]] == [f"p{i}" for i in range(1, 21)]
    assert pools["page2"][0]["deep_retrieval_rank_after_exclusion"] == 11
    assert diagnostics["page1"]["date_valid_positive_document_count_before_arxiv_dedup"] == 30
    assert diagnostics["page2"]["actual_count"] == 10

    single_pools, single_diagnostics = replay.retrieve_bm25_requests(
        _fake_rag(),
        subquery="query",
        before_date="2024-09",
        requests={"page1": {"offset": 0, "count": 10, "exclude_arxiv_ids": []}},
    )
    assert single_pools["page1"] == pools["page1"]
    assert single_diagnostics["page1"] == diagnostics["page1"]


def test_final_planner_trajectory_and_last_event_block_are_canonicalized():
    old = "old checklist"
    new = "new checklist"
    baseline_rows = [
        _paper_row(old, "old-a", rerank_rank=1),
        _paper_row(old, "old-b", rerank_rank=2),
        _paper_row(new, "new-a", rerank_rank=1),
        _paper_row(new, "new-b", rerank_rank=2),
    ]
    local_rows = [
        _paper_row(old, "old-expanded", local=True),
        _paper_row(new, "new-expanded", local=True, rerank_rank=1),
        _paper_row(new, "new-seed", local=True, rerank_rank=2),
        _paper_row(new, "new-seed", local=True, rerank_rank=2),
    ]

    context = replay.build_source_context(
        {
            "idx": 36,
            "query": "query",
            "ground_truth_arxiv_ids": ["new-expanded"],
            "postprocess_results": {
                "query_id": "q36",
                "baseline": {
                    "candidate_arxiv_ids": ["new-a", "new-b"],
                    "selected_arxiv_ids": [],
                },
            },
        },
        [_planner(old), _planner(new)],
        baseline_rows,
        local_rows,
    )

    event = context["events"][0]
    assert event["planner_checklist"] == new
    assert event["baseline_seed_arxiv_ids"] == ["new-a", "new-b"]
    assert event["source_local_graph_pool_arxiv_ids"] == ["new-expanded", "new-seed"]
    assert event["source_local_graph_pool_size"] == 2
    assert context["canonicalization"]["planner"]["duplicate_planner_row_count"] == 1


def test_matching_event_blocks_do_not_merge_across_nonmatching_rows():
    expected, _ = replay.planner_expectations([_planner("new")])
    rows = [
        _paper_row("new", "stale-new", rerank_rank=1),
        _paper_row("old", "interleaved-old", rerank_rank=1),
        _paper_row("new", "final-new", rerank_rank=1),
    ]

    blocks, _, stats = replay.canonical_event_blocks(rows, expected)

    assert [row["paper_arxiv_id"] for row in blocks["q36:retrieval:i1:s1:p1"]] == ["final-new"]
    assert stats["replaced_earlier_event_block_count"] == 1


def test_partial_newer_planner_retry_does_not_replace_committed_complete_trajectory():
    old_first = _planner("old-1")
    old_second = _planner("old-2")
    old_second["iteration_idx"] = 2
    old_second["subqueries"][0].update({"subquery_id": 2, "subquery": "old second"})
    new_partial = _planner("new-partial")

    def row(checklist, paper_id, iteration, subquery_id, subquery, event_id, *, local=False):
        value = _paper_row(checklist, paper_id, local=local, rerank_rank=1)
        value.update(
            {
                "iteration_idx": iteration,
                "subquery_id": subquery_id,
                "subquery": subquery,
                "retrieval_event_id": event_id,
            }
        )
        return value

    baseline = [
        row("old-1", "old-a", 1, 1, "stable text", "old:i1:s1"),
        row("old-2", "old-b", 2, 2, "old second", "old:i2:s2"),
        row("new-partial", "new-a", 1, 1, "stable text", "new:i1:s1"),
    ]
    local = [
        row("old-1", "old-expanded-a", 1, 1, "stable text", "old:i1:s1", local=True),
        row("old-2", "old-expanded-b", 2, 2, "old second", "old:i2:s2", local=True),
        row("new-partial", "new-expanded", 1, 1, "stable text", "new:i1:s1", local=True),
    ]
    context = replay.build_source_context(
        {
            "idx": 36,
            "query": "query",
            "ground_truth_arxiv_ids": [],
            "postprocess_results": {
                "query_id": "q36",
                "baseline": {
                    "candidate_arxiv_ids": ["old-a", "old-b"],
                    "selected_arxiv_ids": [],
                },
            },
        },
        [old_first, old_second, new_partial],
        baseline,
        local,
    )

    assert [event["retrieval_event_id"] for event in context["events"]] == ["old:i1:s1", "old:i2:s2"]
    assert context["canonicalization"]["planner"]["selected_planner_trajectory_index"] == 1


def test_contiguous_selector_slices_use_each_event_actual_topk():
    ids = [f"p{index}" for index in range(10)]
    events = [{"selector_top_k": 2}, {"selector_top_k": 3}, {"selector_top_k": 1}]

    slices = replay.contiguous_selector_slices(ids, events)

    assert [value["selector_input_arxiv_ids"] for value in slices] == [
        ["p0", "p1"],
        ["p2", "p3", "p4"],
        ["p5"],
    ]
    assert [value["rerank_start_rank"] for value in slices] == [1, 3, 6]


def test_text_only_formula_uses_shared_four_factor_weights_with_zero_graph_features():
    assert replay.text_only_formula_score(1.0, 1.0) == 0.70
    assert replay.text_only_formula_score(0.5, 0.25) == 0.25


def test_merged_method_calls_selector_with_chronological_disjoint_slices(monkeypatch):
    deep_ids = [f"p{index}" for index in range(8)]
    ordered = deep_ids[:7]
    features = {
        paper_id: {
            "paper_arxiv_id": paper_id,
            "rerank_score": 0.7 - rank / 100,
            "rerank_rank": rank,
            "intent_score": 0.0,
            "path_count": 0,
            "path_count_normalized": 0.0,
        }
        for rank, paper_id in enumerate(ordered, start=1)
    }
    monkeypatch.setattr(replay, "rerank_text_pool", lambda *args, **kwargs: (ordered, features, ["p7"]))
    events = []
    for index, top_k in enumerate((2, 3, 1), start=1):
        events.append(
            {
                "retrieval_event_id": f"event-{index}",
                "iteration_idx": index,
                "retrieval_page_idx": index,
                "retrieval_offset": (index - 1) * 10,
                "subquery_id": "1",
                "subquery": "stable text",
                "subquery_before_date": "2024-09",
                "subquery_link_type": "continue" if index > 1 else "derive",
                "parent_subquery_id": 0,
                "planner_checklist": f"checklist-{index}",
                "selector_top_k": top_k,
                "source_local_graph_pool_size": (3, 3, 2)[index - 1],
                "source_local_graph_pool_arxiv_ids": (
                    ["p0", "p1", "p6"],
                    ["p2", "p3", "p4"],
                    ["p5", "p6"],
                )[index - 1],
                "retrieval_exclusion_arxiv_ids": [],
            }
        )
    group = {
        "group_order": 1,
        "subquery_id": "1",
        "subquery": "stable text",
        "subquery_before_date": "2024-09",
        "events": events,
        "source_event_ids": [event["retrieval_event_id"] for event in events],
        "source_event_count": 3,
        "source_local_graph_pool_occurrence_budget": 8,
        "source_local_graph_pool_union_count": 7,
        "source_local_graph_pool_overlap_occurrence_count": 1,
        "source_local_graph_pool_union_over_sum": 7 / 8,
        "source_local_graph_pool_union_arxiv_ids": deep_ids[:7],
        "source_baseline_seed_occurrence_count": 6,
        "source_baseline_seed_union_count": 6,
        "source_baseline_seed_union_arxiv_ids": deep_ids[:6],
        "frozen_first_event_exclusion_arxiv_ids": [],
    }
    context = {
        "query_id": "q",
        "benchmark_idx": 0,
        "query": "query",
        "gt_ids": {"p0", "p2", "p5"},
        "events": events,
        "groups": [group],
        "canonicalization": {},
    }
    pool = [
        {
            "paper_arxiv_id": paper_id,
            "_metadata": {"title": paper_id, "abstract": "body", "date": "2020-01"},
        }
        for paper_id in deep_ids
    ]
    selector = RecordingSelector()

    output = asyncio.run(
        replay.build_merged_method_output(
            context=context,
            pools={(replay.MERGED_METHOD, "1"): pool},
            retrieval_diagnostics={(replay.MERGED_METHOD, "1"): {}},
            selector=selector,
            save_level="full",
            run_signature="test",
        )
    )

    assert [call[0] for call in selector.calls] == [
        ["p0", "p1"],
        ["p2", "p3", "p4"],
        ["p5"],
    ]
    assert [call[1] for call in selector.calls] == ["checklist-1", "checklist-2", "checklist-3"]
    assert {call[3] for call in selector.calls} == {""}
    assert output["summary"]["selected_arxiv_ids"] == ["p0", "p2", "p5"]
    assert output["summary"]["rerank_input_arxiv_ids"] == deep_ids[:6]
    assert output["summary"]["deep_pool_count"] == 8
    assert output["summary"]["rerankable_occurrence_count"] == 7
    assert output["paper_rows"][-1]["paper_arxiv_id"] == "p7"
    assert output["paper_rows"][-1]["rerankable"] is False
    assert [row["assigned_retrieval_event_id"] for row in output["paper_rows"][:6]] == [
        "event-1",
        "event-1",
        "event-2",
        "event-2",
        "event-2",
        "event-3",
    ]


def test_selector_free_summary_marks_selection_metrics_unavailable():
    context = {
        "query_id": "q",
        "benchmark_idx": 0,
        "gt_ids": {"a", "b"},
        "events": [{}],
        "groups": [{}],
    }

    summary = replay.build_query_summary(
        context=context,
        method=replay.MERGED_METHOD,
        deep_pool_ids=["a", "x", "y"],
        selector_input_ids=["a", "x"],
        selected_ids=[],
        selector_enabled=False,
        selector_call_count=0,
        deep_requested_occurrences=3,
        deep_actual_occurrences=3,
        rerankable_occurrences=3,
        source_graph_pool_ids=["a", "x", "y"],
    )

    assert summary["deep_pool_recall"] == 0.5
    assert summary["candidate_recall"] == 0.5
    assert summary["selected_count"] is None
    assert summary["selection_recall"] is None

    aggregate = replay.aggregate_query_outputs([{"summary": summary}])
    assert aggregate["avg_selection_recall"] is None
    assert aggregate["total_selected_count"] is None


def test_aggregate_f1_exposes_mean_query_and_main_table_macro_harmonic():
    base = {
        "query_id": "q",
        "benchmark_idx": 0,
        "events": [{}],
        "groups": [{}],
    }
    first = replay.build_query_summary(
        context={**base, "gt_ids": {"a"}},
        method=replay.MERGED_METHOD,
        deep_pool_ids=["a", "x"],
        selector_input_ids=["a", "x"],
        selected_ids=["a"],
        selector_enabled=True,
        selector_call_count=1,
        deep_requested_occurrences=2,
        deep_actual_occurrences=2,
        rerankable_occurrences=2,
        source_graph_pool_ids=["a", "x"],
    )
    second = replay.build_query_summary(
        context={**base, "gt_ids": {"b"}},
        method=replay.MERGED_METHOD,
        deep_pool_ids=["b"],
        selector_input_ids=["b"],
        selected_ids=["b"],
        selector_enabled=True,
        selector_call_count=1,
        deep_requested_occurrences=1,
        deep_actual_occurrences=1,
        rerankable_occurrences=1,
        source_graph_pool_ids=["b"],
    )

    aggregate = replay.aggregate_query_outputs([{"summary": first}, {"summary": second}])

    assert aggregate["mean_query_rerank_input_f1"] == pytest.approx((2 / 3 + 1) / 2)
    assert aggregate["avg_candidate_f1"] == pytest.approx(2 * 1.0 * 0.75 / 1.75)
    assert aggregate["avg_candidate_f1"] != aggregate["mean_query_rerank_input_f1"]


def test_output_directory_rejects_a_different_signature_when_query_files_exist(tmp_path):
    replay.atomic_write_json(tmp_path / "run_manifest.json", {"run_signature": "old"})
    replay.atomic_write_json(tmp_path / replay.MERGED_METHOD / "queries" / "000000.json", {"ok": True})

    with pytest.raises(ValueError, match="different run signature"):
        replay.ensure_compatible_output_dir(tmp_path, "new")
