import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from metrics import MetricsCalculator
from structures import SubQuery
from utils import CheckpointManager


def test_global_and_local_rank_views_are_independent():
    subqueries = {1: SubQuery(id=1, text="graph retrieval")}
    views = MetricsCalculator.calculate_rank_views(
        subqueries=subqueries,
        global_rank_dicts={1: {"gt": {"rank": 80, "total": 1000}}},
        local_rank_dicts={1: {"gt": {"rank": 2, "total": 9}}},
        gt_arxiv_ids={"gt"},
        selected_paper_ids_tracker={"gt"},
        global_selected_min_rank_tracker={},
        local_selected_min_rank_tracker={},
        gt_rank_cutoff=100,
    )

    assert views["gt_rank"][0]["ranks"][0] == {"arxiv_id": "gt", "rank": 80, "total_rank": 1000}
    assert views["local_gt_rank"][0]["ranks"][0] == {"arxiv_id": "gt", "rank": 2, "total_rank": 9}
    assert abs(views["avg_distance"] - 0.2) < 1e-12
    assert abs(views["local_avg_distance"] - 0.98) < 1e-12
    assert views["updated_global_selected_min_rank_tracker"] == {"gt": 80}
    assert views["updated_local_selected_min_rank_tracker"] == {"gt": 2}


def test_local_rank_view_is_absent_when_graph_rerank_is_disabled():
    subqueries = {1: SubQuery(id=1, text="baseline")}
    views = MetricsCalculator.calculate_rank_views(
        subqueries=subqueries,
        global_rank_dicts={1: {"gt": {"rank": 5, "total": 50}}},
        local_rank_dicts=None,
        gt_arxiv_ids={"gt"},
        selected_paper_ids_tracker=set(),
        global_selected_min_rank_tracker={},
        local_selected_min_rank_tracker={"old": 3},
        gt_rank_cutoff=100,
    )

    assert views["local_gt_rank"] == []
    assert views["local_avg_distance"] == -1
    assert views["updated_local_selected_min_rank_tracker"] == {"old": 3}


def test_checkpoint_rebuild_preserves_both_distance_scopes():
    manager = CheckpointManager("unused.jsonl")
    results = {}
    manager._rebuild_deep_research_stats(
        {
            "iteration_results": [
                {
                    "iter_idx": 1,
                    "avg_distance": 0.2,
                    "local_avg_distance": 0.9,
                    "iteration_metrics": {},
                }
            ]
        },
        results,
        max_iterations=1,
    )

    assert results["avg_distance_iter_1"] == [0.2]
    assert results["local_avg_distance_iter_1"] == [0.9]
