import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "replay_graph_rerank_formulas.py"
SPEC = importlib.util.spec_from_file_location("replay_graph_rerank_formulas", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


Candidate = MODULE.Candidate


def _candidate(
    paper_id,
    *,
    is_seed=False,
    is_expanded=True,
    q=0.0,
    sq=0.0,
    intent=0.0,
    path=0.0,
    seeds=1,
    stored_top=False,
):
    return Candidate(
        paper_id=paper_id,
        is_seed=is_seed,
        is_expanded=is_expanded,
        observed_rank=1 if is_seed else None,
        query_score=q,
        subquery_score=sq,
        intent_score=intent,
        original_path_score=path,
        source_seed_count=seeds,
        stored_top=stored_top,
    )


def test_source_support_does_not_reward_high_degree_seed():
    seed = _candidate("seed", is_seed=True, seeds=100)
    one_path = _candidate("one", seeds=1)
    three_paths = _candidate("three", seeds=3)

    assert seed.source_support == 0.0
    assert one_path.source_support == 0.5
    assert three_paths.source_support == 1.0


def test_semantic_only_matches_deep_q_sq_ratio_and_ignores_other_features():
    low_semantic = _candidate(
        "low-semantic",
        q=0.2,
        sq=0.3,
        intent=1.0,
        path=1.0,
    )
    high_semantic = _candidate(
        "high-semantic",
        q=0.8,
        sq=0.7,
        intent=0.0,
        path=0.0,
    )

    scores = MODULE._score_maps([low_semantic, high_semantic])["semantic_only"]

    assert scores["low-semantic"] == pytest.approx((3.0 * 0.2 + 4.0 * 0.3) / 7.0)
    assert scores["high-semantic"] == pytest.approx((3.0 * 0.8 + 4.0 * 0.7) / 7.0)
    assert scores["high-semantic"] > scores["low-semantic"]


def test_semantic_40_60_uses_exact_requested_weights():
    row = _candidate(
        "candidate",
        q=0.25,
        sq=0.75,
        intent=1.0,
        path=1.0,
    )

    score = MODULE._score_maps([row])["semantic_40_60"]["candidate"]

    assert score == pytest.approx(0.40 * 0.25 + 0.60 * 0.75)


def test_graph_novel_quota_preserves_k_and_reserves_novel_candidate():
    rows = [
        _candidate("seed", is_seed=True, q=1.0, sq=1.0, stored_top=True),
        _candidate("common", q=0.9, sq=0.9, intent=1.0, stored_top=True),
        _candidate("novel-a", q=0.7, sq=0.7, intent=1.0),
        _candidate("novel-b", q=0.6, sq=0.6, intent=0.75),
    ]

    selected = MODULE.select_candidates(
        "graph_novel_quota_30",
        rows,
        top_k=2,
        deep_pool_ids={"seed", "common"},
        quota_ratio=0.30,
    )

    selected_ids = [candidate.paper_id for candidate, _ in selected]
    assert len(selected_ids) == 2
    assert any(paper_id.startswith("novel-") for paper_id in selected_ids)


def test_replay_reconstructs_stored_candidates_and_recovers_novel_gt(tmp_path):
    run_dir = tmp_path / "run"
    artifact_dir = run_dir / "onepass_artifacts" / "per_subquery"
    artifact_dir.mkdir(parents=True)
    detailed = {
        "idx": 0,
        "ground_truth_arxiv_ids": ["novel-gt"],
        "postprocess_results": {
            "query_id": "q0",
            "baseline": {"candidate_arxiv_ids": ["seed"]},
            "per_subquery": {"candidate_arxiv_ids": ["seed", "common"]},
            "deep_merged": {
                "query_id": "q0",
                "source_graph_pool_arxiv_ids": ["seed", "common", "novel-gt"],
                "deep_pool_arxiv_ids": ["seed", "common"],
            },
        },
    }
    (run_dir / "detailed_results.jsonl").write_text(json.dumps(detailed) + "\n", encoding="utf-8")

    base = {
        "benchmark_idx": 0,
        "query_id": "q0",
        "retrieval_event_id": "q0-e1",
        "iteration_idx": 1,
        "subquery_id": 1,
        "subquery": "test",
        "selector_top_k": 2,
        "query_score_normalized": 0.0,
        "subquery_score_normalized": 0.0,
        "intent_score": 0.0,
        "path_count_normalized": 0.0,
        "observed_retrieval_rank": None,
        "source_seed_arxiv_ids": [],
        "is_seed": False,
        "is_expanded": True,
        "in_selector_topk": False,
    }
    rows = [
        {
            **base,
            "paper_arxiv_id": "seed",
            "is_seed": True,
            "is_expanded": False,
            "observed_retrieval_rank": 1,
            "query_score_normalized": 1.0,
            "subquery_score_normalized": 1.0,
            "in_selector_topk": True,
        },
        {
            **base,
            "paper_arxiv_id": "common",
            "query_score_normalized": 0.9,
            "subquery_score_normalized": 0.9,
            "intent_score": 1.0,
            "source_seed_arxiv_ids": ["seed"],
            "in_selector_topk": True,
        },
        {
            **base,
            "paper_arxiv_id": "novel-gt",
            "query_score_normalized": 0.7,
            "subquery_score_normalized": 0.7,
            "intent_score": 1.0,
            "source_seed_arxiv_ids": ["seed"],
        },
    ]
    with (artifact_dir / "paper_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "out"
    result = MODULE.replay(
        run_dir,
        output_dir,
        strategies=("stored_current", "graph_novel_quota_30"),
        quota_ratio=0.30,
    )

    assert result["strategies"]["stored_current"]["total_candidate_gt_count"] == 0
    assert result["strategies"]["graph_novel_quota_30"]["total_candidate_gt_count"] == 1
    assert result["strategies"]["graph_novel_quota_30"]["graph_only_candidate_gt_count"] == 1
    assert (output_dir / "event_results.jsonl").exists()
