import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "replay_dynamic_rerank_selector.py"
)
SPEC = importlib.util.spec_from_file_location(
    "replay_dynamic_rerank_selector", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RecordingSelector:
    def __init__(self):
        self.calls = []

    async def decide_for_subquery(
        self,
        *,
        user_query,
        sub_query,
        planner_checklist,
        papers,
        **kwargs,
    ):
        self.calls.append(
            {
                "query": user_query,
                "subquery": sub_query.text,
                "checklist": planner_checklist,
                "papers": [(paper.id, paper.score) for paper in papers],
            }
        )
        kept = [max(papers, key=lambda paper: paper.score)] if papers else []
        reasons = {kept[0].id: "highest arm-specific score"} if kept else {}
        return kept, "test overview", {}, {"reasons": reasons}


class ParseFailOnceSelector(RecordingSelector):
    def __init__(self):
        super().__init__()
        self.attempts = {}

    async def decide_for_subquery(self, *, papers, **kwargs):
        key = papers[0].id
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if self.attempts[key] == 1:
            self.calls.append({"papers": [(paper.id, paper.score) for paper in papers]})
            return [], "", {}, {"reasons": {}}
        return await super().decide_for_subquery(papers=papers, **kwargs)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _fixture_files(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    _write_jsonl(
        benchmark,
        [
            {
                "qid": "q1",
                "query": "find the target paper but exclude surveys",
                "cited_paper": [{"arxiv_id": "2002.00002"}],
                "gt_label": [1],
            }
        ],
    )
    pool = tmp_path / "pool.jsonl"
    _write_jsonl(
        pool,
        [
            {
                "query_id": "q1",
                "benchmark_idx": 0,
                "query": "find the target paper but exclude surveys",
                "retrieval_event_id": "q1:event:1",
                "iteration_idx": 1,
                "subquery_id": 7,
                "subquery": "target method papers",
                "subquery_before_date": "2024-09",
                "subquery_target_k": 2,
                "subquery_link_type": "derive",
                "parent_subquery_id": 0,
                "selector_top_k": 2,
                "planner_checklist": "include method papers; exclude surveys",
                "local_pool_rows": [],
            }
        ],
    )
    ranked = tmp_path / "ranked.jsonl"
    _write_jsonl(
        ranked,
        [
            {
                "query_id": "q1",
                "benchmark_idx": 0,
                "retrieval_event_id": "q1:event:1",
                "method": "legacy_static",
                "selector_top_k": 2,
                "rerank_policy_id": "static",
                "ranked_candidates": [
                    {
                        "paper_arxiv_id": "2001.00001",
                        "rerank_rank": 1,
                        "rerank_score": 0.9,
                        "selected_at_event_top_k": True,
                    },
                    {
                        "paper_arxiv_id": "2002.00002",
                        "rerank_rank": 2,
                        "rerank_score": 0.2,
                        "selected_at_event_top_k": True,
                    },
                ],
            },
            {
                "query_id": "q1",
                "benchmark_idx": 0,
                "retrieval_event_id": "q1:event:1",
                "method": "dynamic_policy",
                "selector_top_k": 2,
                "rerank_policy_id": "dynamic",
                "ranked_candidates": [
                    {
                        "paper_arxiv_id": "2002.00002",
                        "rerank_rank": 1,
                        "rerank_score": 0.95,
                        "selected_at_event_top_k": True,
                    },
                    {
                        "paper_arxiv_id": "2001.00001",
                        "rerank_rank": 2,
                        "rerank_score": 0.1,
                        "selected_at_event_top_k": True,
                    },
                ],
            },
        ],
    )
    paper_db = tmp_path / "paper_db.json"
    paper_db.write_text(
        json.dumps(
            {
                "2001.00001": {
                    "arxiv_id": "2001.00001",
                    "title": "Static favorite",
                    "abstract": "A distractor.",
                    "date": "2020-01-01",
                },
                "2002.00002": {
                    "arxiv_id": "2002.00002",
                    "title": "Dynamic favorite",
                    "abstract": "The target method paper.",
                    "date": "2020-02-01",
                },
            }
        ),
        encoding="utf-8",
    )
    return benchmark, pool, ranked, paper_db


def _run(tmp_path, selector, output_dir, precomputed_paths=None):
    benchmark, pool, ranked, paper_db = _fixture_files(tmp_path)
    return MODULE.replay(
        ranked_candidates_path=ranked,
        pool_records_path=pool,
        benchmark_path=benchmark,
        paper_db_path=paper_db,
        output_dir=output_dir,
        selector=selector,
        selector_config={
            "llm_model": "fake-qwen",
            "llm_gen_params": {"temperature": 0},
            "enable_reasoning": False,
            "browser_mode": "NONE",
        },
        precomputed_paths=precomputed_paths,
        concurrency=2,
    )


def test_replay_uses_each_arms_scores_and_resumes(tmp_path):
    output_dir = tmp_path / "output"
    selector = RecordingSelector()
    summary = _run(tmp_path, selector, output_dir)

    assert len(selector.calls) == 2
    calls_by_first_id = {call["papers"][0][0]: call for call in selector.calls}
    assert calls_by_first_id["2001.00001"]["papers"] == [
        ("2001.00001", 0.9),
        ("2002.00002", 0.2),
    ]
    assert calls_by_first_id["2002.00002"]["papers"] == [
        ("2002.00002", 0.95),
        ("2001.00001", 0.1),
    ]
    assert {call["checklist"] for call in selector.calls} == {
        "include method papers; exclude surveys"
    }
    assert summary["methods"]["legacy_static"]["main_table_selection_f1"] == 0.0
    assert summary["methods"]["dynamic_policy"]["main_table_selection_f1"] == 1.0
    assert summary["dynamic_minus_legacy"]["main_table_selection_f1"] == 1.0

    resumed_selector = RecordingSelector()
    resumed = _run(tmp_path, resumed_selector, output_dir)
    assert resumed_selector.calls == []
    assert resumed["selector_result_source_counts"] == {"api": 2}
    assert resumed["selector_loaded_from_checkpoint_count"] == 2


def test_exact_precomputed_static_decision_is_reused(tmp_path):
    precomputed = tmp_path / "precomputed.jsonl"
    _write_jsonl(
        precomputed,
        [
            {
                "query_id": "q1",
                "benchmark_idx": 0,
                "retrieval_event_id": "q1:event:1",
                "subquery": "target method papers",
                "planner_checklist": "include method papers; exclude surveys",
                "candidate_rows": [
                    {
                        "paper_arxiv_id": "2001.00001",
                        "rerank_rank": 1,
                        "rerank_score": 0.9,
                        "in_selector_topk": True,
                        "selector_selected": True,
                        "selector_reason": "saved",
                    },
                    {
                        "paper_arxiv_id": "2002.00002",
                        "rerank_rank": 2,
                        "rerank_score": 0.2,
                        "in_selector_topk": True,
                        "selector_selected": False,
                    },
                ],
                "selected_arxiv_ids": ["2001.00001"],
                "selector_overview": "saved overview",
            }
        ],
    )
    selector = RecordingSelector()
    summary = _run(
        tmp_path,
        selector,
        tmp_path / "precomputed_output",
        {"legacy_static": precomputed},
    )

    assert len(selector.calls) == 1
    assert selector.calls[0]["papers"][0] == ("2002.00002", 0.95)
    assert summary["selector_result_source_counts"] == {
        "api": 1,
        "precomputed": 1,
    }


def test_silent_legacy_parse_failure_is_retried(tmp_path):
    selector = ParseFailOnceSelector()
    output_dir = tmp_path / "retry_output"
    summary = _run(tmp_path, selector, output_dir)

    assert summary["complete"] is True
    assert selector.attempts == {"2001.00001": 2, "2002.00002": 2}
    decisions = [
        json.loads(line)
        for line in (output_dir / "selector_decisions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert {row["attempt_count"] for row in decisions} == {2}


def test_silently_unparsed_precomputed_decision_is_not_reused(tmp_path):
    precomputed = tmp_path / "empty_precomputed.jsonl"
    _write_jsonl(
        precomputed,
        [
            {
                "query_id": "q1",
                "retrieval_event_id": "q1:event:1",
                "subquery": "target method papers",
                "planner_checklist": "include method papers; exclude surveys",
                "candidate_rows": [
                    {
                        "paper_arxiv_id": "2001.00001",
                        "rerank_rank": 1,
                        "rerank_score": 0.9,
                        "in_selector_topk": True,
                    },
                    {
                        "paper_arxiv_id": "2002.00002",
                        "rerank_rank": 2,
                        "rerank_score": 0.2,
                        "in_selector_topk": True,
                    },
                ],
                "selected_arxiv_ids": [],
                "selector_overview": "",
            }
        ],
    )
    selector = RecordingSelector()
    summary = _run(
        tmp_path,
        selector,
        tmp_path / "empty_precomputed_output",
        {"legacy_static": precomputed},
    )

    assert len(selector.calls) == 2
    assert summary["selector_result_source_counts"] == {"api": 2}


def test_paired_bootstrap_is_deterministic_and_query_paired():
    rows = []
    for idx in range(3):
        rows.append(
            {
                "complete": True,
                "methods": {
                    "legacy_static": {
                        "selection_recall": 0.2,
                        "selection_precision": 0.2,
                        "selection_f1": 0.2,
                        "gt_count": 10,
                        "selected_count": 10,
                        "selected_gt_count": 2,
                    },
                    "dynamic_policy": {
                        "selection_recall": 0.4,
                        "selection_precision": 0.4,
                        "selection_f1": 0.4,
                        "gt_count": 10,
                        "selected_count": 10,
                        "selected_gt_count": 4,
                    },
                },
            }
        )

    result = MODULE.paired_bootstrap_selection_delta(rows, iterations=100, seed=7)

    metric = result["metrics"]["main_table_selection_f1"]
    assert metric["dynamic_minus_legacy"] == pytest.approx(0.2)
    assert metric["ci95_low"] == pytest.approx(0.2)
    assert metric["ci95_high"] == pytest.approx(0.2)
    assert result["query_win_tie_loss"] == {
        "dynamic_wins": 3,
        "ties": 0,
        "dynamic_losses": 0,
    }
