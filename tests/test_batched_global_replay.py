import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from replay_global_batched_selector import (
    DEFAULT_GLOBAL_CHECKLIST,
    EXISTING_METHOD,
    GroupCursor,
    NEW_METHOD,
    existing_formula_rerank,
    new_formula_rerank,
    select_in_independent_batches,
)


class FakeSelector:
    async def decide_for_subquery(self, papers, **kwargs):
        selected = papers[:1]
        reasons = {selected[0].id: "first in batch"} if selected else {}
        return selected, "batch overview", {}, {"reasons": reasons}


def test_existing_formula_replay_preserves_saved_global_order_and_scores():
    rows = [
        {"paper_arxiv_id": "b", "global_final_rank": 2, "global_final_score": 0.4},
        {"paper_arxiv_id": "a", "global_final_rank": 1, "global_final_score": 0.8},
    ]

    ordered, scores, _ = existing_formula_rerank(rows)

    assert ordered == ["a", "b"]
    assert scores == {"a": 0.8, "b": 0.4}


def test_global_new_formula_uses_minmax_max_subquery_intent_and_path_features():
    final_rows = [
        {"paper_arxiv_id": "a", "global_final_rank": 1, "query_score_raw": 1.0},
        {"paper_arxiv_id": "b", "global_final_rank": 2, "query_score_raw": 3.0},
        {"paper_arxiv_id": "c", "global_final_rank": 3, "query_score_raw": 2.0},
    ]
    component_rows = [
        {"paper_arxiv_id": "a", "subquery_id": 1, "subquery_score_raw": 0.0},
        {"paper_arxiv_id": "b", "subquery_id": 1, "subquery_score_raw": 1.0},
        {"paper_arxiv_id": "c", "subquery_id": 1, "subquery_score_raw": 2.0},
        {"paper_arxiv_id": "a", "subquery_id": 2, "subquery_score_raw": 2.0},
        {"paper_arxiv_id": "b", "subquery_id": 2, "subquery_score_raw": 0.0},
        {"paper_arxiv_id": "c", "subquery_id": 2, "subquery_score_raw": 1.0},
    ]
    edges = [
        {"seed_arxiv_id": "a", "expanded_arxiv_id": "b", "edge_type": "citation", "intents": ["methodology"]},
        {"seed_arxiv_id": "a", "expanded_arxiv_id": "c", "edge_type": "reference", "intents": ["background"]},
    ]

    ordered, scores, features = new_formula_rerank(final_rows, component_rows, edges, {"a"})

    assert ordered == ["b", "c", "a"]
    assert abs(scores["a"] - 0.55) < 1e-12
    assert abs(scores["b"] - 0.65) < 1e-12
    assert abs(scores["c"] - 0.6025) < 1e-12
    assert features["b"]["intent_score"] == 1.0
    assert features["a"]["intent_score"] == 0.0
    assert features["a"]["path_count"] == 2


def test_selector_receives_independent_contiguous_batches_of_ten():
    candidate_ids = [f"p{index:02d}" for index in range(25)]
    scores = {paper_id: 1.0 - index / 100.0 for index, paper_id in enumerate(candidate_ids)}
    paper_db = {
        paper_id: {"title": paper_id, "abstract": "abstract", "date": "2020-01"}
        for paper_id in candidate_ids
    }

    selected, batches, reasons = asyncio.run(
        select_in_independent_batches(
            FakeSelector(),
            method=NEW_METHOD,
            query_text="query",
            benchmark_idx=7,
            candidate_ids=candidate_ids,
            scores=scores,
            paper_db=paper_db,
            batch_size=10,
        )
    )

    assert [row["selector_batch_size"] for row in batches] == [10, 10, 5]
    assert batches[0]["input_arxiv_ids"] == candidate_ids[:10]
    assert batches[1]["input_arxiv_ids"] == candidate_ids[10:20]
    assert selected == ["p00", "p10", "p20"]
    assert set(reasons) == {"p00", "p10", "p20"}
    assert {row["checklist"] for row in batches} == {DEFAULT_GLOBAL_CHECKLIST}
    assert {row["old_overview"] for row in batches} == {""}


def test_group_cursor_supports_resume_order_and_missing_optional_groups():
    with tempfile.TemporaryDirectory() as tmp:
        full_path = Path(tmp) / "full.jsonl"
        optional_path = Path(tmp) / "optional.jsonl"
        full_path.write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {"benchmark_idx": 2, "value": "first"},
                    {"benchmark_idx": 0, "value": "resumed-later"},
                )
            )
            + "\n"
        )
        optional_path.write_text(json.dumps({"benchmark_idx": 0, "edge": "only-later"}) + "\n")
        allowed = {0, 2}
        order = {2: 0, 0: 1}

        full = GroupCursor(full_path, allowed, order)
        optional = GroupCursor(optional_path, allowed, order)

        assert full.take(2)[0]["value"] == "first"
        assert optional.take(2) == []
        assert full.take(0)[0]["value"] == "resumed-later"
        assert optional.take(0)[0]["edge"] == "only-later"
