import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from graph_methods import ArtifactWriter, PerSubqueryProcessor
from onepass_postprocess import OnePassPostprocessor, _phase3_max_normalize, aggregate_postprocess_metrics


class FakeS2:
    def expand(self, seed_ids, method, limit):
        rows = []
        for index, seed in enumerate(seed_ids, start=1):
            rows.append(
                {
                    "seed_arxiv_id": seed,
                    "expanded_arxiv_id": "2001.00003",
                    "seed_s2_paper_id": f"S{index}",
                    "expanded_s2_paper_id": "E3",
                    "edge_type": "citation",
                    "edge_rank": index,
                    "intents": ["methodology"],
                    "is_influential": True,
                }
            )
        return rows

    def snapshot_stats(self):
        return {"api_calls": 0}


class FakeSelector:
    def __init__(self):
        self.subqueries = []

    async def decide_for_subquery(self, papers, return_details=False, **kwargs):
        self.subqueries.append(kwargs["sub_query"])
        selected = list(papers[:1])
        details = {"reasons": {selected[0].id: "kept"}} if selected else {"reasons": {}}
        return selected, "overview", {}, details


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def test_one_query_reuses_baseline_and_preserves_multi_subquery_seed_origins():
    paper_db = {
        "2001.00001": {"title": "seed one", "abstract": "graph query", "date": "2001-01"},
        "2001.00002": {"title": "seed two", "abstract": "retrieval", "date": "2001-02"},
        "2001.00003": {"title": "expanded", "abstract": "graph retrieval", "date": "2001-03"},
    }
    fake_s2 = FakeS2()
    fake_selector = FakeSelector()
    processor = PerSubqueryProcessor(paper_db, fake_s2, scoring_backend="bm25", embedding_provider=None)
    with tempfile.TemporaryDirectory() as tmp:
        writer = ArtifactWriter(tmp, "full")
        manager = OnePassPostprocessor(
            selector=fake_selector,
            paper_db=paper_db,
            writer=writer,
            s2_client=fake_s2,
            per_subquery_processor=processor,
            scoring_backend="bm25",
            embedding_provider=None,
            run_per_subquery=True,
            run_global=True,
        )
        base = {
            "query_id": "q1",
            "benchmark_idx": 0,
            "query": "graph query",
            "query_date": "2002-01",
            "iteration_idx": 1,
            # The CLI page-size setting is deliberately smaller than the three
            # actual retrieval records across this complete baseline query.
            "results_per_query": 1,
            "planner_checklist": "check",
            "retrieval_backend": "bm25",
        }
        events = [
            {
                **base,
                "subquery_id": 1,
                "subquery": "graph",
                "subquery_target_k": 7,
                "subquery_link_type": "derive",
                "subquery_before_date": "2002-01",
                "retrieval_page_idx": 1,
                "seed_papers": [{"paper_arxiv_id": "2001.00001", "observed_retrieval_score": 2, "observed_retrieval_rank": 1}],
                "baseline_selected_arxiv_ids": ["2001.00001"],
            },
            {
                **base,
                "subquery_id": 2,
                "subquery": "retrieval",
                "subquery_target_k": 8,
                "subquery_link_type": "expand",
                "subquery_before_date": "2002-01",
                "retrieval_page_idx": 1,
                "seed_papers": [
                    {"paper_arxiv_id": "2001.00001", "observed_retrieval_score": 3, "observed_retrieval_rank": 1},
                    {"paper_arxiv_id": "2001.00002", "observed_retrieval_score": 1, "observed_retrieval_rank": 2},
                ],
                "baseline_selected_arxiv_ids": [],
            },
        ]
        result = manager.process_query({"query": "graph query", "date": "2002-01"}, [], events, {"2001.00003"})
        # seed one occurs in two subqueries, so there are three retrieval
        # records but only two unique baseline papers.
        assert result["global"]["baseline_query_retrieval_count"] == 3
        assert result["global"]["baseline_unique_retrieved_paper_count"] == 2
        assert result["global"]["global_selector_top_k"] == 3
        assert result["global"]["selected_arxiv_ids"]
        assert [(sq.target_k, sq.link_type) for sq in fake_selector.subqueries[:2]] == [(7, "derive"), (8, "expand")]
        assert fake_selector.subqueries[-1].target_k == 3
        decisions = _read_jsonl(Path(tmp) / "global/selector_decisions.jsonl")
        assert len(decisions[0]["candidate_rows"]) == 3
        assert {row["global_selector_top_k"] for row in decisions[0]["candidate_rows"]} == {3}
        components = _read_jsonl(Path(tmp) / "global/subquery_paper_scores.jsonl")
        expanded = [row for row in components if row["paper_arxiv_id"] == "2001.00003"]
        assert {row["subquery_id"] for row in expanded} == {1, 2}
        edges = _read_jsonl(Path(tmp) / "global/expansion_edges.jsonl")
        seed_one_edges = [row for row in edges if row["seed_arxiv_id"] == "2001.00001"]
        assert {row["subquery_id"] for row in seed_one_edges} == {1, 2}


def test_global_phase3_normalization_is_raw_over_positive_maximum():
    normalized = _phase3_max_normalize({"a": 2.0, "b": 1.0, "c": -1.0})
    assert normalized == {"a": 2.0 / (2.0 + 1e-12), "b": 1.0 / (2.0 + 1e-12), "c": -1.0 / (2.0 + 1e-12)}
    assert _phase3_max_normalize({"a": -1.0, "b": -2.0}) == {"a": 0.0, "b": 0.0}


def test_postprocess_metrics_aggregate_unique_queries_and_both_methods():
    stale = {
        "idx": 0,
        "postprocess_results": {
            "per_subquery": {
                "gt_count": 99,
                "candidate_count": 99,
                "selected_count": 99,
                "candidate_gt_ids": [],
                "selected_gt_ids": [],
            }
        },
    }
    query_zero = {
        "idx": 0,
        "postprocess_results": {
            "per_subquery": {
                "gt_count": 2,
                "candidate_count": 4,
                "selected_count": 2,
                "candidate_gt_ids": ["a"],
                "selected_gt_ids": ["a"],
                "candidate_recall": 0.5,
                "candidate_precision": 0.25,
                "selection_recall": 0.5,
                "selection_precision": 0.5,
            },
            "global": {"error": "selector failed"},
        },
    }
    query_one = {
        "idx": 1,
        "postprocess_results": {
            "per_subquery": {
                "gt_count": 1,
                "candidate_count": 2,
                "selected_count": 1,
                "candidate_gt_ids": ["b"],
                "selected_gt_ids": ["b"],
                "candidate_recall": 1.0,
                "candidate_precision": 0.5,
                "selection_recall": 1.0,
                "selection_precision": 1.0,
            },
            "global": {
                "gt_count": 1,
                "candidate_count": 3,
                "selected_count": 1,
                "candidate_gt_ids": ["b"],
                "selected_gt_ids": [],
                "candidate_recall": 1.0,
                "candidate_precision": 1.0 / 3.0,
                "selection_recall": 0.0,
                "selection_precision": 0.0,
            },
        },
    }

    metrics = aggregate_postprocess_metrics([stale, query_zero, query_one])

    assert metrics["source_query_count"] == 2
    per_subquery = metrics["per_subquery"]
    assert per_subquery["evaluated_query_count"] == 2
    assert per_subquery["total_gt_count"] == 3
    assert per_subquery["avg_candidate_recall"] == 0.75
    assert per_subquery["avg_candidate_precision"] == 0.375
    assert per_subquery["micro_candidate_recall"] == 2.0 / 3.0
    assert per_subquery["micro_selection_precision"] == 2.0 / 3.0
    assert metrics["global"]["evaluated_query_count"] == 1
    assert metrics["global"]["failed_query_count"] == 1
