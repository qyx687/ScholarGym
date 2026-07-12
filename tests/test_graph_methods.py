import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from graph_methods import PerSubqueryProcessor


class FakeS2:
    def expand(self, seed_ids, method, limit):
        return [
            {
                "seed_arxiv_id": "2001.00001",
                "expanded_arxiv_id": "2001.00003",
                "seed_s2_paper_id": "S1",
                "expanded_s2_paper_id": "E3",
                "edge_type": "citation",
                "edge_rank": 1,
                "intents": ["methodology"],
                "is_influential": True,
            },
            {
                "seed_arxiv_id": "2001.00002",
                "expanded_arxiv_id": "2001.00003",
                "seed_s2_paper_id": "S2",
                "expanded_s2_paper_id": "E3",
                "edge_type": "reference",
                "edge_rank": 2,
                "intents": ["background"],
                "is_influential": False,
            },
            {
                "seed_arxiv_id": "2001.00001",
                "expanded_arxiv_id": "2003.00004",
                "edge_type": "citation",
                "edge_rank": 3,
                "intents": [],
                "is_influential": False,
            },
        ]

    def snapshot_stats(self):
        return {}


def test_expanded_candidate_has_scores_rank_and_all_seed_origins():
    paper_db = {
        "2001.00001": {"title": "graph retrieval", "abstract": "query method", "date": "2001-01"},
        "2001.00002": {"title": "retrieval", "abstract": "subquery method", "date": "2001-02"},
        "2001.00003": {"title": "expanded graph method", "abstract": "query subquery", "date": "2001-03"},
        "2003.00004": {"title": "future paper", "abstract": "query", "date": "2003-01"},
    }
    processor = PerSubqueryProcessor(
        paper_db,
        FakeS2(),
        scoring_backend="bm25",
        embedding_provider=None,
    )
    event = {
        "query_id": "q1",
        "query": "graph query",
        "query_date": "2002-01",
        "iteration_idx": 1,
        "subquery_id": 7,
        "subquery": "retrieval method",
        "subquery_before_date": "2002-01",
        "results_per_query": 2,
        "seed_papers": [
            {"paper_arxiv_id": "2001.00001", "observed_retrieval_score": 5.0, "observed_retrieval_rank": 1},
            {"paper_arxiv_id": "2001.00002", "observed_retrieval_score": 4.0, "observed_retrieval_rank": 2},
        ],
    }
    result = processor.process(event)
    rows = {row["paper_arxiv_id"]: row for row in result["rows"]}
    expanded = rows["2001.00003"]
    assert expanded["retrieval_score_raw"] is not None
    assert expanded["retrieval_rank"] is not None
    assert expanded["source_seed_arxiv_ids"] == ["2001.00001", "2001.00002"]
    assert expanded["path_count"] == 2
    assert len([edge for edge in result["edges"] if edge["expanded_arxiv_id"] == "2001.00003"]) == 2
    assert "2003.00004" not in rows

    excluded = processor.process(event, exclude_arxiv_ids={"2001.00003"})
    assert "2001.00003" not in {row["paper_arxiv_id"] for row in excluded["rows"]}
    assert excluded["filter_stats"]["previously_selected_count"] == 2
