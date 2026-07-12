import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from graph_methods import ArtifactWriter, PerSubqueryProcessor


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


def test_artifact_query_staging_and_checkpoint_reconciliation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        writer = ArtifactWriter(tmp, "full")

        writer.begin_query("q1", 1)
        writer.append("per_subquery/paper_rows.jsonl", {"query_id": "q1", "benchmark_idx": 1, "paper": "a"})
        assert not (root / "per_subquery/paper_rows.jsonl").exists()
        writer.commit_query()
        assert json.loads((root / "per_subquery/paper_rows.jsonl").read_text()) == {
            "benchmark_idx": 1,
            "paper": "a",
            "query_id": "q1",
        }

        # Simulate a crash during canonical flush for an uncommitted query.
        writer.append("per_subquery/paper_rows.jsonl", {"query_id": "q2", "benchmark_idx": 2, "paper": "b"})
        writer.append("per_subquery/filter_stats.jsonl", {"query_id": "q2", "kept": 3})
        with (root / "per_subquery/filter_stats.jsonl").open("a", encoding="utf-8") as handle:
            handle.write('{"query_id":"q2"')
        stale_stage = root / ".staging/stale/per_subquery"
        stale_stage.mkdir(parents=True)
        (stale_stage / "paper_rows.jsonl").write_text('{"query_id":"q3"}\n')

        stats = writer.reconcile_with_checkpoint({1}, {"q1"})

        rows = [json.loads(line) for line in (root / "per_subquery/paper_rows.jsonl").read_text().splitlines()]
        assert rows == [{"benchmark_idx": 1, "paper": "a", "query_id": "q1"}]
        assert (root / "per_subquery/filter_stats.jsonl").read_text() == ""
        assert not (root / ".staging").exists()
        assert stats["removed_rows"] == 2
        assert stats["removed_malformed_rows"] == 1
        assert stats["removed_staging_directories"] == 1


def test_aborted_artifact_query_never_reaches_canonical_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        writer = ArtifactWriter(tmp, "full")
        writer.begin_query("q-abort", 4)
        writer.append("global/final_paper_rows.jsonl", {"query_id": "q-abort", "benchmark_idx": 4})
        writer.abort_query()

        assert not (Path(tmp) / "global/final_paper_rows.jsonl").exists()
