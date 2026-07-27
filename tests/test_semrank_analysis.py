import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_semrank_qsq.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_semrank_qsq", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_binary_metrics_report_both_paper_cutoffs():
    ranking = [f"p{index}" for index in range(1, 21)]
    metrics = MODULE.binary_metrics(ranking, {"p11"})

    assert metrics["recall@10"] == 0.0
    assert metrics["recall@20"] == 1.0
    assert metrics["map@10"] == 0.0
    assert metrics["map@20"] == 1.0 / 11.0
    assert metrics["ndcg@10"] == 0.0
    assert metrics["ndcg@20"] == 1.0 / MODULE.math.log2(12)


def test_analysis_metrics_and_scientific_audit(tmp_path):
    artifacts = tmp_path / "online_artifacts"
    artifacts.mkdir()
    _write_jsonl(
        artifacts / "query_results.jsonl",
        [
            {
                "query_id": "q1",
                "gt_count": 2,
                "candidate_count": 2,
                "selected_count": 1,
                "candidate_gt_ids": ["p1"],
                "selected_gt_ids": ["p1"],
                "semrank_stats_delta": {
                    "llm.query_concept_llm_calls": 1,
                    "llm.paper_concept_llm_calls": 2,
                    "paper_profile.paper_concept_cache_hits": 1,
                    "paper_profile.paper_concept_cache_misses": 2,
                },
            }
        ],
    )
    _write_jsonl(
        artifacts / "semrank_query_profiles.jsonl",
        [
            {
                "query_id": "q1",
                "query_profile_id": "profile-q1",
                "date_cutoff": "2020-01",
                "candidate_topics": [
                    {"concept": "retrieval", "frequency": 2}
                ],
                "candidate_keyphrases": [],
                "selected_concepts": ["retrieval"],
            },
            {
                "query_id": "q1",
                "query_profile_id": "profile-q1",
                "date_cutoff": "2020-01",
                "candidate_topics": [
                    {"concept": "retrieval", "frequency": 2}
                ],
                "candidate_keyphrases": [],
                "selected_concepts": ["retrieval"],
            },
        ],
    )
    paper_rows = [
        {
            "query_id": "q1",
            "retrieval_event_id": "e1",
            "paper_arxiv_id": "p1",
            "query_score_normalized": 1.0,
            "subquery_score_normalized": 0.5,
            "semrank_base_score": 0.7,
            "semrank_base_score_z": 1.0,
            "semrank_concept_score": 0.9,
            "semrank_concept_score_z": 1.0,
            "rerank_score": 2.0,
            "rerank_rank": 1,
            "rerank_method": "semrank_qsq",
            "semrank_fallback_used": False,
            "candidate_pool_signature": (
                "0a28944dab55ca77f6772f5c895d8db7"
                "e84f8b3dcd1c5d4d2a01b14f60dbae7e"
            ),
            "passed_date_cutoff": True,
            "is_ground_truth": True,
            "is_seed": False,
            "is_expanded": True,
            "in_selector_topk": True,
            "selector_selected": True,
        },
        {
            "query_id": "q1",
            "retrieval_event_id": "e1",
            "paper_arxiv_id": "p2",
            "query_score_normalized": 0.0,
            "subquery_score_normalized": 0.5,
            "semrank_base_score": 0.3,
            "semrank_base_score_z": -1.0,
            "semrank_concept_score": 0.1,
            "semrank_concept_score_z": -1.0,
            "rerank_score": -2.0,
            "rerank_rank": 2,
            "rerank_method": "semrank_qsq",
            "semrank_fallback_used": False,
            "candidate_pool_signature": (
                "0a28944dab55ca77f6772f5c895d8db7"
                "e84f8b3dcd1c5d4d2a01b14f60dbae7e"
            ),
            "passed_date_cutoff": True,
            "is_ground_truth": False,
            "is_seed": True,
            "is_expanded": False,
            "in_selector_topk": True,
            "selector_selected": False,
        },
    ]
    # A resumed failed query can append the same deterministic event again.
    # Analysis must keep the latest complete occurrence rather than double
    # counting it.
    _write_jsonl(
        artifacts / "paper_rows.jsonl",
        paper_rows + paper_rows,
    )
    _write_jsonl(
        artifacts / "semrank_event_profiles.jsonl",
        [
            {
                "retrieval_event_id": "e1",
                "candidate_count": 2,
                "candidate_ids_preserved": True,
                "base_mean": 0.5,
                "base_std": 0.2,
                "concept_mean": 0.5,
                "concept_std": 0.4,
                "candidate_pool_signature": (
                    "0a28944dab55ca77f6772f5c895d8db7"
                    "e84f8b3dcd1c5d4d2a01b14f60dbae7e"
                ),
            }
        ],
    )

    queries = MODULE.latest_by_query(artifacts / "query_results.jsonl")
    e2e = MODULE.end_to_end(queries)
    assert e2e["selection_recall"] == 0.5
    assert e2e["gt_conversion"] == 1.0
    rerank = MODULE.rerank_metrics(artifacts / "paper_rows.jsonl")
    assert rerank["recall@5"] == 1.0
    assert rerank["retrieval_event_count"] == 1
    assert MODULE.graph_unique_survival(
        artifacts / "paper_rows.jsonl"
    )["graph_pool_unique_gt_selected"] == 1
    audit = MODULE.scientific_audit(artifacts)
    assert audit["all_checks_passed"] is True
    assert audit["duplicate_query_profile_rows"] == 1
