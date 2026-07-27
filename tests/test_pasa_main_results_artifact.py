import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_main_results_match_event_query_macro_summary():
    main = json.loads(
        (ROOT / "docs" / "pasa_realscholar_main_results.json").read_text(
            encoding="utf-8"
        )
    )
    event_summary = json.loads(
        (
            ROOT
            / "docs"
            / "pasa_realscholar_event_query_macro_top1000"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    method_names = {
        ("Controlled", "Semantic"): "Controlled/Semantic",
        ("Controlled", "Static-Fusion"): "Controlled/Static-Fusion",
        ("Controlled", "QuDAR-Rerank"): "Controlled/QuDAR-Rerank",
        (
            "Controlled",
            "LLM-Semantic-Rerank",
        ): "Controlled/LLM-Semantic-Rerank",
        ("Controlled", "Ours"): "Controlled/Ours",
        ("Native", "QuDAR"): "Native/QuDAR",
        (
            "Native",
            "LLM-guided retrieval",
        ): "Native/LLM-guided-retrieval",
        ("Native", "Ours"): "Native/Ours",
    }

    assert event_summary["benchmark_query_count"] == 50
    for row in main["rows"]:
        key = (row["track"], row["method"])
        if key not in method_names:
            assert row["method"] == "Oracle-Policy"
            assert row["ranking"] is None
            continue
        generated = event_summary["methods"][method_names[key]]["metrics"]
        ranking = row["ranking"]
        for cutoff in (10, 20):
            assert ranking[f"recall_at_{cutoff}"] == generated[str(cutoff)][
                "recall"
            ]
            assert ranking[f"ndcg_at_{cutoff}"] == generated[str(cutoff)][
                "ndcg"
            ]
            assert ranking[f"map_at_{cutoff}"] == generated[str(cutoff)][
                "map"
            ]


def test_saved_ranking_depth_limit_is_explicit():
    event_summary = json.loads(
        (
            ROOT
            / "docs"
            / "pasa_realscholar_event_query_macro_top1000"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )

    for name, method in event_summary["methods"].items():
        coverage = method["ranking_depth"]["coverage_at_cutoff"]["20"]
        if name.startswith("Controlled/"):
            assert coverage == 0.0
        else:
            assert coverage == 1.0
