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
        if name == "Controlled/Semantic":
            assert coverage == 1.0
        elif name.startswith("Controlled/"):
            assert coverage == 0.0
        else:
            assert coverage == 1.0


def test_corrected_semantic_and_native_ours_sources_are_pinned():
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

    semantic = event_summary["sources"]["Controlled/Semantic"]
    assert semantic["path"].endswith(
        "/onepass_artifacts/deep_event/paper_rows.jsonl"
    )
    assert semantic["rank_field"] == "rerank_rank"

    native_ours = event_summary["sources"]["Native/Ours"]
    assert "/eval_results_online_dynamic_pasa_s2_native_v4/" in native_ours[
        "path"
    ]
    assert "pasa_dynamic_rerank_s2_native_v4_run1" in native_ours["path"]
    assert "eval_results_online_s2_native_v4_20260727" not in native_ours[
        "path"
    ]

    source_audit = main["source_corrections"]
    assert source_audit["controlled_semantic"]["retrieved_gt_count"] == 289
    assert source_audit["controlled_semantic"]["selected_gt_count"] == 220
    assert source_audit["native_ours"]["retrieved_gt_count"] == 310
    assert source_audit["native_ours"]["selected_gt_count"] == 271
