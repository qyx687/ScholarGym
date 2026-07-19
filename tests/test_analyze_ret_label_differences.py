import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_ret_label_differences.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_ret_label_differences", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_annotation_features_include_single_multi_and_derived_labels():
    candidate = {"is_ground_truth": True}
    annotation = {
        "relevance_grade": "direct",
        "semantic_distance": "exact",
        "paper_type": "primary_empirical",
        "scholarly_roles": ["direct_target", "method_component"],
        "information_added": ["empirical_evidence"],
        "matched_aspect_ids": ["A1", "A2", "A3"],
        "exclusion_reasons": ["none"],
        "needs_full_text": False,
    }

    features = MODULE.annotation_features(candidate, annotation)

    assert ("relevance_grade", "direct") in features
    assert ("scholarly_role", "method_component") in features
    assert ("matched_aspect_count", "3-5") in features
    assert ("derived", "direct_or_partial") in features
    assert ("derived", "direct_task_method_match") in features
    assert ("derived", "ground_truth") in features


def test_comparison_uses_query_paired_rates_not_pooled_counts():
    group_totals = Counter({"graph_only": 11, "deep_merged_only": 11})
    feature_counts = Counter(
        {
            ("graph_only", "scholarly_role", "background_or_foundation"): 2,
            ("deep_merged_only", "scholarly_role", "background_or_foundation"): 1,
        }
    )
    query_totals = Counter(
        {
            ("graph_only", "q0"): 1,
            ("deep_merged_only", "q0"): 10,
            ("graph_only", "q1"): 10,
            ("deep_merged_only", "q1"): 1,
        }
    )
    query_features = Counter(
        {
            ("graph_only", "q0", "scholarly_role", "background_or_foundation"): 1,
            ("deep_merged_only", "q0", "scholarly_role", "background_or_foundation"): 0,
            ("graph_only", "q1", "scholarly_role", "background_or_foundation"): 1,
            ("deep_merged_only", "q1", "scholarly_role", "background_or_foundation"): 1,
        }
    )

    rows = MODULE.comparison_rows(
        group_totals,
        feature_counts,
        query_totals,
        query_features,
        bootstrap_samples=100,
    )
    row = next(value for value in rows if value["scope"] == "selection_exclusive")

    assert row["micro_rate_difference"] == pytest.approx(1 / 11)
    assert row["left_macro_rate"] == pytest.approx(0.55)
    assert row["right_macro_rate"] == pytest.approx(0.5)
    assert row["macro_rate_difference"] == pytest.approx(0.05)
