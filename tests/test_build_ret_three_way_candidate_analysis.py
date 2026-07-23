import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_ret_three_way_candidate_analysis.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_ret_three_way_candidate_analysis", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_provenance_retention_uses_original_pool_sources_not_ret_membership(tmp_path):
    first_analysis = tmp_path / "analysis"
    summary_path = first_analysis / "deep_merged_primary" / "group_summary.csv"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        "group,candidate_count,ground_truth_count\n"
        "graph_only,100,10\n"
        "deep_merged_only,80,8\n",
        encoding="utf-8",
    )
    graph_source_gt = ("q", "graph-source-gt")
    selected_only_by_graph_but_full_overlap = ("q", "full-overlap-gt")
    deep_source_gt = ("q", "deep-source-gt")
    partition_sets = {
        "graph": {graph_source_gt, selected_only_by_graph_but_full_overlap},
        "deep_merged": {deep_source_gt},
    }
    original_provenance = {
        graph_source_gt: {
            "partition": "graph_only",
            "is_ground_truth": True,
        },
        selected_only_by_graph_but_full_overlap: {
            "partition": "graph_and_deep_merged",
            "is_ground_truth": True,
        },
        deep_source_gt: {
            "partition": "deep_merged_only",
            "is_ground_truth": True,
        },
    }

    rows = MODULE._provenance_retention_rows(
        partition_sets, original_provenance, first_analysis
    )
    by_partition = {row["provenance_partition"]: row for row in rows}

    assert by_partition["graph_only"]["retained_candidate_count"] == 1
    assert by_partition["graph_only"]["retained_gt_count"] == 1
    assert by_partition["deep_merged_only"]["retained_candidate_count"] == 1
    assert by_partition["deep_merged_only"]["retained_gt_count"] == 1
