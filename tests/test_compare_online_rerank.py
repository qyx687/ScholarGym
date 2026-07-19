import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "code" / "compare_online_rerank.py"
SPEC = importlib.util.spec_from_file_location("compare_online_rerank", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_run(path, records, *, dynamic):
    artifacts = path / "online_artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "query_results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (artifacts / "run_manifest.json").write_text(
        json.dumps(
            {
                "dynamic_rerank_requested": dynamic,
                "rerank_formula_id": (
                    "dynamic_rerank_v1"
                    if dynamic
                    else "q030_sq040_intent015_path015_closed_pool_minmax_v1"
                ),
            }
        ),
        encoding="utf-8",
    )


def test_comparison_uses_main_table_f1_of_macro_precision_and_recall(tmp_path):
    records = [
        {
            "query_id": "q1",
            "gt_count": 10,
            "candidate_count": 1,
            "candidate_gt_ids": ["hit"],
            "selected_count": 1,
            "selected_gt_ids": ["hit"],
        },
        {
            "query_id": "q2",
            "gt_count": 1,
            "candidate_count": 10,
            "candidate_gt_ids": ["hit"],
            "selected_count": 10,
            "selected_gt_ids": ["hit"],
        },
    ]
    _write_run(tmp_path / "static", records, dynamic=False)
    _write_run(tmp_path / "dynamic", records, dynamic=True)

    result = MODULE.compare(
        tmp_path / "static",
        tmp_path / "dynamic",
        bootstrap_samples=10,
        seed=1,
    )

    metrics = result["candidate"]["static"]
    assert metrics["macro_precision"] == pytest.approx(0.55)
    assert metrics["macro_recall"] == pytest.approx(0.55)
    assert metrics["macro_f1"] == pytest.approx(0.55)
    assert metrics["mean_per_query_f1"] == pytest.approx(2.0 / 11.0)
    assert result["comparability"]["comparable_config"] is True
