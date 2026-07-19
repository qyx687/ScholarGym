import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from eval import CitationEvaluator, signature_sha256, validate_resume_signature
from metrics import MetricsCalculator
from utils import CheckpointManager


def test_resume_signature_rejects_mixed_experiment_state(tmp_path):
    artifacts = tmp_path / "onepass_artifacts"
    artifacts.mkdir()
    manifest = artifacts / "run_manifest.json"
    checkpoint = tmp_path / "detailed_results.jsonl"
    expected = signature_sha256({"code": "new", "method": "dense"})
    manifest.write_text(
        json.dumps({"run_signature_sha256": signature_sha256({"code": "old"})}),
        encoding="utf-8",
    )
    checkpoint.write_text('{"idx":0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to resume"):
        validate_resume_signature(str(manifest), str(checkpoint), expected)


def test_stage_a_runs_queries_without_positive_arxiv_ground_truth():
    class Workflow:
        def __init__(self, stage):
            self.onepass_postprocessor = SimpleNamespace(postprocess_stage=stage)
            self.calls = 0

        def run(self, *_args, **_kwargs):
            self.calls += 1
            return {
                "history": [],
                "selected_papers": [],
                "executed_queries": [],
                "postprocess_results": {"postprocess_stage": "materialize"},
            }

    query = {
        "query": "no labelled arxiv paper",
        "cited_paper": [],
        "gt_label": [],
    }
    materialize_workflow = Workflow("materialize")
    evaluator = SimpleNamespace(deep_research_workflow=materialize_workflow)
    result = CitationEvaluator.evaluate_single_query_deep_research(
        evaluator, query, idx=7
    )
    assert materialize_workflow.calls == 1
    assert result["ground_truth_arxiv_ids"] == []
    assert result["ground_truth_metrics_available"] is False
    assert result["iteration_results"] == []

    full_workflow = Workflow("full")
    evaluator = SimpleNamespace(deep_research_workflow=full_workflow)
    assert (
        CitationEvaluator.evaluate_single_query_deep_research(
            evaluator, query, idx=7
        )
        is None
    )
    assert full_workflow.calls == 0


def test_empty_ground_truth_rank_distance_is_explicitly_unavailable():
    result = MetricsCalculator.calculate_gt_rank_and_distance(
        subqueries={},
        rank_dicts={},
        gt_arxiv_ids=set(),
        selected_paper_ids_tracker=set(),
        selected_min_rank_tracker={},
        gt_rank_cutoff=100,
    )

    assert result["cur_iter_distances"] == {}
    assert result["avg_distance"] == -1.0


def test_resume_signature_accepts_same_run_and_empty_manifest_only_directory(tmp_path):
    artifacts = tmp_path / "onepass_artifacts"
    artifacts.mkdir()
    manifest = artifacts / "run_manifest.json"
    checkpoint = tmp_path / "detailed_results.jsonl"
    expected = signature_sha256({"code": "same"})

    # A manifest written before any query is safe to replace.
    manifest.write_text(json.dumps({"run_signature_sha256": "stale"}), encoding="utf-8")
    validate_resume_signature(str(manifest), str(checkpoint), expected)

    manifest.write_text(
        json.dumps({"run_signature_sha256": expected}), encoding="utf-8"
    )
    checkpoint.write_text('{"idx":0}\n', encoding="utf-8")
    validate_resume_signature(str(manifest), str(checkpoint), expected)


def test_resume_signature_rejects_rows_when_manifest_is_missing(tmp_path):
    artifacts = tmp_path / "onepass_artifacts"
    artifacts.mkdir()
    manifest = artifacts / "run_manifest.json"
    checkpoint = tmp_path / "detailed_results.jsonl"
    checkpoint.write_text('{"idx":0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is missing"):
        validate_resume_signature(str(manifest), str(checkpoint), "expected")


def test_resume_signature_allows_explicit_result_preserving_code_upgrade(tmp_path):
    artifacts = tmp_path / "onepass_artifacts"
    artifacts.mkdir()
    manifest = artifacts / "run_manifest.json"
    checkpoint = tmp_path / "detailed_results.jsonl"
    formula_semantics = {
        "rerank_formula_id": "q030_sq040_intent015_path015_closed_pool_minmax_v1",
        "feature_weights_applied": {
            "query_score_normalized": 0.30,
            "subquery_score_normalized": 0.40,
            "intent_score": 0.15,
            "path_count_normalized": 0.15,
        },
    }
    old_payload = {
        "package_source_sha256": "old-code",
        "method": "dense",
        "top_k": 10,
        **formula_semantics,
    }
    new_payload = {
        "package_source_sha256": "new-code",
        "method": "dense",
        "top_k": 10,
        **formula_semantics,
        "postprocess_embedding_cache_policy": (
            "query_scoped_exact_text_singleflight_float32_v1"
        ),
    }
    manifest.write_text(
        json.dumps(
            {
                "run_signature_sha256": signature_sha256(old_payload),
                "run_signature": old_payload,
            }
        ),
        encoding="utf-8",
    )
    checkpoint.write_text('{"idx":0}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Refusing to resume"):
        validate_resume_signature(
            str(manifest), str(checkpoint), signature_sha256(new_payload)
        )

    result = validate_resume_signature(
        str(manifest),
        str(checkpoint),
        signature_sha256(new_payload),
        expected_payload=new_payload,
        allow_compatible_code_change=True,
    )
    assert result["mode"] == "compatible_code_change"

    changed_method = {**new_payload, "top_k": 20}
    with pytest.raises(ValueError, match="Refusing to resume"):
        validate_resume_signature(
            str(manifest),
            str(checkpoint),
            signature_sha256(changed_method),
            expected_payload=changed_method,
            allow_compatible_code_change=True,
        )

    changed_formula = {
        **new_payload,
        "rerank_formula_id": "legacy_q040_sq060_closed_pool_minmax_v1",
        "feature_weights_applied": {
            "query_score_normalized": 0.40,
            "subquery_score_normalized": 0.60,
        },
    }
    with pytest.raises(ValueError, match="Refusing to resume"):
        validate_resume_signature(
            str(manifest),
            str(checkpoint),
            signature_sha256(changed_formula),
            expected_payload=changed_formula,
            allow_compatible_code_change=True,
        )


def test_checkpoint_manager_atomically_drops_only_a_torn_final_row(tmp_path):
    checkpoint = tmp_path / "detailed_results.jsonl"
    checkpoint.write_text('{"idx":0,"ok":true}\n{"idx":1', encoding="utf-8")

    manager = CheckpointManager(str(checkpoint))
    processed, cached = manager.load_checkpoint()

    assert processed == {0}
    assert [row["idx"] for row in cached] == [0]
    assert checkpoint.read_text(encoding="utf-8") == '{"idx":0,"ok":true}\n'


def test_checkpoint_manager_rejects_corruption_before_valid_rows(tmp_path):
    checkpoint = tmp_path / "detailed_results.jsonl"
    checkpoint.write_text('{"idx":0\n{"idx":1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="before the file tail"):
        CheckpointManager(str(checkpoint)).load_checkpoint()
