import importlib.util
import json
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
SCRIPT = CODE_DIR / "compare_online_paper_type_backends.py"
SPEC = importlib.util.spec_from_file_location(
    "compare_online_paper_type_backends", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_run(path, *, backend, result, type_rows):
    artifacts = path / "online_artifacts"
    artifacts.mkdir(parents=True)
    manifest = {
        "dynamic_rerank_requested": True,
        "rerank_formula_id": "dynamic_rerank_v1",
        "paper_type_backend": backend,
        "paper_type_source": "semantic_scholar" if backend == "s2" else "qwen",
        "package_source_sha256": "same-source",
        "rerank_policy_model": "qwen-test",
        "rerank_catalog_version": "v1",
        "rerank_prompt_version": "v3",
    }
    (artifacts / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _write_jsonl(artifacts / "query_results.jsonl", [result])
    _write_jsonl(
        artifacts / "query_rerank_policies.jsonl",
        [
            {
                "query_id": "q1",
                "rerank_policy_id": "shared-policy",
                "compiled_policy": {
                    "paper_type_alignment_enabled": backend == "qwen",
                    "paper_type_rules": [
                        {
                            "types": ["dataset_benchmark"],
                            "action": "prefer",
                        }
                    ],
                    "adjustments": (
                        []
                        if backend == "qwen"
                        else ["paper_type_alignment_forced_off_unsupported_rules"]
                    ),
                },
                "used_fallback": False,
            }
        ],
    )
    _write_jsonl(artifacts / "paper_rows.jsonl", type_rows)


def test_compare_s2_and_qwen_backends_aligns_policies_and_type_evidence(tmp_path):
    s2_result = {
        "query_id": "q1",
        "gt_count": 2,
        "candidate_count": 2,
        "candidate_gt_ids": ["p1"],
        "selected_count": 1,
        "selected_gt_ids": ["p1"],
    }
    qwen_result = {
        "query_id": "q1",
        "gt_count": 2,
        "candidate_count": 2,
        "candidate_gt_ids": ["p1", "p2"],
        "selected_count": 2,
        "selected_gt_ids": ["p1", "p2"],
    }
    _write_run(
        tmp_path / "s2",
        backend="s2",
        result=s2_result,
        type_rows=[
            {
                "paper_arxiv_id": "p1",
                "paper_type_evidence_source": "semantic_scholar",
                "paper_type_probs": {"survey_review": 1.0},
                "paper_type_classifier_confidence": 1.0,
                "paper_type_supported_types": ["survey_review"],
            },
            {
                "paper_arxiv_id": "p2",
                "paper_type_evidence_source": "semantic_scholar",
                "paper_type_probs": {},
                "paper_type_classifier_confidence": 1.0,
                "paper_type_supported_types": ["survey_review"],
            },
        ],
    )
    _write_run(
        tmp_path / "qwen",
        backend="qwen",
        result=qwen_result,
        type_rows=[
            {
                "paper_arxiv_id": "p1",
                "paper_type_evidence_source": "qwen",
                "paper_type_probs": {"survey_review": 0.9},
                "paper_type_classifier_confidence": 0.95,
                "paper_type_supported_types": ["survey_review", "dataset_benchmark"],
            },
            {
                "paper_arxiv_id": "p2",
                "paper_type_evidence_source": "qwen",
                "paper_type_probs": {"dataset_benchmark": 0.8},
                "paper_type_classifier_confidence": 0.94,
                "paper_type_supported_types": ["survey_review", "dataset_benchmark"],
            },
        ],
    )

    result = MODULE.compare(
        tmp_path / "s2",
        tmp_path / "qwen",
        bootstrap_samples=10,
        seed=1,
    )

    assert result["comparability"]["comparable_config"] is True
    assert result["policy_id_mismatch_query_ids"] == []
    assert result["selection"]["qwen"]["macro_f1"] == 1.0
    evidence = result["type_evidence"]
    assert evidence["shared_typed_paper_count"] == 2
    assert evidence["per_type_on_shared_papers"]["survey_review"][
        "both_positive_count"
    ] == 1
    assert evidence["per_type_on_shared_papers"]["dataset_benchmark"][
        "s2_supports_type"
    ] is False
    assert evidence["per_type_on_shared_papers"]["dataset_benchmark"][
        "qwen_positive_count"
    ] == 1


def test_legacy_s2_manifest_requires_explicit_compatibility_flag(tmp_path):
    result = {
        "query_id": "q1",
        "gt_count": 1,
        "candidate_count": 1,
        "candidate_gt_ids": ["p1"],
        "selected_count": 1,
        "selected_gt_ids": ["p1"],
    }
    _write_run(tmp_path / "s2", backend="s2", result=result, type_rows=[])
    _write_run(tmp_path / "qwen", backend="qwen", result=result, type_rows=[])
    s2_manifest_path = tmp_path / "s2" / "online_artifacts" / "run_manifest.json"
    s2_manifest = json.loads(s2_manifest_path.read_text())
    del s2_manifest["paper_type_backend"]
    s2_manifest["paper_type_source"] = "semantic_scholar_publicationTypes"
    s2_manifest["package_source_sha256"] = "legacy-source"
    s2_manifest_path.write_text(json.dumps(s2_manifest), encoding="utf-8")

    strict = MODULE.compare(tmp_path / "s2", tmp_path / "qwen")
    compatible = MODULE.compare(
        tmp_path / "s2",
        tmp_path / "qwen",
        allow_legacy_s2_manifest=True,
    )

    assert strict["comparability"]["comparable_config"] is False
    assert compatible["comparability"]["comparable_config"] is True
    assert compatible["comparability"]["accepted_legacy_differences"][
        "legacy_s2_manifest_without_backend_field"
    ] is True
