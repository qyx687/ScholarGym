import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from eval import signature_sha256, validate_resume_signature
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
    old_payload = {"package_source_sha256": "old-code", "method": "dense", "top_k": 10}
    new_payload = {"package_source_sha256": "new-code", "method": "dense", "top_k": 10}
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
