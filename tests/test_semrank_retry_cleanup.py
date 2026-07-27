import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "clean_retried_query_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location(
    "semrank_retry_cleanup",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rows(path):
    return [row for _, _, row in MODULE.iter_jsonl(path)]


def test_retry_cleanup_uses_iteration_reset_and_exact_deduplication(
    tmp_path,
):
    run = tmp_path / "run"
    artifacts = run / "online_artifacts"
    artifacts.mkdir(parents=True)

    event_rows = [
        {"query_id": "q2", "iteration_idx": 1, "value": "old-1"},
        {"query_id": "q1", "iteration_idx": 1, "value": "other"},
        {"query_id": "q2", "iteration_idx": 2, "value": "old-2a"},
        {"query_id": "q2", "iteration_idx": 2, "value": "old-2b"},
        {"query_id": "q2", "iteration_idx": 4, "value": "old-4"},
        {"query_id": "q2", "iteration_idx": 1, "value": "new-1"},
        {"query_id": "q2", "iteration_idx": 2, "value": "new-2"},
        {"query_id": "q2", "iteration_idx": 5, "value": "new-5"},
    ]
    events = artifacts / "semrank_event_profiles.jsonl"
    _write_jsonl(events, event_rows)

    profile = {"query_id": "q2", "profile_id": "same"}
    profiles = artifacts / "semrank_query_profiles.jsonl"
    _write_jsonl(profiles, [profile, {"query_id": "q1"}, profile])
    results = artifacts / "query_results.jsonl"
    _write_jsonl(results, [{"query_id": "q2", "status": "complete"}])

    preview = MODULE.clean(run, query_id="q2", apply=False)
    event_plan = preview["plans"]["semrank_event_profiles.jsonl"]
    profile_plan = preview["plans"]["semrank_query_profiles.jsonl"]
    assert preview["ambiguous_files"] == []
    assert event_plan["strategy"] == "last_iteration_reset"
    assert event_plan["removed_rows"] == 4
    assert event_plan["remaining_target_rows"] == 3
    assert profile_plan["strategy"] == "identical_query_row_deduplication"
    assert profile_plan["removed_rows"] == 1
    assert len(_rows(events)) == len(event_rows)

    backup = tmp_path / "backup"
    result = MODULE.clean(
        run,
        query_id="q2",
        apply=True,
        backup_dir=backup,
    )
    assert result["dry_run"] is False
    assert [row["value"] for row in _rows(events)] == [
        "other",
        "new-1",
        "new-2",
        "new-5",
    ]
    assert [row["query_id"] for row in _rows(profiles)] == ["q1", "q2"]
    assert _rows(backup / events.name) == event_rows
    assert (backup / "cleanup_summary.json").is_file()
    second_preview = MODULE.clean(run, query_id="q2", apply=False)
    assert second_preview["affected_files"] == []
    assert second_preview["ambiguous_files"] == []
    assert (
        second_preview["plans"]["semrank_event_profiles.jsonl"]["strategy"]
        == "single_monotonic_attempt"
    )


def test_retry_cleanup_refuses_ambiguous_repeated_rows(tmp_path):
    run = tmp_path / "run"
    artifacts = run / "online_artifacts"
    artifacts.mkdir(parents=True)
    target = artifacts / "unknown.jsonl"
    _write_jsonl(
        target,
        [
            {"query_id": "q2", "value": "first"},
            {"query_id": "q2", "value": "second"},
        ],
    )

    preview = MODULE.clean(run, query_id="q2", apply=False)
    assert preview["ambiguous_files"] == ["unknown.jsonl"]
    with pytest.raises(ValueError, match="ambiguous"):
        MODULE.clean(run, query_id="q2", apply=True)
    assert [row["value"] for row in _rows(target)] == ["first", "second"]
