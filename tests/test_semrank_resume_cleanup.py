import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "clean_failed_attempt_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("semrank_resume_cleanup", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_cleanup_dry_run_then_backup_and_atomic_filter(tmp_path):
    run = tmp_path / "run"
    artifacts = run / "online_artifacts"
    artifacts.mkdir(parents=True)
    _write_jsonl(run / "detailed_results.jsonl", [{"idx": 0}, {"idx": 2}])
    rows = [
        {
            "benchmark_idx": 0,
            "query_id": "q0",
            "retrieval_event_id": "e0",
            "paper_arxiv_id": "p0",
        },
        {
            "benchmark_idx": 1,
            "query_id": "q1",
            "retrieval_event_id": "e1",
            "paper_arxiv_id": "partial",
        },
        {
            "benchmark_idx": 2,
            "query_id": "q2",
            "retrieval_event_id": "e2",
            "paper_arxiv_id": "p2",
        },
    ]
    target = artifacts / "paper_rows.jsonl"
    _write_jsonl(target, rows)
    filter_stats = artifacts / "filter_stats.jsonl"
    _write_jsonl(
        filter_stats,
        [
            {"query_id": "q0", "retrieval_event_id": "e0"},
            {"query_id": "q1", "retrieval_event_id": "e1"},
            {"query_id": "q2", "retrieval_event_id": "e2"},
        ],
    )

    preview = MODULE.clean(run, expected_count=3, apply=False)
    assert preview["unfinished_indices"] == [1]
    assert preview["scan"]["paper_rows.jsonl"]["removed_rows"] == 1
    assert preview["scan"]["filter_stats.jsonl"]["removed_rows"] == 1
    assert preview["query_index_mapping_count"] == 3
    assert preview["event_index_mapping_count"] == 3
    assert len(list(MODULE.iter_jsonl(target))) == 3

    backup = tmp_path / "backup"
    result = MODULE.clean(
        run,
        expected_count=3,
        apply=True,
        backup_dir=backup,
    )
    remaining = [row for _, row in MODULE.iter_jsonl(target)]
    original = [
        row for _, row in MODULE.iter_jsonl(backup / "paper_rows.jsonl")
    ]
    assert result["dry_run"] is False
    assert [row["benchmark_idx"] for row in remaining] == [0, 2]
    assert [
        row["query_id"] for _, row in MODULE.iter_jsonl(filter_stats)
    ] == ["q0", "q2"]
    assert original == rows
    assert (backup / "cleanup_summary.json").is_file()
