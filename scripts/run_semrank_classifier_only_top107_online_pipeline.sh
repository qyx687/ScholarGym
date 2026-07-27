#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="logs"
LOG_PATH="${LOG_DIR}/semrank_classifier_only_top107_online.log"
STATUS_PATH="${LOG_DIR}/semrank_classifier_only_top107_online.status"
PYTHON_BIN="${SCHOLARGYM_PYTHON:-/home/quan/miniconda3/envs/scholargym-official/bin/python}"
OUTPUT_ROOT="eval_results_semrank_classifier_only_top107_pasa_realscholar"
EXPECTED_QUERY_COUNT="${SEMRANK_EXPECTED_QUERY_COUNT:-50}"
mkdir -p "$LOG_DIR"
exec >>"$LOG_PATH" 2>&1

write_status() {
  printf "%s\n" "$1" >"$STATUS_PATH"
  printf "[%s] %s\n" "$(date --iso-8601=seconds)" "$1"
}

fail() {
  local exit_code=$?
  write_status "failed:${exit_code}"
  exit "$exit_code"
}
trap fail ERR

write_status "online"
bash scripts/run_semrank_pasa.sh classifier-only-top107-full
"$PYTHON_BIN" - "$OUTPUT_ROOT" "$EXPECTED_QUERY_COUNT" <<'PY'
import json
import sys
from pathlib import Path


output_root = Path(sys.argv[1]).resolve()
expected_count = int(sys.argv[2])
complete_runs = []
diagnostics = []

for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
    checkpoint = run_dir / "detailed_results.jsonl"
    if not checkpoint.is_file():
        diagnostics.append(f"{run_dir.name}: missing detailed_results.jsonl")
        continue

    rows = []
    with checkpoint.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"invalid checkpoint JSON at {checkpoint}:{line_number}: {exc}"
                ) from exc

    indices = [row.get("idx") for row in rows]
    expected_indices = set(range(expected_count))
    if len(rows) != expected_count or set(indices) != expected_indices:
        diagnostics.append(
            f"{run_dir.name}: checkpoint rows={len(rows)}, "
            f"unique_expected_indices={len(set(indices) & expected_indices)}"
        )
        continue

    result_files = sorted(
        run_dir.glob("eval_results_*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not result_files:
        diagnostics.append(f"{run_dir.name}: missing final eval_results JSON")
        continue
    final_result = json.loads(result_files[-1].read_text(encoding="utf-8"))
    successful = int(final_result.get("successful_queries", -1))
    total = int(final_result.get("total_queries", -1))
    if successful != expected_count or total != expected_count:
        diagnostics.append(
            f"{run_dir.name}: final total={total}, successful={successful}"
        )
        continue
    complete_runs.append((result_files[-1].stat().st_mtime_ns, run_dir))

if not complete_runs:
    detail = "; ".join(diagnostics) or "no run directories found"
    raise SystemExit(
        f"online run did not produce {expected_count}/{expected_count} "
        f"successful checkpoints: {detail}"
    )

_, completed_run = max(complete_runs)
print(
    f"validated online run: {completed_run} "
    f"({expected_count}/{expected_count} successful)"
)
PY
write_status "complete"
