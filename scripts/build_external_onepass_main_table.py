#!/usr/bin/env python3
"""Build the five-arm OnePass main table and a compact fairness audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield value


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def harmonic(recall: float, precision: float) -> float:
    return (
        2.0 * recall * precision / (recall + precision)
        if recall + precision
        else 0.0
    )


def baseline_metrics(path: Path) -> Dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for record in iter_jsonl(path):
        postprocess = record.get("postprocess_results")
        baseline = (
            postprocess.get("baseline")
            if isinstance(postprocess, Mapping)
            else None
        )
        if not isinstance(baseline, Mapping):
            raise ValueError(f"missing baseline result in {path}")
        rows.append(baseline)
    if not rows:
        raise ValueError(f"no baseline rows found in {path}")

    sel_r = mean(float(row["selection_recall"]) for row in rows)
    sel_p = mean(float(row["selection_precision"]) for row in rows)
    ret_r = mean(float(row["candidate_recall"]) for row in rows)
    ret_p = mean(float(row["candidate_precision"]) for row in rows)
    retrieved_gt = sum(int(row["candidate_gt_count"]) for row in rows)
    selected_gt = sum(int(row["selected_gt_count"]) for row in rows)
    return {
        "evaluated_query_count": len(rows),
        "avg_selection_recall": sel_r,
        "avg_selection_precision": sel_p,
        "main_table_selection_f1": harmonic(sel_r, sel_p),
        "avg_candidate_recall": ret_r,
        "avg_candidate_precision": ret_p,
        "main_table_candidate_f1": harmonic(ret_r, ret_p),
        "total_candidate_count": sum(
            int(row["candidate_count"]) for row in rows
        ),
        "total_selected_count": sum(
            int(row["selected_count"]) for row in rows
        ),
        "total_candidate_gt_count": retrieved_gt,
        "total_selected_gt_count": selected_gt,
    }


def selector_method(path: Path, method: str) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("complete"):
        raise ValueError(f"Selector replay is incomplete: {path}")
    methods = value.get("methods")
    selected = methods.get(method) if isinstance(methods, Mapping) else None
    if not isinstance(selected, Mapping):
        raise KeyError(f"method {method!r} not found in {path}")
    if int(value.get("selector_failure_count") or 0) != 0:
        raise ValueError(f"Selector failures are nonzero in {path}")
    return dict(selected)


def table_row(
    name: str,
    metrics: Mapping[str, Any],
    *,
    postprocess_llm: str,
    candidate_pool: str,
) -> Dict[str, Any]:
    retrieved_gt = int(metrics["total_candidate_gt_count"])
    selected_gt = int(metrics["total_selected_gt_count"])
    sel_r = float(metrics["avg_selection_recall"])
    sel_p = float(metrics["avg_selection_precision"])
    ret_r = float(metrics["avg_candidate_recall"])
    ret_p = float(metrics["avg_candidate_precision"])
    return {
        "method": name,
        "sel_recall": sel_r,
        "sel_precision": sel_p,
        "sel_f1": harmonic(sel_r, sel_p),
        "ret_recall": ret_r,
        "ret_precision": ret_p,
        "ret_f1": harmonic(ret_r, ret_p),
        "retrieved_gt": retrieved_gt,
        "selected_gt": selected_gt,
        "gap": retrieved_gt - selected_gt,
        "gt_conversion_rate": safe_div(selected_gt, retrieved_gt),
        "candidate_count": int(metrics["total_candidate_count"]),
        "selected_count": int(metrics["total_selected_count"]),
        "postprocess_llm_budget": postprocess_llm,
        "candidate_pool": candidate_pool,
    }


def load_audit(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("candidate_pool_equality"):
        raise ValueError(f"candidate-pool equality failed: {path}")
    if not value.get("selector_top_k_equality"):
        raise ValueError(f"Selector-depth equality failed: {path}")
    return value


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def ranked_output_audit(path: Path) -> Dict[str, Dict[str, int]]:
    output: Dict[str, Dict[str, int]] = {}
    seen: set[tuple[str, str]] = set()
    for row in iter_jsonl(path):
        method = str(row.get("method") or "")
        event_id = str(row.get("retrieval_event_id") or "")
        if not method or not event_id:
            raise ValueError(f"ranked row lacks method/event in {path}")
        key = (method, event_id)
        if key in seen:
            raise ValueError(f"duplicate ranked event {key} in {path}")
        seen.add(key)
        candidates = row.get("ranked_candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"ranked row lacks candidates for {key}")
        selected_count = sum(
            bool(candidate.get("selected_at_event_top_k"))
            for candidate in candidates
            if isinstance(candidate, Mapping)
        )
        method_stats = output.setdefault(
            method,
            {"event_count": 0, "selector_occurrence_count": 0},
        )
        method_stats["event_count"] += 1
        method_stats["selector_occurrence_count"] += selected_count
    return output


def fmt(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| 方法 | Sel R | Sel P | Sel F1 | Ret R | Ret P | Retrieved GT | Selected GT | Gap | GT转化率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {sel_r} | {sel_p} | {sel_f1} | {ret_r} | "
            "{ret_p} | {retrieved_gt} | {selected_gt} | {gap} | "
            "{conversion} |".format(
                method=row["method"],
                sel_r=fmt(float(row["sel_recall"])),
                sel_p=fmt(float(row["sel_precision"])),
                sel_f1=fmt(float(row["sel_f1"])),
                ret_r=fmt(float(row["ret_recall"])),
                ret_p=fmt(float(row["ret_precision"])),
                retrieved_gt=row["retrieved_gt"],
                selected_gt=row["selected_gt"],
                gap=row["gap"],
                conversion=fmt(float(row["gt_conversion_rate"])),
            )
        )
    return "\n".join(lines)


def render_markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| 方法 | Sel R | Sel P | Sel F1 | Ret R | Ret P | "
        "Retrieved GT | Selected GT | Gap | GT转化率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {sel_r} | {sel_p} | {sel_f1} | {ret_r} | "
            "{ret_p} | {retrieved_gt} | {selected_gt} | {gap} | "
            "{conversion} |".format(
                method=row["method"],
                sel_r=fmt(float(row["sel_recall"])),
                sel_p=fmt(float(row["sel_precision"])),
                sel_f1=fmt(float(row["sel_f1"])),
                ret_r=fmt(float(row["ret_recall"])),
                ret_p=fmt(float(row["ret_precision"])),
                retrieved_gt=row["retrieved_gt"],
                selected_gt=row["selected_gt"],
                gap=row["gap"],
                conversion=fmt(float(row["gt_conversion_rate"])),
            )
        )
    return "\n".join(lines)


def render_markdown_report(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> str:
    audit = report["fairness_audit"]
    profile = audit.get("semrank_query_profile_build")
    lines = [
        "# ScholarGym baseline-OnePass 严格后处理对照",
        "",
        render_markdown_table(rows),
        "",
        "## 公平性审计",
        "",
        (
            "- 固定 baseline query/subquery trajectory：是；共 "
            f"{audit['graph_arm_event_count']} events。"
        ),
        (
            "- 后四种方法共用 graph pool：是；共 "
            f"{audit['graph_pool_candidate_occurrences']:,} "
            "candidate occurrences。"
        ),
        (
            "- 每种方法进入 Selector 的 occurrence 深度："
            f"{audit['selector_input_occurrences_per_graph_method']:,}；"
            "逐事件与 baseline 实际页长一致。"
        ),
        (
            "- 方法额外 rerank LLM：Static/QuDAR 为 0；"
            "S2-native/SemRank 为 1 个 query-profile task/query；"
            "四种方法 paper-level LLM 均为 0。"
        ),
        (
            "- SemRank：classifier-only；full 未纳入；"
            "逐 concept 使用 Qwen3-Embedding-0.6B。"
        ),
        (
            "- Selector score：SemRank 的原始 z-score 按事件完整池单调 "
            "min-max 到 [0,1]；候选顺序和 Top-K 不变。"
        ),
        "- 所有后处理均为 open-loop，不写回 baseline memory。",
    ]
    if isinstance(profile, Mapping):
        lines.extend(
            [
                (
                    "- SemRank query profile：辅助检索 top-"
                    f"{profile['initial_top_m']}，聚合前 "
                    f"{profile['feedback_top_n']} 篇 classifier topics，"
                    f"prompt 展示前 {profile['prompt_top_papers']} 篇。"
                ),
                (
                    "- SemRank query-profile 成本："
                    f"{profile['query_profile_tasks']} 个 task，"
                    f"{profile['query_llm_api_attempts']} 次 API attempt，"
                    f"其中 {profile['llm_parse_failures']} 次格式解析失败重试；"
                    f"{profile['fallback_profiles']} 个 profile fallback。"
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def build_report(
    *,
    baseline_details: Path,
    s2_ranked_candidates: Path,
    s2_selector_summary: Path,
    qudar_selector_summary: Path,
    semrank_selector_summary: Path,
    qudar_score_summary: Path,
    semrank_score_summary: Path,
    qudar_artifact_audit: Path,
    semrank_artifact_audit: Path,
    semrank_query_profile_report: Path | None,
    output_dir: Path,
) -> Dict[str, Any]:
    baseline = baseline_metrics(baseline_details)
    static = selector_method(s2_selector_summary, "legacy_static")
    s2_native = selector_method(s2_selector_summary, "dynamic_policy")
    qudar = selector_method(qudar_selector_summary, "dynamic_policy")
    semrank = selector_method(semrank_selector_summary, "dynamic_policy")
    qudar_audit = load_audit(qudar_artifact_audit)
    semrank_audit = load_audit(semrank_artifact_audit)
    qudar_score_audit = load_json(qudar_score_summary)
    semrank_score_audit = load_json(semrank_score_summary)
    semrank_profile_audit = (
        load_json(semrank_query_profile_report)
        if semrank_query_profile_report is not None
        else None
    )
    s2_output_audit = ranked_output_audit(s2_ranked_candidates)
    invariant_fields = (
        "query_count",
        "event_count",
        "candidate_occurrence_count",
        "selector_occurrence_count_per_method",
    )
    for field in invariant_fields:
        if qudar_audit.get(field) != semrank_audit.get(field):
            raise ValueError(
                f"QuDAR/SemRank audit mismatch for {field}: "
                f"{qudar_audit.get(field)} != {semrank_audit.get(field)}"
            )
    if qudar_audit.get("selector_score_transform") != "none":
        raise ValueError("QuDAR score adapter unexpectedly changed scores")
    if semrank_audit.get("selector_score_transform") != "minmax":
        raise ValueError(
            "SemRank Selector scores were not event-wise min-max calibrated"
        )
    expected_events = int(qudar_audit["event_count"])
    expected_occurrences = int(
        qudar_audit["selector_occurrence_count_per_method"]
    )
    for method in ("legacy_static", "dynamic_policy"):
        stats = s2_output_audit.get(method)
        if stats != {
            "event_count": expected_events,
            "selector_occurrence_count": expected_occurrences,
        }:
            raise ValueError(
                f"S2 ranked-output audit failed for {method}: {stats}"
            )
    if int(qudar_score_audit.get("additional_llm_calls") or 0) != 0:
        raise ValueError("QuDAR unexpectedly used an LLM")
    if int(qudar_score_audit.get("candidate_count") or 0) != int(
        qudar_audit["candidate_occurrence_count"]
    ):
        raise ValueError("QuDAR score coverage does not match the graph pool")
    semrank_llm = semrank_score_audit.get("llm_stats")
    if not isinstance(semrank_llm, Mapping):
        raise ValueError("SemRank score summary has no llm_stats")
    if int(semrank_llm.get("paper_concept_llm_calls") or 0) != 0:
        raise ValueError("SemRank classifier-only called the paper LLM")
    if int(semrank_llm.get("query_concept_llm_calls") or 0) != 0:
        raise ValueError(
            "SemRank replay regenerated query concepts instead of using cache"
        )
    if int(semrank_score_audit.get("candidate_count") or 0) != int(
        semrank_audit["candidate_occurrence_count"]
    ):
        raise ValueError("SemRank score coverage does not match the graph pool")
    if int(semrank_score_audit.get("query_profile_count") or 0) != int(
        semrank_audit["query_count"]
    ):
        raise ValueError("SemRank query-profile coverage is incomplete")
    if semrank_profile_audit is not None:
        if int(semrank_profile_audit.get("paper_level_llm_calls") or 0) != 0:
            raise ValueError("SemRank query-profile build called the paper LLM")
        if int(semrank_profile_audit.get("paper_keyphrases_consumed") or 0) != 0:
            raise ValueError(
                "SemRank classifier-only query profile consumed keyphrases"
            )
        built = int(semrank_profile_audit.get("query_profiles_built") or 0)
        cached = int(
            semrank_profile_audit.get("query_profiles_already_cached") or 0
        )
        if built + cached != int(semrank_audit["query_count"]):
            raise ValueError(
                "SemRank query-profile build does not cover every query"
            )

    rows = [
        table_row(
            "ScholarGym baseline",
            baseline,
            postprocess_llm="N/A（原始在线流程）",
            candidate_pool="原始 dense 检索页",
        ),
        table_row(
            "Graph + Static QSQ",
            static,
            postprocess_llm="0",
            candidate_pool="共同 OnePass per_subquery 图池",
        ),
        table_row(
            "Graph + S2-native",
            s2_native,
            postprocess_llm="1/query，0/paper",
            candidate_pool="共同 OnePass per_subquery 图池",
        ),
        table_row(
            "Graph + QuDAR-Confidence-QSQ",
            qudar,
            postprocess_llm="0",
            candidate_pool="共同 OnePass per_subquery 图池",
        ),
        table_row(
            "Graph + SemRank classifier-only",
            semrank,
            postprocess_llm="1/query，0/paper",
            candidate_pool="共同 OnePass per_subquery 图池",
        ),
    ]
    report = {
        "schema_version": "external_onepass_main_table_v1",
        "rows": rows,
        "fairness_audit": {
            "onepass_baseline_trajectory_fixed": True,
            "postprocess_writes_back_to_baseline_memory": False,
            "graph_arm_query_count": qudar_audit["query_count"],
            "graph_arm_event_count": qudar_audit["event_count"],
            "graph_pool_candidate_occurrences": qudar_audit[
                "candidate_occurrence_count"
            ],
            "selector_input_occurrences_per_graph_method": qudar_audit[
                "selector_occurrence_count_per_method"
            ],
            "graph_arm_candidate_pool_equal": True,
            "graph_arm_output_depth_equal": True,
            "paper_level_llm_calls_per_graph_method": 0,
            "rerank_query_profile_budget": {
                "Graph + Static QSQ": "0",
                "Graph + S2-native": "1 task/query",
                "Graph + QuDAR-Confidence-QSQ": "0",
                "Graph + SemRank classifier-only": "1 task/query",
            },
            "selector_event_budget_per_method": qudar_audit["event_count"],
            "selector_score_domain": "[0,1]",
            "semrank_selector_score_calibration": (
                "event-full-pool minmax; monotonic; ranking/Top-K unchanged"
            ),
            "semrank_mode": "classifier_only",
            "semrank_full_included": False,
            "semrank_query_profile_build": (
                {
                    "initial_top_m": int(
                        semrank_profile_audit.get("initial_top_m") or 0
                    ),
                    "feedback_top_n": int(
                        semrank_profile_audit.get("feedback_top_n") or 0
                    ),
                    "prompt_top_papers": int(
                        semrank_profile_audit.get("prompt_top_papers") or 0
                    ),
                    "query_profile_tasks": int(
                        semrank_profile_audit.get("query_profiles_built") or 0
                    ),
                    "query_llm_api_attempts": int(
                        (
                            semrank_profile_audit.get(
                                "query_level_llm_stats"
                            )
                            or {}
                        ).get("query_concept_llm_calls")
                        or 0
                    ),
                    "llm_parse_failures": int(
                        (
                            semrank_profile_audit.get(
                                "query_level_llm_stats"
                            )
                            or {}
                        ).get("llm_parse_failures")
                        or 0
                    ),
                    "fallback_profiles": int(
                        semrank_profile_audit.get(
                            "query_profiles_fallback_built"
                        )
                        or 0
                    ),
                }
                if semrank_profile_audit is not None
                else None
            ),
        },
        "sources": {
            "baseline_details": str(baseline_details.resolve()),
            "s2_ranked_candidates": str(s2_ranked_candidates.resolve()),
            "s2_selector_summary": str(s2_selector_summary.resolve()),
            "qudar_selector_summary": str(qudar_selector_summary.resolve()),
            "semrank_selector_summary": str(
                semrank_selector_summary.resolve()
            ),
            "qudar_score_summary": str(qudar_score_summary.resolve()),
            "semrank_score_summary": str(semrank_score_summary.resolve()),
            "qudar_artifact_audit": str(qudar_artifact_audit.resolve()),
            "semrank_artifact_audit": str(
                semrank_artifact_audit.resolve()
            ),
            "semrank_query_profile_report": (
                str(semrank_query_profile_report.resolve())
                if semrank_query_profile_report is not None
                else None
            ),
        },
    }

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "main_table.json"
    markdown_path = output_dir / "main_table.md"
    temporary_json = json_path.with_name(f".{json_path.name}.tmp-{os.getpid()}")
    temporary_md = markdown_path.with_name(
        f".{markdown_path.name}.tmp-{os.getpid()}"
    )
    temporary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_md.write_text(
        "# ScholarGym baseline-OnePass 严格后处理对照\n\n"
        + markdown_table(rows)
        + "\n\n"
        + "## 公平性审计\n\n"
        + f"- 固定 baseline query/subquery trajectory：是；共 "
        + f"{report['fairness_audit']['graph_arm_event_count']} events。\n"
        + "- 后四种方法共同 graph pool：是；共 "
        + f"{report['fairness_audit']['graph_pool_candidate_occurrences']:,} "
        + "candidate occurrences。\n"
        + "- 每方法进入 Selector 的 occurrence 深度："
        + f"{report['fairness_audit']['selector_input_occurrences_per_graph_method']:,}；"
        + "逐事件与 baseline 实际页长一致。\n"
        + "- 后处理 paper-level LLM：全部为 0；query-level 上限："
        + "1/query。\n"
        + "- SemRank：classifier-only；full 未纳入。\n"
        + "- Selector score：SemRank 的原始 z-score 按事件完整池单调 "
        + "min-max 到 [0,1]；候选顺序和 Top-K 不变。\n"
        + "- 所有后处理均为 open-loop，不写回 baseline memory。\n",
        encoding="utf-8",
    )
    temporary_md.write_text(
        render_markdown_report(report, rows),
        encoding="utf-8",
    )
    os.replace(temporary_json, json_path)
    os.replace(temporary_md, markdown_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-details", required=True, type=Path)
    parser.add_argument("--s2-ranked-candidates", required=True, type=Path)
    parser.add_argument("--s2-selector-summary", required=True, type=Path)
    parser.add_argument("--qudar-selector-summary", required=True, type=Path)
    parser.add_argument("--semrank-selector-summary", required=True, type=Path)
    parser.add_argument("--qudar-score-summary", required=True, type=Path)
    parser.add_argument("--semrank-score-summary", required=True, type=Path)
    parser.add_argument("--qudar-artifact-audit", required=True, type=Path)
    parser.add_argument("--semrank-artifact-audit", required=True, type=Path)
    parser.add_argument("--semrank-query-profile-report", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        baseline_details=args.baseline_details,
        s2_ranked_candidates=args.s2_ranked_candidates,
        s2_selector_summary=args.s2_selector_summary,
        qudar_selector_summary=args.qudar_selector_summary,
        semrank_selector_summary=args.semrank_selector_summary,
        qudar_score_summary=args.qudar_score_summary,
        semrank_score_summary=args.semrank_score_summary,
        qudar_artifact_audit=args.qudar_artifact_audit,
        semrank_artifact_audit=args.semrank_artifact_audit,
        semrank_query_profile_report=args.semrank_query_profile_report,
        output_dir=args.output_dir,
    )
    print(render_markdown_table(report["rows"]))


if __name__ == "__main__":
    main()
