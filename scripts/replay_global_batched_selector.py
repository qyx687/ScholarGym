#!/usr/bin/env python3
"""Offline experimental replay of two global rerank + batched Selector arms.

This script is intentionally not wired into ``code/eval.py``.  It consumes a
completed OnePass run, reuses its date-valid global candidate pool and scores,
and calls only the Selector.  Baseline retrieval and Semantic Scholar expansion
are never rerun.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PACKAGE_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

import config  # noqa: E402
from agent.selector import Selector  # noqa: E402
from graph_methods import (  # noqa: E402
    INTENT_WEIGHTS,
    load_paper_db,
    minmax,
    normalize_arxiv_id,
)
from structures import Paper, SubQuery  # noqa: E402


DEFAULT_GLOBAL_CHECKLIST = (
    "Select papers that directly answer the original query; prefer concrete "
    "method papers, seminal works, and papers matching the requested entity/task."
)
EXISTING_METHOD = "global_original_plus_all_subqueries_max_batched_selector"
NEW_METHOD = "global_query_subquery_intent_path_weighted_batched_selector"
METHODS = (EXISTING_METHOD, NEW_METHOD)
NEW_FORMULA_WEIGHTS = {
    "query_score_normalized": 0.30,
    "max_subquery_score_normalized": 0.40,
    "intent_score": 0.15,
    "path_count_normalized": 0.15,
}


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_detailed_results(path: Path) -> Dict[int, Dict[str, Any]]:
    """Use the last valid record for each benchmark index as the commit set."""
    results: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid detailed JSONL at line {line_number}: {exc}") from exc
            idx = row.get("idx")
            if isinstance(idx, int) and idx >= 0:
                results[idx] = row
    return results


def iter_jsonl_groups(path: Path, allowed_indices: Set[int]) -> Iterator[Tuple[int, List[Dict[str, Any]]]]:
    """Yield contiguous benchmark-index groups without loading a full artifact."""
    current_idx: Optional[int] = None
    current_rows: List[Dict[str, Any]] = []
    seen: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid artifact {path} at line {line_number}: {exc}") from exc
            idx = row.get("benchmark_idx")
            if not isinstance(idx, int) or idx not in allowed_indices:
                continue
            if current_idx is None:
                current_idx = idx
            if idx != current_idx:
                if idx in seen:
                    raise ValueError(f"artifact {path} contains non-contiguous duplicate group for idx={idx}")
                seen.add(current_idx)
                yield current_idx, current_rows
                current_idx = idx
                current_rows = []
            current_rows.append(row)
    if current_idx is not None:
        if current_idx in seen:
            raise ValueError(f"artifact {path} contains non-contiguous duplicate group for idx={current_idx}")
        yield current_idx, current_rows


class GroupCursor:
    def __init__(self, path: Path, allowed_indices: Set[int], order: Mapping[int, int]) -> None:
        self.path = path
        self._order = order
        self._iterator = iter_jsonl_groups(path, allowed_indices)
        self._current: Optional[Tuple[int, List[Dict[str, Any]]]] = None

    def _ensure_current(self) -> None:
        if self._current is None:
            self._current = next(self._iterator, None)

    def take(self, idx: int) -> List[Dict[str, Any]]:
        self._ensure_current()
        target_position = self._order[idx]
        while (
            self._current is not None
            and self._order.get(self._current[0], 10**12) < target_position
        ):
            self._current = next(self._iterator, None)
        if self._current is None or self._current[0] != idx:
            return []
        rows = self._current[1]
        self._current = None
        return rows


def _candidate_ids(final_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    rows = sorted(
        final_rows,
        key=lambda row: (int(row.get("global_final_rank") or 10**12), str(row.get("paper_arxiv_id") or "")),
    )
    return [
        normalize_arxiv_id(row.get("paper_arxiv_id"))
        for row in rows
        if normalize_arxiv_id(row.get("paper_arxiv_id"))
    ]


def existing_formula_rerank(
    final_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, Any]]]:
    ordered = _candidate_ids(final_rows)
    scores = {
        normalize_arxiv_id(row.get("paper_arxiv_id")): float(row.get("global_final_score") or 0.0)
        for row in final_rows
        if normalize_arxiv_id(row.get("paper_arxiv_id"))
    }
    features = {
        paper_id: {
            "query_score_normalized": row.get("query_score_normalized"),
            "max_subquery_score_normalized": row.get("max_subquery_score"),
            "global_alpha": row.get("global_alpha"),
            "rerank_score": scores[paper_id],
        }
        for row in final_rows
        if (paper_id := normalize_arxiv_id(row.get("paper_arxiv_id")))
    }
    return ordered, scores, features


def _max_subquery_features(
    candidate_ids: Sequence[str],
    component_rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    raw_by_subquery: Dict[Any, Dict[str, float]] = defaultdict(dict)
    for row in component_rows:
        paper_id = normalize_arxiv_id(row.get("paper_arxiv_id"))
        if not paper_id:
            continue
        raw_by_subquery[row.get("subquery_id")][paper_id] = float(row.get("subquery_score_raw") or 0.0)
    maximum = {paper_id: 0.0 for paper_id in candidate_ids}
    maximum_id = {paper_id: None for paper_id in candidate_ids}
    for subquery_id, raw_scores in raw_by_subquery.items():
        normalized = minmax(raw_scores, candidate_ids)
        for paper_id in candidate_ids:
            score = float(normalized.get(paper_id, 0.0))
            if score > maximum[paper_id]:
                maximum[paper_id] = score
                maximum_id[paper_id] = subquery_id
    return maximum, maximum_id


def _intent_features(
    candidate_ids: Sequence[str],
    seed_ids: Set[str],
    edges: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    labels_by_paper: Dict[str, Set[str]] = defaultdict(set)
    for edge in edges:
        expanded = normalize_arxiv_id(edge.get("expanded_arxiv_id"))
        if expanded and expanded not in seed_ids:
            labels_by_paper[expanded].update(
                str(label).strip().lower() for label in (edge.get("intents") or []) if str(label).strip()
            )
    labels = {paper_id: sorted(labels_by_paper.get(paper_id, set())) for paper_id in candidate_ids}
    scores = {
        paper_id: max((INTENT_WEIGHTS.get(label, 0.0) for label in labels[paper_id]), default=0.0)
        for paper_id in candidate_ids
    }
    return scores, labels


def _path_features(
    candidate_ids: Sequence[str],
    seed_ids: Set[str],
    edges: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, float]]:
    candidate_set = set(candidate_ids)
    neighbors: Dict[str, Set[str]] = {paper_id: set() for paper_id in candidate_ids}
    seen: Set[Tuple[str, str, str]] = set()
    for edge in edges:
        seed = normalize_arxiv_id(edge.get("seed_arxiv_id"))
        expanded = normalize_arxiv_id(edge.get("expanded_arxiv_id"))
        identity = (seed, expanded, str(edge.get("edge_type") or ""))
        if identity in seen or seed not in seed_ids or seed not in candidate_set or expanded not in candidate_set:
            continue
        seen.add(identity)
        neighbors[seed].add(expanded)
        neighbors[expanded].add(seed)
    counts = {paper_id: len(neighbors[paper_id]) for paper_id in candidate_ids}
    return counts, minmax(counts, candidate_ids)


def new_formula_rerank(
    final_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    seed_ids: Set[str],
) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, Any]]]:
    candidate_ids = _candidate_ids(final_rows)
    query_raw = {
        normalize_arxiv_id(row.get("paper_arxiv_id")): float(row.get("query_score_raw") or 0.0)
        for row in final_rows
        if normalize_arxiv_id(row.get("paper_arxiv_id"))
    }
    query_normalized = minmax(query_raw, candidate_ids)
    max_subquery, max_subquery_id = _max_subquery_features(candidate_ids, component_rows)
    intent_score, intent_labels = _intent_features(candidate_ids, seed_ids, edges)
    path_count, path_normalized = _path_features(candidate_ids, seed_ids, edges)
    scores = {
        paper_id: (
            NEW_FORMULA_WEIGHTS["query_score_normalized"] * query_normalized.get(paper_id, 0.0)
            + NEW_FORMULA_WEIGHTS["max_subquery_score_normalized"] * max_subquery.get(paper_id, 0.0)
            + NEW_FORMULA_WEIGHTS["intent_score"] * intent_score.get(paper_id, 0.0)
            + NEW_FORMULA_WEIGHTS["path_count_normalized"] * path_normalized.get(paper_id, 0.0)
        )
        for paper_id in candidate_ids
    }
    ordered = sorted(candidate_ids, key=lambda paper_id: (-scores[paper_id], -int(paper_id in seed_ids), paper_id))
    features = {
        paper_id: {
            "query_score_raw": query_raw.get(paper_id, 0.0),
            "query_score_normalized": query_normalized.get(paper_id, 0.0),
            "max_subquery_id": max_subquery_id.get(paper_id),
            "max_subquery_score_normalized": max_subquery.get(paper_id, 0.0),
            "intent_labels": intent_labels.get(paper_id, []),
            "intent_score": intent_score.get(paper_id, 0.0),
            "path_count": path_count.get(paper_id, 0),
            "path_count_normalized": path_normalized.get(paper_id, 0.0),
            "feature_weights": dict(NEW_FORMULA_WEIGHTS),
            "rerank_score": scores[paper_id],
        }
        for paper_id in candidate_ids
    }
    return ordered, scores, features


def query_metrics(gt_ids: Set[str], candidate_ids: Sequence[str], selected_ids: Sequence[str]) -> Dict[str, Any]:
    candidates = set(candidate_ids)
    selected = set(selected_ids)
    candidate_hits = sorted(gt_ids & candidates)
    selected_hits = sorted(gt_ids & selected)
    return {
        "gt_count": len(gt_ids),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidate_arxiv_ids": list(candidate_ids),
        "selected_arxiv_ids": list(selected_ids),
        "candidate_gt_ids": candidate_hits,
        "selected_gt_ids": selected_hits,
        "candidate_recall": _safe_div(len(candidate_hits), len(gt_ids)),
        "candidate_precision": _safe_div(len(candidate_hits), len(candidates)),
        "selection_recall": _safe_div(len(selected_hits), len(gt_ids)),
        "selection_precision": _safe_div(len(selected_hits), len(selected)),
    }


async def select_in_independent_batches(
    selector: Selector,
    *,
    method: str,
    query_text: str,
    benchmark_idx: int,
    candidate_ids: Sequence[str],
    scores: Mapping[str, float],
    paper_db: Mapping[str, Mapping[str, Any]],
    batch_size: int,
) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str]]:
    selected: Set[str] = set()
    selector_rows: List[Dict[str, Any]] = []
    all_reasons: Dict[str, str] = {}
    for start in range(0, len(candidate_ids), batch_size):
        batch_ids = list(candidate_ids[start : start + batch_size])
        batch_idx = start // batch_size + 1
        papers = []
        for paper_id in batch_ids:
            metadata = paper_db.get(paper_id) or {}
            papers.append(
                Paper(
                    id=paper_id,
                    arxiv_id=paper_id,
                    title=str(metadata.get("title") or "N/A"),
                    abstract=str(metadata.get("abstract") or "N/A"),
                    date=metadata.get("date") or "",
                    score=float(scores.get(paper_id, 0.0)),
                )
            )
        subquery = SubQuery(
            id=batch_idx,
            text=query_text,
            target_k=len(batch_ids),
            iter_index=1,
        )
        result = await selector.decide_for_subquery(
            user_query=query_text,
            sub_query=subquery,
            planner_checklist=DEFAULT_GLOBAL_CHECKLIST,
            papers=papers,
            iteration_index=1,
            idx=benchmark_idx,
            old_overview="",
            is_after_browsing=False,
            return_details=True,
        )
        kept, overview, _, details = result
        selected_ids = [normalize_arxiv_id(paper.arxiv_id or paper.id) for paper in kept]
        reasons = dict((details or {}).get("reasons") or {})
        selected.update(selected_ids)
        all_reasons.update(reasons)
        selector_rows.append(
            {
                "method": method,
                "benchmark_idx": benchmark_idx,
                "selector_batch_idx": batch_idx,
                "selector_batch_size": len(batch_ids),
                "rerank_start_rank": start + 1,
                "rerank_end_rank": start + len(batch_ids),
                "input_arxiv_ids": batch_ids,
                "selected_arxiv_ids": selected_ids,
                "selector_overview": overview or "",
                "selector_reasons": reasons,
                "checklist": DEFAULT_GLOBAL_CHECKLIST,
                "old_overview": "",
            }
        )
    ordered_selected = [paper_id for paper_id in candidate_ids if paper_id in selected]
    return ordered_selected, selector_rows, all_reasons


def _candidate_budget(final_rows: Sequence[Mapping[str, Any]], pool_size: int) -> int:
    values = [
        int(row.get("global_selector_top_k") or row.get("baseline_query_retrieval_count") or 0)
        for row in final_rows
    ]
    budget = max(values, default=0)
    return min(budget, pool_size)


def build_query_output(
    *,
    method: str,
    detail: Mapping[str, Any],
    final_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    selector: Selector,
    paper_db: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    save_level: str,
) -> Dict[str, Any]:
    benchmark_idx = int(detail["idx"])
    query_text = str(detail.get("query") or "")
    gt_ids = {normalize_arxiv_id(value) for value in detail.get("ground_truth_arxiv_ids") or []}
    query_id = str((detail.get("postprocess_results") or {}).get("query_id") or f"idx-{benchmark_idx}")
    seed_ids = {
        normalize_arxiv_id(row.get("paper_arxiv_id"))
        for row in baseline_rows
        if normalize_arxiv_id(row.get("paper_arxiv_id"))
    }
    if method == EXISTING_METHOD:
        ordered, scores, features = existing_formula_rerank(final_rows)
    elif method == NEW_METHOD:
        ordered, scores, features = new_formula_rerank(final_rows, component_rows, edges, seed_ids)
    else:
        raise ValueError(f"unsupported method: {method}")
    budget = _candidate_budget(final_rows, len(ordered))
    selector_ids = ordered[:budget]
    selected_ids, batches, reasons = asyncio.run(
        select_in_independent_batches(
            selector,
            method=method,
            query_text=query_text,
            benchmark_idx=benchmark_idx,
            candidate_ids=selector_ids,
            scores=scores,
            paper_db=paper_db,
            batch_size=batch_size,
        )
    )
    summary = {
        **query_metrics(gt_ids, selector_ids, selected_ids),
        "method": method,
        "query_id": query_id,
        "benchmark_idx": benchmark_idx,
        "candidate_pool_count": len(ordered),
        "selector_budget": budget,
        "selector_batch_size": batch_size,
        "selector_batch_count": len(batches),
        "selector_batches_independent": True,
        "default_checklist": DEFAULT_GLOBAL_CHECKLIST,
    }
    output: Dict[str, Any] = {
        "schema_version": "1.0-experimental",
        "method": method,
        "llm_model": config.LLM_MODEL_NAME,
        "query_id": query_id,
        "benchmark_idx": benchmark_idx,
        "query": query_text,
        "summary": summary,
        "selector_batches": batches,
    }
    if save_level == "full":
        selected_set = set(selected_ids)
        output["paper_rows"] = [
            {
                "method": method,
                "query_id": query_id,
                "benchmark_idx": benchmark_idx,
                "paper_arxiv_id": paper_id,
                "is_seed": paper_id in seed_ids,
                "rerank_score": float(scores.get(paper_id, 0.0)),
                "rerank_rank": rank,
                "in_selector_budget": rank <= budget,
                "selector_batch_idx": ((rank - 1) // batch_size + 1) if rank <= budget else None,
                "selector_batch_rank": ((rank - 1) % batch_size + 1) if rank <= budget else None,
                "selector_selected": paper_id in selected_set,
                "selector_reason": reasons.get(paper_id, ""),
                **features.get(paper_id, {}),
            }
            for rank, paper_id in enumerate(ordered, start=1)
        ]
    return output


def aggregate_query_outputs(outputs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summaries = [row.get("summary") or {} for row in outputs]
    count = len(summaries)
    metric_names = (
        "candidate_count",
        "selected_count",
        "candidate_recall",
        "candidate_precision",
        "selection_recall",
        "selection_precision",
        "selector_batch_count",
    )
    result: Dict[str, Any] = {"evaluated_query_count": count}
    for name in metric_names:
        result[f"avg_{name}"] = (
            sum(float(summary.get(name) or 0.0) for summary in summaries) / count if count else 0.0
        )
    total_gt = sum(int(summary.get("gt_count") or 0) for summary in summaries)
    total_candidates = sum(int(summary.get("candidate_count") or 0) for summary in summaries)
    total_selected = sum(int(summary.get("selected_count") or 0) for summary in summaries)
    candidate_hits = sum(len(summary.get("candidate_gt_ids") or []) for summary in summaries)
    selected_hits = sum(len(summary.get("selected_gt_ids") or []) for summary in summaries)
    result.update(
        {
            "total_gt_count": total_gt,
            "total_candidate_count": total_candidates,
            "total_selected_count": total_selected,
            "total_candidate_gt_count": candidate_hits,
            "total_selected_gt_count": selected_hits,
            "micro_candidate_recall": _safe_div(candidate_hits, total_gt),
            "micro_candidate_precision": _safe_div(candidate_hits, total_candidates),
            "micro_selection_recall": _safe_div(selected_hits, total_gt),
            "micro_selection_precision": _safe_div(selected_hits, total_selected),
        }
    )
    return result


def load_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("batched_replay_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_selector(config_path: Path, llm_model: Optional[str]) -> Selector:
    cfg = load_config(config_path)
    config.LLM_MODEL_NAME = llm_model or cfg.LLM_MODEL_NAME
    config.IS_LOCAL_LLM = cfg.IS_LOCAL_LLM
    config.LLM_GEN_PARAMS = cfg.LLM_GEN_PARAMS
    config.ENABLE_REASONING = cfg.ENABLE_REASONING
    config.ENABLE_STRUCTURED_OUTPUT = cfg.ENABLE_STRUCTURED_OUTPUT
    config.BROWSER_MODE = "NONE"
    config.DEBUG = False
    config.SAVE_AGENT_TRACES = False
    return Selector(config.LLM_MODEL_NAME, config.LLM_GEN_PARAMS, config.IS_LOCAL_LLM)


def completed_outputs(
    method_dir: Path,
    *,
    batch_size: int,
    llm_model: str,
) -> Dict[int, Dict[str, Any]]:
    outputs: Dict[int, Dict[str, Any]] = {}
    for path in sorted((method_dir / "queries").glob("*.json")) if (method_dir / "queries").exists() else []:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        idx = value.get("benchmark_idx")
        summary = value.get("summary") or {}
        if (
            isinstance(idx, int)
            and value.get("method") == method_dir.name
            and value.get("llm_model") == llm_model
            and summary.get("selector_batch_size") == batch_size
        ):
            outputs[idx] = value
    return outputs


def write_method_summary(method_dir: Path, outputs: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = [outputs[idx] for idx in sorted(outputs)]
    summary = aggregate_query_outputs(ordered)
    summary["method"] = method_dir.name
    query_results_path = method_dir / "query_results.jsonl"
    tmp = query_results_path.with_suffix(query_results_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as handle:
        for output in ordered:
            handle.write(json.dumps(output.get("summary") or {}, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, query_results_path)
    atomic_write_json(method_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline_run_dir", required=True, help="Completed OnePass run containing detailed_results.jsonl and onepass_artifacts")
    parser.add_argument("--paper_db", required=True)
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs/config_qwen30b_api.py"))
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--save_level", choices=("minimal", "full"), default="full")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Recompute method/query files that already exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    run_dir = Path(args.baseline_run_dir).resolve()
    artifacts = run_dir / "onepass_artifacts"
    detailed_path = run_dir / "detailed_results.jsonl"
    required = {
        "detailed_results": detailed_path,
        "baseline paper rows": artifacts / "baseline/paper_rows.jsonl",
        "global final rows": artifacts / "global/final_paper_rows.jsonl",
        "global component rows": artifacts / "global/subquery_paper_scores.jsonl",
        "global expansion edges": artifacts / "global/expansion_edges.jsonl",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required full-mode artifacts:\n" + "\n".join(missing))

    detailed = load_detailed_results(detailed_path)
    indices = list(detailed)
    if args.limit is not None:
        selected_indices = set(sorted(detailed)[: max(0, args.limit)])
        indices = [idx for idx in indices if idx in selected_indices]
    allowed = set(indices)
    order = {idx: position for position, idx in enumerate(indices)}
    selector = configure_selector(Path(args.config).resolve(), args.llm_model)
    paper_db = load_paper_db(args.paper_db)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "1.0-experimental",
            "baseline_run_dir": str(run_dir),
            "detailed_results_path": str(detailed_path),
            "paper_db_path": str(Path(args.paper_db).resolve()),
            "config_path": str(Path(args.config).resolve()),
            "llm_model": config.LLM_MODEL_NAME,
            "enable_reasoning": config.ENABLE_REASONING,
            "browser_mode": config.BROWSER_MODE,
            "methods": args.methods,
            "batch_size": args.batch_size,
            "selector_batches_independent": True,
            "selector_checklist": DEFAULT_GLOBAL_CHECKLIST,
            "candidate_budget": "existing global_selector_top_k (= baseline query retrieval occurrence count)",
            "new_formula_weights": NEW_FORMULA_WEIGHTS,
            "prompts_saved": False,
            "save_level": args.save_level,
        },
    )

    cursors = {
        "baseline": GroupCursor(required["baseline paper rows"], allowed, order),
        "final": GroupCursor(required["global final rows"], allowed, order),
        "components": GroupCursor(required["global component rows"], allowed, order),
        "edges": GroupCursor(required["global expansion edges"], allowed, order),
    }
    method_outputs = {
        method: {
            idx: value
            for idx, value in completed_outputs(
                output_dir / method,
                batch_size=args.batch_size,
                llm_model=config.LLM_MODEL_NAME,
            ).items()
            if idx in allowed
        }
        for method in args.methods
    }
    for position, idx in enumerate(indices, start=1):
        baseline_rows = cursors["baseline"].take(idx)
        final_rows = cursors["final"].take(idx)
        component_rows = cursors["components"].take(idx)
        edges = cursors["edges"].take(idx)
        if not final_rows:
            raise ValueError(f"no global final candidate rows for committed idx={idx}")
        for method in args.methods:
            query_path = output_dir / method / "queries" / f"{idx:06d}.json"
            if idx in method_outputs[method] and not args.force:
                continue
            print(f"[{position}/{len(indices)}] idx={idx} method={method}", flush=True)
            output = build_query_output(
                method=method,
                detail=detailed[idx],
                final_rows=final_rows,
                component_rows=component_rows,
                edges=edges,
                baseline_rows=baseline_rows,
                selector=selector,
                paper_db=paper_db,
                batch_size=args.batch_size,
                save_level=args.save_level,
            )
            atomic_write_json(query_path, output)
            method_outputs[method][idx] = output

    all_method_outputs: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for method in METHODS:
        outputs = method_outputs.get(method)
        if outputs is None:
            outputs = {
                idx: value
                for idx, value in completed_outputs(
                    output_dir / method,
                    batch_size=args.batch_size,
                    llm_model=config.LLM_MODEL_NAME,
                ).items()
                if idx in allowed
            }
        if outputs:
            all_method_outputs[method] = outputs
    overall = {
        "source_committed_query_count": len(indices),
        "batch_size": args.batch_size,
        "methods": {
            method: write_method_summary(output_dir / method, outputs)
            for method, outputs in all_method_outputs.items()
        },
    }
    atomic_write_json(output_dir / "evaluation_summary.json", overall)
    print(f"Saved replay outputs to {output_dir}")


if __name__ == "__main__":
    main()
