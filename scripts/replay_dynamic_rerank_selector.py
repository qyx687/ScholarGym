#!/usr/bin/env python3
"""Replay saved static/dynamic rerank Top-K candidates through Selector.

The experiment freezes the completed OnePass trajectory.  For every saved
retrieval event it reuses the original query, subquery, Planner checklist,
iteration, and Selector Top-K budget, while replacing only the ``<candidates>``
block with one rerank arm's papers.  Each paper carries that arm's own
``rerank_score``.

No Planner, retrieval, graph expansion, embedding, Semantic Scholar, or rerank
policy call is made here.  Selector calls are checkpointed independently so a
long API replay can be resumed safely.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from paper_type import normalize_paper_id  # noqa: E402
from runtime_env import load_env_file  # noqa: E402
from structures import Paper, SubQuery  # noqa: E402


SCHEMA_VERSION = "dynamic_rerank_selector_replay_v1"
METHODS = ("legacy_static", "dynamic_policy")
CONTEXT_FIELDS = (
    "query_id",
    "benchmark_idx",
    "query",
    "retrieval_event_id",
    "iteration_idx",
    "subquery_id",
    "subquery",
    "subquery_before_date",
    "subquery_target_k",
    "subquery_link_type",
    "parent_subquery_id",
    "selector_top_k",
    "planner_checklist",
)


class SelectorResponseParseError(RuntimeError):
    """Raised when legacy Selector parsing silently collapses to an empty dict."""


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(recall: float, precision: float) -> float:
    return 2.0 * recall * precision / (recall + precision) if recall + precision else 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ordered_unique(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield line_number, value


@dataclass(frozen=True)
class BenchmarkQuery:
    benchmark_idx: int
    query_id: str
    query: str
    gt_ids: frozenset[str]


def _extract_gt_ids(record: Mapping[str, Any]) -> Set[str]:
    papers = record.get("cited_paper") or record.get("ground_truth_papers") or []
    labels = record.get("gt_label") or record.get("gt_labels")
    if not isinstance(papers, list):
        return set()
    if not isinstance(labels, list):
        labels = [1] * len(papers)
    output: Set[str] = set()
    for paper, label in zip(papers, labels):
        if label != 1:
            continue
        raw_id = paper.get("arxiv_id") if isinstance(paper, Mapping) else paper
        paper_id = normalize_paper_id(raw_id)
        if paper_id:
            output.add(paper_id)
    return output


def load_benchmark(
    path: Path,
    limit: Optional[int] = None,
) -> Tuple[List[BenchmarkQuery], Dict[str, BenchmarkQuery]]:
    ordered: List[BenchmarkQuery] = []
    by_id: Dict[str, BenchmarkQuery] = {}
    for logical_idx, (_, row) in enumerate(iter_jsonl(path)):
        query_id = str(row.get("qid") or row.get("query_id") or f"idx-{logical_idx}")
        if query_id in by_id:
            raise ValueError(f"duplicate benchmark query_id={query_id}")
        item = BenchmarkQuery(
            benchmark_idx=logical_idx,
            query_id=query_id,
            query=str(row.get("query") or ""),
            gt_ids=frozenset(_extract_gt_ids(row)),
        )
        ordered.append(item)
        by_id[query_id] = item
    if limit is not None:
        ordered = ordered[: max(0, int(limit))]
        by_id = {item.query_id: item for item in ordered}
    if not ordered:
        raise ValueError(f"no benchmark queries selected from {path}")
    return ordered, by_id


def _candidate_sort_key(row: Mapping[str, Any]) -> Tuple[int, int, str]:
    def integer(name: str, fallback: int) -> int:
        try:
            return int(row.get(name))
        except (TypeError, ValueError):
            return fallback

    return (
        integer("rerank_rank", 10**12),
        integer("artifact_rank", 10**12),
        normalize_paper_id(row.get("paper_arxiv_id")),
    )


def load_ranked_requests(
    path: Path,
    benchmark_by_id: Mapping[str, BenchmarkQuery],
    methods: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed_queries = set(benchmark_by_id)
    allowed_methods = set(methods)
    requests: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    methods_by_event: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for line_number, row in iter_jsonl(path):
        query_id = str(row.get("query_id") or "")
        method = str(row.get("method") or "")
        if query_id not in allowed_queries or method not in allowed_methods:
            continue
        event_id = str(row.get("retrieval_event_id") or "")
        if not event_id:
            raise ValueError(f"ranked artifact line {line_number} has no retrieval_event_id")
        key = (query_id, event_id, method)
        if key in seen:
            raise ValueError(f"duplicate ranked request key={key}")
        seen.add(key)
        methods_by_event[(query_id, event_id)].add(method)
        top_k = int(row.get("selector_top_k") or 0)
        raw_candidates = row.get("ranked_candidates")
        if not isinstance(raw_candidates, list):
            raise ValueError(f"ranked artifact line {line_number} has no candidate list")
        selected_rows = sorted(
            (
                candidate
                for candidate in raw_candidates
                if isinstance(candidate, Mapping)
                and bool(candidate.get("selected_at_event_top_k"))
            ),
            key=_candidate_sort_key,
        )
        candidates: List[Dict[str, Any]] = []
        candidate_ids: Set[str] = set()
        for candidate in selected_rows:
            paper_id = normalize_paper_id(candidate.get("paper_arxiv_id"))
            score = candidate.get("rerank_score")
            if not paper_id:
                raise ValueError(f"empty paper id at ranked artifact line {line_number}")
            if paper_id in candidate_ids:
                raise ValueError(
                    f"duplicate paper_id={paper_id} for event={event_id}, method={method}"
                )
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(float(score)):
                raise ValueError(
                    f"missing/non-finite rerank_score for {paper_id} at line {line_number}"
                )
            candidate_ids.add(paper_id)
            candidates.append(
                {
                    "paper_arxiv_id": paper_id,
                    "rerank_rank": candidate.get("rerank_rank"),
                    "artifact_rank": candidate.get("artifact_rank"),
                    "rerank_score": float(score),
                    "rerank_policy_id": candidate.get("rerank_policy_id")
                    or row.get("rerank_policy_id"),
                }
            )
        if len(candidates) > top_k:
            raise ValueError(
                f"event={event_id}, method={method} has {len(candidates)} selected rows but top_k={top_k}"
            )
        benchmark = benchmark_by_id[query_id]
        artifact_idx = row.get("benchmark_idx")
        if artifact_idx is not None and int(artifact_idx) != benchmark.benchmark_idx:
            raise ValueError(
                f"benchmark index mismatch for query_id={query_id}: "
                f"artifact={artifact_idx}, benchmark={benchmark.benchmark_idx}"
            )
        requests.append(
            {
                "query_id": query_id,
                "benchmark_idx": benchmark.benchmark_idx,
                "retrieval_event_id": event_id,
                "method": method,
                "selector_top_k": top_k,
                "rerank_policy_id": row.get("rerank_policy_id"),
                "candidates": candidates,
                "ranked_artifact_line": line_number,
            }
        )
    expected_methods = set(methods)
    incomplete = {
        key: sorted(expected_methods - present)
        for key, present in methods_by_event.items()
        if present != expected_methods
    }
    if incomplete:
        examples = list(incomplete.items())[:10]
        raise ValueError(f"ranked artifact has incomplete method pairs: {examples}")
    if not requests:
        raise ValueError(f"no ranked Selector requests selected from {path}")
    return requests


def load_event_contexts(
    pool_records_path: Path,
    required_keys: Set[Tuple[str, str]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    contexts: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for line_number, row in iter_jsonl(pool_records_path):
        key = (str(row.get("query_id") or ""), str(row.get("retrieval_event_id") or ""))
        if key not in required_keys:
            continue
        if key in contexts:
            raise ValueError(f"duplicate pool event key={key} at line {line_number}")
        contexts[key] = {field: row.get(field) for field in CONTEXT_FIELDS}
        if len(contexts) == len(required_keys):
            break
    missing = sorted(required_keys - set(contexts))
    if missing:
        raise ValueError(f"pool records are missing {len(missing)} ranked events: {missing[:10]}")
    return contexts


def attach_event_contexts(
    requests: Sequence[Dict[str, Any]],
    contexts: Mapping[Tuple[str, str], Mapping[str, Any]],
    benchmark_by_id: Mapping[str, BenchmarkQuery],
) -> None:
    for request in requests:
        key = (request["query_id"], request["retrieval_event_id"])
        context = contexts[key]
        if int(context.get("selector_top_k") or 0) != int(request["selector_top_k"]):
            raise ValueError(
                f"selector_top_k mismatch for {key}, method={request['method']}: "
                f"pool={context.get('selector_top_k')}, ranked={request['selector_top_k']}"
            )
        benchmark = benchmark_by_id[request["query_id"]]
        request.update(
            {
                field: context.get(field)
                for field in CONTEXT_FIELDS
                if field not in {"query_id", "benchmark_idx", "retrieval_event_id", "selector_top_k"}
            }
        )
        request["query"] = str(context.get("query") or benchmark.query)
        request["subquery"] = str(context.get("subquery") or request["query"])
        request["planner_checklist"] = str(context.get("planner_checklist") or "")


def _iter_top_level_json_object(
    path: Path,
    chunk_size: int = 4 * 1024 * 1024,
) -> Iterator[Tuple[str, Any]]:
    """Stream key/value pairs from a large top-level JSON object."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def read_more(preserve_from: int) -> None:
            nonlocal buffer, position, eof
            prefix = buffer[preserve_from:]
            chunk = handle.read(chunk_size)
            eof = chunk == ""
            buffer = prefix + chunk
            position = 0

        read_more(0)

        def skip_ws() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                read_more(position)

        def expect(character: str) -> None:
            nonlocal position
            skip_ws()
            if position >= len(buffer) and not eof:
                read_more(position)
                skip_ws()
            if position >= len(buffer) or buffer[position] != character:
                nearby = buffer[position : position + 40]
                raise ValueError(f"expected {character!r} in {path}, found {nearby!r}")
            position += 1

        def decode_value() -> Any:
            nonlocal buffer, position, eof
            skip_ws()
            start = position
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, start)
                    position = end
                    return value
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"invalid JSON object in {path}: {exc}") from exc
                    prefix = buffer[start:]
                    chunk = handle.read(chunk_size)
                    eof = chunk == ""
                    buffer = prefix + chunk
                    position = 0
                    start = 0

        skip_ws()
        expect("{")
        skip_ws()
        first = True
        while True:
            skip_ws()
            if position < len(buffer) and buffer[position] == "}":
                position += 1
                break
            if not first:
                expect(",")
            key = decode_value()
            if not isinstance(key, str):
                raise ValueError(f"non-string key in {path}: {key!r}")
            expect(":")
            value = decode_value()
            yield key, value
            first = False
            if position > chunk_size:
                read_more(position)

        skip_ws()
        trailing = buffer[position:] + (handle.read() if not eof else "")
        if trailing.strip():
            raise ValueError(f"unexpected trailing data in {path}")


def load_needed_paper_metadata(
    paper_db_path: Path,
    paper_ids: Set[str],
) -> Dict[str, Dict[str, Any]]:
    remaining = set(paper_ids)
    output: Dict[str, Dict[str, Any]] = {}
    for key, value in _iter_top_level_json_object(paper_db_path):
        if not isinstance(value, Mapping):
            continue
        paper_id = normalize_paper_id(value.get("arxiv_id") or key)
        if paper_id not in remaining:
            continue
        output[paper_id] = {
            "title": str(value.get("title") or "N/A"),
            "abstract": str(value.get("abstract") or "N/A"),
            "date": value.get("date") or "",
        }
        remaining.remove(paper_id)
        if not remaining:
            break
    if remaining:
        raise ValueError(
            f"paper DB is missing metadata for {len(remaining)} Selector candidates: "
            f"{sorted(remaining)[:20]}"
        )
    return output


def _load_config(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("dynamic_selector_replay_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_selector(
    config_path: Path,
    llm_model: Optional[str],
) -> Tuple[Any, Dict[str, Any]]:
    """Configure the same Selector implementation used by OnePass."""

    import config  # Imported after environment loading.
    import api as llm_api
    from agent.selector import Selector

    # Per-request decisions are persisted below; suppress one INFO line per
    # call so multi-thousand-call replays remain readable.
    logging.getLogger("agent").setLevel(logging.WARNING)

    cfg = _load_config(config_path)
    config.LLM_MODEL_NAME = llm_model or cfg.LLM_MODEL_NAME
    config.IS_LOCAL_LLM = cfg.IS_LOCAL_LLM
    config.LLM_GEN_PARAMS = dict(cfg.LLM_GEN_PARAMS)
    config.ENABLE_REASONING = cfg.ENABLE_REASONING
    config.ENABLE_STRUCTURED_OUTPUT = cfg.ENABLE_STRUCTURED_OUTPUT
    config.ENABLE_LLM_FILTERING = getattr(cfg, "ENABLE_LLM_FILTERING", True)
    if not config.ENABLE_LLM_FILTERING:
        raise ValueError("Selector replay requires ENABLE_LLM_FILTERING=True")
    config.BROWSER_MODE = "NONE"
    config.DEBUG = False
    config.SAVE_AGENT_TRACES = False
    selector = Selector(config.LLM_MODEL_NAME, config.LLM_GEN_PARAMS, config.IS_LOCAL_LLM)
    provider = llm_api._resolve_provider(config.LLM_MODEL_NAME, config.IS_LOCAL_LLM)
    return selector, {
        "llm_model": config.LLM_MODEL_NAME,
        "is_local_llm": config.IS_LOCAL_LLM,
        "llm_gen_params": dict(config.LLM_GEN_PARAMS),
        "enable_reasoning": config.ENABLE_REASONING,
        "enable_structured_output": config.ENABLE_STRUCTURED_OUTPUT,
        "enable_llm_filtering": config.ENABLE_LLM_FILTERING,
        "browser_mode": config.BROWSER_MODE,
        "provider_base_url_sha256": hashlib.sha256(
            str(provider.get("base_url") or "").encode("utf-8")
        ).hexdigest(),
    }


def selector_prompt_fingerprint() -> Dict[str, str]:
    paths = {
        "selector": CODE_DIR / "agent/selector.py",
        "prompt": CODE_DIR / "prompt.py",
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def attach_metadata_and_signatures(
    requests: Sequence[Dict[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    selector_config: Mapping[str, Any],
    prompt_fingerprint: Mapping[str, str],
) -> None:
    for request in requests:
        prompt_candidates: List[Dict[str, Any]] = []
        for candidate in request["candidates"]:
            paper_id = candidate["paper_arxiv_id"]
            item = metadata[paper_id]
            prompt_candidates.append(
                {
                    **candidate,
                    "title": str(item.get("title") or "N/A"),
                    "abstract": str(item.get("abstract") or "N/A"),
                    "date": item.get("date") or "",
                }
            )
        request["prompt_candidates"] = prompt_candidates
        request["input_signature"] = _sha256_json(
            {
                "schema_version": SCHEMA_VERSION,
                "method": request["method"],
                "query_id": request["query_id"],
                "retrieval_event_id": request["retrieval_event_id"],
                "query": request["query"],
                "subquery": request["subquery"],
                "planner_checklist": request["planner_checklist"],
                "iteration_idx": request.get("iteration_idx"),
                "subquery_id": request.get("subquery_id"),
                "subquery_before_date": request.get("subquery_before_date"),
                "subquery_target_k": request.get("subquery_target_k"),
                "subquery_link_type": request.get("subquery_link_type"),
                "parent_subquery_id": request.get("parent_subquery_id"),
                "selector_top_k": request["selector_top_k"],
                "prompt_candidates": prompt_candidates,
                "selector_config": selector_config,
                "prompt_fingerprint": prompt_fingerprint,
            }
        )


def _checkpoint_path(output_dir: Path, request: Mapping[str, Any]) -> Path:
    event_digest = hashlib.sha256(
        str(request["retrieval_event_id"]).encode("utf-8")
    ).hexdigest()[:16]
    return (
        output_dir
        / "checkpoints"
        / str(request["method"])
        / f"{int(request['benchmark_idx']):06d}_{event_digest}.json"
    )


def _load_checkpoint(path: Path, request: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    silently_unparsed = bool(
        isinstance(value, dict)
        and value.get("selector_executed")
        and value.get("selector_input_arxiv_ids")
        and not value.get("selector_selected_arxiv_ids")
        and not value.get("selector_reasons")
        and not value.get("selector_overview")
    )
    if (
        isinstance(value, dict)
        and value.get("status") == "success"
        and not silently_unparsed
        and value.get("input_signature") == request.get("input_signature")
        and value.get("method") == request.get("method")
        and value.get("retrieval_event_id") == request.get("retrieval_event_id")
    ):
        # Preserve how the decision was originally produced (API versus
        # precomputed OnePass); track resume provenance separately.
        value["loaded_from_checkpoint"] = True
        return value
    return None


def _base_result(request: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "query_id": request["query_id"],
        "benchmark_idx": request["benchmark_idx"],
        "retrieval_event_id": request["retrieval_event_id"],
        "iteration_idx": request.get("iteration_idx"),
        "subquery_id": request.get("subquery_id"),
        "subquery": request.get("subquery"),
        "planner_checklist": request.get("planner_checklist"),
        "method": request["method"],
        "selector_top_k": request["selector_top_k"],
        "rerank_policy_id": request.get("rerank_policy_id"),
        "selector_input_arxiv_ids": [
            candidate["paper_arxiv_id"] for candidate in request["candidates"]
        ],
        "selector_input_rerank_scores": {
            candidate["paper_arxiv_id"]: candidate["rerank_score"]
            for candidate in request["candidates"]
        },
        "input_signature": request["input_signature"],
    }


def load_precomputed_decisions(
    method_paths: Mapping[str, Path],
    requests: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    request_by_key = {
        (request["query_id"], request["retrieval_event_id"], request["method"]): request
        for request in requests
    }
    outputs: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    silently_unparsed_count = 0
    for method, path in method_paths.items():
        if method not in METHODS:
            raise ValueError(f"unsupported precomputed method={method}")
        for line_number, row in iter_jsonl(path):
            key = (
                str(row.get("query_id") or ""),
                str(row.get("retrieval_event_id") or ""),
                method,
            )
            request = request_by_key.get(key)
            if request is None:
                continue
            if key in outputs:
                raise ValueError(f"duplicate precomputed decision key={key} at {path}:{line_number}")
            candidate_rows = row.get("candidate_rows") or []
            ordered_rows = sorted(
                (
                    candidate
                    for candidate in candidate_rows
                    if isinstance(candidate, Mapping)
                    and bool(candidate.get("in_selector_topk", True))
                ),
                key=lambda candidate: (
                    int(candidate.get("selector_input_rank") or candidate.get("rerank_rank") or 10**12),
                    normalize_paper_id(candidate.get("paper_arxiv_id")),
                ),
            )
            expected_ids = [candidate["paper_arxiv_id"] for candidate in request["candidates"]]
            actual_ids = [normalize_paper_id(candidate.get("paper_arxiv_id")) for candidate in ordered_rows]
            if actual_ids != expected_ids:
                raise ValueError(
                    f"precomputed candidates do not match ranked artifact for key={key}: "
                    f"expected={expected_ids}, actual={actual_ids}"
                )
            for expected, actual in zip(request["candidates"], ordered_rows):
                score = actual.get("rerank_score")
                if not isinstance(score, (int, float)) or not math.isclose(
                    float(score), float(expected["rerank_score"]), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        f"precomputed rerank score mismatch for key={key}, "
                        f"paper_id={expected['paper_arxiv_id']}"
                    )
            if str(row.get("planner_checklist") or "") != str(request["planner_checklist"]):
                raise ValueError(f"precomputed checklist mismatch for key={key}")
            if str(row.get("subquery") or "") != str(request["subquery"]):
                raise ValueError(f"precomputed subquery mismatch for key={key}")
            selected_set = {
                normalize_paper_id(value) for value in row.get("selected_arxiv_ids") or []
            }
            selected_ids = [paper_id for paper_id in expected_ids if paper_id in selected_set]
            reasons = {
                normalize_paper_id(candidate.get("paper_arxiv_id")): str(
                    candidate.get("selector_reason") or ""
                )
                for candidate in ordered_rows
                if candidate.get("selector_reason")
            }
            overview = str(row.get("selector_overview") or "")
            if expected_ids and not selected_ids and not reasons and not overview:
                # Older OnePass Selector parsing converted malformed JSON to
                # an all-empty decision without raising.  Do not reuse it.
                silently_unparsed_count += 1
                continue
            outputs[key] = {
                **_base_result(request),
                "status": "success",
                "result_source": "precomputed",
                "attempt_count": 0,
                "selector_executed": True,
                "selector_selected_arxiv_ids": selected_ids,
                "selector_reasons": reasons,
                "selector_overview": overview,
                "precomputed_decisions_path": str(path.resolve()),
            }
    if silently_unparsed_count:
        print(
            f"Ignored {silently_unparsed_count} silently unparsed precomputed "
            "Selector decisions; they will be called again",
            flush=True,
        )
    return outputs


def _subquery_for_request(request: Mapping[str, Any]) -> SubQuery:
    try:
        subquery_id = int(request.get("subquery_id") or 0)
    except (TypeError, ValueError):
        subquery_id = 0
    try:
        source_id = int(request.get("parent_subquery_id"))
    except (TypeError, ValueError):
        source_id = None
    return SubQuery(
        id=subquery_id,
        text=str(request.get("subquery") or request.get("query") or ""),
        before_date=request.get("subquery_before_date"),
        target_k=int(request.get("subquery_target_k") or request.get("selector_top_k") or 0),
        link_type=request.get("subquery_link_type"),
        source_subquery_id=source_id,
        iter_index=int(request.get("iteration_idx") or 1),
    )


async def _call_selector(selector: Any, request: Mapping[str, Any]) -> Dict[str, Any]:
    if not request["prompt_candidates"]:
        return {
            **_base_result(request),
            "status": "success",
            "result_source": "empty_input",
            "attempt_count": 0,
            "selector_executed": False,
            "selector_selected_arxiv_ids": [],
            "selector_reasons": {},
            "selector_overview": "",
        }
    papers = [
        Paper(
            id=candidate["paper_arxiv_id"],
            arxiv_id=candidate["paper_arxiv_id"],
            title=candidate["title"],
            abstract=candidate["abstract"],
            date=candidate["date"],
            score=float(candidate["rerank_score"]),
        )
        for candidate in request["prompt_candidates"]
    ]
    kept, overview, _, details = await selector.decide_for_subquery(
        user_query=str(request.get("query") or ""),
        sub_query=_subquery_for_request(request),
        planner_checklist=str(request.get("planner_checklist") or ""),
        papers=papers,
        iteration_index=int(request.get("iteration_idx") or 1),
        idx=int(request.get("benchmark_idx") or 0),
        old_overview="",
        is_after_browsing=False,
        return_details=True,
    )
    selected_set = {
        normalize_paper_id(paper.arxiv_id or paper.id)
        for paper in kept
        if normalize_paper_id(paper.arxiv_id or paper.id)
    }
    input_ids = [candidate["paper_arxiv_id"] for candidate in request["candidates"]]
    raw_reasons = dict((details or {}).get("reasons") or {})
    reasons = {
        normalize_paper_id(key): str(value)
        for key, value in raw_reasons.items()
        if normalize_paper_id(key) in set(input_ids)
    }
    if not selected_set and not reasons and not (overview or "").strip():
        raise SelectorResponseParseError(
            "Selector returned no selected IDs, reasons, or overview; "
            "the legacy parser most likely rejected malformed JSON"
        )
    return {
        **_base_result(request),
        "status": "success",
        "result_source": "api",
        "selector_executed": True,
        "selector_selected_arxiv_ids": [
            paper_id for paper_id in input_ids if paper_id in selected_set
        ],
        "selector_reasons": reasons,
        "selector_overview": overview or "",
    }


async def execute_requests(
    selector: Any,
    requests: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    concurrency: int,
    max_attempts: int,
    force: bool,
    precomputed: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    progress_every: int = 25,
) -> List[Dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    results: List[Optional[Dict[str, Any]]] = [None] * len(requests)
    pending: List[Tuple[int, Mapping[str, Any], Path]] = []
    source_counts: Counter[str] = Counter()
    for index, request in enumerate(requests):
        checkpoint_path = _checkpoint_path(output_dir, request)
        key = (request["query_id"], request["retrieval_event_id"], request["method"])
        precomputed_result = precomputed.get(key)
        if precomputed_result is not None:
            value = dict(precomputed_result)
            atomic_write_json(checkpoint_path, value)
            results[index] = value
            source_counts["precomputed"] += 1
            continue
        cached = None if force else _load_checkpoint(checkpoint_path, request)
        if cached is not None:
            results[index] = cached
            source_counts["checkpoint"] += 1
            continue
        pending.append((index, request, checkpoint_path))

    print(
        f"Selector requests={len(requests)}, pending_api={len(pending)}, "
        f"precomputed={source_counts['precomputed']}, checkpoint={source_counts['checkpoint']}",
        flush=True,
    )
    semaphore = asyncio.Semaphore(concurrency)
    completed_api = 0

    async def run_one(
        index: int,
        request: Mapping[str, Any],
        checkpoint_path: Path,
    ) -> None:
        nonlocal completed_api
        async with semaphore:
            last_error: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    value = await _call_selector(selector, request)
                    value["attempt_count"] = attempt
                    atomic_write_json(checkpoint_path, value)
                    results[index] = value
                    source_counts[value["result_source"]] += 1
                    break
                except Exception as exc:  # Keep the long replay resumable.
                    last_error = exc
                    if attempt < max_attempts:
                        await asyncio.sleep(min(20.0, float(2 ** (attempt - 1))))
            if results[index] is None:
                error_value = {
                    **_base_result(request),
                    "status": "error",
                    "result_source": "api_error",
                    "attempt_count": max_attempts,
                    "selector_executed": True,
                    "selector_selected_arxiv_ids": [],
                    "selector_reasons": {},
                    "selector_overview": "",
                    "error_type": type(last_error).__name__ if last_error else "UnknownError",
                    "error": str(last_error or "unknown Selector failure"),
                }
                atomic_write_json(checkpoint_path, error_value)
                results[index] = error_value
                source_counts["api_error"] += 1
            completed_api += 1
            if (
                completed_api == len(pending)
                or completed_api % max(1, progress_every) == 0
                or results[index].get("status") != "success"
            ):
                print(
                    f"Selector API progress {completed_api}/{len(pending)} "
                    f"(errors={source_counts['api_error']})",
                    flush=True,
                )

    await asyncio.gather(*(run_one(*item) for item in pending))
    if any(value is None for value in results):
        raise AssertionError("internal error: missing Selector request result")
    return [dict(value) for value in results if value is not None]


def _query_metrics(
    gt_ids: Set[str],
    candidate_ids: Sequence[str],
    selected_ids: Sequence[str],
) -> Dict[str, Any]:
    candidates = set(candidate_ids)
    selected = set(selected_ids)
    candidate_hits = candidates & gt_ids
    selected_hits = selected & gt_ids
    candidate_recall = _safe_div(len(candidate_hits), len(gt_ids))
    candidate_precision = _safe_div(len(candidate_hits), len(candidates))
    selection_recall = _safe_div(len(selected_hits), len(gt_ids))
    selection_precision = _safe_div(len(selected_hits), len(selected))
    return {
        "gt_count": len(gt_ids),
        "candidate_count": len(candidates),
        "candidate_gt_count": len(candidate_hits),
        "candidate_arxiv_ids": list(candidate_ids),
        "candidate_gt_ids": sorted(candidate_hits),
        "candidate_recall": candidate_recall,
        "candidate_precision": candidate_precision,
        "candidate_f1": _f1(candidate_recall, candidate_precision),
        "selected_count": len(selected),
        "selected_gt_count": len(selected_hits),
        "selected_arxiv_ids": list(selected_ids),
        "selected_gt_ids": sorted(selected_hits),
        "selection_recall": selection_recall,
        "selection_precision": selection_precision,
        "selection_f1": _f1(selection_recall, selection_precision),
    }


def _aggregate(method: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_gt = sum(int(row.get("gt_count") or 0) for row in rows)
    total_candidates = sum(int(row.get("candidate_count") or 0) for row in rows)
    total_candidate_hits = sum(int(row.get("candidate_gt_count") or 0) for row in rows)
    total_selected = sum(int(row.get("selected_count") or 0) for row in rows)
    total_selected_hits = sum(int(row.get("selected_gt_count") or 0) for row in rows)
    micro_candidate_recall = _safe_div(total_candidate_hits, total_gt)
    micro_candidate_precision = _safe_div(total_candidate_hits, total_candidates)
    micro_selection_recall = _safe_div(total_selected_hits, total_gt)
    micro_selection_precision = _safe_div(total_selected_hits, total_selected)
    output: Dict[str, Any] = {
        "method": method,
        "evaluated_query_count": len(rows),
        "total_gt_count": total_gt,
        "total_candidate_count": total_candidates,
        "total_candidate_gt_count": total_candidate_hits,
        "total_selected_count": total_selected,
        "total_selected_gt_count": total_selected_hits,
        "micro_candidate_recall": micro_candidate_recall,
        "micro_candidate_precision": micro_candidate_precision,
        "micro_candidate_f1": _f1(micro_candidate_recall, micro_candidate_precision),
        "micro_selection_recall": micro_selection_recall,
        "micro_selection_precision": micro_selection_precision,
        "micro_selection_f1": _f1(micro_selection_recall, micro_selection_precision),
    }
    for metric in (
        "candidate_count",
        "candidate_recall",
        "candidate_precision",
        "candidate_f1",
        "selected_count",
        "selection_recall",
        "selection_precision",
        "selection_f1",
    ):
        output[f"avg_{metric}"] = _mean([float(row.get(metric) or 0.0) for row in rows])
    output["mean_query_candidate_f1"] = output["avg_candidate_f1"]
    output["macro_candidate_f1_from_avg_recall_precision"] = _f1(
        output["avg_candidate_recall"], output["avg_candidate_precision"]
    )
    output["main_table_candidate_f1"] = output[
        "macro_candidate_f1_from_avg_recall_precision"
    ]
    output["mean_query_selection_f1"] = output["avg_selection_f1"]
    output["macro_selection_f1_from_avg_recall_precision"] = _f1(
        output["avg_selection_recall"], output["avg_selection_precision"]
    )
    output["main_table_selection_f1"] = output[
        "macro_selection_f1_from_avg_recall_precision"
    ]
    return output


def _numeric_delta(dynamic: Mapping[str, Any], legacy: Mapping[str, Any]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key in sorted(set(dynamic) & set(legacy)):
        left, right = dynamic[key], legacy[key]
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        ):
            output[key] = float(left) - float(right)
    return output


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = max(0.0, min(1.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_selection_delta(
    per_query: Sequence[Mapping[str, Any]],
    *,
    iterations: int = 20_000,
    seed: int = 20260719,
) -> Dict[str, Any]:
    paired = [
        row
        for row in per_query
        if row.get("complete")
        and all(method in (row.get("methods") or {}) for method in METHODS)
    ]
    if not paired:
        return {
            "paired_query_count": 0,
            "iterations": 0,
            "seed": seed,
            "metrics": {},
        }

    def sample_metrics(method: str, indices: Sequence[int]) -> Dict[str, float]:
        rows = [paired[index]["methods"][method] for index in indices]
        avg_recall = _mean([float(row.get("selection_recall") or 0.0) for row in rows])
        avg_precision = _mean(
            [float(row.get("selection_precision") or 0.0) for row in rows]
        )
        total_gt = sum(int(row.get("gt_count") or 0) for row in rows)
        total_selected = sum(int(row.get("selected_count") or 0) for row in rows)
        total_hits = sum(int(row.get("selected_gt_count") or 0) for row in rows)
        micro_recall = _safe_div(total_hits, total_gt)
        micro_precision = _safe_div(total_hits, total_selected)
        return {
            "main_table_selection_f1": _f1(avg_recall, avg_precision),
            "mean_query_selection_f1": _mean(
                [float(row.get("selection_f1") or 0.0) for row in rows]
            ),
            "micro_selection_f1": _f1(micro_recall, micro_precision),
        }

    all_indices = list(range(len(paired)))
    point = {
        method: sample_metrics(method, all_indices)
        for method in METHODS
    }
    rng = random.Random(seed)
    distributions: Dict[str, List[float]] = {
        metric: []
        for metric in (
            "main_table_selection_f1",
            "mean_query_selection_f1",
            "micro_selection_f1",
        )
    }
    for _ in range(max(1, int(iterations))):
        indices = [rng.randrange(len(paired)) for _ in paired]
        legacy = sample_metrics("legacy_static", indices)
        dynamic = sample_metrics("dynamic_policy", indices)
        for metric in distributions:
            distributions[metric].append(dynamic[metric] - legacy[metric])

    metric_outputs: Dict[str, Any] = {}
    for metric, deltas in distributions.items():
        probability_nonpositive = _safe_div(
            sum(delta <= 0.0 for delta in deltas), len(deltas)
        )
        probability_nonnegative = _safe_div(
            sum(delta >= 0.0 for delta in deltas), len(deltas)
        )
        metric_outputs[metric] = {
            "legacy_static": point["legacy_static"][metric],
            "dynamic_policy": point["dynamic_policy"][metric],
            "dynamic_minus_legacy": (
                point["dynamic_policy"][metric] - point["legacy_static"][metric]
            ),
            "bootstrap_mean_delta": _mean(deltas),
            "ci95_low": _percentile(deltas, 0.025),
            "ci95_high": _percentile(deltas, 0.975),
            "probability_delta_gt_zero": _safe_div(
                sum(delta > 0.0 for delta in deltas), len(deltas)
            ),
            "two_sided_bootstrap_p": min(
                1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)
            ),
        }

    query_deltas = [
        float(row["methods"]["dynamic_policy"].get("selection_f1") or 0.0)
        - float(row["methods"]["legacy_static"].get("selection_f1") or 0.0)
        for row in paired
    ]
    return {
        "paired_query_count": len(paired),
        "iterations": max(1, int(iterations)),
        "seed": seed,
        "metrics": metric_outputs,
        "query_win_tie_loss": {
            "dynamic_wins": sum(delta > 1e-12 for delta in query_deltas),
            "ties": sum(abs(delta) <= 1e-12 for delta in query_deltas),
            "dynamic_losses": sum(delta < -1e-12 for delta in query_deltas),
        },
    }


def aggregate_results(
    ordered_benchmark: Sequence[BenchmarkQuery],
    requests: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    request_groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    result_groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for request in requests:
        request_groups[(request["query_id"], request["method"])].append(request)
    for result in results:
        if result.get("status") == "success":
            result_groups[(result["query_id"], result["method"])].append(result)

    per_query: List[Dict[str, Any]] = []
    method_rows: Dict[str, List[Dict[str, Any]]] = {method: [] for method in methods}
    for benchmark in ordered_benchmark:
        method_values: Dict[str, Any] = {}
        query_complete = True
        for method in methods:
            event_requests = request_groups.get((benchmark.query_id, method), [])
            event_results = result_groups.get((benchmark.query_id, method), [])
            if not event_requests or len(event_results) != len(event_requests):
                query_complete = False
                continue
            candidate_ids = _ordered_unique(
                candidate["paper_arxiv_id"]
                for request in event_requests
                for candidate in request["candidates"]
            )
            selected_by_event = {
                result["retrieval_event_id"]: result.get("selector_selected_arxiv_ids") or []
                for result in event_results
            }
            selected_ids = _ordered_unique(
                paper_id
                for request in event_requests
                for paper_id in selected_by_event.get(request["retrieval_event_id"], [])
            )
            metrics = {
                **_query_metrics(set(benchmark.gt_ids), candidate_ids, selected_ids),
                "event_count": len(event_requests),
            }
            method_values[method] = metrics
            method_rows[method].append(metrics)
        row: Dict[str, Any] = {
            "benchmark_idx": benchmark.benchmark_idx,
            "query_id": benchmark.query_id,
            "query": benchmark.query,
            "gt_count": len(benchmark.gt_ids),
            "complete": query_complete and len(method_values) == len(methods),
            "methods": method_values,
        }
        if all(method in method_values for method in METHODS):
            row["dynamic_minus_legacy"] = _numeric_delta(
                method_values["dynamic_policy"], method_values["legacy_static"]
            )
        per_query.append(row)
    summaries = {method: _aggregate(method, method_rows[method]) for method in methods}
    return per_query, summaries


def replay(
    *,
    ranked_candidates_path: Path,
    pool_records_path: Path,
    benchmark_path: Path,
    paper_db_path: Path,
    output_dir: Path,
    selector: Any,
    selector_config: Mapping[str, Any],
    methods: Sequence[str] = METHODS,
    precomputed_paths: Optional[Mapping[str, Path]] = None,
    limit: Optional[int] = None,
    concurrency: int = 4,
    max_attempts: int = 3,
    force: bool = False,
) -> Dict[str, Any]:
    paths = {
        "ranked_candidates": ranked_candidates_path.resolve(),
        "pool_records": pool_records_path.resolve(),
        "benchmark": benchmark_path.resolve(),
        "paper_db": paper_db_path.resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_benchmark, benchmark_by_id = load_benchmark(paths["benchmark"], limit)
    requests = load_ranked_requests(paths["ranked_candidates"], benchmark_by_id, methods)
    required_event_keys = {
        (request["query_id"], request["retrieval_event_id"]) for request in requests
    }
    contexts = load_event_contexts(paths["pool_records"], required_event_keys)
    attach_event_contexts(requests, contexts, benchmark_by_id)
    paper_ids = {
        candidate["paper_arxiv_id"]
        for request in requests
        for candidate in request["candidates"]
    }
    metadata = load_needed_paper_metadata(paths["paper_db"], paper_ids)
    prompt_fingerprint = selector_prompt_fingerprint()
    attach_metadata_and_signatures(requests, metadata, selector_config, prompt_fingerprint)
    precomputed = load_precomputed_decisions(precomputed_paths or {}, requests)
    results = asyncio.run(
        execute_requests(
            selector,
            requests,
            output_dir,
            concurrency=concurrency,
            max_attempts=max_attempts,
            force=force,
            precomputed=precomputed,
        )
    )
    per_query, method_summaries = aggregate_results(
        ordered_benchmark, requests, results, methods
    )
    bootstrap = (
        paired_bootstrap_selection_delta(per_query)
        if all(method in methods for method in METHODS)
        else None
    )
    source_counts = Counter(str(result.get("result_source") or "unknown") for result in results)
    failure_count = sum(result.get("status") != "success" for result in results)
    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "complete": failure_count == 0,
        "evaluated_query_count": len(ordered_benchmark),
        "complete_paired_query_count": sum(bool(row.get("complete")) for row in per_query),
        "selector_request_count": len(requests),
        "selector_failure_count": failure_count,
        "selector_result_source_counts": dict(sorted(source_counts.items())),
        "selector_loaded_from_checkpoint_count": sum(
            bool(result.get("loaded_from_checkpoint")) for result in results
        ),
        "methods": method_summaries,
        "candidate_prompt_contract": (
            "each arm uses its own rerank Top-K papers and its own raw rerank_score; "
            "query/subquery/checklist/iteration/Selector prompt/model are frozen"
        ),
        "open_loop": True,
        "planner_rerun": False,
        "paths": {name: str(path) for name, path in paths.items()},
        "precomputed_decision_paths": {
            method: str(path.resolve()) for method, path in (precomputed_paths or {}).items()
        },
        "selector_config": dict(selector_config),
        "selector_prompt_fingerprint": prompt_fingerprint,
        "selector_parser_sha256": _sha256_file(CODE_DIR / "utils.py"),
    }
    if all(method in method_summaries for method in METHODS):
        summary["dynamic_minus_legacy"] = _numeric_delta(
            method_summaries["dynamic_policy"], method_summaries["legacy_static"]
        )
        summary["paired_bootstrap_selection_delta"] = bootstrap
    atomic_write_jsonl(output_dir / "selector_decisions.jsonl", results)
    atomic_write_jsonl(output_dir / "per_query_results.jsonl", per_query)
    atomic_write_json(output_dir / "summary.json", summary)
    if bootstrap is not None:
        atomic_write_json(output_dir / "bootstrap_selection_delta.json", bootstrap)
    atomic_write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "methods": list(methods),
            "limit": limit,
            "concurrency": concurrency,
            "max_attempts": max_attempts,
            "paths": summary["paths"],
            "precomputed_decision_paths": summary["precomputed_decision_paths"],
            "selector_config": dict(selector_config),
            "selector_prompt_fingerprint": prompt_fingerprint,
            "selector_parser_sha256": summary["selector_parser_sha256"],
            "candidate_prompt_contract": summary["candidate_prompt_contract"],
            "api_credentials_persisted": False,
        },
    )
    return summary


def _parse_method_paths(values: Sequence[str]) -> Dict[str, Path]:
    output: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"--precomputed-decisions expects METHOD=PATH, received {value!r}"
            )
        method, raw_path = value.split("=", 1)
        method = method.strip()
        if method not in METHODS:
            raise ValueError(f"unsupported precomputed method={method}")
        if method in output:
            raise ValueError(f"duplicate precomputed path for method={method}")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        output[method] = path
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-candidates", required=True)
    parser.add_argument("--pool-records", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--paper-db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/config_qwen30b_api.py"),
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--precomputed-decisions",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Reuse a saved Selector arm only after exact candidate/order/score/context validation.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config_path = Path(args.config).expanduser().resolve()
    selector, selector_config = configure_selector(config_path, args.llm_model)
    precomputed_paths = _parse_method_paths(args.precomputed_decisions)
    summary = replay(
        ranked_candidates_path=Path(args.ranked_candidates),
        pool_records_path=Path(args.pool_records),
        benchmark_path=Path(args.benchmark),
        paper_db_path=Path(args.paper_db),
        output_dir=Path(args.output_dir),
        selector=selector,
        selector_config=selector_config,
        methods=args.methods,
        precomputed_paths=precomputed_paths,
        limit=args.limit,
        concurrency=args.concurrency,
        max_attempts=args.max_attempts,
        force=args.force,
    )
    for method, values in summary["methods"].items():
        print(
            f"{method}: selection F1={values['main_table_selection_f1']:.6f}, "
            f"recall={values['avg_selection_recall']:.6f}, "
            f"precision={values['avg_selection_precision']:.6f}",
            flush=True,
        )
    if "dynamic_minus_legacy" in summary:
        delta = summary["dynamic_minus_legacy"]
        print(
            f"dynamic - static selection F1={delta['main_table_selection_f1']:+.6f}",
            flush=True,
        )
    print(f"Saved Selector replay to {Path(args.output_dir).resolve()}", flush=True)
    if not summary["complete"]:
        raise RuntimeError(
            f"Selector replay has {summary['selector_failure_count']} failed requests; rerun to resume"
        )


if __name__ == "__main__":
    main()
