"""LLM prompts and strict parsers for SemRank concept construction."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI

from api import _call_llm, _resolve_provider

from .models import (
    AuxiliaryPaper,
    SEMRANK_PAPER_PROMPT_VERSION,
    SEMRANK_QUERY_PROMPT_VERSION,
    TopicCandidate,
    normalize_concept,
    stable_unique,
)


def _extract_json_object(text: str) -> Mapping[str, Any]:
    value = str(text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    candidates = [value]
    if start >= 0 and end > start and value[start : end + 1] != value:
        candidates.append(value[start : end + 1])
    parsed = None
    last_error: Optional[json.JSONDecodeError] = None
    for candidate in candidates:
        for payload in (candidate, candidate.replace("\\'", "'")):
            try:
                parsed = json.loads(payload)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        if parsed is not None:
            break
    if parsed is None:
        if start < 0 or end <= start:
            raise ValueError("LLM response does not contain a JSON object")
        assert last_error is not None
        raise last_error
    if not isinstance(parsed, Mapping):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def _list_field(value: Mapping[str, Any], key: str) -> List[Any]:
    selected = value.get(key)
    if not isinstance(selected, (list, tuple)):
        return []
    return list(selected)


def _recover_json_string_array(
    text: str,
    key: str,
    *,
    require_closing_bracket: bool,
    implicit_stop_field: Optional[str] = None,
) -> Tuple[List[str], bool]:
    """Recover only fully decoded strings from one truncated JSON array."""

    value = str(text or "").replace("\\'", "'")
    match = re.search(
        rf'"{re.escape(key)}"\s*:\s*\[',
        value,
    )
    if match is None:
        raise ValueError(f"LLM response is missing JSON field {key!r}")
    decoder = json.JSONDecoder()
    index = match.end()
    output: List[str] = []
    closed = False
    while True:
        while index < len(value) and value[index].isspace():
            index += 1
        if index >= len(value):
            break
        if value[index] == "]":
            closed = True
            break
        if value[index] == ",":
            index += 1
            continue
        if implicit_stop_field is not None and re.match(
            rf'"{re.escape(implicit_stop_field)}"\s*:',
            value[index:],
        ):
            # The model omitted only the array bracket before the next known
            # object field. The already-decoded strings remain unambiguous.
            closed = True
            break
        try:
            item, end = decoder.raw_decode(value, index)
        except json.JSONDecodeError:
            # The generation ended inside this value. Never guess its suffix.
            break
        if not isinstance(item, str):
            raise ValueError(f"LLM JSON field {key!r} must contain strings")
        output.append(item)
        index = end
        while index < len(value) and value[index].isspace():
            index += 1
        if index < len(value) and value[index] == "]":
            closed = True
            break
        if index < len(value) and value[index] == ",":
            index += 1
            continue
        break
    if require_closing_bracket and not closed:
        raise ValueError(f"LLM JSON field {key!r} is truncated")
    return output, closed


def _recover_truncated_paper_object(text: str) -> Mapping[str, Any]:
    """Salvage complete list items without inventing truncated content."""

    selected_topics, _ = _recover_json_string_array(
        text,
        "selected_topics",
        require_closing_bracket=True,
        implicit_stop_field="keyphrases",
    )
    keyphrases, _ = _recover_json_string_array(
        text,
        "keyphrases",
        require_closing_bracket=False,
    )
    if not selected_topics and not keyphrases:
        raise ValueError("truncated LLM response has no complete concepts")
    return {
        "selected_topics": selected_topics,
        "keyphrases": keyphrases,
    }


def paper_concept_prompt(
    title: str,
    abstract: str,
    candidates: Sequence[TopicCandidate],
) -> str:
    candidate_names = [normalize_concept(item.concept) for item in candidates]
    return f"""You are constructing the scientific concept index used by SemRank.

Given a paper and topic candidates from the official multi-label classifier:
1. Select only classifier topics that match the paper's core theme. Every
   selected topic must be copied exactly from CANDIDATE_TOPICS.
2. Extract a concise, complete list of fine-grained keyphrases that actually
   occur in the title or abstract. Do not invent facts.

Return JSON only:
{{"selected_topics": ["exact candidate"], "keyphrases": ["phrase"]}}

TITLE:
{title}

ABSTRACT:
{abstract}

CANDIDATE_TOPICS:
{json.dumps(candidate_names, ensure_ascii=False)}
"""


def query_concept_prompt(
    query: str,
    papers: Sequence[AuxiliaryPaper],
    candidate_topics: Sequence[Mapping[str, Any]],
    candidate_keyphrases: Sequence[Mapping[str, Any]],
) -> str:
    paper_rows = [
        {
            "rank": int(item.rank),
            "title": item.title,
            "abstract": item.abstract,
        }
        for item in papers
    ]
    return f"""You are selecting the original-query concept profile for SemRank.

Select concepts that accurately identify papers relevant to ORIGINAL_QUERY.
You may only copy exact concept strings from CANDIDATE_TOPICS or
CANDIDATE_KEYPHRASES. Do not generate synonyms or new concepts. Do not use any
subquery. Remove duplicates.

Return JSON only:
{{"selected_concepts": ["exact candidate concept"]}}

ORIGINAL_QUERY:
{query}

TOP_RETRIEVED_PAPERS:
{json.dumps(paper_rows, ensure_ascii=False)}

CANDIDATE_TOPICS_WITH_FREQUENCY:
{json.dumps(list(candidate_topics), ensure_ascii=False)}

CANDIDATE_KEYPHRASES_WITH_FREQUENCY:
{json.dumps(list(candidate_keyphrases), ensure_ascii=False)}
"""


class SemRankLLMClient:
    def __init__(
        self,
        model: str,
        *,
        is_local: bool = False,
        enable_thinking: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        workers: int = 8,
        call: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.model = str(model)
        self.is_local = bool(is_local)
        self.enable_thinking = bool(enable_thinking)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.workers = max(1, int(workers))
        provider = _resolve_provider(self.model, self.is_local)
        self.provider_base_url = str(provider.get("base_url") or "")
        self._client = None
        if call is None:
            self._client = OpenAI(
                api_key=provider["api_key"],
                base_url=provider["base_url"],
            )
            self._call = self._call_with_persistent_client
        else:
            self._call = call
        self._executor = (
            ThreadPoolExecutor(max_workers=self.workers)
            if self.workers > 1
            else None
        )
        self._lock = Lock()
        self._stats: Dict[str, int] = {
            "paper_concept_llm_calls": 0,
            "query_concept_llm_calls": 0,
            "llm_failures": 0,
            "llm_parse_failures": 0,
        }
        self.llm_id = (
            f"{self.model}|local={int(self.is_local)}"
            f"|thinking={int(self.enable_thinking)}"
            f"|temperature={self.temperature:g}|top_p={self.top_p:g}"
            f"|max_tokens={self.max_tokens}"
            f"|base_url={self.provider_base_url}"
        )

    def _call_with_persistent_client(
        self,
        prompt: str,
        model: str,
        gen_params: Mapping[str, Any],
        *,
        is_local: bool = False,
        enable_thinking: bool = False,
    ) -> Any:
        if self._client is None:
            return _call_llm(
                prompt,
                model,
                dict(gen_params),
                is_local=is_local,
                enable_thinking=enable_thinking,
            )
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(gen_params.get("max_tokens", 8192)),
            "temperature": float(gen_params.get("temperature", 0.0)),
            "top_p": float(gen_params.get("top_p", 1.0)),
            "stream": False,
        }
        if not is_local and model.lower().startswith("qwen3"):
            kwargs["extra_body"] = {
                "enable_thinking": bool(enable_thinking)
            }
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content or ""
        if enable_thinking:
            reasoning = getattr(
                choice.message, "reasoning_content", None
            )
            if reasoning:
                return reasoning, content
        return content

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def _invoke(self, prompt: str, kind: str) -> str:
        for attempt in range(3):
            with self._lock:
                self._stats[f"{kind}_concept_llm_calls"] += 1
            try:
                result = self._call(
                    prompt,
                    self.model,
                    {
                        "max_tokens": self.max_tokens,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                    },
                    is_local=self.is_local,
                    enable_thinking=self.enable_thinking,
                )
                if isinstance(result, tuple):
                    result = result[-1]
                if isinstance(result, Mapping):
                    return json.dumps(result, ensure_ascii=False)
                return str(result or "")
            except Exception:
                with self._lock:
                    self._stats["llm_failures"] += 1
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def refine_paper(
        self,
        title: str,
        abstract: str,
        candidates: Sequence[TopicCandidate],
    ) -> Tuple[List[str], List[str], str, Optional[str]]:
        prompt = paper_concept_prompt(title, abstract, candidates)
        raw = ""
        for _ in range(3):
            try:
                raw = self._invoke(prompt, "paper")
            except Exception as exc:
                return (
                    [],
                    [],
                    raw,
                    f"paper_concept_llm_call_failed:{type(exc).__name__}",
                )
            try:
                selected, phrases = self.parse_paper_response(
                    raw,
                    title,
                    abstract,
                    candidates,
                )
                return selected, phrases, raw, None
            except (ValueError, json.JSONDecodeError):
                with self._lock:
                    self._stats["llm_parse_failures"] += 1
        return [], [], raw, "paper_concept_llm_parse_failed"

    def parse_paper_response(
        self,
        raw: str,
        title: str,
        abstract: str,
        candidates: Sequence[TopicCandidate],
    ) -> Tuple[List[str], List[str]]:
        """Parse one saved response without making an LLM call."""

        try:
            value = _extract_json_object(raw)
        except (ValueError, json.JSONDecodeError):
            value = _recover_truncated_paper_object(raw)
        vocabulary = {
            normalize_concept(item.concept): normalize_concept(item.concept)
            for item in candidates
            if normalize_concept(item.concept)
        }
        selected = [
            vocabulary[normalized]
            for normalized in stable_unique(
                _list_field(value, "selected_topics")
            )
            if normalized in vocabulary
        ]
        paper_text = normalize_concept(f"{title} {abstract}")
        phrases = [
            phrase
            for phrase in stable_unique(_list_field(value, "keyphrases"))
            if phrase in paper_text
        ]
        return selected, phrases

    def refine_papers(
        self,
        requests: Sequence[
            Tuple[str, str, Sequence[TopicCandidate]]
        ],
    ) -> List[Tuple[List[str], List[str], str, Optional[str]]]:
        if not requests:
            return []
        if self.workers == 1 or len(requests) == 1:
            return [
                self.refine_paper(title, abstract, candidates)
                for title, abstract, candidates in requests
            ]
        results: List[
            Optional[Tuple[List[str], List[str], str, Optional[str]]]
        ] = [None] * len(requests)
        assert self._executor is not None
        futures = {
            self._executor.submit(
                self.refine_paper,
                title,
                abstract,
                candidates,
            ): index
            for index, (title, abstract, candidates) in enumerate(requests)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
        return [item for item in results if item is not None]

    def select_query_concepts(
        self,
        query: str,
        papers: Sequence[AuxiliaryPaper],
        candidate_topics: Sequence[Mapping[str, Any]],
        candidate_keyphrases: Sequence[Mapping[str, Any]],
    ) -> Tuple[List[str], str]:
        prompt = query_concept_prompt(
            query,
            papers,
            candidate_topics,
            candidate_keyphrases,
        )
        raw = ""
        value: Optional[Mapping[str, Any]] = None
        for _ in range(3):
            raw = self._invoke(prompt, "query")
            try:
                value = _extract_json_object(raw)
                break
            except (ValueError, json.JSONDecodeError):
                with self._lock:
                    self._stats["llm_parse_failures"] += 1
        if value is None:
            raise ValueError("query concept LLM returned invalid JSON")
        candidates = list(candidate_topics) + list(candidate_keyphrases)
        vocabulary = {
            normalize_concept(item.get("concept")): normalize_concept(
                item.get("concept")
            )
            for item in candidates
            if normalize_concept(item.get("concept"))
        }
        selected = [
            vocabulary[normalized]
            for normalized in stable_unique(
                _list_field(value, "selected_concepts")
            )
            if normalized in vocabulary
        ]
        return selected, raw

    def snapshot_stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)


__all__ = [
    "SEMRANK_PAPER_PROMPT_VERSION",
    "SEMRANK_QUERY_PROMPT_VERSION",
    "SemRankLLMClient",
    "paper_concept_prompt",
    "query_concept_prompt",
]
