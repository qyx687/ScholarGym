"""Concept encoding backends with persistent, identity-safe caching."""

from __future__ import annotations

from collections import Counter, OrderedDict
from threading import Lock
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .cache import SemRankCache
from .models import (
    SEMRANK_CONCEPT_NORMALIZATION_VERSION,
    normalize_concept,
    stable_hash,
)


class ConceptEncoder(Protocol):
    encoder_id: str

    def encode(self, concepts: Sequence[str]) -> np.ndarray:
        ...


class EmbeddingProviderConceptEncoder:
    """Use the experiment's dense embedding provider for SemRank concepts."""

    def __init__(self, provider: Any) -> None:
        required = ("backend", "model", "base_url", "embed")
        missing = [name for name in required if not hasattr(provider, name)]
        if missing:
            raise TypeError(
                "SemRank embedding provider is missing: "
                + ", ".join(missing)
            )
        self.provider = provider
        self.backend = str(provider.backend)
        self.model_name = str(provider.model)
        self.base_url = str(provider.base_url).rstrip("/")
        self.batch_size = max(1, int(getattr(provider, "batch_size", 1)))
        if self.backend not in {"ollama", "api"}:
            raise ValueError(
                "SemRank provider concept encoder requires ollama or api"
            )
        if not self.model_name or not self.base_url:
            raise ValueError(
                "SemRank provider concept encoder requires model and base URL"
            )
        self.encoder_id = (
            "dense_embedding_provider"
            f"|backend={self.backend}"
            f"|model={self.model_name}"
            f"|base_url={self.base_url}"
            f"|batch={self.batch_size}"
            "|normalized_concept_input|f32|l2"
            f"|{SEMRANK_CONCEPT_NORMALIZATION_VERSION}"
        )

    def encode(self, concepts: Sequence[str]) -> np.ndarray:
        normalized = [normalize_concept(value) for value in concepts]
        if not normalized:
            return np.zeros((0, 0), dtype=np.float32)
        if any(not value for value in normalized):
            raise ValueError("SemRank concept encoder received an empty concept")
        encoded = np.asarray(self.provider.embed(normalized), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != len(normalized):
            raise RuntimeError(
                "SemRank dense embedding provider returned an unexpected shape"
            )
        norms = np.linalg.norm(encoded, axis=1, keepdims=True)
        if bool((norms <= 1e-12).any()):
            raise RuntimeError(
                "SemRank dense embedding provider returned a zero vector"
            )
        return (encoded / norms).astype(np.float32)


class Specter2MeanPoolEncoder:
    """Official SPECTER2 backbone with task-required non-special mean pooling."""

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        *,
        revision: str = "3447645e1def9117997203454fa4495937bfbd83",
        device: str = "cuda:0",
        batch_size: int = 32,
        max_length: int = 512,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model_name = str(model_name)
        self.revision = str(revision)
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self.encoder_id = (
            f"{self.model_name}@{self.revision}|mean_non_special|f32|l2|maxlen="
            f"{self.max_length}|{SEMRANK_CONCEPT_NORMALIZATION_VERSION}"
        )
        self._tokenizer = tokenizer
        self._model = model

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "SemRank concept encoding requires torch and transformers"
            ) from exc
        if not torch.cuda.is_available() and self.device.startswith("cuda"):
            raise RuntimeError(
                f"SemRank concept encoder requested {self.device}, but CUDA "
                "is unavailable"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            revision=self.revision,
        )
        self._model.to(self.device)
        self._model.eval()

    def encode(self, concepts: Sequence[str]) -> np.ndarray:
        normalized = [normalize_concept(value) for value in concepts]
        if not normalized:
            return np.zeros((0, 0), dtype=np.float32)
        if any(not value for value in normalized):
            raise ValueError("SemRank concept encoder received an empty concept")
        self._load()
        import torch

        batches: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(normalized), self.batch_size):
                batch = normalized[start : start + self.batch_size]
                tokens = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                )
                special = tokens.pop("special_tokens_mask").to(self.device)
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                output = self._model(**tokens)
                hidden = output.last_hidden_state
                mask = tokens["attention_mask"].bool() & ~special.bool()
                counts = mask.sum(dim=1, keepdim=True)
                if bool((counts <= 0).any()):
                    raise RuntimeError(
                        "SPECTER2 produced no non-special tokens for a concept"
                    )
                pooled = (
                    hidden * mask.unsqueeze(-1).to(hidden.dtype)
                ).sum(dim=1) / counts.to(hidden.dtype)
                pooled = torch.nn.functional.normalize(pooled.float(), dim=1)
                batches.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(batches, axis=0)


class CachedConceptEncoder:
    """Cache each normalized concept once for a specific encoder identity.

    The bounded in-process LRU avoids repeatedly decoding the same persistent
    vectors for every retrieval event.  It is only a read-through performance
    layer: persistent keys, encoder namespaces, stored values, and returned
    arrays are unchanged.
    """

    def __init__(
        self,
        backend: ConceptEncoder,
        cache: SemRankCache,
        *,
        memory_cache_max_items: int = 20_000,
    ) -> None:
        self.backend = backend
        self.cache = cache
        self.encoder_id = str(backend.encoder_id)
        self.memory_cache_max_items = max(
            0, int(memory_cache_max_items)
        )
        self._memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._memory_lock = Lock()
        self._stats: Counter[str] = Counter()

    def _key(self, concept: str) -> str:
        return stable_hash(
            {
                "encoder_id": self.encoder_id,
                "concept": concept,
                "normalization": SEMRANK_CONCEPT_NORMALIZATION_VERSION,
            }
        )

    def _get_memory(self, concept: str) -> Optional[np.ndarray]:
        if self.memory_cache_max_items <= 0:
            return None
        with self._memory_lock:
            vector = self._memory_cache.pop(concept, None)
            if vector is None:
                return None
            self._memory_cache[concept] = vector
            self._stats["memory_hits"] += 1
            return vector

    def _put_memory(self, concept: str, vector: np.ndarray) -> None:
        if self.memory_cache_max_items <= 0:
            return
        stored = np.asarray(vector, dtype=np.float32).copy()
        with self._memory_lock:
            self._memory_cache.pop(concept, None)
            self._memory_cache[concept] = stored
            while len(self._memory_cache) > self.memory_cache_max_items:
                self._memory_cache.popitem(last=False)
                self._stats["memory_evictions"] += 1

    def encode(self, concepts: Sequence[str]) -> np.ndarray:
        normalized = [normalize_concept(value) for value in concepts]
        if not normalized:
            return np.zeros((0, 0), dtype=np.float32)
        if any(not value for value in normalized):
            raise ValueError("SemRank concept encoder received an empty concept")

        vectors: Dict[str, np.ndarray] = {}
        missing: List[str] = []
        for concept in dict.fromkeys(normalized):
            memory = self._get_memory(concept)
            if memory is not None:
                vectors[concept] = memory
                self._stats["hits"] += 1
                continue
            key = self._key(concept)
            cached = self.cache.get_embedding(key, encoder_id=self.encoder_id)
            if cached is None:
                missing.append(concept)
                self._stats["misses"] += 1
            else:
                vectors[concept] = cached
                self._put_memory(concept, cached)
                self._stats["hits"] += 1

        if missing:
            encoded = np.asarray(self.backend.encode(missing), dtype=np.float32)
            if encoded.ndim != 2 or encoded.shape[0] != len(missing):
                raise RuntimeError(
                    "SemRank concept encoder returned an unexpected shape"
                )
            norms = np.linalg.norm(encoded, axis=1, keepdims=True)
            encoded = encoded / np.maximum(norms, 1e-12)
            cache_records = []
            for concept, vector in zip(missing, encoded):
                vector = np.asarray(vector, dtype=np.float32)
                vectors[concept] = vector
                self._put_memory(concept, vector)
                cache_records.append(
                    (
                        self._key(concept),
                        self.encoder_id,
                        concept,
                        vector,
                    )
                )
                self._stats["encoded"] += 1
            self.cache.put_embeddings(cache_records)

        result = np.stack([vectors[concept] for concept in normalized]).astype(
            np.float32
        )
        dimensions = {int(vector.size) for vector in result}
        if len(dimensions) != 1:
            raise RuntimeError("SemRank concept cache contains mixed dimensions")
        return result

    def snapshot_stats(self) -> Dict[str, int]:
        value = {
            key: int(count) for key, count in self._stats.items()
        }
        with self._memory_lock:
            value["memory_cache_items"] = len(self._memory_cache)
        value["memory_cache_max_items"] = self.memory_cache_max_items
        return value
