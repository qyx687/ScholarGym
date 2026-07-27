"""Minimal compatible loader for the official SemRank topic classifier.

The architecture follows the public SemRank/TELEClass checkpoint interface.
No upstream source file or trained weight is vendored in this repository.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .models import TopicCandidate, normalize_concept


class TopicClassifier(Protocol):
    classifier_id: str
    label_space_id: str

    def predict_batch(
        self, texts: Sequence[str], *, top_k: int
    ) -> List[List[TopicCandidate]]:
        ...


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_labels(
    path: str | Path,
) -> Tuple[List[str], List[str], str]:
    label_path = Path(path)
    if not label_path.is_file():
        raise FileNotFoundError(
            f"official SemRank topic labels not found: {label_path}"
        )
    label_ids: List[str] = []
    names: List[str] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split("\t")
        if len(parts) < 2:
            raise ValueError(
                f"invalid SemRank label row at {label_path}:{line_number}"
            )
        label_id = str(parts[0]).strip()
        name = normalize_concept(parts[1])
        if not label_id or not name:
            raise ValueError(
                f"empty SemRank label at {label_path}:{line_number}"
            )
        label_ids.append(label_id)
        names.append(name)
    if not names:
        raise ValueError(f"SemRank label space is empty: {label_path}")
    return label_ids, names, _file_sha256(label_path)


class OfficialSemRankTopicClassifier:
    """SPECTER2 + label bilinear matching used by the official checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        labels_path: str | Path,
        *,
        encoder_name: str = "allenai/specter2_base",
        encoder_revision: str = (
            "3447645e1def9117997203454fa4495937bfbd83"
        ),
        device: str = "cuda:0",
        batch_size: int = 64,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.labels_path = Path(labels_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "official SemRank topic classifier checkpoint not found: "
                f"{self.checkpoint_path}"
            )
        self.label_ids, self.label_names, labels_sha = load_official_labels(
            self.labels_path
        )
        self.checkpoint_sha256 = _file_sha256(self.checkpoint_path)
        self.label_space_id = f"semrank_labels_sha256:{labels_sha}"
        self.encoder_name = str(encoder_name)
        self.encoder_revision = str(encoder_revision)
        self.device = str(device)
        self.batch_size = max(1, int(batch_size))
        self.classifier_id = (
            "official_semrank_specter2_lbm_v1"
            f"|encoder={self.encoder_name}@{self.encoder_revision}"
            "|pooler_output|padding=max_length|maxlen=512|f32"
            f"|checkpoint_sha256={self.checkpoint_sha256}"
            f"|labels_sha256={labels_sha}"
        )
        self._tokenizer = tokenizer
        self._model = model
        self._stats: Counter[str] = Counter()

    def _load(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "official SemRank classifier requires torch and transformers"
            ) from exc
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"SemRank topic classifier requested {self.device}, but CUDA "
                "is unavailable"
            )
        try:
            state = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:  # Older supported torch releases.
            state = torch.load(self.checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict) or "label_embedding_weights" not in state:
            raise ValueError(
                "official SemRank checkpoint is missing "
                "'label_embedding_weights'"
            )
        label_embeddings = state["label_embedding_weights"]
        if int(label_embeddings.shape[0]) != len(self.label_names):
            raise ValueError(
                "SemRank classifier/label count mismatch: checkpoint has "
                f"{int(label_embeddings.shape[0])}, labels file has "
                f"{len(self.label_names)}"
            )
        embedding_dim = int(label_embeddings.shape[1])

        class BilinearLabelModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.doc_encoder = AutoModel.from_pretrained(
                    self_outer.encoder_name,
                    revision=self_outer.encoder_revision,
                )
                self.label_embedding_weights = nn.Parameter(
                    torch.empty_like(label_embeddings),
                    requires_grad=False,
                )
                self.interaction_weight = nn.Parameter(
                    torch.empty(
                        self.doc_encoder.config.hidden_size,
                        embedding_dim,
                    ),
                    requires_grad=False,
                )

            def forward(self, input_ids: Any, attention_mask: Any) -> Any:
                output = self.doc_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                pooled = getattr(output, "pooler_output", None)
                if pooled is None:
                    raise RuntimeError(
                        "SPECTER2 classifier backbone has no pooler_output; "
                        "the official checkpoint contract cannot be applied"
                    )
                return (
                    pooled
                    @ self.interaction_weight
                    @ self.label_embedding_weights.T
                )

        self_outer = self
        model_instance = BilinearLabelModel()
        remapped = dict(state)
        if "interaction.weight" in remapped:
            remapped["interaction_weight"] = remapped.pop("interaction.weight")
        unexpected_interaction = [
            key for key in remapped if key.startswith("interaction.")
        ]
        if unexpected_interaction:
            raise ValueError(
                "unsupported official SemRank interaction state: "
                f"{unexpected_interaction}"
            )
        missing, unexpected = model_instance.load_state_dict(
            remapped, strict=False
        )
        allowed_missing = set()
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(
                "official SemRank checkpoint does not match the expected "
                f"architecture (missing={list(missing)}, "
                f"unexpected={list(unexpected)})"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.encoder_name,
            revision=self.encoder_revision,
        )
        self._model = model_instance.to(self.device)
        self._model.eval()

    def predict_batch(
        self, texts: Sequence[str], *, top_k: int
    ) -> List[List[TopicCandidate]]:
        values = [str(text or "") for text in texts]
        if not values:
            return []
        self._load()
        import torch

        requested = min(max(1, int(top_k)), len(self.label_names))
        output: List[List[TopicCandidate]] = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                self._stats["forward_batches"] += 1
                self._stats["papers_scored"] += len(batch)
                tokens = self._tokenizer(
                    batch,
                    add_special_tokens=True,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                scores = self._model(
                    tokens["input_ids"].to(self.device),
                    tokens["attention_mask"].to(self.device),
                )
                for row in scores.float().cpu().numpy():
                    order = sorted(
                        range(len(row)),
                        key=lambda index: (
                            -float(row[index])
                            if np.isfinite(row[index])
                            else float("inf"),
                            self.label_names[index],
                            self.label_ids[index],
                        ),
                    )[:requested]
                    output.append(
                        [
                            TopicCandidate(
                                concept=self.label_names[index],
                                score=float(row[index]),
                                label_id=self.label_ids[index],
                            )
                            for index in order
                        ]
                    )
        return output

    def snapshot_stats(self) -> Dict[str, int]:
        return {key: int(value) for key, value in self._stats.items()}
