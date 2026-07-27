"""Strict date-valid auxiliary retrieval used only to construct C(q)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .models import AuxiliaryPaper, stable_hash


class ScholarGymDateValidRetriever:
    def __init__(
        self,
        rag_system: Any,
        *,
        qdrant_url: str,
        qdrant_collection: str,
        embedding_model: str,
        paper_db: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.rag_system = rag_system
        self.paper_db = paper_db
        self.identity_record = {
            "implementation": "qdrant_progressive_overfetch_strict_date_v1",
            "qdrant_url": str(qdrant_url or ""),
            "qdrant_collection": str(qdrant_collection),
            "embedding_model": str(embedding_model),
        }
        self.retriever_identity = stable_hash(self.identity_record)

    def retrieve_date_valid(
        self,
        query: str,
        *,
        top_k: int,
        before_date: str,
    ) -> Sequence[AuxiliaryPaper]:
        rows = self.rag_system.search_citations_vector_date_valid(
            query,
            top_k=int(top_k),
            before_date=str(before_date or ""),
        )
        output = []
        for rank, (paper_id, score, qdrant_metadata) in enumerate(
            rows, start=1
        ):
            metadata = self.paper_db.get(str(paper_id)) or qdrant_metadata or {}
            output.append(
                AuxiliaryPaper(
                    paper_arxiv_id=str(paper_id),
                    score=float(score),
                    title=str(metadata.get("title") or ""),
                    abstract=str(metadata.get("abstract") or ""),
                    date=str(metadata.get("date") or ""),
                    rank=rank,
                )
            )
        return output
