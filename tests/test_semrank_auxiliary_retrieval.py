import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from rag import CitationRAGSystem


class FakeDocument:
    def __init__(self, metadata):
        self.metadata = metadata


class ProgressiveVectorStore:
    def __init__(self):
        self.calls = []

    def similarity_search_with_score(self, *, query, k):
        self.calls.append(k)
        rows = []
        for index in range(k):
            if index < 2100:
                date = "2099-01"
            elif index == 2100:
                date = ""
            elif index == 2101:
                date = "unknown"
            else:
                date = "2019-01"
            rows.append(
                (
                    FakeDocument(
                        {
                            "arxiv_id": f"{index:04d}.00001",
                            "date": date,
                            "title": f"Paper {index}",
                            "abstract": "Abstract",
                        }
                    ),
                    float(k - index),
                )
            )
        return rows


def test_auxiliary_vector_retrieval_overfetches_until_top_m_after_date_filter():
    rag = CitationRAGSystem.__new__(CitationRAGSystem)
    rag.qdrant_vector_store = ProgressiveVectorStore()

    rows = rag.search_citations_vector_date_valid(
        "original query",
        top_k=1000,
        before_date="2020-01",
    )

    assert rag.qdrant_vector_store.calls == [2000, 4000]
    assert len(rows) == 1000
    assert all(item[2]["date"] <= "2020-01" for item in rows)
    assert all(item[2]["date"] for item in rows)
    assert rows[0][0] == "2102.00001"
