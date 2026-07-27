import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from graph_methods import PerSubqueryProcessor
from semrank import (
    PaperConceptProfile,
    QueryConceptProfile,
    SEMRANK_FORMULA_ID,
    SemRankConfig,
    SemRankQSQReranker,
)


class FixedDenseEncoder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        output = []
        for text in texts:
            lowered = text.lower()
            if text == "original query" or "seed one" in lowered:
                output.append([1.0, 0.0])
            elif text == "current subquery" or "seed two" in lowered:
                output.append([0.0, 1.0])
            else:
                output.append([2 ** -0.5, 2 ** -0.5])
        return np.asarray(output, dtype=np.float32)


class FixedConceptEncoder:
    encoder_id = "fixed-concept-encoder"

    def encode(self, concepts):
        mapping = {
            "query concept": [1.0, 0.0],
            "paper one": [1.0, 0.0],
            "paper two": [0.0, 1.0],
            "paper three": [2 ** -0.5, 2 ** -0.5],
        }
        return np.asarray([mapping[value] for value in concepts], dtype=np.float32)

    def snapshot_stats(self):
        return {}


class FakeGraph:
    def expand(self, seed_ids, method, limit):
        return [
            {
                "seed_arxiv_id": "2001.00001",
                "expanded_arxiv_id": "2001.00003",
                "edge_type": "citation",
                "edge_rank": 1,
                "intents": ["methodology"],
            },
            {
                "seed_arxiv_id": "2001.00002",
                "expanded_arxiv_id": "2099.00004",
                "edge_type": "reference",
                "edge_rank": 1,
                "intents": ["result"],
            },
        ]

    def snapshot_stats(self):
        return {}


class FakeQueryProfiles:
    def snapshot_stats(self):
        return {}


class FakePaperConcepts:
    def __init__(self, profiles):
        self.profiles = profiles
        self.requested = []

    def get_or_build(self, metadata):
        self.requested.append(list(metadata))
        return {paper_id: self.profiles[paper_id] for paper_id in metadata}

    def snapshot_stats(self):
        return {}


def _paper_profile(paper_id, concepts):
    return PaperConceptProfile(
        paper_arxiv_id=paper_id,
        profile_id=f"profile-{paper_id}",
        title_abstract_hash=f"hash-{paper_id}",
        candidate_topics=[],
        selected_topics=concepts,
        keyphrases=[],
        concepts=concepts,
        status="ok",
        fallback_reason=None,
        pipeline_version="pipeline",
        classifier_id="classifier",
        label_space_id="labels",
        llm_model="llm",
        prompt_version="prompt",
        concept_encoder="fixed-concept-encoder",
    )


def _query_profile(concepts):
    return QueryConceptProfile(
        query_id="q1",
        query_profile_id="query-profile-q1",
        query="original query",
        date_cutoff="2002-01",
        initial_retrieval_paper_ids=["1999.99999"],
        feedback_paper_ids=["1999.99999"],
        candidate_topics=[],
        candidate_keyphrases=[],
        selected_concepts=concepts,
        selection_status="ok" if concepts else "fallback",
        fallback_reason=None if concepts else "empty",
        prompt_version="prompt",
        llm_model="llm",
        concept_encoder="fixed-concept-encoder",
        topic_pipeline_version="pipeline",
        retriever_identity="auxiliary",
        initial_retrieval_target=1,
        initial_retrieval_count=1,
        initial_retrieval_complete=True,
    )


def _event():
    return {
        "schema_version": "1.0",
        "query_id": "q1",
        "query": "original query",
        "query_date": "2002-01",
        "subquery": "current subquery",
        "subquery_before_date": "2002-01",
        "iteration_idx": 1,
        "subquery_id": 1,
        "retrieval_event_id": "q1:event1",
        "selector_top_k": 2,
        "seed_papers": [
            {
                "paper_arxiv_id": "2001.00001",
                "observed_retrieval_score": 2.0,
                "observed_retrieval_rank": 1,
            },
            {
                "paper_arxiv_id": "2001.00002",
                "observed_retrieval_score": 1.0,
                "observed_retrieval_rank": 2,
            },
        ],
    }


def _paper_db():
    return {
        "2001.00001": {
            "title": "Seed one",
            "abstract": "Original representation.",
            "date": "2001-01",
        },
        "2001.00002": {
            "title": "Seed two",
            "abstract": "Subquery representation.",
            "date": "2001-02",
        },
        "2001.00003": {
            "title": "Expanded",
            "abstract": "Balanced representation.",
            "date": "2001-03",
        },
        "2099.00004": {
            "title": "Future",
            "abstract": "Must be excluded.",
            "date": "2099-01",
        },
        "1999.99999": {
            "title": "Auxiliary only",
            "abstract": "Must not enter the Agent pool.",
            "date": "1999-01",
        },
    }


def _semrank_processor(query_concepts=("query concept",)):
    profiles = {
        "2001.00001": _paper_profile("2001.00001", ["paper one"]),
        "2001.00002": _paper_profile("2001.00002", ["paper two"]),
        "2001.00003": _paper_profile("2001.00003", ["paper three"]),
    }
    paper_service = FakePaperConcepts(profiles)
    reranker = SemRankQSQReranker(
        SemRankConfig(),
        FakeQueryProfiles(),
        paper_service,
        FixedConceptEncoder(),
    )
    reranker.active_query_profile = _query_profile(list(query_concepts))
    processor = PerSubqueryProcessor(
        _paper_db(),
        FakeGraph(),
        scoring_backend="embedding",
        embedding_provider=FixedDenseEncoder(),
        rerank_method="semrank_qsq",
        semrank_reranker=reranker,
    )
    return processor, paper_service


def test_semrank_only_reorders_the_shared_date_valid_candidate_pool():
    static = PerSubqueryProcessor(
        _paper_db(),
        FakeGraph(),
        scoring_backend="embedding",
        embedding_provider=FixedDenseEncoder(),
        rerank_method="static",
    ).process(_event())
    processor, paper_service = _semrank_processor()
    semrank = processor.process(_event())

    static_ids = {row["paper_arxiv_id"] for row in static["rows"]}
    semrank_ids = {row["paper_arxiv_id"] for row in semrank["rows"]}
    assert static_ids == semrank_ids == {
        "2001.00001",
        "2001.00002",
        "2001.00003",
    }
    assert "2099.00004" not in semrank_ids
    assert "1999.99999" not in semrank_ids
    assert len(paper_service.requested) == 1
    assert set(paper_service.requested[0]) == semrank_ids
    assert len(semrank["top_rows"]) == len(static["top_rows"]) == 2


def test_semrank_rows_are_auditable_and_ignore_graph_features():
    processor, _ = _semrank_processor()
    result = processor.process(_event())
    rows = result["rows"]
    assert [row["rerank_rank"] for row in rows] == [1, 2, 3]
    for row in rows:
        assert row["rerank_method"] == "semrank_qsq"
        assert row["rerank_formula_id"] == SEMRANK_FORMULA_ID
        assert row["rerank_score"] == (
            row["semrank_base_score_z"]
            + row["semrank_concept_score_z"]
        )
        assert row["semrank_fallback_used"] is False
    assert result["semrank_event_profile"]["candidate_ids_preserved"] is True
    pool_signature = result["semrank_event_profile"][
        "candidate_pool_signature"
    ]
    assert pool_signature
    assert {
        row["candidate_pool_signature"] for row in rows
    } == {pool_signature}

    copied = [dict(row) for row in rows]
    copied[0]["intent_score"] = 1000.0
    copied[0]["path_count_normalized"] = 1000.0
    # The recorded formula remains entirely reconstructible without either.
    assert copied[0]["rerank_score"] == rows[0]["rerank_score"]


def test_empty_query_concepts_fall_back_to_z_base_only():
    processor, _ = _semrank_processor(query_concepts=())
    result = processor.process(_event())
    for row in result["rows"]:
        assert row["semrank_fallback_used"] is True
        assert row["rerank_score"] == row["semrank_base_score_z"]
        assert row["semrank_concept_score_z"] == 0.0
