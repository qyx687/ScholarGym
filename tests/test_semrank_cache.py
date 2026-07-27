import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from semrank import (
    CachedConceptEncoder,
    EmbeddingProviderConceptEncoder,
    PaperConceptProfile,
    QueryConceptProfile,
    SemRankCache,
    SemRankConfig,
    paper_text_profile_identity,
    query_text_profile_identity,
    stable_hash,
)


class CountingEncoder:
    encoder_id = "fixed-encoder-v1"

    def __init__(self):
        self.calls = []

    def encode(self, concepts):
        self.calls.append(list(concepts))
        vectors = {
            "alpha": [1.0, 0.0],
            "beta": [0.0, 1.0],
        }
        return np.asarray([vectors[value] for value in concepts], dtype=np.float32)


class FakeDenseEmbeddingProvider:
    backend = "ollama"
    model = "qwen3-embedding:0.6b"
    base_url = "http://127.0.0.1:11434"
    batch_size = 64

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)


def test_qwen_concept_adapter_reuses_and_identifies_dense_provider():
    provider = FakeDenseEmbeddingProvider()
    encoder = EmbeddingProviderConceptEncoder(provider)

    vectors = encoder.encode([" Mixed   Case ", "BETA"])

    assert provider.calls == [["mixed case", "beta"]]
    assert np.allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])
    assert "model=qwen3-embedding:0.6b" in encoder.encoder_id
    assert "backend=ollama" in encoder.encoder_id
    assert "batch=64" in encoder.encoder_id


def test_main_config_decouples_concept_and_classifier_encoders():
    config = SemRankConfig()

    assert config.concept_encoder_backend == "ollama"
    assert config.concept_encoder == "qwen3-embedding:0.6b"
    assert config.topic_classifier_encoder == "allenai/specter2_base"
    assert config.topic_classifier_encoder_revision


def test_text_profile_keys_do_not_change_with_vector_encoder():
    paper_specter = {
        "paper_arxiv_id": "2001.00001",
        "title_abstract_hash": "content",
        "classifier_id": "official-classifier",
        "concept_encoder": "specter2-vector-namespace",
    }
    paper_qwen = {
        **paper_specter,
        "concept_encoder": "qwen3-vector-namespace",
    }
    query_specter = {
        "query_id": "q1",
        "query_llm": "fixed-llm",
        "paper_concept_encoder": "specter2-vector-namespace",
        "concept_encoder": "specter2-vector-namespace",
    }
    query_qwen = {
        **query_specter,
        "paper_concept_encoder": "qwen3-vector-namespace",
        "concept_encoder": "qwen3-vector-namespace",
    }

    assert stable_hash(paper_text_profile_identity(paper_specter)) == (
        stable_hash(paper_text_profile_identity(paper_qwen))
    )
    assert stable_hash(query_text_profile_identity(query_specter)) == (
        stable_hash(query_text_profile_identity(query_qwen))
    )


def test_concept_embedding_is_persistent_and_identity_safe(tmp_path):
    cache_path = tmp_path / "semrank.sqlite3"
    first_backend = CountingEncoder()
    with SemRankCache(cache_path) as cache:
        first = CachedConceptEncoder(first_backend, cache)
        matrix = first.encode(["Alpha", "beta", "alpha"])
        assert matrix.shape == (3, 2)
        assert first_backend.calls == [["alpha", "beta"]]
        repeated_in_process = first.encode(["beta", "alpha"])
        assert np.allclose(
            repeated_in_process,
            np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        )
        assert first.snapshot_stats()["memory_hits"] == 2
        assert first.snapshot_stats()["memory_cache_items"] == 2

    second_backend = CountingEncoder()
    with SemRankCache(cache_path) as cache:
        second = CachedConceptEncoder(second_backend, cache)
        repeated = second.encode(["alpha", "beta"])
        assert second_backend.calls == []
        assert np.allclose(repeated, np.eye(2, dtype=np.float32))
        assert cache.table_counts()["concept_embeddings"] == 2


def test_qwen_and_specter_vectors_use_distinct_cache_namespaces(tmp_path):
    class AlternateEncoder(CountingEncoder):
        encoder_id = "different-specter2-namespace"

        def encode(self, concepts):
            self.calls.append(list(concepts))
            return np.asarray(
                [[0.5, 0.5] for _ in concepts],
                dtype=np.float32,
            )

    cache_path = tmp_path / "semrank.sqlite3"
    qwen_backend = CountingEncoder()
    specter_backend = AlternateEncoder()
    with SemRankCache(cache_path) as cache:
        qwen = CachedConceptEncoder(qwen_backend, cache)
        specter = CachedConceptEncoder(specter_backend, cache)
        qwen_vectors = qwen.encode(["alpha", "beta"])
        specter_vectors = specter.encode(["alpha", "beta"])

        assert qwen_backend.calls == [["alpha", "beta"]]
        assert specter_backend.calls == [["alpha", "beta"]]
        assert not np.allclose(qwen_vectors, specter_vectors)
        assert cache.table_counts()["concept_embeddings"] == 4


def test_profile_cache_round_trip_and_changed_key_misses(tmp_path):
    paper = PaperConceptProfile(
        paper_arxiv_id="2001.00001",
        profile_id="paper-key-v1",
        title_abstract_hash="content",
        candidate_topics=[{"concept": "retrieval", "score": 1.0}],
        selected_topics=["retrieval"],
        keyphrases=["dense retrieval"],
        concepts=["retrieval", "dense retrieval"],
        status="ok",
        fallback_reason=None,
        pipeline_version="pipeline",
        classifier_id="classifier",
        label_space_id="labels",
        llm_model="llm",
        prompt_version="paper-prompt",
        concept_encoder="encoder",
    )
    query = QueryConceptProfile(
        query_id="q1",
        query_profile_id="query-key-v1",
        query="retrieval query",
        date_cutoff="2020-01",
        initial_retrieval_paper_ids=["2001.00001"],
        feedback_paper_ids=["2001.00001"],
        candidate_topics=[{"concept": "retrieval", "frequency": 1}],
        candidate_keyphrases=[],
        selected_concepts=["retrieval"],
        selection_status="ok",
        fallback_reason=None,
        prompt_version="query-prompt",
        llm_model="llm",
        concept_encoder="encoder",
        topic_pipeline_version="pipeline",
        retriever_identity="retriever",
        initial_retrieval_target=1,
        initial_retrieval_count=1,
        initial_retrieval_complete=True,
    )
    with SemRankCache(tmp_path / "profiles.sqlite3") as cache:
        cache.put_paper_profile("paper-key-v1", {"version": 1}, paper)
        cache.put_query_profile("query-key-v1", {"version": 1}, query)

        assert cache.get_paper_profile("paper-key-v1").cache_hit is True
        assert cache.get_query_profile("query-key-v1").cache_hit is True
        assert cache.get_paper_profile("paper-key-v2") is None
        assert cache.get_query_profile("query-key-v2") is None
        assert cache.paper_profile_status_counts() == {"ok": 1}
