import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from semrank import (
    AuxiliaryPaper,
    PaperConceptProfile,
    PaperConceptService,
    QueryProfileBuilder,
    SemRankCache,
    SemRankConfig,
    SemRankLLMClient,
    TopicCandidate,
)
import semrank.prompts as prompt_module
import semrank.paper_concepts as paper_concepts_module
import semrank.query_profile as query_profile_module


class FixedEncoder:
    def __init__(self, encoder_id="fixed-concepts-v1"):
        self.encoder_id = encoder_id
        self.calls = []

    def encode(self, concepts):
        self.calls.append(list(concepts))
        return np.asarray(
            [[1.0, float(index)] for index, _ in enumerate(concepts)],
            dtype=np.float32,
        )

    def snapshot_stats(self):
        return {"calls": len(self.calls)}


class FailingEncoder(FixedEncoder):
    def encode(self, concepts):
        self.calls.append(list(concepts))
        raise ConnectionError("embedding service unavailable")


class FixedRetriever:
    retriever_identity = "retriever-v1"

    def __init__(self, papers):
        self.papers = list(papers)
        self.calls = 0

    def retrieve_date_valid(self, query, *, top_k, before_date):
        self.calls += 1
        return self.papers[:top_k]


class FixedPaperProfiles:
    pipeline_version = "paper-pipeline-v1"

    def __init__(self, profiles, *, classifier_id="classifier-v1"):
        self.profiles = profiles
        self.calls = 0
        self.classifier = SimpleNamespace(
            classifier_id=classifier_id,
            label_space_id="labels-v1",
        )
        self.llm = SimpleNamespace(llm_id="paper-llm-v1")
        self.encoder = SimpleNamespace(encoder_id="fixed-concepts-v1")

    def get_or_build(self, metadata):
        self.calls += 1
        return {paper_id: self.profiles[paper_id] for paper_id in metadata}


class FixedQueryLLM:
    model = "fake-query-llm"
    llm_id = "fake-query-llm|temperature=0"

    def __init__(self):
        self.calls = 0

    def select_query_concepts(
        self,
        query,
        papers,
        candidate_topics,
        candidate_keyphrases,
    ):
        self.calls += 1
        vocabulary = candidate_topics + candidate_keyphrases
        return [vocabulary[0]["concept"]], '{"selected_concepts":[]}'


def _paper_profile(paper_id, topics, phrases):
    concepts = list(dict.fromkeys(list(topics) + list(phrases)))
    return PaperConceptProfile(
        paper_arxiv_id=paper_id,
        profile_id=f"profile-{paper_id}",
        title_abstract_hash=f"hash-{paper_id}",
        candidate_topics=[],
        selected_topics=topics,
        keyphrases=phrases,
        concepts=concepts,
        status="ok",
        fallback_reason=None,
        pipeline_version="paper-pipeline-v1",
        classifier_id="classifier-v1",
        label_space_id="labels-v1",
        llm_model="paper-llm",
        prompt_version="paper-prompt",
        concept_encoder="fixed-concepts-v1",
    )


def _auxiliary_papers():
    return [
        AuxiliaryPaper(
            paper_arxiv_id=f"200{i}.00001",
            score=float(4 - i),
            title=f"Paper {i}",
            abstract=f"Abstract {i}",
            date=f"200{i}-01",
            rank=i,
        )
        for i in range(1, 4)
    ]


def test_one_query_profile_is_reused_across_all_events_and_resume(tmp_path):
    config = SemRankConfig(
        initial_top_m=3,
        feedback_top_n=2,
        prompt_top_papers=2,
        candidate_topic_k=2,
        candidate_phrase_k=2,
        cache_path=str(tmp_path / "semrank.sqlite3"),
    )
    profiles = {
        "2001.00001": _paper_profile(
            "2001.00001", ["retrieval"], ["dense search"]
        ),
        "2002.00001": _paper_profile(
            "2002.00001", ["retrieval", "ranking"], ["reranking"]
        ),
    }
    retriever = FixedRetriever(_auxiliary_papers())
    paper_service = FixedPaperProfiles(profiles)
    llm = FixedQueryLLM()
    encoder = FixedEncoder()

    with SemRankCache(config.cache_path) as cache:
        builder = QueryProfileBuilder(
            config, cache, paper_service, llm, encoder
        )
        generated = [
            builder.start_query(
                query_id="q1",
                query="find retrieval papers",
                date_cutoff="2004-01",
                retriever=retriever,
            )
            for _ in range(5)
        ]
        assert llm.calls == 1
        assert retriever.calls == 1
        assert paper_service.calls == 1
        assert len({item.query_profile_id for item in generated}) == 1
        assert all(
            item.selected_concepts == generated[0].selected_concepts
            for item in generated
        )
        assert generated[0].candidate_topics == [
            {"concept": "retrieval", "frequency": 2},
            {"concept": "ranking", "frequency": 1},
        ]

    resumed_retriever = FixedRetriever(_auxiliary_papers())
    resumed_llm = FixedQueryLLM()
    resumed_encoder = FixedEncoder("qwen3-concepts")
    with SemRankCache(config.cache_path) as cache:
        resumed = QueryProfileBuilder(
            config,
            cache,
            paper_service,
            resumed_llm,
            resumed_encoder,
        ).start_query(
            query_id="q1",
            query="find retrieval papers",
            date_cutoff="2004-01",
            retriever=resumed_retriever,
        )
        assert resumed.cache_hit is True
        assert resumed_llm.calls == 0
        assert resumed_retriever.calls == 0
        assert resumed_encoder.calls == [resumed.selected_concepts]
        assert resumed.concept_encoder == "qwen3-concepts"


def test_query_text_profile_ignores_encoder_but_tracks_pipeline_changes(
    tmp_path,
):
    config = SemRankConfig(
        initial_top_m=3,
        feedback_top_n=2,
        prompt_top_papers=2,
        candidate_topic_k=2,
        candidate_phrase_k=2,
        cache_path=str(tmp_path / "semrank.sqlite3"),
    )
    profiles = {
        "2001.00001": _paper_profile("2001.00001", ["a"], ["x"]),
        "2002.00001": _paper_profile("2002.00001", ["b"], ["y"]),
    }
    with SemRankCache(config.cache_path) as cache:
        retriever = FixedRetriever(_auxiliary_papers())
        builder = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles(profiles),
            FixedQueryLLM(),
            FixedEncoder("encoder-v1"),
        )
        first = builder.start_query(
            query_id="q1",
            query="first",
            date_cutoff="2004-01",
            retriever=retriever,
        )
        second = builder.start_query(
            query_id="q2",
            query="second",
            date_cutoff="2004-01",
            retriever=retriever,
        )
        changed_encoder = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles(profiles),
            FixedQueryLLM(),
            FixedEncoder("encoder-v2"),
        ).start_query(
            query_id="q1",
            query="first",
            date_cutoff="2004-01",
            retriever=retriever,
        )
        changed_paper_pipeline = QueryProfileBuilder(
            SemRankConfig(
                initial_top_m=3,
                feedback_top_n=2,
                prompt_top_papers=2,
                candidate_topic_k=2,
                candidate_phrase_k=2,
                classifier_topic_k=99,
                cache_path=str(tmp_path / "semrank.sqlite3"),
            ),
            cache,
            FixedPaperProfiles(profiles),
            FixedQueryLLM(),
            FixedEncoder("encoder-v1"),
        ).start_query(
            query_id="q1",
            query="first",
            date_cutoff="2004-01",
            retriever=retriever,
        )
    assert first.query_profile_id != second.query_profile_id
    assert first.query_profile_id == changed_encoder.query_profile_id
    assert changed_encoder.concept_encoder == "encoder-v2"
    assert first.query_profile_id != changed_paper_pipeline.query_profile_id


def test_future_auxiliary_paper_is_rejected_before_frequency_count(tmp_path):
    config = SemRankConfig(
        initial_top_m=1,
        feedback_top_n=1,
        prompt_top_papers=1,
        candidate_topic_k=1,
        candidate_phrase_k=1,
        cache_path=str(tmp_path / "semrank.sqlite3"),
    )
    future = AuxiliaryPaper(
        paper_arxiv_id="2099.00001",
        score=100.0,
        title="Future",
        abstract="High-frequency future concept",
        date="2099-01",
        rank=1,
    )
    with SemRankCache(config.cache_path) as cache:
        builder = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles({}),
            FixedQueryLLM(),
            FixedEncoder(),
        )
        with pytest.raises(AssertionError, match="strict date-valid"):
            builder.start_query(
                query_id="q1",
                query="query",
                date_cutoff="2020-01",
                retriever=FixedRetriever([future]),
            )


def test_query_signature_tracks_classifier_and_prompt_versions(
    tmp_path,
    monkeypatch,
):
    config = SemRankConfig(
        initial_top_m=3,
        feedback_top_n=2,
        prompt_top_papers=2,
        candidate_topic_k=2,
        candidate_phrase_k=2,
        cache_path=str(tmp_path / "semrank.sqlite3"),
    )
    profiles = {
        "2001.00001": _paper_profile("2001.00001", ["a"], ["x"]),
        "2002.00001": _paper_profile("2002.00001", ["b"], ["y"]),
    }
    retriever = FixedRetriever(_auxiliary_papers())
    with SemRankCache(config.cache_path) as cache:
        first = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles(profiles, classifier_id="checkpoint-v1"),
            FixedQueryLLM(),
            FixedEncoder(),
        ).start_query(
            query_id="q1",
            query="first",
            date_cutoff="2004-01",
            retriever=retriever,
        )
        changed_classifier = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles(profiles, classifier_id="checkpoint-v2"),
            FixedQueryLLM(),
            FixedEncoder(),
        ).start_query(
            query_id="q1",
            query="first",
            date_cutoff="2004-01",
            retriever=retriever,
        )
        monkeypatch.setattr(
            query_profile_module,
            "SEMRANK_QUERY_PROMPT_VERSION",
            "query-prompt-v2",
        )
        changed_prompt = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles(profiles, classifier_id="checkpoint-v1"),
            FixedQueryLLM(),
            FixedEncoder(),
        ).start_query(
            query_id="q1",
            query="first",
            date_cutoff="2004-01",
            retriever=retriever,
        )

    assert first.query_profile_id != changed_classifier.query_profile_id
    assert first.query_profile_id != changed_prompt.query_profile_id


class CountingClassifier:
    classifier_id = "classifier-v1"
    label_space_id = "labels-v1"

    def __init__(self):
        self.calls = 0

    def predict_batch(self, texts, *, top_k):
        self.calls += 1
        return [
            [TopicCandidate("retrieval", 1.0, "L1")] for _ in texts
        ]


class TwoTopicCountingClassifier(CountingClassifier):
    def predict_batch(self, texts, *, top_k):
        self.calls += 1
        return [
            [
                TopicCandidate("retrieval", 1.0, "L1"),
                TopicCandidate("ranking", 0.5, "L2"),
            ][:top_k]
            for _ in texts
        ]


class CountingPaperLLM:
    model = "paper-llm"
    llm_id = "paper-llm|temperature=0"

    def __init__(self):
        self.calls = 0

    def refine_papers(self, requests):
        self.calls += len(requests)
        return [
            (
                ["retrieval"],
                ["dense retrieval"],
                '{"selected_topics":["retrieval"]}',
            )
            for _ in requests
        ]


class FlakyPaperLLM(CountingPaperLLM):
    def refine_papers(self, requests):
        self.calls += len(requests)
        if self.calls == len(requests):
            return [
                (
                    [],
                    [],
                    "",
                    "paper_concept_llm_call_failed:ConnectionError",
                )
                for _ in requests
            ]
        return [
            (
                ["retrieval"],
                ["dense retrieval"],
                '{"selected_topics":["retrieval"]}',
                None,
            )
            for _ in requests
        ]


def test_paper_concepts_are_global_and_built_once(tmp_path):
    config = SemRankConfig(cache_path=str(tmp_path / "semrank.sqlite3"))
    classifier = CountingClassifier()
    llm = CountingPaperLLM()
    encoder = FixedEncoder()
    metadata = {
        "2001.00001": {
            "title": "Dense retrieval",
            "abstract": "A dense retrieval method.",
        }
    }
    with SemRankCache(config.cache_path) as cache:
        service = PaperConceptService(
            config, cache, classifier, llm, encoder
        )
        first = service.get_or_build(metadata)["2001.00001"]
        second = service.get_or_build(metadata)["2001.00001"]

    assert classifier.calls == 1
    assert llm.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.concepts == ["retrieval", "dense retrieval"]


def test_classifier_only_uses_all_candidate_topics_and_zero_paper_llm(
    tmp_path,
):
    cache_path = str(tmp_path / "semrank.sqlite3")
    metadata = {
        "2001.00001": {
            "title": "Dense retrieval",
            "abstract": "A dense retrieval method.",
        }
    }
    with SemRankCache(cache_path) as cache:
        full = PaperConceptService(
            SemRankConfig(cache_path=cache_path),
            cache,
            TwoTopicCountingClassifier(),
            CountingPaperLLM(),
            FixedEncoder(),
        ).get_or_build(metadata)["2001.00001"]

        classifier = TwoTopicCountingClassifier()
        paper_llm = CountingPaperLLM()
        service = PaperConceptService(
            SemRankConfig(
                cache_path=cache_path,
                paper_concept_mode="classifier_only",
            ),
            cache,
            classifier,
            paper_llm,
            FixedEncoder(),
        )
        classifier_only = service.get_or_build(metadata)["2001.00001"]
        cached = service.get_or_build(metadata)["2001.00001"]

    assert classifier.calls == 1
    assert paper_llm.calls == 0
    assert classifier_only.profile_id != full.profile_id
    assert classifier_only.selected_topics == ["retrieval", "ranking"]
    assert classifier_only.keyphrases == []
    assert classifier_only.concepts == ["retrieval", "ranking"]
    assert classifier_only.llm_model == "none"
    assert classifier_only.raw_llm_output is None
    assert cached.cache_hit is True
    assert service.extraction_llm_id == "none"
    assert service.snapshot_stats() == {
        "paper_classifier_only_profiles_built": 1,
        "paper_concept_cache_hits": 1,
        "paper_concept_cache_misses": 1,
        "paper_concepts_total": 2,
        "topic_classifier_calls": 1,
        "topic_classifier_papers": 1,
    }


def test_classifier_only_query_profile_sees_topics_but_no_keyphrases(
    tmp_path,
):
    config = SemRankConfig(
        initial_top_m=3,
        feedback_top_n=2,
        prompt_top_papers=2,
        candidate_topic_k=2,
        candidate_phrase_k=2,
        classifier_topic_k=2,
        paper_concept_mode="classifier_only",
        cache_path=str(tmp_path / "semrank.sqlite3"),
    )
    classifier = TwoTopicCountingClassifier()
    paper_llm = CountingPaperLLM()
    query_llm = FixedQueryLLM()
    with SemRankCache(config.cache_path) as cache:
        service = PaperConceptService(
            config,
            cache,
            classifier,
            paper_llm,
            FixedEncoder(),
        )
        profile = QueryProfileBuilder(
            config,
            cache,
            service,
            query_llm,
            FixedEncoder(),
        ).start_query(
            query_id="q1",
            query="find retrieval papers",
            date_cutoff="2004-01",
            retriever=FixedRetriever(_auxiliary_papers()),
        )

    assert classifier.calls == 1
    assert paper_llm.calls == 0
    assert query_llm.calls == 1
    assert profile.candidate_topics == [
        {"concept": "ranking", "frequency": 2},
        {"concept": "retrieval", "frequency": 2},
    ]
    assert profile.candidate_keyphrases == []
    assert profile.selected_concepts == ["ranking"]
    assert profile.topic_pipeline_version == (
        "official_semrank_classifier_only_topics_v1"
    )


def test_cached_paper_text_profile_reuses_llm_but_reencodes_concepts(
    tmp_path,
):
    config = SemRankConfig(cache_path=str(tmp_path / "semrank.sqlite3"))
    metadata = {
        "2001.00001": {
            "title": "Dense retrieval",
            "abstract": "A dense retrieval method.",
        }
    }
    with SemRankCache(config.cache_path) as cache:
        PaperConceptService(
            config,
            cache,
            CountingClassifier(),
            CountingPaperLLM(),
            FixedEncoder("specter2-concepts"),
        ).get_or_build(metadata)

        classifier = CountingClassifier()
        llm = CountingPaperLLM()
        qwen = FixedEncoder("qwen3-concepts")
        reused = PaperConceptService(
            config,
            cache,
            classifier,
            llm,
            qwen,
        ).get_or_build(metadata)["2001.00001"]

    assert reused.cache_hit is True
    assert reused.concept_encoder == "qwen3-concepts"
    assert classifier.calls == 0
    assert llm.calls == 0
    assert qwen.calls == [["retrieval", "dense retrieval"]]
    assert metadata["2001.00001"]["title"] not in qwen.calls[0]


def test_embedding_failure_does_not_repeat_paper_classifier_or_llm(
    tmp_path,
):
    config = SemRankConfig(cache_path=str(tmp_path / "semrank.sqlite3"))
    metadata = {
        "2001.00001": {
            "title": "Dense retrieval",
            "abstract": "A dense retrieval method.",
        }
    }
    classifier = CountingClassifier()
    llm = CountingPaperLLM()
    with SemRankCache(config.cache_path) as cache:
        with pytest.raises(ConnectionError, match="embedding service"):
            PaperConceptService(
                config,
                cache,
                classifier,
                llm,
                FailingEncoder(),
            ).get_or_build(metadata)

        qwen = FixedEncoder("qwen3-concepts")
        reused = PaperConceptService(
            config,
            cache,
            classifier,
            llm,
            qwen,
        ).get_or_build(metadata)["2001.00001"]

    assert classifier.calls == 1
    assert llm.calls == 1
    assert reused.cache_hit is True
    assert qwen.calls == [["retrieval", "dense retrieval"]]


def test_embedding_failure_does_not_repeat_query_llm(tmp_path):
    config = SemRankConfig(
        initial_top_m=3,
        feedback_top_n=2,
        prompt_top_papers=2,
        candidate_topic_k=2,
        candidate_phrase_k=2,
        cache_path=str(tmp_path / "semrank.sqlite3"),
    )
    profiles = {
        "2001.00001": _paper_profile("2001.00001", ["a"], ["x"]),
        "2002.00001": _paper_profile("2002.00001", ["b"], ["y"]),
    }
    query_llm = FixedQueryLLM()
    with SemRankCache(config.cache_path) as cache:
        with pytest.raises(ConnectionError, match="embedding service"):
            QueryProfileBuilder(
                config,
                cache,
                FixedPaperProfiles(profiles),
                query_llm,
                FailingEncoder(),
            ).start_query(
                query_id="q1",
                query="first",
                date_cutoff="2004-01",
                retriever=FixedRetriever(_auxiliary_papers()),
            )

        resumed_llm = FixedQueryLLM()
        resumed_retriever = FixedRetriever(_auxiliary_papers())
        qwen = FixedEncoder("qwen3-concepts")
        reused = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles(profiles),
            resumed_llm,
            qwen,
        ).start_query(
            query_id="q1",
            query="first",
            date_cutoff="2004-01",
            retriever=resumed_retriever,
        )

    assert query_llm.calls == 1
    assert resumed_llm.calls == 0
    assert resumed_retriever.calls == 0
    assert reused.cache_hit is True
    assert qwen.calls == [reused.selected_concepts]


def test_only_failed_paper_profiles_are_retried_from_cache(tmp_path):
    config = SemRankConfig(cache_path=str(tmp_path / "semrank.sqlite3"))
    classifier = CountingClassifier()
    llm = FlakyPaperLLM()
    metadata = {
        "2001.00001": {
            "title": "Dense retrieval",
            "abstract": "A dense retrieval method.",
        }
    }
    with SemRankCache(config.cache_path) as cache:
        service = PaperConceptService(
            config, cache, classifier, llm, FixedEncoder()
        )
        failed = service.get_or_build(metadata)["2001.00001"]
        recovered = service.get_or_build(metadata)["2001.00001"]
        cached = service.get_or_build(metadata)["2001.00001"]
        audit = service.drain_audit_records()

    assert failed.status == "failed"
    assert recovered.status == "ok"
    assert recovered.cache_hit is False
    assert cached.status == "ok"
    assert cached.cache_hit is True
    assert classifier.calls == 2
    assert llm.calls == 2
    assert (
        service.snapshot_stats()["paper_concept_failed_cache_retries"] == 1
    )
    assert [row["status"] for row in audit] == ["failed", "ok"]


class FailingQueryLLM(FixedQueryLLM):
    def select_query_concepts(
        self,
        query,
        papers,
        candidate_topics,
        candidate_keyphrases,
    ):
        self.calls += 1
        raise ConnectionError("temporary")


def test_failed_query_profile_is_retried_on_resume(tmp_path):
    config = SemRankConfig(
        initial_top_m=3,
        feedback_top_n=2,
        prompt_top_papers=2,
        candidate_topic_k=2,
        candidate_phrase_k=2,
        cache_path=str(tmp_path / "semrank.sqlite3"),
    )
    profiles = {
        "2001.00001": _paper_profile("2001.00001", ["a"], ["x"]),
        "2002.00001": _paper_profile("2002.00001", ["b"], ["y"]),
    }
    with SemRankCache(config.cache_path) as cache:
        failed = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles(profiles),
            FailingQueryLLM(),
            FixedEncoder(),
        ).start_query(
            query_id="q1",
            query="first",
            date_cutoff="2004-01",
            retriever=FixedRetriever(_auxiliary_papers()),
        )
        recovering_llm = FixedQueryLLM()
        recovered_builder = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles(profiles),
            recovering_llm,
            FixedEncoder(),
        )
        recovered = recovered_builder.start_query(
            query_id="q1",
            query="first",
            date_cutoff="2004-01",
            retriever=FixedRetriever(_auxiliary_papers()),
        )

    assert failed.selection_status == "fallback"
    assert failed.fallback_reason == (
        "query_concept_selection_failed:ConnectionError"
    )
    assert recovered.selection_status == "ok"
    assert recovered.cache_hit is False
    assert recovering_llm.calls == 1
    assert (
        recovered_builder.snapshot_stats()[
            "query_profile_failed_cache_retries"
        ]
        == 1
    )


def test_query_profile_fails_closed_on_failed_feedback_profile(tmp_path):
    config = SemRankConfig(
        initial_top_m=3,
        feedback_top_n=2,
        prompt_top_papers=2,
        candidate_topic_k=2,
        candidate_phrase_k=2,
        cache_path=str(tmp_path / "semrank.sqlite3"),
    )
    failed = _paper_profile("2001.00001", [], [])
    failed = PaperConceptProfile(
        **{
            **failed.to_dict(),
            "status": "failed",
            "fallback_reason": (
                "paper_concept_llm_call_failed:ConnectionError"
            ),
        }
    )
    profiles = {
        "2001.00001": failed,
        "2002.00001": _paper_profile("2002.00001", ["b"], ["y"]),
    }
    with SemRankCache(config.cache_path) as cache:
        builder = QueryProfileBuilder(
            config,
            cache,
            FixedPaperProfiles(profiles),
            FixedQueryLLM(),
            FixedEncoder(),
        )
        with pytest.raises(RuntimeError, match="failed feedback profiles"):
            builder.start_query(
                query_id="q1",
                query="first",
                date_cutoff="2004-01",
                retriever=FixedRetriever(_auxiliary_papers()),
            )

    assert builder.snapshot_stats()[
        "query_profile_failed_feedback_papers"
    ] == 1


def test_paper_profile_signature_tracks_classifier_and_prompt(
    tmp_path,
    monkeypatch,
):
    config = SemRankConfig(cache_path=str(tmp_path / "semrank.sqlite3"))
    metadata = {
        "2001.00001": {
            "title": "Dense retrieval",
            "abstract": "A dense retrieval method.",
        }
    }
    with SemRankCache(config.cache_path) as cache:
        first_classifier = CountingClassifier()
        PaperConceptService(
            config,
            cache,
            first_classifier,
            CountingPaperLLM(),
            FixedEncoder(),
        ).get_or_build(metadata)

        second_classifier = CountingClassifier()
        second_classifier.classifier_id = "classifier-v2"
        PaperConceptService(
            config,
            cache,
            second_classifier,
            CountingPaperLLM(),
            FixedEncoder(),
        ).get_or_build(metadata)

        monkeypatch.setattr(
            paper_concepts_module,
            "SEMRANK_PAPER_PROMPT_VERSION",
            "paper-prompt-v2",
        )
        third_classifier = CountingClassifier()
        PaperConceptService(
            config,
            cache,
            third_classifier,
            CountingPaperLLM(),
            FixedEncoder(),
        ).get_or_build(metadata)

    assert first_classifier.calls == 1
    assert second_classifier.calls == 1
    assert third_classifier.calls == 1


def test_llm_parser_rejects_out_of_vocabulary_query_and_paper_topics():
    responses = iter(
        [
            (
                '{"selected_topics":["retrieval","invented topic"],'
                '"keyphrases":["dense retrieval","not in the paper"]}'
            ),
            (
                '{"selected_concepts":["retrieval","invented query term",'
                '"dense retrieval"]}'
            ),
        ]
    )

    def fake_call(*args, **kwargs):
        return next(responses)

    client = SemRankLLMClient(
        "fake",
        workers=1,
        call=fake_call,
    )
    topics, phrases, _, error = client.refine_paper(
        "Dense retrieval",
        "A dense retrieval paper.",
        [TopicCandidate("retrieval", 1.0, "L1")],
    )
    assert error is None
    assert topics == ["retrieval"]
    assert phrases == ["dense retrieval"]

    selected, _ = client.select_query_concepts(
        "retrieval",
        [],
        [{"concept": "retrieval", "frequency": 2}],
        [{"concept": "dense retrieval", "frequency": 1}],
    )
    assert selected == ["retrieval", "dense retrieval"]


def test_paper_llm_malformed_json_is_auditable_fallback():
    calls = []

    def fake_call(*args, **kwargs):
        calls.append(args[0])
        return "not-json"

    client = SemRankLLMClient(
        "fake",
        workers=1,
        call=fake_call,
    )
    topics, phrases, raw, error = client.refine_paper(
        "Dense retrieval",
        "A dense retrieval paper.",
        [TopicCandidate("retrieval", 1.0, "L1")],
    )

    assert topics == []
    assert phrases == []
    assert raw == "not-json"
    assert error == "paper_concept_llm_parse_failed"
    assert len(calls) == 3
    stats = client.snapshot_stats()
    assert stats["paper_concept_llm_calls"] == 3
    assert stats["llm_parse_failures"] == 3


def test_paper_llm_repairs_invalid_escaped_apostrophe_without_retry():
    raw = (
        r'{"selected_topics":["retrieval"],'
        r'"keyphrases":["Sarl\'os transform"]}'
    )
    client = SemRankLLMClient(
        "fake",
        workers=1,
        call=lambda *args, **kwargs: raw,
    )

    topics, phrases, returned_raw, error = client.refine_paper(
        "Sarl'os transform",
        "A retrieval method.",
        [TopicCandidate("retrieval", 1.0, "L1")],
    )

    assert topics == ["retrieval"]
    assert phrases == ["sarl'os transform"]
    assert returned_raw == raw
    assert error is None
    stats = client.snapshot_stats()
    assert stats["paper_concept_llm_calls"] == 1
    assert stats["llm_parse_failures"] == 0


def test_paper_llm_repairs_missing_outer_brace_without_retry():
    raw = (
        '{"selected_topics":["retrieval"],'
        '"keyphrases":["dense retrieval"]'
    )
    client = SemRankLLMClient(
        "fake",
        workers=1,
        call=lambda *args, **kwargs: raw,
    )

    topics, phrases, returned_raw, error = client.refine_paper(
        "Dense retrieval",
        "A dense retrieval method.",
        [TopicCandidate("retrieval", 1.0, "L1")],
    )

    assert topics == ["retrieval"]
    assert phrases == ["dense retrieval"]
    assert returned_raw == raw
    assert error is None
    stats = client.snapshot_stats()
    assert stats["paper_concept_llm_calls"] == 1
    assert stats["llm_parse_failures"] == 0


def test_paper_llm_discards_only_truncated_keyphrase_item():
    raw = (
        '{"selected_topics":["retrieval"],'
        '"keyphrases":["dense retrieval","unfinished'
    )
    client = SemRankLLMClient(
        "fake",
        workers=1,
        call=lambda *args, **kwargs: raw,
    )

    topics, phrases, returned_raw, error = client.refine_paper(
        "Dense retrieval",
        "A dense retrieval method.",
        [TopicCandidate("retrieval", 1.0, "L1")],
    )

    assert topics == ["retrieval"]
    assert phrases == ["dense retrieval"]
    assert returned_raw == raw
    assert error is None
    stats = client.snapshot_stats()
    assert stats["paper_concept_llm_calls"] == 1
    assert stats["llm_parse_failures"] == 0


def test_paper_llm_repairs_missing_topic_array_bracket():
    raw = (
        '{"selected_topics":["retrieval","ranking",'
        '"keyphrases":["dense retrieval"]}'
    )
    client = SemRankLLMClient(
        "fake",
        workers=1,
        call=lambda *args, **kwargs: raw,
    )

    topics, phrases, returned_raw, error = client.refine_paper(
        "Dense retrieval",
        "A ranking method.",
        [
            TopicCandidate("retrieval", 1.0, "L1"),
            TopicCandidate("ranking", 0.9, "L2"),
        ],
    )

    assert topics == ["retrieval", "ranking"]
    assert phrases == ["dense retrieval"]
    assert returned_raw == raw
    assert error is None
    stats = client.snapshot_stats()
    assert stats["paper_concept_llm_calls"] == 1
    assert stats["llm_parse_failures"] == 0


def test_paper_llm_transport_failure_is_auditable_fallback():
    calls = []

    def fake_call(*args, **kwargs):
        calls.append(args[0])
        raise ConnectionError("offline")

    client = SemRankLLMClient(
        "fake",
        workers=1,
        call=fake_call,
    )
    topics, phrases, raw, error = client.refine_paper(
        "Dense retrieval",
        "A dense retrieval paper.",
        [TopicCandidate("retrieval", 1.0, "L1")],
    )

    assert topics == []
    assert phrases == []
    assert raw == ""
    assert error == "paper_concept_llm_call_failed:ConnectionError"
    assert len(calls) == 3
    stats = client.snapshot_stats()
    assert stats["paper_concept_llm_calls"] == 3
    assert stats["llm_failures"] == 3


def test_default_llm_client_reuses_one_persistent_connection(monkeypatch):
    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                '{"selected_topics":["retrieval"],'
                                '"keyphrases":["dense retrieval"]}'
                            )
                        )
                    )
                ]
            )

    class FakeOpenAI:
        instances = []

        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=FakeCompletions()
            )
            self.closed = False
            self.instances.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(prompt_module, "OpenAI", FakeOpenAI)
    client = SemRankLLMClient(
        "qwen3-test",
        workers=2,
    )
    requests = [
        (
            "Dense retrieval",
            "A dense retrieval paper.",
            [TopicCandidate("retrieval", 1.0, "L1")],
        ),
        (
            "Dense retrieval",
            "A second dense retrieval paper.",
            [TopicCandidate("retrieval", 1.0, "L1")],
        ),
    ]
    results = client.refine_papers(requests)
    client.close()

    assert len(FakeOpenAI.instances) == 1
    instance = FakeOpenAI.instances[0]
    assert instance.chat.completions.calls == 2
    assert instance.closed is True
    assert all(result[3] is None for result in results)
