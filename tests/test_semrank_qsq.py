import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from semrank import (
    base_semantic_scores,
    mean_max_concept_cosine,
    population_zscore,
    semrank_scores,
)


def test_concept_score_is_mean_query_of_max_paper_cosine():
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    paper = np.asarray(
        [[1.0, 0.0], [2 ** -0.5, 2 ** -0.5]],
        dtype=np.float32,
    )

    expected = (1.0 + 2 ** -0.5) / 2.0
    assert np.isclose(mean_max_concept_cosine(query, paper), expected)
    assert mean_max_concept_cosine(query, np.zeros((0, 2))) == 0.0


def test_base_and_final_formula_are_exact():
    query = [0.0, 0.5, 1.0]
    subquery = [1.0, 0.5, 0.0]
    concepts = [0.1, 0.4, 0.9]

    base = base_semantic_scores(query, subquery)
    assert np.allclose(base, [0.6, 0.5, 0.4])

    result = semrank_scores(query, subquery, concepts)
    expected_base_z = (base - base.mean()) / base.std(ddof=0)
    concept_array = np.asarray(concepts)
    expected_concept_z = (
        concept_array - concept_array.mean()
    ) / concept_array.std(ddof=0)
    assert np.allclose(result["base_z"], expected_base_z)
    assert np.allclose(result["concept_z"], expected_concept_z)
    assert np.allclose(
        result["final"],
        expected_base_z + expected_concept_z,
    )


def test_population_zscore_degenerate_cases_are_zero():
    empty, mean, std = population_zscore([])
    assert empty.size == 0
    assert mean == std == 0.0

    for values in ([7.0], [2.0, 2.0, 2.0], [1.0, 1.0 + 1e-14]):
        normalized, _, _ = population_zscore(values)
        assert np.array_equal(normalized, np.zeros(len(values)))


def test_empty_query_concepts_use_base_only():
    result = semrank_scores(
        [0.0, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        [0.0, 0.0, 0.0],
        base_only=True,
    )
    assert np.array_equal(result["final"], result["base_z"])
    assert np.array_equal(result["concept_z"], np.zeros(3))


def test_graph_features_cannot_change_semrank_formula():
    first = semrank_scores([0.5, 0.5], [0.5, 0.5], [0.2, 0.2])
    # There is deliberately no intent/path/type argument in the pure formula.
    assert first["final"][0] == first["final"][1]
