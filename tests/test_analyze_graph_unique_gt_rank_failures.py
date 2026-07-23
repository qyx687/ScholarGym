import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_graph_unique_gt_rank_failures.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_graph_unique_gt_rank_failures", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _cause_row(**overrides):
    row = {
        "hybrid_selected": False,
        "tie_break_loss": False,
        "semantic_control_selected": False,
        "max_structure_counterfactual_selected": False,
    }
    row.update(overrides)
    return row


def test_paper_cause_precedence_uses_best_available_occurrence():
    assert MODULE.classify_paper([_cause_row(tie_break_loss=True)]) == "tie_break_loss"
    assert MODULE.classify_paper(
        [_cause_row(max_structure_counterfactual_selected=True), _cause_row(semantic_control_selected=True)]
    ) == "displaced_by_structure"
    assert MODULE.classify_paper(
        [_cause_row(max_structure_counterfactual_selected=True)]
    ) == "insufficient_structural_signal"
    assert MODULE.classify_paper(
        [_cause_row()]
    ) == "semantic_deficit_beyond_max_structure"


def test_counterfactual_rank_preserves_graph_seed_tie_break():
    target = {
        "paper_arxiv_id": "target",
        "candidate_type": "expanded",
        "query_score_normalized": 0.5,
        "subquery_score_normalized": 0.5,
    }
    seed = {
        "paper_arxiv_id": "seed",
        "candidate_type": "seed",
        "observed_retrieval_rank": 1,
        "query_score_normalized": 0.5,
        "subquery_score_normalized": 0.5,
    }

    rank = MODULE._counterfactual_rank(
        [target, seed],
        target,
        0.5,
        query_weight=0.5,
        subquery_weight=0.5,
        intent_weight=0.0,
        path_weight=0.0,
    )

    assert rank == 2
