import math

import pytest

from resume_search.eval.metrics import agreement, field_accuracy, ndcg_at_k, precision_at_k

GRADES = {"a": 2, "b": 1, "c": 0, "d": 0}


def test_ndcg_perfect_order_is_one():
    assert ndcg_at_k(["a", "b", "c", "d"], GRADES, 5) == pytest.approx(1.0)


def test_ndcg_reversed_order_is_lower():
    expected = (0 + 0 + 1 / math.log2(4) + 2 / math.log2(5)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["d", "c", "b", "a"], GRADES, 5) == pytest.approx(expected)
    assert ndcg_at_k(["d", "c", "b", "a"], GRADES, 5) < 1.0


def test_ndcg_all_zero_pool_scores_zero():
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 5) == 0.0


def test_ndcg_cutoff_ignores_ranks_past_k():
    assert ndcg_at_k(["c", "d", "a", "b"], GRADES, 2) == 0.0


def test_precision_at_k_threshold_and_small_pool():
    assert precision_at_k(["a", "c", "b"], GRADES, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a", "c", "b"], GRADES, 3, threshold=2) == pytest.approx(1 / 3)
    assert precision_at_k(["a", "b"], GRADES, 3) == 1.0
    assert precision_at_k([], GRADES, 3) == 0.0


def test_field_accuracy_groups_roles_and_counts_partial_half():
    rows = [
        {"field": "name", "grade": "correct", "reason": ""},
        {"field": "skills", "grade": "partial", "reason": ""},
        {"field": "experience[0]", "grade": "correct", "reason": ""},
        {"field": "experience[1]", "grade": "hallucinated", "reason": ""},
        {"field": "experience[2]", "grade": "missing", "reason": ""},
        {"field": "experience[3]", "grade": "partial", "reason": ""},
    ]
    acc = field_accuracy(rows)
    role = acc["experience_role"]
    assert (role["total"], role["correct"], role["hallucinated"], role["missing"]) == (4, 1, 1, 1)
    assert role["accuracy"] == 0.25
    assert role["soft_accuracy"] == pytest.approx(0.375)
    assert acc["skills"]["accuracy"] == 0.0 and acc["skills"]["soft_accuracy"] == 0.5
    assert acc["overall"]["total"] == 6


def test_agreement_levels():
    human = [("a", "correct"), ("b", "partial"), ("c", "wrong"), ("d", "correct"), ("e", "missing")]
    judge = [("a", "correct"), ("b", "correct"), ("c", "hallucinated"), ("d", "wrong"), ("z", "correct")]
    fix = lambda rows: [{"field": f, "grade": g} for f, g in rows]
    out = agreement(fix(human), fix(judge))
    assert out["n"] == 4
    assert out["exact"] == 0.25
    assert out["within_one_level"] == 0.75
    assert out["disagreements"][-1] == {"field": "d", "human": "correct", "judge": "wrong"}
