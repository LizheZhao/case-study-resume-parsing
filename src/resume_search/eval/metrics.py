"""Pure scoring functions for the eval suites. No I/O, no model calls."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

GRADES = ("correct", "partial", "wrong", "hallucinated", "missing")
# correct > partial > wrong, missing and hallucinated
LEVEL = {"correct": 2, "partial": 1, "wrong": 0, "missing": 0, "hallucinated": 0}


def field_group(field: str) -> str:
    """experience[3] belongs to the experience_role group; everything else is its own group."""
    return "experience_role" if field.startswith("experience[") else field


def _pair(row) -> tuple[str, str]:
    if isinstance(row, Mapping):
        return row["field"], row["grade"]
    return row.field, row.grade


def field_accuracy(grades: Iterable) -> dict[str, dict[str, float]]:
    """Per field group: a count per grade, `total`, `accuracy` (correct / total) and
    `soft_accuracy` (partial counts 0.5). The `overall` key covers every row."""
    out: dict[str, dict[str, float]] = {}
    for row in grades:
        field, grade = _pair(row)
        for key in (field_group(field), "overall"):
            g = out.setdefault(key, {**dict.fromkeys(GRADES, 0), "total": 0})
            g[grade] += 1
            g["total"] += 1
    for g in out.values():
        g["accuracy"] = g["correct"] / g["total"]
        g["soft_accuracy"] = (g["correct"] + 0.5 * g["partial"]) / g["total"]
    return out


def ndcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Linear-gain nDCG. The ideal order sorts every graded candidate; an id with no grade
    counts as 0. A pool with no relevant candidate has nothing to rank and scores 0.0."""

    def dcg(rels: Sequence[int]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))

    ideal = dcg(sorted(grades.values(), reverse=True))
    if ideal == 0:
        return 0.0
    return dcg([grades.get(i, 0) for i in ranked_ids]) / ideal


def precision_at_k(
    ranked_ids: Sequence[str], grades: Mapping[str, int], k: int, threshold: int = 1
) -> float:
    """Share of the top k with grade >= threshold. A pool smaller than k divides by its size."""
    top = list(ranked_ids[:k])
    if not top:
        return 0.0
    return sum(grades.get(i, 0) >= threshold for i in top) / len(top)


def agreement(human: Iterable, judge: Iterable) -> dict:
    """Compare two gradings of the same fields. `within_one_level` counts a pair as agreeing
    unless one side says correct and the other wrong, missing or hallucinated."""
    h = dict(_pair(r) for r in human)
    j = dict(_pair(r) for r in judge)
    shared = [f for f in h if f in j]
    if not shared:
        return {"exact": 0.0, "within_one_level": 0.0, "n": 0, "disagreements": []}
    exact = sum(h[f] == j[f] for f in shared)
    near = sum(abs(LEVEL[h[f]] - LEVEL[j[f]]) <= 1 for f in shared)
    return {
        "exact": exact / len(shared),
        "within_one_level": near / len(shared),
        "n": len(shared),
        "disagreements": [{"field": f, "human": h[f], "judge": j[f]} for f in shared if h[f] != j[f]],
    }
