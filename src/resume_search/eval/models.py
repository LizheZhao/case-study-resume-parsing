"""Data shapes shared by the judge, the metrics and the runners (design section 10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from resume_search.search.requisition import Requisition

FieldGrade = Literal["correct", "partial", "wrong", "hallucinated", "missing"]
RelevanceGrade = Literal[0, 1, 2]

FIELDS = (
    "name", "contact", "education", "experience", "skills", "certifications", "languages",
    "investment_approach", "search_summary",
)


def expected_fields(n_roles: int) -> list[str]:
    """Every field a parsing judgement must grade, in order: top-level fields, then each role."""
    return [*FIELDS, *(f"experience[{i}]" for i in range(n_roles))]


class FieldRow(BaseModel):
    field: str
    grade: FieldGrade
    reason: str


class ParsingGrades(BaseModel):
    """What the judge fills. candidate_id is attached by code, not asked of the model."""

    grades: list[FieldRow]


class ParsingJudgement(BaseModel):
    candidate_id: str
    grades: list[FieldRow]


class RankRow(BaseModel):
    candidate_id: str
    grade: RelevanceGrade
    reason: str


class RankingGrades(BaseModel):
    grades: list[RankRow]


class RankingJudgement(BaseModel):
    requisition_id: str
    grades: list[RankRow]


class EvalRequisition(BaseModel):
    id: str
    title: str
    requisition: Requisition


def load_requisitions(path: str | Path) -> list[EvalRequisition]:
    """One JSON object per line: id, title, then the Requisition fields flat."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        out.append(
            EvalRequisition(
                id=raw.pop("id"), title=raw.pop("title"), requisition=Requisition.model_validate(raw)
            )
        )
    return out
