"""Month arithmetic over roles. Every stint of every role is a range; ranges are merged per bucket
so overlapping jobs are not double counted. Roles without stints fall back to duration_months and
are summed as-is."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from resume_search.parse.models import EmploymentType, Experience, Relevance

# Upper bounds are inclusive so the design's ">120" Principal and "<6" Intern hold exactly.
_BANDS = ((5, "Intern/Student"), (24, "Junior"), (60, "Mid"), (120, "Senior"))


def total_months(roles: Iterable[Experience], today: date) -> int:
    spans: list[tuple[int, int]] = []
    loose = 0
    for e in roles:
        role = e.spans(today)
        if role:
            spans += role
        else:
            loose += e.duration_months or 0
    merged = 0
    cur_start = cur_end = None
    for s, t in sorted(spans):
        if cur_end is None or s > cur_end:
            if cur_end is not None:
                merged += cur_end - cur_start
            cur_start, cur_end = s, t
        else:
            cur_end = max(cur_end, t)
    if cur_end is not None:
        merged += cur_end - cur_start
    return merged + loose


def is_relevant_full_time(e: Experience) -> bool:
    return e.employment_type is EmploymentType.full_time and e.relevance in (
        Relevance.core,
        Relevance.adjacent,
    )


def is_core_full_time(e: Experience) -> bool:
    return e.employment_type is EmploymentType.full_time and e.relevance is Relevance.core


def is_internship(e: Experience) -> bool:
    return e.employment_type is EmploymentType.internship


def seniority_band(months_relevant: int) -> str:
    for upper, label in _BANDS:
        if months_relevant <= upper:
            return label
    return "Principal"
