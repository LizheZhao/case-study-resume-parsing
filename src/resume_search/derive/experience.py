"""Month arithmetic over roles. Ranges are merged per bucket so overlapping jobs are not double
counted; roles without a start fall back to duration_months and are summed as-is."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from resume_search.parse.models import EmploymentType, Experience, Relevance

# Upper bounds are inclusive so the design's ">120" Principal and "<6" Intern hold exactly.
_BANDS = ((5, "Intern/Student"), (24, "Junior"), (60, "Mid"), (120, "Senior"))


def month_index(value: str) -> int:
    """YYYY-MM -> months since year 0; a bare YYYY means June."""
    month = int(value[5:7]) if len(value) >= 7 else 6
    return int(value[:4]) * 12 + month - 1


def role_span(e: Experience, today: date) -> tuple[int, int] | None:
    if e.start is None:
        return None
    start = month_index(e.start)
    if e.is_current:
        end = today.year * 12 + today.month - 1
    elif e.end:
        end = month_index(e.end)
    elif e.duration_months is not None:
        end = start + e.duration_months
    else:
        return None
    return start, max(start, end)


def role_months(e: Experience, today: date) -> int:
    span = role_span(e, today)
    if span:
        return span[1] - span[0]
    return e.duration_months or 0


def total_months(roles: Iterable[Experience], today: date) -> int:
    spans: list[tuple[int, int]] = []
    loose = 0
    for e in roles:
        span = role_span(e, today)
        if span:
            spans.append(span)
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
