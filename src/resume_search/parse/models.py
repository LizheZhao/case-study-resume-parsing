"""Pydantic models for the LLM output (design section 4.1)."""

from __future__ import annotations

import re
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from resume_search.taxonomy import iso2_codes, market_labels


class Relevance(str, Enum):
    core = "core"
    adjacent = "adjacent"
    unrelated = "unrelated"


class EmploymentType(str, Enum):
    full_time = "full_time"
    internship = "internship"
    part_time = "part_time"
    academic = "academic"


class FirmType(str, Enum):
    hedge_fund = "Hedge fund"
    asset_manager = "Asset manager"
    sell_side_bank = "Sell-side bank"
    sell_side_broker = "Sell-side broker"
    pe_vc = "PE/VC"
    consulting = "Consulting"
    corporate = "Corporate"
    academic = "Academic"
    other = "Other"


class AssetClass(str, Enum):
    equities = "Equities"
    fixed_income = "Fixed Income"
    credit = "Credit"
    fx = "FX"
    commodities = "Commodities"
    derivatives = "Derivatives"
    multi_asset = "Multi-asset"


class Sector(str, Enum):
    technology = "Technology"
    healthcare = "Healthcare"
    financials = "Financials"
    energy = "Energy"
    industrials = "Industrials"
    consumer = "Consumer"
    credit = "Credit"
    macro = "Macro"
    other = "Other"


class Approach(str, Enum):
    fundamental = "Fundamental"
    systematic = "Systematic"
    both = "Both"
    unclear = "Unclear"


_DATE = re.compile(r"^\d{4}(-(0[1-9]|1[0-2]))?$")
_LONG_DIGITS = re.compile(r"\d{7,}")
_WS = re.compile(r"\s+")
_QUOTE_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _check_date(v: str | None) -> str | None:
    if v is not None and not _DATE.match(v):
        raise ValueError(f"date '{v}' must be YYYY or YYYY-MM")
    return v


def _check_country(v: str | None) -> str | None:
    if v is not None and v not in iso2_codes():
        raise ValueError(f"country '{v}' is not an ISO-2 code in taxonomy/countries.yaml")
    return v


def month_index(value: str) -> int:
    """YYYY-MM -> months since year 0; a bare YYYY means June."""
    month = int(value[5:7]) if len(value) >= 7 else 6
    return int(value[:4]) * 12 + month - 1


def stint_span(start: str, end: str) -> tuple[int, int]:
    """Half-open month range of one stint. A stint whose start equals its end is one month."""
    s = month_index(start)
    return s, max(month_index(end), s + 1)


def _check_order(start: str | None, end: str | None) -> None:
    if start and end:
        n = min(len(start), len(end))  # compare at the coarser precision
        if start[:n] > end[:n]:
            raise ValueError(f"start {start} is after end {end}")


class Evidence(BaseModel):
    quote: str = Field(max_length=200, description="Verbatim span from the resume")


class Labeled(BaseModel):
    label: str
    evidence: Evidence


class Contact(BaseModel):
    email: str | None
    phone: str | None
    location_text: str | None


class Education(BaseModel):
    institution: str
    degree: str | None
    degree_level: str | None
    field: str | None
    start: str | None
    end: str | None
    country: str | None

    _dates = field_validator("start", "end")(_check_date)
    _country = field_validator("country")(_check_country)

    @model_validator(mode="after")
    def _order(self) -> Education:
        _check_order(self.start, self.end)
        return self


class Experience(BaseModel):
    employer: str
    title: str
    location_text: str | None
    country: str | None
    start: list[str]
    end: list[str | None]
    is_current: bool
    employment_type: EmploymentType
    relevance: Relevance
    firm_type: FirmType
    asset_classes: list[AssetClass] = []
    sectors: list[Labeled] = []
    other_sector_tags: list[str] = []
    markets_covered: list[Labeled] = []
    duration_months: int | None
    bullets: list[str]

    _country = field_validator("country")(_check_country)

    @field_validator("start", "end")
    @classmethod
    def _dates(cls, v: list[str | None]) -> list[str | None]:
        for d in v:
            _check_date(d)
        return v

    @property
    def first_start(self) -> str | None:
        return self.start[0] if self.start else None

    @property
    def last_end(self) -> str | None:
        return self.end[-1] if self.end else None

    def spans(self, today: date) -> list[tuple[int, int]]:
        """Half-open month range per stint; the open end of a current role is today's month."""
        now = today.year * 12 + today.month - 1
        out = []
        for start, end in zip(self.start, self.end, strict=True):
            if end is None:
                s = month_index(start)
                out.append((s, max(now, s + 1)))
            else:
                out.append(stint_span(start, end))
        return out

    def months(self, today: date) -> int:
        """Sum over stints; duration_months is the fallback when the resume gives no dates."""
        spans = self.spans(today)
        return sum(t - s for s, t in spans) if spans else (self.duration_months or 0)

    @model_validator(mode="after")
    def _consistent(self) -> Experience:
        if len(self.start) != len(self.end):
            raise ValueError(
                f"start has {len(self.start)} entries and end has {len(self.end)}; "
                "they are parallel lists, one entry per stint"
            )
        last = len(self.start) - 1
        for i, (s, e) in enumerate(zip(self.start, self.end, strict=True)):
            if e is None:
                if not (self.is_current and i == last):
                    raise ValueError(f"end[{i}] is null; only the last end of a current role may be")
            else:
                if self.is_current and i == last:
                    raise ValueError("the last end must be null when is_current is true")
                _check_order(s, e)
        prev_end = None
        for i, (s, e) in enumerate(zip(self.start, self.end, strict=True)):
            if prev_end is not None and month_index(s) < prev_end:
                raise ValueError(f"stint {i} ({s}) overlaps or precedes stint {i - 1}")
            prev_end = stint_span(s, e)[1] if e else month_index(s) + 1
        sector_names = {s.value for s in Sector}
        for s in self.sectors:
            if s.label not in sector_names:
                raise ValueError(f"sector label '{s.label}' is not one of {sorted(sector_names)}")
        allowed = iso2_codes() | market_labels()
        for m in self.markets_covered:
            if m.label not in allowed:
                raise ValueError(f"markets_covered label '{m.label}' is not an ISO-2 code or market label")
        return self


class Skills(BaseModel):
    programming: list[str] = []
    tools: list[str] = []
    domain: list[str] = []


class Language(BaseModel):
    name: str
    level: str | None


class Profile(BaseModel):
    name: str
    contact: Contact
    education: list[Education]
    skills: Skills
    certifications: list[str] = []
    languages: list[Language] = []


class ExperienceList(BaseModel):
    experience: list[Experience]


class Assessment(BaseModel):
    investment_approach: Labeled
    search_summary: str = Field(min_length=200, max_length=700)

    @field_validator("search_summary")
    @classmethod
    def _no_contact_leak(cls, v: str) -> str:
        if "@" in v or _LONG_DIGITS.search(v):
            raise ValueError("search_summary must not contain '@' or a run of 7+ digits")
        return v

    @model_validator(mode="after")
    def _approach_label(self) -> Assessment:
        valid = {a.value for a in Approach}
        if self.investment_approach.label not in valid:
            raise ValueError(f"investment_approach label must be one of {sorted(valid)}")
        return self


class ParsedResume(Assessment, ExperienceList, Profile):
    """The stored record. Bases are ordered so fields dump as profile, experience, assessment."""


# Smallest output models for targeted regeneration (design 5.5).
class NameOnly(BaseModel):
    name: str


class ContactOnly(BaseModel):
    contact: Contact


class CertificationsOnly(BaseModel):
    certifications: list[str]


class SkillsOnly(BaseModel):
    skills: Skills


ExperienceItem = Experience
EducationItem = Education


def _norm(s: str) -> str:
    return _WS.sub(" ", s.translate(_QUOTE_FOLD)).strip()


def _evidence_errors(
    text: str, approach: Labeled | None, experiences: list[Experience]
) -> list[str]:
    haystack = _norm(text)
    errors: list[str] = []

    def check(path: str, labeled: Labeled) -> None:
        if _norm(labeled.evidence.quote) not in haystack:
            errors.append(f"{path}.evidence.quote not found verbatim in resume: "
                          f"'{labeled.evidence.quote}'")

    if approach is not None:
        check("investment_approach", approach)
    for i, exp in enumerate(experiences):
        for j, s in enumerate(exp.sectors):
            check(f"experience[{i}].sectors[{j}]", s)
        for j, m in enumerate(exp.markets_covered):
            check(f"experience[{i}].markets_covered[{j}]", m)
    return errors


def check_evidence(parsed: ParsedResume, text: str) -> list[str]:
    """Every Evidence.quote must be a substring of the resume after whitespace folding."""
    return _evidence_errors(text, parsed.investment_approach, parsed.experience)


def check_experience_evidence(experiences: list[Experience], text: str) -> list[str]:
    return _evidence_errors(text, None, experiences)
