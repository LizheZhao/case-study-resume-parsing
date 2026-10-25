from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

from pydantic import BaseModel, ValidationError, field_validator

from resume_search import taxonomy as tx
from resume_search.parse.client import _validation_lines
from resume_search.parse.models import Approach
from resume_search.search.llm import structured_call

Mode = Literal["must", "preferred"]
BANDS = ("Intern/Student", "Junior", "Mid", "Senior", "Principal")
DEGREES = ("Bachelors", "Masters", "MBA", "PhD")
APPROACHES = tuple(a.value for a in Approach if a is not Approach.unclear)


class Criterion[T](BaseModel):
    value: T
    mode: Mode = "preferred"


@dataclass(frozen=True)
class Place:
    kind: Literal["country", "region"]
    code: str


def resolve_location(value: str) -> Place:
    """ISO-2 code first (so "US" is the country), then a region name, then a country name or
    alias such as "United Kingdom" or "London"."""
    text = value.strip()
    if text.upper() in tx.iso2_codes():
        return Place("country", text.upper())
    for region in tx._load("countries")["regions"]:
        if region.casefold() == text.casefold():
            return Place("region", region)
    for iso, c in tx.countries().items():
        if text.casefold() in (n.casefold() for n in (c["name"], *c.get("aliases", []))):
            return Place("country", iso)
    raise ValueError(f"unknown location {value!r}")


def _pick(value: str, allowed: tuple[str, ...] | list[str], what: str) -> str:
    for a in allowed:
        if a.casefold() == value.strip().casefold():
            return a
    raise ValueError(f"unknown {what} {value!r}; allowed: {', '.join(allowed)}")


def _values(c: Criterion | None) -> list[str]:
    if c is None:
        return []
    return [c.value] if isinstance(c.value, str) else list(c.value)


class Requisition(BaseModel):
    free_text: str | None = None
    location: Criterion[list[str]] | None = None
    location_scope: Literal["current", "worked", "either"] = "either"
    approach: Criterion[str] | None = None
    sectors: Criterion[list[str]] | None = None
    markets_covered: Criterion[list[str]] | None = None
    years_min: float | None = None
    years_max: float | None = None
    years_mode: Mode = "must"
    seniority_bands: Criterion[list[str]] | None = None
    skills_required: list[str] = []
    skills_preferred: list[str] = []
    degree_min: Criterion[str] | None = None
    certifications: Criterion[list[str]] | None = None
    languages: Criterion[list[str]] | None = None
    asset_classes: Criterion[list[str]] | None = None

    @field_validator("location", mode="before")
    @classmethod
    def _location_list(cls, c):
        """A single place, as older requisitions and the CLI give it, is a one-item list."""
        if isinstance(c, Criterion) and isinstance(c.value, str):
            return {"value": [c.value], "mode": c.mode}
        if isinstance(c, dict) and isinstance(c.get("value"), str):
            return {**c, "value": [c["value"]]}
        return c

    @field_validator("location")
    @classmethod
    def _location(cls, c):
        if c is not None:
            for v in c.value:
                resolve_location(v)
        return c

    @field_validator("approach")
    @classmethod
    def _approach(cls, c):
        return c and Criterion[str](value=_pick(c.value, APPROACHES, "approach"), mode=c.mode)

    @field_validator("degree_min")
    @classmethod
    def _degree(cls, c):
        return c and Criterion[str](value=_pick(c.value, DEGREES, "degree"), mode=c.mode)

    @field_validator("sectors", "seniority_bands", "asset_classes")
    @classmethod
    def _enum_list(cls, c, info):
        if c is None:
            return c
        allowed = {
            "sectors": tx.sectors(), "seniority_bands": list(BANDS), "asset_classes": tx.asset_classes()
        }[info.field_name]
        return Criterion[list[str]](
            value=[_pick(v, allowed, info.field_name) for v in c.value], mode=c.mode
        )

    def is_empty(self) -> bool:
        return not (
            (self.free_text or "").strip()
            or self.years_min is not None
            or self.years_max is not None
            or self.skills_required
            or self.skills_preferred
            or any(
                _values(c)
                for c in (
                    self.location, self.approach, self.sectors, self.markets_covered,
                    self.seniority_bands, self.degree_min, self.certifications,
                    self.languages, self.asset_classes,
                )
            )
        )


Field = Literal[
    "location", "approach", "sectors", "markets_covered", "seniority_bands", "degree_min",
    "certifications", "languages", "asset_classes",
]
_SINGLE = ("approach", "degree_min")


class DraftCriterion(BaseModel):
    field: Field
    values: list[str]
    mode: Mode


class RequisitionDraft(BaseModel):
    """What the model fills. Requisition's nested optional generics exceed the structured-output
    schema limit, so the model gets this flat list and `to_requisition` builds the real thing."""

    criteria: list[DraftCriterion] = []
    location_scope: Literal["current", "worked", "either"] = "either"
    years_min: float | None = None
    years_max: float | None = None
    years_mode: Mode = "must"
    skills_required: list[str] = []
    skills_preferred: list[str] = []

    def to_requisition(self, free_text: str) -> Requisition:
        kw: dict = {}
        for c in self.criteria:
            if not c.values:
                continue
            kw[c.field] = (
                Criterion[str](value=c.values[0], mode=c.mode)
                if c.field in _SINGLE
                else Criterion[list[str]](value=c.values, mode=c.mode)
            )
        return Requisition(
            free_text=free_text, location_scope=self.location_scope, years_min=self.years_min,
            years_max=self.years_max, years_mode=self.years_mode,
            skills_required=self.skills_required, skills_preferred=self.skills_preferred, **kw,
        )


_SYSTEM = """\
You turn a pasted job description or keyword string from a hedge fund recruiter into a
search requisition. The reply is constrained to the output schema.

Rules
- Fill only what the text states. Leave everything else null or empty. Never infer a
  criterion from the firm, the title alone, or common practice.
- Every filled criterion has mode "preferred", except skills explicitly called required or
  must-have (put those in skills_required) and an explicit years-of-experience range or
  minimum, which gets years_mode "must". A years figure hedged as "ideally" or "about" is
  "preferred". Skills described as a plus, nice to have, or preferred go in skills_preferred.
- criteria holds one entry per filled criterion: field, values, mode. approach and
  degree_min take exactly one value; the other fields take a list. Omit a field the text does
  not mention.
- location: ISO-2 codes or region names, any of which satisfies the role. Codes: {iso2}.
  Regions: {regions}.
  location_scope is "current" when the text says based in, located in, or must be in a
  place, otherwise "either". "worked" only when it asks for work history in a place.
- approach: one of {approaches}.
- sectors: from {sectors}.
- asset_classes: from {asset_classes}.
- seniority_bands: from {bands}, only when the text names a level such as junior or senior.
- degree_min: one of {degrees}, the lowest acceptable.
- certifications: names as written, for example "CFA".
- markets_covered: ISO-2 codes or market labels ({market_labels}) for geographies the
  candidate must have covered.
- Certifications such as the CFA go in certifications only, never in the skill lists.
- skills_required and skills_preferred: use the canonical skill names where one fits:
  {skills}. Otherwise use the short lowercase name from the text.
- years_min and years_max: numbers of years. "2-5 years" is 2 and 5. "5+ years" is
  years_min 5 and years_max null.
"""


@cache
def system_prompt() -> str:
    skills = sorted({v["canonical"] for v in tx.skill_variants().values()})
    return _SYSTEM.format(
        iso2=", ".join(sorted(tx.iso2_codes())),
        regions=", ".join(tx._load("countries")["regions"]),
        approaches=", ".join(APPROACHES),
        sectors=", ".join(tx.sectors()),
        asset_classes=", ".join(tx.asset_classes()),
        bands=", ".join(BANDS),
        degrees=", ".join(DEGREES),
        market_labels=", ".join(sorted(tx.market_labels())),
        skills=", ".join(skills),
    )


def _convert_errors(draft: RequisitionDraft, text: str) -> list[str]:
    try:
        draft.to_requisition(text)
    except ValidationError as exc:
        return _validation_lines(exc)
    return []


def parse_requisition(text: str, client=None, attempts: list | None = None) -> Requisition:
    draft = structured_call(
        client, system_prompt(), f"<job_description>\n{text}\n</job_description>",
        RequisitionDraft, validate=lambda d: _convert_errors(d, text), attempts=attempts,
        call="requisition",
    )
    return draft.to_requisition(text)
