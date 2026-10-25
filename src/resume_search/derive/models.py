"""Data shapes for the derived layer (design 4.2) and the stored record."""

from __future__ import annotations

from pydantic import BaseModel

from resume_search.parse.models import ParsedResume

# Bump when derive() output changes; `rederive` then rebuilds every candidate without API calls.
SCHEMA_VERSION = 2


class CountryMonths(BaseModel):
    country: str
    months: int


class SectorEvidence(BaseModel):
    label: str
    evidence: str


class DerivedFields(BaseModel):
    current_country: str | None
    current_region: str | None
    countries_worked: list[CountryMonths]
    regions_worked: list[str]
    markets_covered: list[str]
    months_full_time_relevant: int
    months_full_time_core: int
    months_internship: int
    seniority_band: str
    investment_approach: str
    approach_evidence: str
    sectors: list[SectorEvidence]
    other_sector_tags: list[str]
    asset_classes: list[str]
    firm_types: list[str]
    highest_degree: str | None
    degree_fields: list[str]
    skills_normalized: list[str]
    has_cfa: bool
    cfa_level: str | None
    search_summary: str
    quality: dict


class CandidateRecord(BaseModel):
    meta: dict
    extracted: ParsedResume
    derived: DerivedFields
