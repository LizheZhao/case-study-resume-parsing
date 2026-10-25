from __future__ import annotations

from datetime import date

from resume_search.derive import experience as exp
from resume_search.derive import location, skills
from resume_search.derive.models import (
    CandidateRecord,
    CountryMonths,
    DerivedFields,
    SectorEvidence,
)
from resume_search.parse.models import ParsedResume
from resume_search.taxonomy import Taxonomy

__all__ = ["CandidateRecord", "DerivedFields", "derive"]

_DEGREE_ORDER = {"Bachelors": 1, "Masters": 2, "MBA": 3, "PhD": 4}


def _highest_degree(parsed: ParsedResume) -> str | None:
    levels = [e.degree_level for e in parsed.education if e.degree_level]
    if not levels:
        return None
    return max(levels, key=lambda lv: _DEGREE_ORDER.get(lv, 0))


def _dedupe(values) -> list[str]:
    return list(dict.fromkeys(values))


def _sectors(parsed: ParsedResume, tax: Taxonomy) -> tuple[list[SectorEvidence], list[str]]:
    found: dict[str, str] = {}
    for e in parsed.experience:
        for s in e.sectors:
            found.setdefault(s.label, s.evidence.quote)
    tags = sorted({t.strip().lower() for e in parsed.experience for t in e.other_sector_tags if t.strip()})
    for tag in tags:
        if core := tax.tag_to_core.get(tag):
            found.setdefault(core, tag)
    return [SectorEvidence(label=k, evidence=v) for k, v in found.items()], tags


def _today(meta: dict) -> date:
    parsed_at = meta.get("parsed_at")
    return date.fromisoformat(parsed_at[:10]) if parsed_at else date.today()


def derive(
    parsed: ParsedResume, meta: dict, taxonomy: Taxonomy, *, today: date | None = None
) -> DerivedFields:
    today = today or _today(meta)
    roles = parsed.experience
    relevant = exp.total_months([e for e in roles if exp.is_relevant_full_time(e)], today)
    country = location.current_country(parsed, taxonomy)
    worked = location.countries_worked(parsed, today)
    sectors, tags = _sectors(parsed, taxonomy)
    has_cfa, cfa_level = skills.cfa_status(parsed.certifications)
    verification = meta.get("verification") or {}
    return DerivedFields(
        current_country=country,
        current_region=taxonomy.country_region.get(country) if country else None,
        countries_worked=[CountryMonths(country=c, months=m) for c, m in worked],
        regions_worked=_dedupe(
            taxonomy.country_region[c] for c, _ in worked if c in taxonomy.country_region
        ),
        markets_covered=location.markets_covered(parsed, taxonomy),
        months_full_time_relevant=relevant,
        months_full_time_core=exp.total_months([e for e in roles if exp.is_core_full_time(e)], today),
        months_internship=exp.total_months([e for e in roles if exp.is_internship(e)], today),
        seniority_band=exp.seniority_band(relevant),
        investment_approach=parsed.investment_approach.label,
        approach_evidence=parsed.investment_approach.evidence.quote,
        sectors=sectors,
        other_sector_tags=tags,
        asset_classes=_dedupe(a.value for e in roles for a in e.asset_classes),
        firm_types=_dedupe(e.firm_type.value for e in roles),
        highest_degree=_highest_degree(parsed),
        degree_fields=_dedupe(e.field for e in parsed.education if e.field),
        skills_normalized=skills.skills_normalized(parsed, taxonomy),
        has_cfa=has_cfa,
        cfa_level=cfa_level,
        search_summary=parsed.search_summary,
        quality={
            "text_quality": meta.get("text_quality"),
            "skill_disagreement_count": 0,
            "unresolved_count": len(verification.get("unresolved", [])),
        },
    )
