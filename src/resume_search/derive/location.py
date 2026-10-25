from __future__ import annotations

import re
from datetime import date

from resume_search.derive.experience import month_index, total_months
from resume_search.parse.models import Contact, EmploymentType, ParsedResume
from resume_search.taxonomy import Taxonomy


def _gazetteer(text: str, tax: Taxonomy) -> str | None:
    """Match comma-separated parts of a free-text location against country names and aliases,
    last part first ('Mumbai, India' hits the country before the city)."""
    parts = [p.strip().lower() for p in reversed(text.split(",")) if p.strip()]
    for part in parts:
        for iso, names in tax.country_names.items():
            if part in (n.lower() for n in names):
                return iso
    return None


def _from_phone(phone: str, tax: Taxonomy) -> str | None:
    compact = re.sub(r"[^\d+]", "", phone)
    if not compact.startswith("+"):
        return None
    for prefix in sorted(tax.phone_prefix, key=len, reverse=True):
        if compact.startswith(prefix):
            isos = tax.phone_prefix[prefix]
            return isos[0] if len(isos) == 1 else None
    return None


def _contact_country(contact: Contact, tax: Taxonomy) -> str | None:
    if contact.location_text and (iso := _gazetteer(contact.location_text, tax)):
        return iso
    if contact.phone:
        return _from_phone(contact.phone, tax)
    return None


def current_country(parsed: ParsedResume, tax: Taxonomy) -> str | None:
    """Country of the current role with the latest start, else of the latest role by start
    (first listed when nothing is dated). Roles with no stated country are skipped, then the
    contact block decides."""
    roles = parsed.experience

    def start_key(i: int) -> int:
        s = roles[i].start
        return month_index(s) if s else -1

    order = sorted(range(len(roles)), key=lambda i: (start_key(i), -i), reverse=True)
    current = [i for i in order if roles[i].is_current]
    for i in current + order:
        if roles[i].country:
            return roles[i].country
    return _contact_country(parsed.contact, tax)


def countries_worked(parsed: ParsedResume, today: date) -> list[tuple[str, int]]:
    by_country: dict[str, list] = {}
    for e in parsed.experience:
        if e.employment_type is EmploymentType.full_time and e.country:
            by_country.setdefault(e.country, []).append(e)
    return [(c, total_months(roles, today)) for c, roles in by_country.items()]


def markets_covered(parsed: ParsedResume, tax: Taxonomy) -> list[str]:
    """Labels as stated, plus the countries and regions each label expands to."""
    out: set[str] = set()
    for e in parsed.experience:
        for m in e.markets_covered:
            out.add(m.label)
            countries, regions = tax.market_expansion.get(m.label, (frozenset(), frozenset()))
            out |= countries | regions
    return sorted(out)
