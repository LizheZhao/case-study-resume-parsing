from __future__ import annotations

import re
from datetime import date

from resume_search.derive.experience import total_months
from resume_search.parse.models import (
    Education,
    EmploymentType,
    Experience,
    ParsedResume,
    month_index,
)
from resume_search.taxonomy import Taxonomy


_POSTCODE = re.compile(r"\s+\d[\d\s-]*$")


def _gazetteer(text: str, tax: Taxonomy) -> str | None:
    """Match comma-separated parts of a free-text location against country names and aliases,
    last part first ('Mumbai, India' hits the country before the city). A trailing postcode is
    dropped, so 'Boston, MA 01125' matches on 'MA'."""
    parts = [_POSTCODE.sub("", p.strip()).lower() for p in reversed(text.split(",")) if p.strip()]
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


_NOW = 10**9  # sorts a current role after every dated end


def _recency(end: str | None, *, current: bool = False, start: str | None = None) -> tuple[int, int]:
    """Sort key, latest first: current entries by start, then dated entries by end; undated -1."""
    if current:
        return _NOW, month_index(start) if start else -1
    return (month_index(end), 0) if end else (-1, 0)


def current_country(parsed: ParsedResume, tax: Taxonomy) -> str | None:
    """Where the candidate is now. A location stated in the contact block wins. Otherwise the
    most recent work or education entry with a stated country (current roles first, then by
    end date; extracurricular posts skipped). Then the head office of the latest role's firm,
    then the phone prefix."""
    contact = parsed.contact
    if contact.location_text and (iso := _gazetteer(contact.location_text, tax)):
        return iso
    roles = [e for e in parsed.experience if e.employment_type is not EmploymentType.extracurricular]
    dated: list[tuple[tuple[int, int], int, Experience | Education]] = []
    for i, e in enumerate(roles):
        key = _recency(e.end[-1] if e.end else None, current=e.is_current,
                       start=e.start[-1] if e.start else None)
        dated.append((key, -i, e))
    for i, e in enumerate(parsed.education):
        dated.append((_recency(e.end or e.start), -len(roles) - i, e))
    dated.sort(key=lambda t: t[:2], reverse=True)
    if iso := next((e.country for *_, e in dated if e.country), None):
        return iso
    latest_roles = [e for *_, e in dated if isinstance(e, Experience)]
    if iso := next((e.firm_hq_country for e in latest_roles if e.firm_hq_country), None):
        return iso
    return _from_phone(contact.phone, tax) if contact.phone else None


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
