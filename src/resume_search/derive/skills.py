from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from resume_search.parse.models import ParsedResume
from resume_search.taxonomy import Taxonomy

FUZZY_THRESHOLD = 90
# Short tokens ("sas", "sap", "r") are one edit from unrelated skills; only exact-match them.
FUZZY_MIN_LEN = 5

_CFA = re.compile(r"\bCFA\b|chartered financial analyst", re.IGNORECASE)
_CFA_PENDING = re.compile(r"\blevel\b|candidate|pursuing|in progress|enrolled|passed", re.IGNORECASE)
_CFA_LEVEL = re.compile(r"\blevel\s*(III|II|I|3|2|1)\b", re.IGNORECASE)
_ARABIC = {"1": "I", "2": "II", "3": "III"}


def normalize_skill(raw: str, tax: Taxonomy) -> str:
    key = raw.strip().lower()
    if key in tax.skill_canonical:
        return tax.skill_canonical[key]
    if len(key) >= FUZZY_MIN_LEN:
        hit = process.extractOne(
            key, tax.skill_canonical.keys(), scorer=fuzz.ratio, score_cutoff=FUZZY_THRESHOLD
        )
        if hit:
            return tax.skill_canonical[hit[0]]
    return key


_COMPOUND = re.compile(r"\s*/\s*")


def split_compound(raw: str, tax: Taxonomy) -> list[str]:
    """"C# / .NET" names two skills; split only when every part is a known variant,
    so a single skill whose name contains a slash (e.g. "long/short") stays whole."""
    parts = _COMPOUND.split(raw.strip())
    if len(parts) > 1 and all(p.lower() in tax.skill_canonical for p in parts):
        return parts
    return [raw]


def skills_normalized(parsed: ParsedResume, tax: Taxonomy) -> list[str]:
    s = parsed.skills
    raw = [p for r in [*s.programming, *s.tools, *s.domain] if r.strip() for p in split_compound(r, tax)]
    return sorted({n for r in raw if (n := normalize_skill(r, tax))})


def cfa_status(certifications: list[str]) -> tuple[bool, str | None]:
    """has_cfa is true only for a charterholder or bare CFA; a level or candidacy mention
    makes the entry a candidate, whose level is reported instead."""
    has, level = False, None
    for cert in certifications:
        if not _CFA.search(cert):
            continue
        if m := _CFA_LEVEL.search(cert):
            level = _ARABIC.get(m.group(1), m.group(1).upper())
        if not _CFA_PENDING.search(cert):
            has = True
    return has, level
