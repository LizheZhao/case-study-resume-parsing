"""Cross-verification of LLM output against deterministic text parsing (design 5.5).

The registry CHECKS is data: adding a check is adding a row. A row whose `field` contains
`[i]` runs once per list item; `anchor_key` names the sibling key (employer, institution) whose
value lets `extract` find that item in the text. `anchor_key="$"` hands the extractor the whole
doc instead.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import cache
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from resume_search import taxonomy
from resume_search.parse.models import (
    ExperienceList,
    Labeled,
    Profile,
    _norm,
)

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Found:
    value: Any
    snippet: str = ""


@dataclass(frozen=True)
class Check:
    field: str
    extract: Callable[[str, str, str | None], Found]
    compare: Callable[[Any, Any], bool]
    regen_target: str
    anchor_key: str | None = None
    applies: Callable[[dict], bool] | None = None


@dataclass(frozen=True)
class Disagreement:
    field: str
    llm_value: Any
    text_value: Any
    evidence_snippet: str
    regen_target: str
    index: int | None = None


@dataclass(frozen=True)
class VerificationReport:
    checks_run: int
    disagreements: list[Disagreement] = field(default_factory=list)


def _snip(text: str, start: int, end: int, pad: int = 50) -> str:
    return _WS.sub(" ", text[max(0, start - pad) : end + pad]).strip()


def _plain(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# name ---------------------------------------------------------------------------------------

_FILENAME_NOISE = {
    "resume", "cv", "curriculum", "vitae", "cfa", "frm", "caia", "cpa", "acca", "final", "updated",
}


def _name_candidates(text: str, filename: str) -> list[str]:
    stem = re.sub(r"[_\-.,()\[\]]+", " ", Path(filename).stem)
    tokens = [
        t for t in stem.split()
        if t.lower() not in _FILENAME_NOISE and not re.fullmatch(r"\d{4,8}", t)
    ]
    cands = [" ".join(tokens)] if tokens else []
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if first:
        cands.append(first)
    return cands


def _extract_name(text: str, filename: str, anchor: str | None = None) -> Found:
    cands = _name_candidates(text, filename)
    return Found(cands, " | ".join(cands))


def _compare_name(llm: str | None, cands: list[str]) -> bool:
    if not llm:
        return False
    return any(fuzz.token_set_ratio(_plain(llm), _plain(c)) >= 80 for c in cands)


# email and phone ----------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)*")
_NOT_PHONE_LABEL = re.compile(r"(?:issn|isbn|doi|pmid)\W*$", re.IGNORECASE)
_PHONE = re.compile(r"(?<![\w.])(?:\+|00)?\d[\d ()\t.\-]{5,}\d(?![\w])")


def _extract_email(text: str, filename: str, anchor: str | None = None) -> Found:
    matches = list(_EMAIL.finditer(text))
    emails = sorted({m.group().lower() for m in matches})
    return Found(emails, _snip(text, matches[0].start(), matches[0].end()) if matches else "")


def _compare_email(llm: str | None, emails: list[str]) -> bool:
    if llm is None:
        return not emails
    return llm.strip().lower() in emails


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _is_year_run(candidate: str) -> bool:
    groups = re.findall(r"\d+", candidate)
    return all(re.fullmatch(r"(?:19|20)\d{2}", g) for g in groups)


def _phone_candidates(text: str) -> list[re.Match]:
    out = []
    for m in _PHONE.finditer(text):
        n = len(_digits(m.group()))
        labelled = _NOT_PHONE_LABEL.search(text[max(0, m.start() - 8) : m.start()])
        if 7 <= n <= 15 and not _is_year_run(m.group()) and not labelled:
            out.append(m)
    return out


def _extract_phone(text: str, filename: str, anchor: str | None = None) -> Found:
    matches = _phone_candidates(text)
    nums = [m.group().strip() for m in matches]
    return Found(nums, _snip(text, matches[0].start(), matches[0].end()) if matches else "")


def _compare_phone(llm: str | None, nums: list[str]) -> bool:
    if llm is None:
        return not nums
    return _digits(llm) in {_digits(n) for n in nums}


# certifications -----------------------------------------------------------------------------

_CERTS: dict[str, re.Pattern] = {
    "CFA": re.compile(
        r"\bCFA\b(?!\s+(?:Institute|Society|Research|Challenge))|Chartered Financial Analyst",
    ),
    "FRM": re.compile(r"\bFRM\b|Financial Risk Manager"),
    "CAIA": re.compile(r"\bCAIA\b|Chartered Alternative Investment Analyst"),
    "CPA": re.compile(r"\bCPA\b|Certified Public Accountant"),
    "ACCA": re.compile(r"\bACCA\b"),
}


def _cert_tokens(s: str) -> set[str]:
    return {name for name, rx in _CERTS.items() if rx.search(s)}


def _extract_certs(text: str, filename: str, anchor: str | None = None) -> Found:
    tokens = _cert_tokens(text) | _cert_tokens(Path(filename).stem)
    snippet = ""
    for name in sorted(tokens):
        m = _CERTS[name].search(text)
        if m:
            snippet = _snip(text, m.start(), m.end())
            break
    return Found(sorted(tokens), snippet or Path(filename).stem)


def _compare_certs(llm: list[str], tokens: list[str]) -> bool:
    llm_tokens: set[str] = set()
    for c in llm:
        llm_tokens |= _cert_tokens(c)
    return llm_tokens == set(tokens)


# skills -------------------------------------------------------------------------------------


@cache
def _skill_matchers() -> tuple[re.Pattern, re.Pattern, dict[str, str]]:
    """Word-boundary regexes over every skills.yaml variant, built once. One- and two-letter
    variants and 'wind' match uppercase only, so 'R', 'ML' and 'WIND' hit but 'r' and 'wind' do not."""
    canon: dict[str, str] = {}
    loose: list[str] = []
    strict: list[str] = []
    for variant, info in taxonomy.skill_variants().items():
        if info["category"] == "cert":
            continue
        canon[variant] = info["canonical"]
        body = r"\s+".join(re.escape(w) for w in variant.split())
        (strict if len(variant) <= 2 or variant == "wind" else loose).append(body)
    wrap = r"(?<!\w)(?:{})(?!\w)"
    loose_rx = re.compile(wrap.format("|".join(sorted(loose, key=len, reverse=True))), re.IGNORECASE)
    upper = [s.upper() for s in strict]
    strict_rx = re.compile(wrap.format("|".join(sorted(upper, key=len, reverse=True))))
    return loose_rx, strict_rx, canon


def _skill_hits(s: str) -> dict[str, re.Match]:
    loose_rx, strict_rx, canon = _skill_matchers()
    hits: dict[str, re.Match] = {}
    for rx in (loose_rx, strict_rx):
        for m in rx.finditer(s):
            key = _WS.sub(" ", m.group().lower())
            if key in canon:
                hits.setdefault(canon[key], m)
    return hits


def _extract_skills(text: str, filename: str, anchor: str | None = None) -> Found:
    hits = _skill_hits(text)
    first = next(iter(hits.values()), None)
    return Found(sorted(hits), _snip(text, first.start(), first.end()) if first else "")


def _compare_skills(llm: dict, matched: list[str]) -> bool:
    _, _, canon = _skill_matchers()
    have: set[str] = set()
    for group in ("programming", "tools", "domain"):
        for item in llm.get(group, []):
            low = _WS.sub(" ", item.lower().strip())
            if low in canon:
                have.add(canon[low])
            have |= set(_skill_hits(item))
    return set(matched) <= have


# date grounding -----------------------------------------------------------------------------

_MONTH_WORDS: dict[int, tuple[str, ...]] = {
    1: ("jan", "january", "janvier", "janeiro"),
    2: ("feb", "february", "fev", "fevrier", "fevereiro"),
    3: ("mar", "march", "mars", "marco"),
    4: ("apr", "april", "avr", "avril", "abr", "abril"),
    5: ("may", "mai", "maio"),
    6: ("jun", "june", "juin", "junho"),
    7: ("jul", "july", "juil", "juillet", "julho"),
    8: ("aug", "august", "aout", "ago", "agosto"),
    9: ("sep", "sept", "september", "septembre", "set", "setembro"),
    10: ("oct", "october", "octobre", "out", "outubro"),
    11: ("nov", "november", "novembre", "novembro"),
    12: ("dec", "december", "decembre", "dez", "dezembro"),
}
_MONTH_NUM = {w: n for n, ws in _MONTH_WORDS.items() for w in ws}
_LETTERS = r"[A-Za-zÀ-ÿ]"
_MONTH_YEAR = re.compile(
    rf"(?<!{_LETTERS})(?P<mon>{_LETTERS}{{3,10}})\.?[\s,\-–/'’‘.]*(?P<yr>\d{{4}}|\d{{2}})(?!\d)"
)
_MONTH_DAY_YEAR = re.compile(
    rf"(?<!{_LETTERS})(?P<mon>{_LETTERS}{{3,10}})\.?\s*\d{{1,2}}(?:st|nd|rd|th)?,?\s*(?P<yr>\d{{4}})(?!\d)"
)
_NUM_MY = re.compile(r"(?<!\d)(?P<m>0?[1-9]|1[0-2])\s*[/.\-]\s*(?P<y>(?:19|20)\d{2})(?!\d)")
_NUM_YM = re.compile(r"(?<!\d)(?P<y>(?:19|20)\d{2})\s*[/.\-]\s*(?P<m>0?[1-9]|1[0-2])(?!\d)")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _full_year(s: str) -> int:
    if len(s) == 4:
        return int(s)
    yy = int(s)
    return 2000 + yy if yy <= 69 else 1900 + yy


def grounded_dates(text: str) -> tuple[set[str], set[str]]:
    """(YYYY-MM tokens, YYYY tokens) the text can justify. Month names, 'Sept.2017',
    "May'22", "Oct-'20", 'Jun 2021', '03/2020' and '2020-03' ground a month; any year ground a year."""
    months: set[str] = set()
    for rx in (_MONTH_YEAR, _MONTH_DAY_YEAR):
        for m in rx.finditer(text):
            n = _MONTH_NUM.get(_plain(m["mon"]))
            if n:
                months.add(f"{_full_year(m['yr'])}-{n:02d}")
    for rx in (_NUM_MY, _NUM_YM):
        for m in rx.finditer(text):
            months.add(f"{m['y']}-{int(m['m']):02d}")
    years = {y.group() for y in _YEAR.finditer(text)} | {mo[:4] for mo in months}
    return months, years


def _extract_dates(text: str, filename: str, anchor: str | None = None) -> Found:
    months, years = grounded_dates(text)
    return Found({"months": sorted(months), "years": sorted(years)}, "")


def _compare_date(llm: list[str | None], grounded: dict) -> bool:
    """Every stint date must be grounded; a null end is a current role and has nothing to ground."""
    return all(
        d is None or d in grounded["months" if len(d) == 7 else "years"] for d in llm
    )


# duration -----------------------------------------------------------------------------------

_DURATION = re.compile(
    r"(\d+)\s*(years?|yrs?)(?:\s*(\d+)\s*months?)?|(\d+)\s*months?|(\d+)\s*weeks?", re.IGNORECASE
)
_ORG_SUFFIX = re.compile(
    r"\b(?:ltd|limited|inc|llc|llp|plc|pvt|private|corp|corporation|co|company|pte|gmbh)\b\.?",
    re.IGNORECASE,
)


def duration_months(s: str) -> int | None:
    """First duration phrase in `s` as whole months; weeks are floored to a quarter-month."""
    m = _DURATION.search(s)
    return _months_of(m) if m else None


def _months_of(m: re.Match) -> int:
    if m.group(1):
        return int(m.group(1)) * 12 + int(m.group(3) or 0)
    if m.group(4):
        return int(m.group(4))
    return int(m.group(5)) // 4


def _employer_forms(employer: str) -> list[str]:
    forms = [_WS.sub(" ", employer.lower()).strip()]
    short = _WS.sub(" ", _ORG_SUFFIX.sub("", employer.lower())).strip(" .,")
    if short and short not in forms:
        forms.append(short)
    return forms


def _extract_duration(text: str, filename: str, anchor: str | None = None) -> Found:
    if not anchor:
        return Found(None)
    for form in _employer_forms(anchor):
        for line in text.splitlines():
            flat = _WS.sub(" ", line.lower())
            pos = flat.find(form)
            if pos < 0:
                continue
            ms = list(_DURATION.finditer(line))
            if ms:
                nearest = min(ms, key=lambda m: abs(m.start() - pos))
                return Found(_months_of(nearest), _WS.sub(" ", line).strip()[:160])
    return Found(None)


def _compare_duration(llm: int | None, text_months: int | None) -> bool:
    if text_months is None:
        return True
    return llm is not None and abs(llm - text_months) <= 1


def _extract_stint_months(text: str, filename: str, anchor: int | None = None) -> Found:
    """The role's months from its stint dates, computed by code and handed over as the anchor."""
    return Found(anchor)


# role coverage on duration-only resumes -----------------------------------------------------

_HEADER_DURATION = re.compile(
    r"(?:\b(?:Duration|Others?)\s*[|:]\s*|\(\s*)(?:" + _DURATION.pattern + ")", re.IGNORECASE
)


def _extract_header_durations(text: str, filename: str, anchor: str | None = None) -> Found:
    """Durations written as a role header ('Duration | 7 months', 'Others: 8 months', 'Project
    (8 weeks):'). Free
    prose such as 'within 3 months of joining' is not a header, so it never counts as a role."""
    ms = list(_HEADER_DURATION.finditer(text))
    return Found([_months_of(m) for m in ms], _snip(text, ms[0].start(), ms[0].end()) if ms else "")


def _compare_role_coverage(roles: list[dict], header_months: list[int]) -> bool:
    """On a duration-only resume every header duration must belong to some LLM role. This is the
    experience-count row: employer lines have no reliable pattern, duration headers do."""
    if not header_months or all(r["start"] for r in roles):
        return True
    pool = [r["duration_months"] for r in roles if r["duration_months"] is not None]
    for m in header_months:
        hit = next((p for p in pool if abs(p - m) <= 1), None)
        if hit is None:
            return False
        pool.remove(hit)
    return True


# degree level -------------------------------------------------------------------------------

_A = r"(?<![\w.])"  # acronyms must not sit inside a longer dotted form such as 'B.M.S'
_NOT_STATE = r"(?<!,\s)"  # 'Cambridge, MA' is a state, not a Master of Arts
_DEGREE_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(rf"{_A}M\.?B\.?B\.?S\b"), "Other"),
    (re.compile(rf"{_A}MBA\b|Master of Business Administration"), "MBA"),
    (
        re.compile(
            rf"{_A}PGDM\b|{_A}PGP\b|Post[- ]?Graduate Diploma in Management", re.IGNORECASE
        ),
        "Masters",
    ),
    (re.compile(rf"{_A}Ph\.?\s?D\b|\bDPhil\b|Doctor(?:ate)? of Philosophy"), "PhD"),
    (
        re.compile(
            rf"{_A}M\.?Sc\b|{_NOT_STATE}{_A}M\.?S\b|{_NOT_STATE}{_A}M\.?A\b|\bMEng\b"
            rf"|{_A}M\.?Tech\b|\bMFE\b|\bMFin\b|\bMPhil\b|\bMSF\b|\bLLM\b|\bMasters?\b"
            r"|\bMaster['’]s\b|Dipl[oô]me d['’]Ing[eé]nieur"
        ),
        "Masters",
    ),
    (
        re.compile(
            rf"{_A}B\.?Sc\b|{_A}B\.?S\b|{_A}B\.?A\b|\bBBA\b|\bBEng\b|{_A}B\.E\b"
            rf"|{_A}B\.?Tech\b|{_A}B\.?Com\b|{_A}B\.?M\.?S\b|\bBEcon\b|\bBFin\b"
            r"|\bBachelors?\b|\bBachelor['’]s\b"
        ),
        "Bachelors",
    ),
)


def _level_key(level: str | None) -> str | None:
    if level is None:
        return None
    return level.lower().replace("'", "").replace("’", "").rstrip("s")


def _extract_degree(text: str, filename: str, anchor: str | None = None) -> Found:
    if not anchor:
        return Found(None)
    needle = _WS.sub(" ", anchor.lower()).strip()
    for line in text.splitlines():
        if needle not in _WS.sub(" ", line.lower()):
            continue
        levels = {lvl for rx, lvl in _DEGREE_RULES if rx.search(line)}
        if len(levels) == 1:
            return Found(levels.pop(), _WS.sub(" ", line).strip()[:160])
    return Found(None)


def _compare_degree(llm: str | None, text_level: str | None) -> bool:
    if text_level is None:
        return True
    return _level_key(llm) == _level_key(text_level)


# country from phone prefix ------------------------------------------------------------------


def _prefix_countries(number: str) -> list[str]:
    n = number.strip()
    if n.startswith("00"):
        n = "+" + n[2:]
    if not n.startswith("+"):
        return []
    digits = "+" + _digits(n)
    for prefix in sorted(taxonomy.phone_prefixes(), key=len, reverse=True):
        if digits.startswith(prefix):
            return taxonomy.phone_prefixes()[prefix]
    return []


def _extract_phone_country(text: str, filename: str, anchor: str | None = None) -> Found:
    for m in _phone_candidates(text):
        iso = _prefix_countries(m.group())
        if iso:
            return Found(iso, _snip(text, m.start(), m.end()))
    return Found([])


def _compare_country(llm: str | None, iso: list[str]) -> bool:
    return not iso or llm is None or llm in iso


# investment approach ------------------------------------------------------------------------


def _extract_approach_role(text: str, filename: str, doc: dict | None = None) -> Found:
    """Which roles' title, employer or bullets contain the approach quote, after the same
    whitespace and quote folding check_evidence uses."""
    quote = _norm(doc["investment_approach"]["evidence"]["quote"])
    roles = [
        {"employer": e["employer"], "title": e["title"],
         "employment_type": e["employment_type"], "relevance": e["relevance"]}
        for e in doc["experience"]
        if quote and quote in _norm(" ".join([e["title"], e["employer"], *e["bullets"]]))
    ]
    snippet = "; ".join(f"{r['title']} at {r['employer']}" for r in roles)
    return Found(roles, snippet)


def _compare_approach(llm: dict, roles: list[dict]) -> bool:
    """Fundamental, Systematic or Both needs a quote from a full-time core or adjacent role."""
    return llm["label"] == "Unclear" or any(
        r["employment_type"] == "full_time" and r["relevance"] in ("core", "adjacent")
        for r in roles
    )


# registry -----------------------------------------------------------------------------------

CHECKS: tuple[Check, ...] = (
    Check("name", _extract_name, _compare_name, "name"),
    Check("contact.email", _extract_email, _compare_email, "contact"),
    Check("contact.phone", _extract_phone, _compare_phone, "contact"),
    Check("certifications", _extract_certs, _compare_certs, "certifications"),
    Check("skills", _extract_skills, _compare_skills, "skills"),
    Check("experience[i].start", _extract_dates, _compare_date, "experience_item"),
    Check("experience[i].end", _extract_dates, _compare_date, "experience_item"),
    Check(
        "experience[i].duration_months", _extract_duration, _compare_duration,
        "experience_item", anchor_key="employer",
        # With start dates the derive layer ignores duration, and a duration near the employer
        # name may be a career total ("covering healthcare for 9 years").
        applies=lambda e: not e["start"] or e["duration_months"] is not None,
    ),
    Check(
        "experience[i].duration_months", _extract_stint_months, _compare_duration,
        "experience_item", anchor_key="stint_months",
        applies=lambda e: bool(e["start"]) and e["duration_months"] is not None,
    ),
    Check("experience", _extract_header_durations, _compare_role_coverage, "experience"),
    Check(
        "education[i].degree_level", _extract_degree, _compare_degree,
        "education_item", anchor_key="institution",
    ),
    Check(
        "derived.current_country", _extract_phone_country, _compare_country, "experience_item"
    ),
    Check(
        "investment_approach", _extract_approach_role, _compare_approach, "assessment",
        anchor_key="$", applies=lambda doc: "investment_approach" in doc,
    ),
)


def current_role_index(experiences: list[dict]) -> int | None:
    """The is_current role with the latest start, else the role with the latest end or start."""
    if not experiences:
        return None
    current = [i for i, e in enumerate(experiences) if e["is_current"]]
    def latest_start(i: int) -> str:
        return experiences[i]["start"][-1] if experiences[i]["start"] else ""

    if current:
        return max(current, key=latest_start)
    return max(range(len(experiences)), key=lambda i: (
        (experiences[i]["end"] or experiences[i]["start"] or [""])[-1] or ""))


def _walk(doc: Any, path: str) -> Any:
    for part in path.replace("]", "").replace("[", ".").split("."):
        doc = doc[int(part)] if part.isdigit() else doc[part]
    return doc


def build_doc(
    profile: Profile,
    experiences: ExperienceList | list,
    approach: Labeled | None = None,
    today: date | None = None,
) -> dict:
    exps = experiences.experience if isinstance(experiences, ExperienceList) else experiences
    today = today or date.today()
    doc = profile.model_dump(mode="json")
    doc["experience"] = [e.model_dump(mode="json") for e in exps]
    for item, e in zip(doc["experience"], exps, strict=True):
        item["stint_months"] = e.months(today) if e.start else None
    if approach is not None:
        doc["investment_approach"] = approach.model_dump(mode="json")
    idx = current_role_index(doc["experience"])
    doc["derived"] = {"current_country": doc["experience"][idx]["country"] if idx is not None else None}
    return doc


def verify(
    profile: Profile,
    experiences: ExperienceList | list,
    text: str,
    filename: str,
    only: frozenset[str] | None = None,
    approach: Labeled | None = None,
) -> VerificationReport:
    """Run every registry row. `only` restricts to concrete field paths such as
    'experience[2].start', for re-checking one regenerated group. The approach row runs only
    when `approach` is given."""
    doc = build_doc(profile, experiences, approach)
    cur = current_role_index(doc["experience"])
    run = 0
    found: list[Disagreement] = []
    for check in CHECKS:
        if "[i]" in check.field:
            head = check.field.split("[i]")[0]
            targets: list[tuple[str, int | None, str | None]] = []
            for i, item in enumerate(_walk(doc, head)):
                if check.applies and not check.applies(item):
                    continue
                anchor = item[check.anchor_key] if check.anchor_key else None
                targets.append((check.field.replace("[i]", f"[{i}]"), i, anchor))
        else:
            if check.applies and not check.applies(doc):
                continue
            index = cur if check.regen_target == "experience_item" else None
            targets = [(check.field, index, doc if check.anchor_key == "$" else None)]
        for path, index, anchor in targets:
            if only is not None and path not in only:
                continue
            run += 1
            llm = _walk(doc, path)
            got = check.extract(text, filename, anchor)
            if not check.compare(llm, got.value):
                shown = [(r["employer"], r["duration_months"]) for r in llm] if path == "experience" else llm
                found.append(
                    Disagreement(path, shown, got.value, got.snippet, check.regen_target, index)
                )
    return VerificationReport(run, found)
