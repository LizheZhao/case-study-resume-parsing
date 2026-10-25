from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from rapidfuzz import fuzz

from resume_search import taxonomy as tx
from resume_search.search.requisition import (
    DEGREES,
    Criterion,
    Place,
    Requisition,
    _values,
    resolve_location,
)
from resume_search.search.weights import Weights


class ChipStatus(StrEnum):
    matched = "matched"
    unmatched = "unmatched"


@dataclass(frozen=True)
class Chip:
    criterion: str
    label: str
    status: ChipStatus
    detail: str | None = None


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    name: str
    search_summary: str
    seniority_band: str | None
    approach: str | None
    months_relevant: int
    highest_degree: str | None
    has_cfa: bool
    current_country: str | None
    current_region: str | None
    sectors: dict[str, str]
    assets: frozenset[str]
    skills: frozenset[str]
    worked: dict[str, int]
    covered: frozenset[str]
    certs: tuple[str, ...]
    languages: frozenset[str]


# Prune ------------------------------------------------------------------------------------------


def canonical_skill(name: str) -> str:
    key = name.strip().lower()
    hit = tx.skill_variants().get(key)
    return hit["canonical"] if hit else key


def _in(values: list) -> str:
    return "(" + ",".join("?" * len(values)) + ")"


def _exists(table: str, cond: str) -> str:
    return f"EXISTS (SELECT 1 FROM {table} t WHERE t.candidate_id = c.candidate_id AND {cond})"


def _location_clause(place: Place, scope: str) -> tuple[str, list]:
    if place.kind == "country":
        cur, cur_p = "c.current_country = ?", [place.code]
        worked = _exists("candidate_countries", "t.kind = 'worked' AND t.country = ?")
    else:
        cur, cur_p = "c.current_region = ?", [place.code]
        worked = _exists(
            "candidate_countries",
            "t.kind = 'worked' AND t.country IN (SELECT iso2 FROM taxonomy_country WHERE region = ?)",
        )
    if scope == "current":
        return cur, cur_p
    if scope == "worked":
        return worked, [place.code]
    return f"({cur} OR {worked})", cur_p + [place.code]


def _lower_in(values: list[str]) -> tuple[str, list[str]]:
    return _in(values), [v.casefold() for v in values]


def prune_sql(req: Requisition) -> tuple[str, list]:
    """Candidate ids satisfying every must criterion. Skills match on the canonical token only;
    fuzzy matching is a scoring concern."""
    where: list[str] = []
    params: list = []

    def add(clause: str, p: list) -> None:
        where.append(clause)
        params.extend(p)

    if req.location and req.location.mode == "must":
        clauses = [_location_clause(resolve_location(v), req.location_scope) for v in req.location.value]
        add("(" + " OR ".join(c for c, _ in clauses) + ")", [p for _, ps in clauses for p in ps])
    if req.approach and req.approach.mode == "must":
        add("(c.investment_approach = ? OR c.investment_approach = 'Both')", [req.approach.value])
    if req.sectors and req.sectors.mode == "must":
        ph, p = _in(req.sectors.value), list(req.sectors.value)
        add(_exists("candidate_sectors", f"t.sector IN {ph}"), p)
    if req.markets_covered and req.markets_covered.mode == "must":
        ph, p = _lower_in(req.markets_covered.value)
        add(_exists("candidate_countries", f"t.kind = 'covered' AND lower(t.country) IN {ph}"), p)
    if req.asset_classes and req.asset_classes.mode == "must":
        ph, p = _in(req.asset_classes.value), list(req.asset_classes.value)
        add(_exists("candidate_assets", f"t.asset_class IN {ph}"), p)
    if req.seniority_bands and req.seniority_bands.mode == "must":
        add(f"c.seniority_band IN {_in(req.seniority_bands.value)}", list(req.seniority_bands.value))
    if req.years_mode == "must":
        if req.years_min is not None:
            add("c.months_full_time_relevant >= ?", [req.years_min * 12])
        if req.years_max is not None:
            add("c.months_full_time_relevant <= ?", [req.years_max * 12])
    for skill in req.skills_required:
        add(_exists("candidate_skills", "t.skill = ?"), [canonical_skill(skill)])
    if req.degree_min and req.degree_min.mode == "must":
        ok = list(DEGREES[DEGREES.index(req.degree_min.value):])
        add(f"c.highest_degree IN {_in(ok)}", ok)
    if req.certifications and req.certifications.mode == "must":
        for cert in req.certifications.value:
            like = _exists("candidate_certs", "lower(t.cert) LIKE ?")
            if cert.casefold() == "cfa":
                like = f"(c.has_cfa = 1 OR {like})"
            add(like, [f"%{cert.casefold()}%"])
    if req.languages and req.languages.mode == "must":
        for lang in req.languages.value:
            add(_exists("candidate_languages", "lower(t.language) = ?"), [lang.casefold()])

    sql = "SELECT c.candidate_id FROM candidates c"
    if where:
        sql += " WHERE " + " AND ".join(where)
    return sql, params


# Loading ----------------------------------------------------------------------------------------


def _chunks(ids: list[str], n: int = 500):
    for i in range(0, len(ids), n):
        yield ids[i : i + n]


def load_candidates(conn: sqlite3.Connection, ids: Iterable[str]) -> list[Candidate]:
    """Candidate rows plus every facet in eight grouped queries, not one per candidate."""
    ids = list(ids)
    facets: dict[str, dict[str, list]] = {
        k: defaultdict(list)
        for k in ("sectors", "assets", "skills", "worked", "covered", "certs", "languages")
    }
    rows = []
    for chunk in _chunks(ids):
        ph = _in(chunk)
        rows += conn.execute(f"SELECT * FROM candidates WHERE candidate_id IN {ph}", chunk).fetchall()
        for table, cols, key, fn in (
            ("candidate_sectors", "sector, evidence", "sectors", lambda r: (r[0], r[1])),
            ("candidate_assets", "asset_class", "assets", lambda r: r[0]),
            ("candidate_skills", "skill", "skills", lambda r: r[0]),
            ("candidate_certs", "cert", "certs", lambda r: r[0]),
            ("candidate_languages", "language", "languages", lambda r: r[0]),
        ):
            for r in conn.execute(
                f"SELECT candidate_id, {cols} FROM {table} WHERE candidate_id IN {ph}", chunk
            ):
                facets[key][r[0]].append(fn(r[1:]))
        for r in conn.execute(
            f"SELECT candidate_id, country, months, kind FROM candidate_countries "
            f"WHERE candidate_id IN {ph}", chunk,
        ):
            if r["kind"] == "worked":
                facets["worked"][r[0]].append((r["country"], r["months"] or 0))
            else:
                facets["covered"][r[0]].append(r["country"])
    return [
        Candidate(
            candidate_id=r["candidate_id"], name=r["name"], search_summary=r["search_summary"],
            seniority_band=r["seniority_band"], approach=r["investment_approach"],
            months_relevant=r["months_full_time_relevant"] or 0,
            highest_degree=r["highest_degree"], has_cfa=bool(r["has_cfa"]),
            current_country=r["current_country"], current_region=r["current_region"],
            sectors=dict(facets["sectors"][r["candidate_id"]]),
            assets=frozenset(facets["assets"][r["candidate_id"]]),
            skills=frozenset(facets["skills"][r["candidate_id"]]),
            worked=dict(facets["worked"][r["candidate_id"]]),
            covered=frozenset(facets["covered"][r["candidate_id"]]),
            certs=tuple(facets["certs"][r["candidate_id"]]),
            languages=frozenset(facets["languages"][r["candidate_id"]]),
        )
        for r in rows
    ]


# Scoring ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Term:
    criterion: str
    mode: str
    weight: float
    fraction: float
    chips: tuple[Chip, ...]


def _term(
    criterion: str, mode: str, weight: float, fraction: float, label: str,
    detail: str | None, matched: bool | None = None,
) -> _Term:
    ok = fraction > 0 if matched is None else matched
    chip = Chip(criterion, label, ChipStatus.matched if ok else ChipStatus.unmatched, detail)
    return _Term(criterion, mode, weight, fraction, (chip,))


def _location(
    cand: Candidate, place: Place, scope: str, w: Weights
) -> tuple[float, str | None]:
    regions = {iso: c["region"] for iso, c in tx.countries().items()}
    target_region = regions[place.code] if place.kind == "country" else place.code

    def current() -> tuple[float, str | None]:
        if place.kind == "country" and cand.current_country == place.code:
            return w.location_current, f"Currently in {place.code}"
        if cand.current_region == target_region and cand.current_country:
            factor = w.location_region_factor if place.kind == "country" else 1.0
            return w.location_current * factor, f"Currently in {cand.current_country} ({target_region})"
        return 0.0, None

    def worked() -> tuple[float, str | None]:
        # An explicit "worked" scope makes work history the primary signal, so it scores in full.
        base = 1.0 if scope == "worked" else w.location_worked
        if place.kind == "country" and place.code in cand.worked:
            return base, f"Worked in {place.code} {cand.worked[place.code] / 12:.1f}y"
        in_region = {c: m for c, m in cand.worked.items() if regions.get(c) == target_region}
        if in_region:
            c, m = max(in_region.items(), key=lambda kv: kv[1])
            factor = w.location_region_factor if place.kind == "country" else 1.0
            return base * factor, f"Worked in {c} ({target_region}) {m / 12:.1f}y"
        return 0.0, None

    if scope == "current":
        return current()
    if scope == "worked":
        return worked()
    return max(current(), worked(), key=lambda t: t[0])


def _years(cand: Candidate, lo: float | None, hi: float | None, decay: float) -> float:
    y = cand.months_relevant / 12
    dist = max((lo - y) if lo is not None else 0.0, (y - hi) if hi is not None else 0.0, 0.0)
    return max(0.0, 1.0 - dist / decay) if decay > 0 else float(dist == 0)


def _skill_match(cand: Candidate, skill: str, threshold: float) -> str | None:
    token = canonical_skill(skill)
    if token in cand.skills:
        return token
    best = max(cand.skills, key=lambda s: fuzz.ratio(token, s), default=None)
    if best is not None and fuzz.ratio(token, best) >= threshold:
        return best
    return None


def _range_label(lo: float | None, hi: float | None) -> str:
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g}y"
    return f"{lo:g}y+" if lo is not None else f"up to {hi:g}y"


def _terms(req: Requisition, cand: Candidate, w: Weights) -> list[_Term]:
    out: list[_Term] = []

    def crit(c: Criterion | None) -> tuple[list[str], str]:
        return _values(c), (c.mode if c else "preferred")

    if req.location:
        places = [resolve_location(v) for v in req.location.value]
        frac, detail = max((_location(cand, p, req.location_scope, w) for p in places), key=lambda t: t[0])
        codes = ", ".join(p.code for p in places)
        label = f"Location {codes}" + ("" if req.location_scope == "either" else f" ({req.location_scope})")
        out.append(_term("location", req.location.mode, w.weight("location"), frac, label,
                         detail or "No match"))

    if req.approach:
        want = req.approach.value
        ok = cand.approach == want or cand.approach == "Both"
        out.append(_term("approach", req.approach.mode, w.weight("approach"), float(ok),
                         f"Approach {want}", cand.approach))

    vals, mode = crit(req.sectors)
    if vals:
        chips = tuple(
            Chip("sector", v, ChipStatus.matched if v in cand.sectors else ChipStatus.unmatched,
                 cand.sectors.get(v))
            for v in vals
        )
        frac = sum(v in cand.sectors for v in vals) / len(vals)
        out.append(_Term("sector", mode, w.weight("sector"), frac, chips))

    vals, mode = crit(req.markets_covered)
    if vals:
        cov = {c.casefold() for c in cand.covered}
        hit = [v for v in vals if v.casefold() in cov]
        out.append(_term("markets_covered", mode, w.weight("markets_covered"), float(bool(hit)),
                         "Markets " + "/".join(vals), ", ".join(hit) or "No overlap"))

    if req.years_min is not None or req.years_max is not None:
        frac = _years(cand, req.years_min, req.years_max, w.years_decay)
        out.append(_term("years", req.years_mode, w.weight("years_in_range"), frac,
                         f"Years {_range_label(req.years_min, req.years_max)}",
                         f"{cand.months_relevant / 12:.1f}y relevant", matched=frac >= 1.0))

    vals, mode = crit(req.seniority_bands)
    if vals:
        out.append(_term("seniority", mode, w.weight("seniority_band"),
                         float(cand.seniority_band in vals), "Level " + "/".join(vals),
                         cand.seniority_band))

    for skills, smode, key in (
        (req.skills_required, "must", "skill_preferred"),
        (req.skills_preferred, "preferred", "skill_preferred"),
    ):
        for s in skills:
            found = _skill_match(cand, s, w.fuzzy_skill_threshold)
            out.append(_term("skill", smode, w.weight(key), float(found is not None), s, found))

    if req.degree_min:
        need = DEGREES.index(req.degree_min.value)
        have = DEGREES.index(cand.highest_degree) if cand.highest_degree in DEGREES else -1
        out.append(_term("degree", req.degree_min.mode, w.weight("degree"), float(have >= need),
                         f"Degree {req.degree_min.value}+", cand.highest_degree))

    vals, mode = crit(req.certifications)
    for v in vals:
        key = v.casefold()
        found = next((c for c in cand.certs if key in c.casefold()), None)
        if found is None and key == "cfa" and cand.has_cfa:
            found = "CFA"
        out.append(_term("certification", mode, w.weight("certification"), float(found is not None),
                         v, found))

    vals, mode = crit(req.languages)
    have_langs = {x.casefold() for x in cand.languages}
    for v in vals:
        out.append(_term("language", mode, w.weight("language"), float(v.casefold() in have_langs), v, None))

    vals, mode = crit(req.asset_classes)
    if vals:
        hit = [a for a in vals if a in cand.assets]
        out.append(_term("asset_class", mode, w.weight("asset_class"), float(bool(hit)),
                         "Assets " + "/".join(vals), ", ".join(hit) or "No overlap"))
    return out


def rule_score(
    req: Requisition, cand: Candidate, weights: Weights
) -> tuple[float, list[tuple[str, float, float]], list[Chip]]:
    """Weighted mean of match fractions over the preferred criteria; chips cover every criterion
    the user set. Score is 0 when there are no preferred criteria."""
    terms = _terms(req, cand, weights)
    scored = [t for t in terms if t.mode == "preferred"]
    total = sum(t.weight for t in scored)
    score = sum(t.weight * t.fraction for t in scored) / total if total else 0.0
    return score, [(t.criterion, t.weight, t.fraction) for t in scored], [c for t in terms for c in t.chips]
