"""Shared Streamlit helpers: cached resources, the requisition <-> widget mapping, display labels
and chip rendering."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from resume_search import taxonomy as tx
from resume_search.search import Chip, ChipStatus, Criterion, Requisition, SearchResult, Weights
from resume_search.search import search as run_ranked_search
from resume_search.search.requisition import resolve_location
from resume_search.search.vectors import Embedder, VectorIndex
from resume_search.store.db import connect

load_dotenv()

DEFAULT_DB = "exports/candidates.sqlite"
ANY = "(any)"
YEARS_MAX = 25
# Criteria that carry a must/preferred mode on the Requisition, keyed to their toggle.
MODE_FIELDS = (
    "location", "approach", "sectors", "markets_covered", "seniority_bands", "degree_min",
    "certifications", "languages", "asset_classes",
)
# ISO codes shown differently from the taxonomy key.
DISPLAY_CODE = {"GB": "UK"}
UNKNOWN = "Unknown"


def db_path() -> str:
    return os.environ.get("RESUME_DB", DEFAULT_DB)


def uploads_dir() -> Path:
    return Path(db_path()).resolve().parent / "uploads"


# Cached resources -------------------------------------------------------------------------------


@st.cache_resource
def get_conn(path: str):
    return connect(path, check_same_thread=False)


def conn():
    return get_conn(db_path())


@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource
def get_index(path: str) -> VectorIndex:
    return VectorIndex.load(get_conn(path))


@st.cache_resource
def get_weights() -> Weights:
    return Weights.load()


@st.cache_resource
def get_taxonomy() -> dict:
    countries = tx.countries()
    skills = tx.skill_variants().values()
    return {
        "regions": list(tx._load("countries")["regions"]),
        "countries": {iso: c["name"] for iso, c in countries.items()},
        "region_of": {iso: c["region"] for iso, c in countries.items()},
        "sectors": tx.sectors(),
        "asset_classes": tx.asset_classes(),
        "firm_types": tx.firm_types(),
        "markets": sorted(tx.market_labels() | tx.iso2_codes()),
        "skills": sorted({s["canonical"] for s in skills if s["category"] != "cert"}),
        "certs": sorted({s["canonical"].upper() for s in skills if s["category"] == "cert"}),
    }


def _distinct(path: str, sql: str) -> list[str]:
    return [r[0] for r in get_conn(path).execute(sql) if r[0]]


@st.cache_data
def picker_options(path: str) -> dict[str, list[str]]:
    """Taxonomy values plus what the stored candidates already carry, deduplicated ignoring case.
    The pickers still accept a typed value that is in neither."""
    t = get_taxonomy()

    def merge(*lists: list[str]) -> list[str]:
        seen: dict[str, str] = {}
        for v in (v for vs in lists for v in vs):
            seen.setdefault(v.casefold(), v)
        return sorted(seen.values(), key=str.casefold)

    return {
        "skills": merge(t["skills"], _distinct(path, "SELECT DISTINCT skill FROM candidate_skills")),
        "certs": merge(t["certs"], _distinct(path, "SELECT DISTINCT cert FROM candidate_certs")),
        "languages": merge(_distinct(path, "SELECT DISTINCT language FROM candidate_languages")),
    }


@st.cache_data(show_spinner="Searching")
def run_search(req_json: str, path: str, top_k: int) -> list[SearchResult]:
    return run_ranked_search(
        get_conn(path), Requisition.model_validate_json(req_json), get_weights(),
        embedder=get_embedder(), index=get_index(path), top_k=top_k,
    )


def clear_caches() -> None:
    """After ingest: new rows invalidate search results and the vector index. The embedder and
    the connection stay."""
    get_index.clear()
    st.cache_data.clear()


# Display labels ---------------------------------------------------------------------------------


def country_code(iso: str | None) -> str:
    return DISPLAY_CODE.get(iso, iso) if iso else UNKNOWN


def country_option(iso: str) -> str:
    return f"{country_code(iso)} – {get_taxonomy()['countries'][iso]}"


def location_label(country: str | None, region: str | None) -> str:
    """'APAC – HK', 'Europe – UK'; one name when region and country coincide ('US')."""
    if not country:
        return region or UNKNOWN
    code = country_code(country)
    return code if not region or region == country else f"{region} – {code}"


def profile_line(band: str | None, country: str | None, region: str | None) -> str:
    return f"Seniority: {band or UNKNOWN} · Location: {location_label(country, region)}"


# Requisition <-> widget state -------------------------------------------------------------------


def default_state() -> dict:
    return {
        "f_regions": [], "f_countries": [], "f_scope": "either", "f_approach": ANY,
        "f_sectors": [], "f_markets": [], "f_years": (0, YEARS_MAX), "f_bands": [],
        "f_degree": ANY, "f_certs": [], "f_skills_req": [], "f_skills_pref": [],
        "f_languages": [], "f_assets": [], "jd_text": "",
        **{f"m_{f}": False for f in MODE_FIELDS}, "m_years": True,
    }


def init_state() -> None:
    for k, v in default_state().items():
        st.session_state.setdefault(k, v)


def country_choices(regions: list[str]) -> list[str]:
    """ISO codes for the country picker: the selected regions' countries, or all of them."""
    t = get_taxonomy()
    isos = [i for i in t["countries"] if not regions or t["region_of"][i] in regions]
    return sorted(isos, key=country_option)


def places(regions: list[str], countries: list[str]) -> list[str]:
    """The location values the widgets mean: each picked country, plus each picked region that
    has no picked country inside it."""
    region_of = get_taxonomy()["region_of"]
    narrowed = {region_of[c] for c in countries}
    return list(countries) + [r for r in regions if r not in narrowed]


def state_from_requisition(req: Requisition) -> dict:
    """Widget values that reproduce `req`."""
    s = default_state()
    s["jd_text"] = req.free_text or ""
    if req.location:
        for v in req.location.value:
            place = resolve_location(v)
            region = place.code if place.kind == "region" else get_taxonomy()["region_of"][place.code]
            if place.kind == "country":
                s["f_countries"].append(place.code)
            if region not in s["f_regions"]:
                s["f_regions"].append(region)
    s["f_scope"] = req.location_scope
    if req.approach:
        s["f_approach"] = req.approach.value
    for field, key in (
        ("sectors", "f_sectors"), ("markets_covered", "f_markets"), ("seniority_bands", "f_bands"),
        ("certifications", "f_certs"), ("languages", "f_languages"), ("asset_classes", "f_assets"),
    ):
        if (c := getattr(req, field)) is not None:
            s[key] = list(c.value)
    lo, hi = req.years_min, req.years_max
    s["f_years"] = (int(lo or 0), int(min(hi, YEARS_MAX)) if hi is not None else YEARS_MAX)
    s["m_years"] = req.years_mode == "must"
    if req.degree_min:
        s["f_degree"] = req.degree_min.value
    s["f_skills_req"] = list(req.skills_required)
    s["f_skills_pref"] = list(req.skills_preferred)
    for f in MODE_FIELDS:
        s[f"m_{f}"] = (c := getattr(req, f)) is not None and c.mode == "must"
    return s


def requisition_from_state(s: dict) -> Requisition:
    def crit(field: str, value):
        return Criterion(value=value, mode="must" if s[f"m_{field}"] else "preferred")

    lo, hi = s["f_years"]
    kw: dict = {}
    if loc := places(s["f_regions"], s["f_countries"]):
        kw["location"] = crit("location", loc)
    if s["f_approach"] != ANY:
        kw["approach"] = crit("approach", s["f_approach"])
    for field, key in (
        ("sectors", "f_sectors"), ("markets_covered", "f_markets"), ("seniority_bands", "f_bands"),
        ("certifications", "f_certs"), ("languages", "f_languages"), ("asset_classes", "f_assets"),
    ):
        if s[key]:
            kw[field] = crit(field, list(s[key]))
    if s["f_degree"] != ANY:
        kw["degree_min"] = crit("degree_min", s["f_degree"])
    return Requisition(
        free_text=(s["jd_text"] or "").strip() or None,
        location_scope=s["f_scope"],
        years_min=float(lo) if lo > 0 else None,
        years_max=float(hi) if hi < YEARS_MAX else None,
        years_mode="must" if s["m_years"] else "preferred",
        skills_required=list(s["f_skills_req"]), skills_preferred=list(s["f_skills_pref"]), **kw,
    )


# Chips ------------------------------------------------------------------------------------------

CHIP_CSS = """<style>
.rs-chip{display:inline-block;font-size:.8rem;line-height:1.5;padding:.05rem .55rem;margin:0 .3rem .3rem 0;
  border-radius:.4rem;cursor:help}
.rs-ok{background:rgba(33,195,84,.14);color:#177233}
.rs-no{background:rgba(255,164,33,.18);color:#8a4b00}
@media (prefers-color-scheme: dark){.rs-ok{color:#7ee2a0}.rs-no{color:#ffc56e}}
</style>"""


def chip_tooltip(chip: Chip) -> str:
    verdict = "Meets filter" if chip.status == ChipStatus.matched else "Misses filter"
    return f"{verdict}: {chip.label}" + (f" ({chip.detail})" if chip.detail else "")


def chip_html(chip: Chip) -> str:
    cls = "rs-ok" if chip.status == ChipStatus.matched else "rs-no"
    return (f'<span class="rs-chip {cls}" title="{html.escape(chip_tooltip(chip))}">'
            f"{html.escape(chip.label)}</span>")


def chips(result: SearchResult) -> None:
    if result.chips:
        st.markdown(CHIP_CSS + "".join(chip_html(c) for c in result.chips), unsafe_allow_html=True)


def req_cache_key(req: Requisition) -> str:
    return json.dumps(json.loads(req.model_dump_json()), sort_keys=True)
