"""Shared Streamlit helpers: cached resources, the requisition <-> widget mapping, chip rendering."""

from __future__ import annotations

import json
import os
from dataclasses import replace
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
# Criteria that carry a must/preferred mode on the Requisition.
MODE_FIELDS = (
    "location", "approach", "sectors", "markets_covered", "seniority_bands", "degree_min",
    "certifications", "languages", "asset_classes",
)
MODE_LABELS = {
    "location": "Location", "approach": "Approach", "sectors": "Sectors",
    "markets_covered": "Markets covered", "seniority_bands": "Seniority",
    "degree_min": "Degree", "certifications": "Certifications", "languages": "Languages",
    "asset_classes": "Asset classes",
}


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
    return {
        "regions": list(tx._load("countries")["regions"]),
        "countries": {iso: c["name"] for iso, c in countries.items()},
        "sectors": tx.sectors(),
        "asset_classes": tx.asset_classes(),
        "markets": sorted(tx.market_labels() | tx.iso2_codes()),
    }


def weights_to_dict(w: Weights) -> dict:
    return {**{f: getattr(w, f) for f in Weights.__dataclass_fields__ if f != "rule"}, "rule": dict(w.rule)}


def weights_from_dict(d: dict) -> Weights:
    return replace(get_weights(), **{**d, "rule": dict(d["rule"])})


@st.cache_data(show_spinner="Searching")
def run_search(req_json: str, weights: dict, path: str, top_k: int = 20) -> list[SearchResult]:
    return run_ranked_search(
        get_conn(path), Requisition.model_validate_json(req_json), weights_from_dict(weights),
        embedder=get_embedder(), index=get_index(path), top_k=top_k,
    )


def clear_caches() -> None:
    """After ingest: new rows invalidate search results and the vector index. The embedder and
    the connection stay."""
    get_index.clear()
    st.cache_data.clear()


# Requisition <-> widget state -------------------------------------------------------------------


def location_options() -> dict[str, str]:
    t = get_taxonomy()
    out = {ANY: ""}
    out.update({f"Region: {r}": r for r in t["regions"]})
    out.update({f"{name} ({iso})": iso for iso, name in sorted(t["countries"].items(), key=lambda kv: kv[1])})
    return out


def _csv(text: str) -> list[str]:
    return [p.strip() for p in (text or "").split(",") if p.strip()]


def default_state() -> dict:
    return {
        "f_location": ANY, "f_scope": "either", "f_approach": ANY, "f_sectors": [],
        "f_markets": [], "f_years": (0, YEARS_MAX), "f_years_hard": True, "f_bands": [],
        "f_degree": ANY, "f_cfa": False, "f_certs": "", "f_skills_req": "", "f_skills_pref": "",
        "f_languages": "", "f_assets": [], "f_must": [], "jd_text": "",
    }


def init_state() -> None:
    for k, v in default_state().items():
        st.session_state.setdefault(k, v)


def state_from_requisition(req: Requisition) -> dict:
    """Widget values that reproduce `req`. Values outside a widget's options are dropped."""
    s = default_state()
    s["jd_text"] = req.free_text or ""
    if req.location:
        place = resolve_location(req.location.value)
        s["f_location"] = next((k for k, v in location_options().items() if v == place.code), ANY)
    s["f_scope"] = req.location_scope
    if req.approach:
        s["f_approach"] = req.approach.value
    if req.sectors:
        s["f_sectors"] = list(req.sectors.value)
    if req.markets_covered:
        s["f_markets"] = [m for m in req.markets_covered.value if m in get_taxonomy()["markets"]]
    lo, hi = req.years_min, req.years_max
    s["f_years"] = (int(lo or 0), int(min(hi, YEARS_MAX)) if hi is not None else YEARS_MAX)
    s["f_years_hard"] = req.years_mode == "must"
    if req.seniority_bands:
        s["f_bands"] = list(req.seniority_bands.value)
    if req.degree_min:
        s["f_degree"] = req.degree_min.value
    certs = list(req.certifications.value) if req.certifications else []
    s["f_cfa"] = any(c.upper() == "CFA" for c in certs)
    s["f_certs"] = ", ".join(c for c in certs if c.upper() != "CFA")
    s["f_skills_req"] = ", ".join(req.skills_required)
    s["f_skills_pref"] = ", ".join(req.skills_preferred)
    if req.languages:
        s["f_languages"] = ", ".join(req.languages.value)
    if req.asset_classes:
        s["f_assets"] = list(req.asset_classes.value)
    s["f_must"] = [f for f in MODE_FIELDS if (c := getattr(req, f)) is not None and c.mode == "must"]
    return s


def set_criteria(s: dict) -> list[str]:
    """The mode-carrying criteria the widget state sets, in MODE_FIELDS order."""
    present = {
        "location": s["f_location"] != ANY, "approach": s["f_approach"] != ANY,
        "sectors": bool(s["f_sectors"]), "markets_covered": bool(s["f_markets"]),
        "seniority_bands": bool(s["f_bands"]), "degree_min": s["f_degree"] != ANY,
        "certifications": bool(s["f_cfa"] or _csv(s["f_certs"])),
        "languages": bool(_csv(s["f_languages"])), "asset_classes": bool(s["f_assets"]),
    }
    return [f for f in MODE_FIELDS if present[f]]


def requisition_from_state(s: dict) -> Requisition:
    must = set(s["f_must"])

    def crit(field: str, value):
        mode = "must" if field in must else "preferred"
        return Criterion(value=value, mode=mode)

    lo, hi = s["f_years"]
    certs = (["CFA"] if s["f_cfa"] else []) + _csv(s["f_certs"])
    kw: dict = {}
    if s["f_location"] != ANY:
        kw["location"] = crit("location", location_options()[s["f_location"]])
    if s["f_approach"] != ANY:
        kw["approach"] = crit("approach", s["f_approach"])
    for field, key, vals in (
        ("sectors", "f_sectors", s["f_sectors"]), ("markets_covered", "f_markets", s["f_markets"]),
        ("seniority_bands", "f_bands", s["f_bands"]), ("certifications", None, certs),
        ("languages", None, _csv(s["f_languages"])), ("asset_classes", "f_assets", s["f_assets"]),
    ):
        if vals:
            kw[field] = crit(field, list(vals))
    if s["f_degree"] != ANY:
        kw["degree_min"] = crit("degree_min", s["f_degree"])
    return Requisition(
        free_text=(s["jd_text"] or "").strip() or None,
        location_scope=s["f_scope"],
        years_min=float(lo) if lo > 0 else None,
        years_max=float(hi) if hi < YEARS_MAX else None,
        years_mode="must" if s["f_years_hard"] else "preferred",
        skills_required=_csv(s["f_skills_req"]), skills_preferred=_csv(s["f_skills_pref"]), **kw,
    )


def requisition_rows(req: Requisition) -> list[dict]:
    rows = []
    for f in MODE_FIELDS:
        c = getattr(req, f)
        if c is not None:
            v = c.value if isinstance(c.value, str) else ", ".join(c.value)
            rows.append({"criterion": MODE_LABELS[f], "value": v, "mode": c.mode})
    if req.location:
        rows.append({"criterion": "Location scope", "value": req.location_scope, "mode": ""})
    if req.years_min is not None or req.years_max is not None:
        rows.append({"criterion": "Years", "value": f"{req.years_min or 0} to {req.years_max or 'any'}",
                     "mode": req.years_mode})
    rows += [{"criterion": "Skill", "value": s, "mode": "must"} for s in req.skills_required]
    rows += [{"criterion": "Skill", "value": s, "mode": "preferred"} for s in req.skills_preferred]
    return rows


# Chips ------------------------------------------------------------------------------------------


def chip_markdown(chip: Chip) -> str:
    label = chip.label.replace("[", "(").replace("]", ")")
    detail = f" ({chip.detail})" if chip.detail else ""
    if chip.status == ChipStatus.matched:
        return f":green-badge[{label}{detail}]"
    return f":orange-badge[{label}{detail}]"


def chips(result: SearchResult) -> None:
    if not result.chips:
        return
    try:
        st.markdown(" ".join(chip_markdown(c) for c in result.chips))
    except Exception:  # noqa: BLE001  badge syntax unavailable in an older Streamlit
        st.markdown("  ".join(
            f"{'✓' if c.status == ChipStatus.matched else '✗'} {c.label}"
            + (f" ({c.detail})" if c.detail else "")
            for c in result.chips
        ))


def req_cache_key(req: Requisition) -> str:
    return json.dumps(json.loads(req.model_dump_json()), sort_keys=True)

