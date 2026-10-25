"""Loader for the YAML files in taxonomy/. The single source of truth for ISO-2 codes,
sector and firm-type lists, and the vocabulary treated as known words."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

TAXONOMY_DIR = Path(
    os.environ.get("RESUME_SEARCH_TAXONOMY", Path(__file__).resolve().parents[2] / "taxonomy")
)

_TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


@cache
def _load(name: str) -> dict:
    return yaml.safe_load((TAXONOMY_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


@cache
def countries() -> dict[str, dict]:
    return _load("countries")["countries"]


@cache
def iso2_codes() -> frozenset[str]:
    return frozenset(countries())


@cache
def market_labels() -> frozenset[str]:
    data = _load("countries")
    return frozenset(data["market_labels"]) | frozenset(data["regions"])


def sectors() -> list[str]:
    return _load("sectors")["core"]


def firm_types() -> list[str]:
    return _load("firm_types")["types"]


def asset_classes() -> list[str]:
    return _load("asset_classes")["classes"]


@cache
def taxonomy_terms() -> frozenset[str]:
    """Lowercased word tokens from skills, firms, countries, aliases, sectors."""
    phrases: list[str] = []
    phrases += _load("skills")["skills"]
    phrases += _load("firm_types")["known_firms"]
    for c in countries().values():
        phrases += [c["name"], *c.get("aliases", [])]
    phrases += _load("sectors")["tag_to_core"]
    phrases += _load("asset_classes")["synonyms"]
    phrases += _load("countries")["market_labels"]
    return frozenset(t.lower() for p in phrases for t in tokenize(str(p)))


@cache
def skill_variants() -> dict[str, dict]:
    """variant (lowercase) -> {canonical, category} from skills.yaml."""
    return {str(k).lower(): v for k, v in _load("skills")["skills"].items()}


@cache
def phone_prefixes() -> dict[str, list[str]]:
    """Calling prefix ('+852') -> ISO-2 codes that share it, from countries.yaml."""
    out: dict[str, list[str]] = {}
    for iso, c in countries().items():
        for prefix in c.get("phone", []):
            out.setdefault(prefix, []).append(iso)
    return out


@dataclass(frozen=True)
class Taxonomy:
    """Plain-data view of the YAML files so derive() stays pure and tests can hand-build one."""

    country_region: dict[str, str]
    country_names: dict[str, tuple[str, ...]]
    phone_prefix: dict[str, tuple[str, ...]]
    # market label -> (ISO-2 codes it expands to, regions it expands to)
    market_expansion: dict[str, tuple[frozenset[str], frozenset[str]]]
    skill_canonical: dict[str, str]
    skill_category: dict[str, str]  # canonical -> programming, tools, domain or cert
    skill_umbrellas: dict[str, frozenset[str]]  # umbrella key -> member keys
    tag_to_core: dict[str, str]


def load_taxonomy_data() -> Taxonomy:
    country_region = {iso: c["region"] for iso, c in countries().items()}
    expansion: dict[str, tuple[frozenset[str], frozenset[str]]] = {
        iso: (frozenset({iso}), frozenset({region})) for iso, region in country_region.items()
    }
    for label, target in _load("countries")["market_labels"].items():
        if target is None:
            expansion[label] = (frozenset(), frozenset())
        elif isinstance(target, str):
            expansion[label] = (frozenset(), frozenset({target}))
        else:
            expansion[label] = (
                frozenset(target),
                frozenset(country_region[c] for c in target),
            )
    for region in _load("countries")["regions"]:
        expansion.setdefault(region, (frozenset(), frozenset({region})))
    return Taxonomy(
        country_region=country_region,
        country_names={
            iso: (c["name"], *c.get("aliases", [])) for iso, c in countries().items()
        },
        phone_prefix={p: tuple(v) for p, v in phone_prefixes().items()},
        market_expansion=expansion,
        skill_canonical={k: v["canonical"] for k, v in skill_variants().items()},
        skill_category={v["canonical"]: v["category"] for v in skill_variants().values()},
        skill_umbrellas={
            name.lower(): frozenset(m.lower() for m in u["members"])
            for u in _load("skills").get("umbrellas", []) for name in u["names"]
        },
        tag_to_core={str(k).lower(): v for k, v in _load("sectors")["tag_to_core"].items()},
    )
