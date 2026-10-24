"""Loader for the YAML files in taxonomy/. The single source of truth for ISO-2 codes,
sector and firm-type lists, and the vocabulary treated as known words."""

from __future__ import annotations

import os
import re
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
