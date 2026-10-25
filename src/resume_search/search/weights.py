from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

from resume_search import taxonomy

# Criteria the design table has no weight for (seniority band, language) count as one skill.
DEFAULT_WEIGHT = 1.0


@dataclass(frozen=True)
class Weights:
    rule: Mapping[str, float]
    location_current: float = 1.0
    location_worked: float = 0.5
    location_region_factor: float = 0.5
    blend_rule: float = 0.5
    blend_keyword: float = 0.2
    blend_embedding: float = 0.3
    years_decay: float = 2.0
    fuzzy_skill_threshold: float = 90.0

    def weight(self, key: str) -> float:
        return float(self.rule.get(key, DEFAULT_WEIGHT))

    @classmethod
    def load(cls, path: str | Path | None = None) -> Weights:
        raw = yaml.safe_load(Path(path or taxonomy.TAXONOMY_DIR / "weights.yaml").read_text("utf-8"))
        loc, blend = raw["location"], raw["blend"]
        return cls(
            rule=MappingProxyType({k: float(v) for k, v in raw["rule"].items()}),
            location_current=float(loc["current_match"]),
            location_worked=float(loc["worked_match"]),
            location_region_factor=float(loc["region_only_factor"]),
            blend_rule=float(blend["rule"]),
            blend_keyword=float(blend["keyword"]),
            blend_embedding=float(blend["embedding"]),
            years_decay=float(raw["years_decay_outside_range"]),
            fuzzy_skill_threshold=float(raw["fuzzy_skill_threshold"]),
        )
