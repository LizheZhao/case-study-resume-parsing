from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from resume_search.store.db import list_candidates

# facet column -> (table, value column, extra WHERE)
_FACETS = {
    "sectors": ("candidate_sectors", "sector", ""),
    "tags": ("candidate_tags", "tag", ""),
    "asset_classes": ("candidate_assets", "asset_class", ""),
    "skills": ("candidate_skills", "skill", ""),
    "countries_worked": ("candidate_countries", "country", "AND kind = 'worked'"),
    "certs": ("candidate_certs", "cert", ""),
}


def export_json(conn: sqlite3.Connection, path: str | Path) -> int:
    records = [r.model_dump(mode="json") for r in list_candidates(conn)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(records)


def export_csv(conn: sqlite3.Connection, path: str | Path) -> int:
    """The candidates table without record_json (it is the JSON export), plus facet columns."""
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(candidates)") if r["name"] != "record_json"]
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM candidates ORDER BY name").fetchall()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([*cols, *_FACETS])
        for row in rows:
            facets = [
                ";".join(
                    r[0]
                    for r in conn.execute(
                        f"SELECT {col} FROM {table} WHERE candidate_id = ? {extra} ORDER BY rowid",
                        (row["candidate_id"],),
                    )
                    if r[0] is not None
                )
                for table, col, extra in _FACETS.values()
            ]
            w.writerow([*tuple(row), *facets])
    return len(rows)
