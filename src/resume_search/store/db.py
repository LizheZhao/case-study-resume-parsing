from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import yaml

from resume_search import taxonomy as tx
from resume_search.derive import _today
from resume_search.derive.models import CandidateRecord
from resume_search.search.vectors import MODEL_NAME, to_blob

SCHEMA = (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")

_CHILD_TABLES = (
    "experience", "education", "candidate_sectors", "candidate_tags", "candidate_assets",
    "candidate_skills", "candidate_countries", "candidate_certs", "candidate_languages",
    "candidate_fts", "embeddings",
)


def connect(path: str | Path, *, check_same_thread: bool = True) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    if not any(r["name"] == "stints_json" for r in conn.execute("PRAGMA table_info(experience)")):
        conn.execute("ALTER TABLE experience ADD COLUMN stints_json TEXT")
    return conn


def _flatten(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = float(v)
    return out


def load_taxonomy(conn: sqlite3.Connection) -> None:
    sectors = tx._load("sectors")
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO taxonomy_country VALUES (?,?,?)",
            [(iso, c["name"], c["region"]) for iso, c in tx.countries().items()],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO taxonomy_sector VALUES (?,?)",
            [(s, None) for s in sectors["core"]]
            + [(str(t), core) for t, core in sectors["tag_to_core"].items()],
        )
        conn.executemany(
            "INSERT OR REPLACE INTO taxonomy_skill_synonym VALUES (?,?,?)",
            [(k, v["canonical"], v["category"]) for k, v in tx.skill_variants().items()],
        )
        weights = yaml.safe_load((tx.TAXONOMY_DIR / "weights.yaml").read_text(encoding="utf-8"))
        conn.executemany(
            "INSERT OR REPLACE INTO ranking_weights VALUES (?,?)", _flatten(weights).items()
        )


def upsert_candidate(
    conn: sqlite3.Connection,
    record: CandidateRecord,
    fts_body: str,
    vectors: list[tuple[str, int, np.ndarray]],
) -> None:
    """Replace one candidate's rows atomically: a failure leaves the previous version intact."""
    meta, ex, d = record.meta, record.extracted, record.derived
    cid = meta["candidate_id"]
    today = _today(meta)
    with conn:
        for table in _CHILD_TABLES:
            conn.execute(f"DELETE FROM {table} WHERE candidate_id = ?", (cid,))
        conn.execute(
            "INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid, ex.name, ex.contact.email, ex.contact.phone, ex.contact.location_text,
                d.current_country, d.current_region,
                d.months_full_time_relevant, d.months_full_time_core, d.months_internship,
                d.seniority_band, d.investment_approach, d.approach_evidence,
                d.highest_degree, int(d.has_cfa), d.cfa_level, d.search_summary,
                record.model_dump_json(),
                meta.get("prompt_version"), meta.get("schema_version"), meta.get("parse_status"),
                meta.get("parsed_at"), meta.get("file_sha256"), meta.get("source_file"),
            ),
        )
        conn.executemany(
            "INSERT INTO experience VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (cid, i, e.employer, e.title, e.country, e.first_start, e.last_end,
                 int(e.is_current), e.employment_type.value, e.relevance.value,
                 e.firm_type.value, e.months(today),
                 json.dumps({"start": e.start, "end": e.end}))
                for i, e in enumerate(ex.experience)
            ],
        )
        conn.executemany(
            "INSERT INTO education VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (cid, i, e.institution, e.degree, e.degree_level, e.field, e.start, e.end, e.country)
                for i, e in enumerate(ex.education)
            ],
        )
        conn.executemany(
            "INSERT INTO candidate_sectors VALUES (?,?,?)",
            [(cid, s.label, s.evidence) for s in d.sectors],
        )
        conn.executemany("INSERT INTO candidate_tags VALUES (?,?)", [(cid, t) for t in d.other_sector_tags])
        conn.executemany("INSERT INTO candidate_assets VALUES (?,?)", [(cid, a) for a in d.asset_classes])
        conn.executemany(
            "INSERT INTO candidate_skills VALUES (?,?,?)", [(cid, s, "llm") for s in d.skills_normalized]
        )
        conn.executemany(
            "INSERT INTO candidate_countries VALUES (?,?,?,?)",
            [(cid, c.country, c.months, "worked") for c in d.countries_worked]
            + [(cid, m, None, "covered") for m in d.markets_covered],
        )
        conn.executemany("INSERT INTO candidate_certs VALUES (?,?)", [(cid, c) for c in ex.certifications])
        conn.executemany(
            "INSERT INTO candidate_languages VALUES (?,?,?)",
            [(cid, lang.name, lang.level) for lang in ex.languages],
        )
        conn.execute("INSERT INTO candidate_fts (candidate_id, body) VALUES (?,?)", (cid, fts_body))
        conn.executemany(
            "INSERT INTO embeddings VALUES (?,?,?,?,?,?)",
            [(cid, kind, idx, MODEL_NAME, int(vec.shape[0]), to_blob(vec)) for kind, idx, vec in vectors],
        )


def get_candidate(conn: sqlite3.Connection, candidate_id: str) -> CandidateRecord | None:
    row = conn.execute(
        "SELECT record_json FROM candidates WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()
    return CandidateRecord.model_validate_json(row["record_json"]) if row else None


def list_candidates(conn: sqlite3.Connection) -> list[CandidateRecord]:
    rows = conn.execute("SELECT record_json FROM candidates ORDER BY name").fetchall()
    return [CandidateRecord.model_validate_json(r["record_json"]) for r in rows]
