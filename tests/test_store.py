import csv
import json

import numpy as np
import pytest
from builders import SUMMARY, record, resume, role

from resume_search.search.vectors import (
    Embedder,
    VectorIndex,
    candidate_vectors,
    from_blob,
    to_blob,
)
from resume_search.store.db import (
    connect,
    get_candidate,
    list_candidates,
    load_taxonomy,
    upsert_candidate,
)
from resume_search.store.export import export_csv, export_json
from resume_search.store.fts import fts_body


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.sqlite")
    yield c
    c.close()


def _count(conn, table, cid):
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE candidate_id = ?", (cid,)).fetchone()[0]


def test_schema_applies_twice(tmp_path):
    connect(tmp_path / "t.sqlite").close()
    conn = connect(tmp_path / "t.sqlite")
    load_taxonomy(conn)
    load_taxonomy(conn)
    assert conn.execute("SELECT COUNT(*) FROM taxonomy_country").fetchone()[0] > 30
    assert conn.execute("SELECT weight FROM ranking_weights WHERE criterion='blend.rule'").fetchone()[0] == 0.5


def test_upsert_twice_is_idempotent(conn, embedder):
    rec = record(resume([role(), role(start="2018-01", end="2019-01")], certifications=["CFA"]))
    vecs = candidate_vectors(rec, embedder)
    for _ in range(2):
        upsert_candidate(conn, rec, "resume body", vecs)
    cid = rec.meta["candidate_id"]
    assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 1
    assert _count(conn, "experience", cid) == 2
    assert _count(conn, "embeddings", cid) == 3
    assert _count(conn, "candidate_fts", cid) == 1
    assert _count(conn, "candidate_certs", cid) == 1
    assert get_candidate(conn, cid) == rec
    assert len(list_candidates(conn)) == 1


def test_fts_body_drops_contact_lines():
    text = (
        "Test Person\ntest.person@example.com | +1 (212) 555-0100\nlinkedin.com/in/test\n"
        "Phone: 212 555 0100\nAnalyst, Acme, 2018-01 - 2020-03\nBuilt models in Python."
    )
    body = fts_body(text)
    assert "@" not in body
    assert "555" not in body
    assert "linkedin" not in body
    assert "2018-01 - 2020-03" in body
    assert "Built models" in body


def test_blob_roundtrip():
    v = np.random.default_rng(0).standard_normal(384).astype(np.float32)
    assert np.array_equal(from_blob(to_blob(v)), v)


def test_export_csv_one_row_per_candidate(conn, tmp_path, embedder):
    for cid, sec in (("aaaaaaaaaaaa", "Technology"), ("bbbbbbbbbbbb", "Healthcare")):
        r = resume([role(sectors=[{"label": sec, "evidence": {"quote": "q"}}])], name=cid)
        rec = record(r, cid)
        upsert_candidate(conn, rec, "body", candidate_vectors(rec, embedder))
    assert export_csv(conn, tmp_path / "c.csv") == 2
    with open(tmp_path / "c.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["sectors"] for r in rows} == {"Technology", "Healthcare"}
    assert "record_json" not in rows[0]
    assert export_json(conn, tmp_path / "c.json") == 2
    assert len(json.loads((tmp_path / "c.json").read_text())) == 2


def test_vector_search_finds_own_summary(conn, embedder):
    other = (
        "Senior fundamental healthcare equity analyst (9.0y relevant). Pharmaceuticals and "
        "medical devices in India. Lead Analyst at an Indian brokerage (sell-side broker), "
        "Mumbai; covers 32 companies with channel checks and earnings models. MBA, a business "
        "school in Mumbai."
    )
    a = record(resume(name="Quant"), "aaaaaaaaaaaa")
    b = record(resume(name="Healthcare", search_summary=other), "bbbbbbbbbbbb")
    for rec in (a, b):
        upsert_candidate(conn, rec, "body", candidate_vectors(rec, embedder))
    idx = VectorIndex.load(conn)
    top = idx.search(embedder.embed([SUMMARY])[0], top_k=2)
    assert top[0].candidate_id == "aaaaaaaaaaaa"
    assert len(top) == 2
    top = idx.search(embedder.embed(["indian pharma sell-side analyst"])[0], top_k=2)
    assert top[0].candidate_id == "bbbbbbbbbbbb"
