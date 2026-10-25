import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from test_eval_runs import JudgeClient, _gold, _reqs
from test_search import FakeEmbedder, _people

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))

import common

from resume_search.eval.parsing_eval import run_parsing_eval
from resume_search.eval.ranking_eval import run_ranking_eval
from resume_search.ingest import pipeline
from resume_search.search import Criterion, Requisition
from resume_search.search.vectors import candidate_vectors
from resume_search.store.db import connect, load_taxonomy, upsert_candidate

ALICE = "a" * 12


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    db = tmp_path / "t.sqlite"
    conn = connect(db)
    load_taxonomy(conn)
    emb = FakeEmbedder()
    for rec in _people():
        body = f"{rec.extracted.name} " + " ".join(
            b for r in rec.extracted.experience for b in r.bullets
        )
        upsert_candidate(conn, rec, body, candidate_vectors(rec, emb))
    conn.close()
    monkeypatch.setenv("RESUME_DB", str(db))
    monkeypatch.setattr(common, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.chdir(tmp_path)
    st.cache_resource.clear()
    st.cache_data.clear()
    return tmp_path


def open_page(name: str) -> AppTest:
    home = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    return home.switch_page(f"pages/{name}").run()


def no_errors(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_home_counts(app_env):
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    no_errors(at)
    assert [m.value for m in at.metric] == ["4", "0"]


def test_search_filters_and_results(app_env):
    at = open_page("1_Search.py")
    no_errors(at)
    assert at.sidebar.selectbox(key="f_location") and at.text_area(key="jd_text")
    assert [b.label for b in at.button] == ["Parse requisition", "Search"]
    at.sidebar.multiselect(key="f_sectors").set_value(["Healthcare"])
    at.sidebar.text_input(key="f_skills_pref").set_value("python").run()
    at.sidebar.multiselect(key="f_must").set_value(["sectors"])
    at.button[1].click().run()
    no_errors(at)
    assert at.subheader[0].value == "1 candidates"
    assert any("Alice Adams" in str(p.proto.label) for p in at.get("page_link"))
    assert any(":green-badge[" in m.value for m in at.markdown)


def test_search_parse_requisition_prefills_sidebar(app_env, monkeypatch):
    parsed = Requisition(
        free_text="tech",
        sectors=Criterion[list[str]](value=["Technology"], mode="must"),
        skills_preferred=["python"],
    )
    monkeypatch.setattr("resume_search.search.parse_requisition", lambda text: parsed)
    at = open_page("1_Search.py")
    at.text_area(key="jd_text").set_value("tech").run()
    at.button[0].click().run()
    no_errors(at)
    assert at.sidebar.multiselect(key="f_sectors").value == ["Technology"]
    assert at.sidebar.multiselect(key="f_must").value == ["sectors"]
    assert at.sidebar.text_input(key="f_skills_pref").value == "python"
    assert at.dataframe


def test_search_rerank_toggle_shows_reasons(app_env, monkeypatch):
    def fake_rerank(req, results):
        return [(r.candidate_id, f"reason for {r.name}") for r in reversed(results)]

    monkeypatch.setattr("resume_search.search.rerank_llm", fake_rerank)
    at = open_page("1_Search.py")
    at.text_area(key="jd_text").set_value("technology semiconductors")
    at.button[1].click().run()
    at.toggle[0].set_value(True).run()
    no_errors(at)
    assert any(m.value.startswith("**Rerank reason.**") for m in at.markdown)


def test_candidate_page_for_query_param(app_env):
    home = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    at = home.switch_page("pages/2_Candidate.py")
    at.query_params["candidate_id"] = ALICE
    at.run()
    no_errors(at)
    assert at.selectbox[0].value == ALICE
    assert at.header[0].value == "Alice Adams"
    assert any(m.label == "Approach" and m.value == "Fundamental" for m in at.metric)
    assert at.expander


def test_insights_charts(app_env):
    at = open_page("3_Insights.py")
    no_errors(at)
    assert len(at.get("plotly_chart")) == 9


def test_ingest_page_and_rederive(app_env, monkeypatch):
    monkeypatch.setattr(pipeline, "rederive", lambda path: 4)
    at = open_page("4_Ingest.py")
    no_errors(at)
    labels = {b.label: b for b in at.button}
    assert {"Ingest", "Re-run failed", "Rederive all (no API)", "Refresh"} <= set(labels)
    assert labels["Ingest"].disabled and labels["Re-run failed"].disabled
    labels["Rederive all (no API)"].click()
    at.run()
    no_errors(at)
    assert any("Rederived 4" in s.value for s in at.success)


def test_evals_empty_then_populated(app_env):
    at = open_page("5_Evals.py")
    no_errors(at)
    assert any("No eval results yet" in i.value for i in at.info)
    assert len(at.dataframe) == 1

    results = app_env / "evals" / "results"
    client = JudgeClient(grade_of=lambda cid: 2 if cid == ALICE else 0)
    run_parsing_eval(
        app_env / "t.sqlite", results, client, gold_path=_gold(app_env), text_dir=app_env / "none"
    )
    run_ranking_eval(
        app_env / "t.sqlite", results, client, _reqs(app_env), embedder=FakeEmbedder()
    )
    at = open_page("5_Evals.py")
    no_errors(at)
    assert not any("No eval results yet" in i.value for i in at.info)
    assert [s.value for s in at.subheader] == ["Parsing", "Ranking"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Candidates"] == "4" and metrics["Hallucinated / missing"] == "0 / 0"
    assert any("Judge versus human" in m.value and "draft" in m.value for m in at.markdown)
    labels = [e.label for e in at.expander]
    assert any(lbl.startswith("Alice Adams:") for lbl in labels)
    assert any(lbl.startswith("Healthcare: pool 4") for lbl in labels)
    ranking = next(df for df in at.dataframe if "Full blend nDCG@5" in df.value.columns)
    assert ranking.value["requisition"].tolist() == ["Healthcare", "Tech in HK", "Mean"]


def test_requisition_widget_round_trip(app_env):
    req = Requisition(
        free_text="jd", location=Criterion[str](value="GB", mode="must"), location_scope="current",
        approach=Criterion[str](value="Fundamental"),
        sectors=Criterion[list[str]](value=["Healthcare"], mode="must"),
        years_min=2.0, years_max=8.0, years_mode="preferred",
        seniority_bands=Criterion[list[str]](value=["Mid"]),
        skills_required=["python"], skills_preferred=["sql"],
        degree_min=Criterion[str](value="Masters"),
        certifications=Criterion[list[str]](value=["CFA"]),
    )
    state = common.state_from_requisition(req)
    assert common.requisition_from_state(state) == req
