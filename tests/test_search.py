import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from builders import record, resume, role

from resume_search.cli import _years
from resume_search.search import (
    Criterion,
    Requisition,
    Weights,
    keyword_scores,
    parse_requisition,
    prune_sql,
    rerank_llm,
    rule_score,
    search,
)
from resume_search.search.requisition import DraftCriterion, RequisitionDraft
from resume_search.search.rules import Candidate, ChipStatus
from resume_search.search.vectors import DIM, candidate_vectors
from resume_search.store.db import connect, load_taxonomy, upsert_candidate

W = Weights.load()


class FakeEmbedder:
    """Hashed bag of words, so cosine tracks word overlap without loading a model."""

    def embed(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1, norm)


def _sector(label, quote):
    return [{"label": label, "evidence": {"quote": quote}}]


def _person(cid, name, roles, skills=(), certs=(), approach="Systematic", summary=None):
    kw = {
        "name": name,
        "skills": {"programming": list(skills), "tools": [], "domain": []},
        "certifications": list(certs),
        "investment_approach": {"label": approach, "evidence": {"quote": "model"}},
    }
    if summary:
        kw["search_summary"] = summary
    return record(resume(roles, **kw), candidate_id=cid)


def _people():
    def summary(s):
        return f"Mid analyst (4.0y relevant). {s} " + "Example Capital hedge fund role. " * 6

    return [
        _person(
            "a" * 12, "Alice Adams",
            [role(country="GB", start="2020-01", end="2024-01", sectors=_sector("Healthcare", "biotech coverage"),
                  bullets=["Covered biotech and pharma equities."])],
            skills=["Python", "SQL"], certs=["CFA"], approach="Fundamental",
            summary=summary("Healthcare equities biotech pharma."),
        ),
        _person(
            "b" * 12, "Bob Brown",
            [
                role(country="GB", start="2016-01", end="2018-01", sectors=_sector("Technology", "software")),
                role(country="US", start="2018-02", end="2022-02", is_current=False,
                     sectors=_sector("Technology", "semiconductors")),
                role(country="US", start="2022-03", is_current=True, end=None,
                     sectors=_sector("Technology", "chips")),
            ],
            skills=["Python", "C++"], summary=summary("Technology semiconductors quant."),
        ),
        _person(
            "c" * 12, "Carol Chen",
            [role(country="HK", start="2021-01", end="2023-01", is_current=False,
                  sectors=_sector("Technology", "internet"))],
            skills=["R"], summary=summary("Technology internet Greater China."),
        ),
        _person(
            "d" * 12, "Carl Chen",
            [role(country="HK", start="2021-01", end="2023-01", is_current=False,
                  sectors=_sector("Technology", "internet"))],
            skills=["R"], summary=summary("Technology internet Greater China."),
        ),
    ]


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    c = connect(tmp_path_factory.mktemp("db") / "t.sqlite")
    load_taxonomy(c)
    emb = FakeEmbedder()
    for rec in _people():
        upsert_candidate(
            c, rec, f"{rec.extracted.name} {' '.join(b for r in rec.extracted.experience for b in r.bullets)}",
            candidate_vectors(rec, emb),
        )
    return c


def _ids(conn, req):
    names = {r["candidate_id"]: r["name"] for r in conn.execute("SELECT candidate_id, name FROM candidates")}
    return {names[r[0]].split()[0] for r in conn.execute(*prune_sql(req))}


def _loc(value, scope, mode="must"):
    return Requisition(location=Criterion[str](value=value, mode=mode), location_scope=scope)


def cand(**kw) -> Candidate:
    base = {
        "candidate_id": "x", "name": "X", "search_summary": "", "seniority_band": "Mid",
        "approach": "Fundamental", "months_relevant": 48, "highest_degree": "Masters",
        "has_cfa": False, "current_country": "GB", "current_region": "Europe", "sectors": {},
        "assets": frozenset(), "skills": frozenset(), "worked": {}, "covered": frozenset(),
        "certs": (), "languages": frozenset(),
    }
    return Candidate(**{**base, **kw})


def test_prune_location_current_worked_either(conn):
    assert _ids(conn, _loc("GB", "current")) == {"Alice"}
    assert _ids(conn, _loc("GB", "worked")) == {"Alice", "Bob"}
    assert _ids(conn, _loc("GB", "either")) == {"Alice", "Bob"}
    assert _ids(conn, _loc("US", "current")) == {"Bob"}
    assert _ids(conn, _loc("APAC", "current")) == {"Carol", "Carl"}
    assert _ids(conn, _loc("Europe", "worked")) == {"Alice", "Bob"}


def test_prune_location_matches_any_listed_place(conn):
    req = Requisition(location=Criterion[list[str]](value=["US", "APAC"], mode="must"),
                      location_scope="current")
    assert _ids(conn, req) == {"Bob", "Carol", "Carl"}


def test_prune_preferred_criteria_do_not_filter(conn):
    assert _ids(conn, _loc("GB", "current", mode="preferred")) == {"Alice", "Bob", "Carol", "Carl"}


def test_prune_must_skill_uses_canonical_token(conn):
    assert _ids(conn, Requisition(skills_required=["python"])) == {"Alice", "Bob"}
    assert _ids(conn, Requisition(skills_required=["C++"])) == {"Bob"}
    assert _ids(conn, Requisition(skills_required=["python", "r"])) == set()


def test_prune_must_cert_sector_and_years(conn):
    assert _ids(conn, Requisition(certifications=Criterion[list[str]](value=["CFA"], mode="must"))) == {"Alice"}
    assert _ids(conn, Requisition(sectors=Criterion[list[str]](value=["Healthcare"], mode="must"))) == {"Alice"}
    months = {r["name"]: r["months_full_time_relevant"] for r in conn.execute("SELECT * FROM candidates")}
    req = Requisition(years_min=months["Alice Adams"] / 12, years_max=months["Alice Adams"] / 12)
    assert "Alice" in _ids(conn, req)
    assert _ids(conn, Requisition(years_min=100)) == set()


@pytest.mark.parametrize(
    ("c", "target", "scope", "expected"),
    [
        (cand(current_country="GB"), "GB", "either", 1.0),
        (cand(current_country="US", current_region="US", worked={"GB": 24}), "GB", "either", 0.5),
        (cand(current_country="US", current_region="US", worked={"GB": 24}), "GB", "worked", 1.0),
        (cand(current_country="FR", current_region="Europe"), "GB", "either", 0.5),
        (cand(current_country="US", current_region="US", worked={"FR": 24}), "GB", "either", 0.25),
        (cand(current_country="HK", current_region="APAC"), "APAC", "either", 1.0),
        (cand(current_country="US", current_region="US", worked={"HK": 24}), "APAC", "either", 0.5),
        (cand(current_country="US", current_region="US", worked={"GB": 24}), "GB", "current", 0.0),
    ],
)
def test_location_rule(c, target, scope, expected):
    req = _loc(target, scope, mode="preferred")
    assert rule_score(req, c, W)[0] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("months", "expected"),
    [(24, 1.0), (60, 1.0), (72, 0.5), (84, 0.0), (120, 0.0), (12, 0.5), (0, 0.0)],
)
def test_years_decay_inside_boundary_and_outside(months, expected):
    req = Requisition(years_min=2, years_max=5, years_mode="preferred")
    assert rule_score(req, cand(months_relevant=months), W)[0] == pytest.approx(expected)


def test_sector_fraction_and_evidence_chips():
    req = Requisition(sectors=Criterion[list[str]](value=["Healthcare", "Technology"]))
    c = cand(sectors={"Healthcare": "biotech coverage"})
    score, terms, chips = rule_score(req, c, W)
    assert score == pytest.approx(0.5)
    assert terms == [("sector", 3.0, 0.5)]
    by = {ch.label: ch for ch in chips}
    assert by["Healthcare"].status is ChipStatus.matched
    assert by["Healthcare"].detail == "biotech coverage"
    assert by["Technology"].status is ChipStatus.unmatched


def test_chips_for_must_and_failures_do_not_enter_score():
    req = Requisition(
        location=Criterion[str](value="GB", mode="must"),
        skills_required=["python", "rust"],
        skills_preferred=["sql"],
        years_min=10,
    )
    c = cand(skills=frozenset({"python", "sql"}), worked={"GB": 48}, months_relevant=48)
    score, terms, chips = rule_score(req, c, W)
    assert terms == [("skill", 1.0, 1.0)]
    assert score == 1.0
    by = {ch.label: ch for ch in chips}
    assert by["Location GB"].status is ChipStatus.matched
    assert by["Location GB"].detail == "Currently in GB"
    assert by["python"].status is ChipStatus.matched
    assert by["rust"].status is ChipStatus.unmatched
    assert by["Years 10y+"].status is ChipStatus.unmatched
    assert by["Years 10y+"].detail == "4.0y relevant"


def test_worked_chip_detail_and_fuzzy_skill():
    req = Requisition(location=Criterion[str](value="GB"), skills_preferred=["financial modeling"])
    c = cand(current_country="US", current_region="US", worked={"GB": 48}, skills=frozenset({"financial-modeling"}))
    _, _, chips = rule_score(req, c, W)
    by = {ch.criterion: ch for ch in chips}
    assert by["location"].detail == "Worked in GB 4.0y"
    assert by["skill"].status is ChipStatus.matched
    assert by["skill"].detail == "financial-modeling"


def test_approach_both_matches_either():
    req = Requisition(approach=Criterion[str](value="Systematic"))
    assert rule_score(req, cand(approach="Both"), W)[0] == 1.0
    assert rule_score(req, cand(approach="Fundamental"), W)[0] == 0.0


def test_keyword_scores_and_escaping(conn):
    ids = [r[0] for r in conn.execute("SELECT candidate_id FROM candidates")]
    scores = keyword_scores(conn, ids, "biotech pharma")
    assert list(scores) == ["a" * 12]
    assert scores["a" * 12] > 0
    assert keyword_scores(conn, ids, '"" AND OR NEAR( c++ *') is not None
    assert keyword_scores(conn, ids, "the of") == {}
    assert keyword_scores(conn, ids, "") == {}


def _by_name(results):
    return {r.name: r for r in results}


def test_search_blend_with_free_text(conn):
    req = Requisition(
        free_text="biotech pharma equities",
        sectors=Criterion[list[str]](value=["Healthcare"]),
    )
    res = search(conn, req, W, embedder=FakeEmbedder())
    top = res[0]
    assert top.name == "Alice Adams"
    b = top.breakdown
    assert b.keyword == 1.0
    assert all(0.0 <= r.breakdown.keyword <= 1.0 and 0.0 <= r.breakdown.embedding <= 1.0 for r in res)
    assert min(r.breakdown.embedding for r in res) == 0.0 and max(r.breakdown.embedding for r in res) == 1.0
    assert b.final == pytest.approx(0.5 * b.rule + 0.2 * b.keyword + 0.3 * b.embedding)
    assert "biotech" in top.keywords


def test_search_without_free_text_is_rule_only(conn):
    req = Requisition(sectors=Criterion[list[str]](value=["Healthcare"]))
    res = search(conn, req, W, embedder=FakeEmbedder())
    assert res[0].name == "Alice Adams"
    assert all(r.breakdown.final == r.breakdown.rule for r in res)
    assert all(r.breakdown.keyword == 0.0 and r.breakdown.embedding == 0.0 for r in res)


def test_search_text_only_uses_keyword_and_embedding(conn):
    res = search(conn, Requisition(free_text="semiconductors quant"), W, embedder=FakeEmbedder())
    assert res[0].name == "Bob Brown"
    b = res[0].breakdown
    assert b.final == pytest.approx((0.2 * b.keyword + 0.3 * b.embedding) / 0.5)


def test_ties_break_by_name(conn):
    req = Requisition(sectors=Criterion[list[str]](value=["Technology"]))
    names = [r.name for r in search(conn, req, W, embedder=FakeEmbedder()) if r.breakdown.final == 1.0]
    assert names == sorted(names)
    assert names.index("Carl Chen") < names.index("Carol Chen")


def test_search_top_k_and_must_only(conn):
    res = search(conn, _loc("GB", "worked"), W, top_k=1)
    assert [r.name for r in res] == ["Alice Adams"]
    assert res[0].chips[0].status is ChipStatus.matched


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls.append(kw)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(
            parsed_output=out, stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        )


def _bad_draft():
    return RequisitionDraft(criteria=[DraftCriterion(field="location", values=["Narnia"], mode="must")])


def test_parse_requisition_copies_input_and_retries_once():
    good = RequisitionDraft(
        criteria=[
            DraftCriterion(field="location", values=["US"], mode="preferred"),
            DraftCriterion(field="sectors", values=["healthcare"], mode="preferred"),
        ],
        skills_required=["python"],
    )
    client = FakeClient([_bad_draft(), good])
    attempts = []
    req = parse_requisition("Need a Python quant in NYC", client, attempts)
    assert req.free_text == "Need a Python quant in NYC"
    assert req.sectors.value == ["Healthcare"]
    assert req.location.value == ["US"] and req.skills_required == ["python"]
    assert len(client.calls) == 2
    assert "validation_errors" in client.calls[1]["messages"][0]["content"]
    assert [a.error is None for a in attempts] == [False, True]
    assert "claude-sonnet-5-5" == client.calls[0]["model"]
    assert not {"temperature", "tool_choice", "thinking"} & set(client.calls[0])


def test_parse_requisition_fails_after_second_error():
    with pytest.raises(ValueError, match="failed after retry"):
        parse_requisition("x", FakeClient([_bad_draft(), _bad_draft()]))


def test_requisition_validation_and_is_empty():
    assert Requisition().is_empty()
    assert not Requisition(free_text="python").is_empty()
    assert not Requisition(years_max=3).is_empty()
    with pytest.raises(ValueError):
        Requisition(approach=Criterion[str](value="Astrology"))
    assert Requisition(location=Criterion[str](value="united kingdom")).location.value == ["united kingdom"]


def test_rerank_llm_reorders_and_retries_on_bad_ids(conn):
    from resume_search.search.rank import _Ranked, _Reranking

    res = search(conn, Requisition(), W)[:3]
    ids = [r.candidate_id for r in res]
    bad = _Reranking(order=[_Ranked(candidate_id=ids[0], reason="a")])
    good = _Reranking(order=[_Ranked(candidate_id=i, reason=f"why {i[0]}") for i in reversed(ids)])
    client = FakeClient([bad, good])
    out = rerank_llm(Requisition(free_text="q"), res, client)
    assert [i for i, _ in out] == ids[::-1]
    assert out[0][1].startswith("why")
    assert len(client.calls) == 2
    assert "missing candidate_id" in client.calls[1]["messages"][0]["content"]


def test_cli_years_parsing():
    assert _years("3-8") == (3.0, 8.0)
    assert _years("5+") == (5.0, None)
    assert _years("4") == (4.0, 4.0)
