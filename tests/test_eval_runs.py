import json
import re
from types import SimpleNamespace

import pytest
from test_search import FakeEmbedder, _people

from resume_search.eval import judge as judge_mod
from resume_search.eval.models import FieldRow, ParsingGrades, RankingGrades, RankRow
from resume_search.eval.parsing_eval import run_parsing_eval
from resume_search.eval.ranking_eval import VARIANTS, run_ranking_eval
from resume_search.eval.results import latest_result
from resume_search.search.vectors import candidate_vectors
from resume_search.store.db import connect, load_taxonomy, upsert_candidate

ALICE, BOB = "a" * 12, "b" * 12


class JudgeClient:
    """Answers by output schema: parsing grades for every requested field, ranking grades
    for every candidate id in the prompt."""

    def __init__(self, grade_of=lambda cid: 1, bad_first=False):
        self.calls = []
        self.grade_of = grade_of
        self.bad_first = bad_first
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kw):
        self.calls.append(kw)
        user = kw["messages"][0]["content"]
        if kw["output_format"] is ParsingGrades:
            fields = user.split("<fields_to_grade>\n")[1].split("\n</fields_to_grade>")[0].split("\n")
            if self.bad_first and len(self.calls) == 1:
                fields = fields[:-1]
            out = ParsingGrades(
                grades=[FieldRow(field=f, grade="correct" if f != "skills" else "partial",
                                 reason=f"checked {f}") for f in fields]
            )
        else:
            ids = re.findall(r'<candidate id="(\w+)"', user)
            out = RankingGrades(
                grades=[RankRow(candidate_id=i, grade=self.grade_of(i), reason="r") for i in ids]
            )
        return SimpleNamespace(
            parsed_output=out, stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "t.sqlite"
    conn = connect(path)
    load_taxonomy(conn)
    emb = FakeEmbedder()
    for rec in _people():
        body = f"{rec.extracted.name} " + " ".join(b for r in rec.extracted.experience for b in r.bullets)
        upsert_candidate(conn, rec, body, candidate_vectors(rec, emb))
    conn.close()
    return path


def _gold(tmp_path):
    rows = [{"field": f, "grade": "correct", "reason": "x"}
            for f in ("name", "contact", "education", "experience", "skills", "certifications",
                      "languages", "investment_approach", "search_summary", "experience[0]")]
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps({"status": "draft", "candidate_id": ALICE, "name": "Alice Adams",
                                "grades": rows}) + "\n")
    return path


def test_parsing_eval_writes_documented_shape(db, tmp_path):
    out = tmp_path / "results"
    client = JudgeClient()
    path, _ = run_parsing_eval(
        db, out, client, gold_path=_gold(tmp_path), text_dir=tmp_path / "none"
    )
    assert len(client.calls) == 4
    assert client.calls[0]["model"] == "claude-opus-5-5"
    assert not {"temperature", "tool_choice", "thinking"} & set(client.calls[0])
    doc = json.loads(path.read_text())
    assert re.fullmatch(r"parsing-\d{8}T\d{6}Z\.json", path.name)
    assert {"suite", "run_at", "task_model", "judge_model", "prompt_version", "metrics", "items",
            "cost_usd", "tokens"} <= set(doc)
    assert doc["suite"] == "parsing" and doc["judge_model"] == "claude-opus-5-5"
    m = doc["metrics"]
    roles = sum(len(r.extracted.experience) for r in _people())
    assert m["candidates"] == 4 and m["fields"] == 4 * 9 + roles
    assert m["per_field"]["skills"]["partial"] == 4
    assert m["per_field"]["experience_role"]["total"] == roles
    assert m["agreement"]["gold_status"] == ["draft"]
    assert m["agreement"]["per_candidate"]["Alice Adams"]["n"] == 10
    assert 0 < m["agreement"]["overall"]["exact"] < 1
    item = next(i for i in doc["items"] if i["candidate_id"] == ALICE)
    assert item["text_source"] == "fts_body" and item["grades"][0]["field"] == "name"
    assert doc["tokens"]["calls"] == 4 and doc["cost_usd"] == pytest.approx(4 * (4.0 + 4.0) / 1000)
    assert latest_result(out, "parsing")[1]["metrics"]["candidates"] == 4


def test_parsing_eval_only_and_retry_on_missing_field(db, tmp_path):
    client = JudgeClient(bad_first=True)
    _, body = run_parsing_eval(db, tmp_path / "r", client, only=[ALICE], text_dir=tmp_path)
    assert body["metrics"]["candidates"] == 1
    assert len(client.calls) == 2
    assert "missing field" in client.calls[1]["messages"][0]["content"]


def test_parsing_eval_reads_text_file(db, tmp_path):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    (text_dir / "x.txt").write_text("FULL RESUME TEXT")
    client = JudgeClient()
    _, body = run_parsing_eval(db, tmp_path / "r", client, only=[ALICE], text_dir=text_dir)
    assert body["items"][0]["text_source"] == "text_file"
    assert "FULL RESUME TEXT" in client.calls[0]["messages"][0]["content"]


def _reqs(tmp_path):
    lines = [
        {"id": "r-health", "title": "Healthcare", "free_text": "healthcare biotech pharma equities",
         "sectors": {"value": ["Healthcare"], "mode": "preferred"}},
        {"id": "r-tech", "title": "Tech in HK", "free_text": "technology internet",
         "location": {"value": "HK", "mode": "must"}, "location_scope": "current",
         "sectors": {"value": ["Technology"], "mode": "preferred"}},
    ]
    p = tmp_path / "reqs.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines))
    return p


def test_ranking_eval_shape_ablations_and_cache(db, tmp_path):
    out = tmp_path / "results"
    client = JudgeClient(grade_of=lambda cid: 2 if cid == ALICE else 0)
    path, _ = run_ranking_eval(db, out, client, _reqs(tmp_path), embedder=FakeEmbedder())
    assert len(client.calls) == 2
    doc = json.loads(path.read_text())
    assert re.fullmatch(r"ranking-\d{8}T\d{6}Z\.json", path.name)
    assert {"suite", "run_at", "task_model", "judge_model", "prompt_version", "metrics", "items",
            "cost_usd", "tokens"} <= set(doc)
    assert set(doc["metrics"]["mean"]) == set(VARIANTS)
    assert set(doc["metrics"]["mean"]["full"]) == {"ndcg5", "p3"}
    health, tech = doc["items"]
    assert health["pool_size"] == 4 and tech["pool_size"] == 2
    assert doc["metrics"]["pool_sizes"] == {"r-health": 4, "r-tech": 2}
    for item in doc["items"]:
        for v in VARIANTS:
            assert 0 <= item[f"{v}_ndcg5"] <= 1 and 0 <= item[f"{v}_p3"] <= 1
        assert {"grade", "reason", "rank_full", "rank_rule_only"} <= set(item["candidates"][0])
    assert health["full_ndcg5"] == 1.0  # Alice is the only relevant candidate and ranks first
    assert tech["full_ndcg5"] == 0.0  # no relevant candidate in the pool
    assert doc["cost_usd"] == pytest.approx(2 * 0.008)

    again = JudgeClient()
    _, body2 = run_ranking_eval(db, out, again, _reqs(tmp_path), embedder=FakeEmbedder())
    assert again.calls == []
    assert body2["cost_usd"] == 0 and body2["metrics"]["judge_calls_this_run"] == 0
    assert all(i["from_cache"] for i in body2["items"])
    assert body2["items"][0]["full_ndcg5"] == health["full_ndcg5"]
    assert (out / "ranking-grades-cache.json").is_file()
    assert latest_result(out, "ranking")[1]["suite"] == "ranking"  # the cache file is not a result


def test_judge_cost_uses_opus_prices():
    a = judge_mod.Attempt("m", 0, 1_000_000, 1_000_000, None, 1_000_000, 1_000_000)
    assert judge_mod.judge_cost([a]) == pytest.approx(4 + 20 + 0.4 + 5)
