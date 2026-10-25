from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from resume_search.eval.judge import (
    JUDGE_MODEL,
    RANKING_PROMPT_VERSION,
    judge_cost,
    judge_ranking,
    ranking_candidate_payload,
    token_totals,
)
from resume_search.eval.metrics import ndcg_at_k, precision_at_k
from resume_search.eval.models import EvalRequisition, load_requisitions
from resume_search.eval.results import now_utc, write_results
from resume_search.parse.client import Attempt, default_client
from resume_search.search import Weights, search
from resume_search.store.db import connect, get_candidate

DEFAULT_REQUISITIONS = "evals/requisitions.jsonl"
CACHE_NAME = "ranking-grades-cache.json"
FULL = Weights.load()
# name -> (rule, keyword, embedding) blend; "full" keeps the weights.yaml blend
VARIANTS = {
    "full": None,
    "rule_only": (1.0, 0.0, 0.0),
    "keyword_only": (0.0, 1.0, 0.0),
    "embedding_only": (0.0, 0.0, 1.0),
}


def _weights(variant: str) -> Weights:
    blend = VARIANTS[variant]
    if blend is None:
        return FULL
    return replace(FULL, blend_rule=blend[0], blend_keyword=blend[1], blend_embedding=blend[2])


def _candidate_payload(record) -> dict:
    d = record.derived
    return {
        "candidate_id": record.meta["candidate_id"],
        "name": record.extracted.name,
        "search_summary": d.search_summary,
        "seniority_band": d.seniority_band,
        "investment_approach": d.investment_approach,
        "years_relevant": round(d.months_full_time_relevant / 12, 1),
        "sectors": [s.label for s in d.sectors],
        "asset_classes": d.asset_classes,
        "current_country": d.current_country,
        "countries_worked": [c.country for c in d.countries_worked],
        "markets_covered": d.markets_covered,
        "skills": d.skills_normalized,
        "highest_degree": d.highest_degree,
        "cfa": d.has_cfa,
    }


def _input_hash(req: EvalRequisition, cand: dict) -> str:
    blob = req.requisition.model_dump_json(exclude_none=True) + ranking_candidate_payload(cand)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_key(req_id: str, cid: str) -> str:
    return f"{req_id}|{cid}|{RANKING_PROMPT_VERSION}"


def _load_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _mean(vals: list[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def run_ranking_eval(
    db_path: str | Path,
    out_dir: str | Path,
    client=None,
    requisitions_path: str | Path = DEFAULT_REQUISITIONS,
    embedder=None,
) -> tuple[Path, dict]:
    conn = connect(db_path)
    reqs = load_requisitions(requisitions_path)
    cache_path = Path(out_dir) / CACHE_NAME
    cache = _load_cache(cache_path)
    n_candidates = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    attempts: list[Attempt] = []
    items, judged_now = [], 0

    for req in reqs:
        pool = {}
        ranked = {}
        for variant in VARIANTS:
            res = search(
                conn, req.requisition, _weights(variant), embedder=embedder, top_k=max(n_candidates, 10)
            )
            ranked[variant] = [r.candidate_id for r in res]
            pool = {r.candidate_id: r for r in res}
        payloads = [
            _candidate_payload(get_candidate(conn, cid)) for cid in sorted(ranked["full"])
        ]
        hashes = {p["candidate_id"]: _input_hash(req, p) for p in payloads}
        cached = {
            p["candidate_id"]: cache.get(_cache_key(req.id, p["candidate_id"]))
            for p in payloads
        }
        hit = all(c and c["input_hash"] == hashes[cid] for cid, c in cached.items())
        if payloads and not hit:
            if client is None:
                client = default_client()
            judgement = judge_ranking(req, payloads, client, attempts)
            judged_now += 1
            for g in judgement.grades:
                cache[_cache_key(req.id, g.candidate_id)] = {
                    "grade": g.grade, "reason": g.reason, "input_hash": hashes[g.candidate_id],
                }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        grades = {cid: cache[_cache_key(req.id, cid)]["grade"] for cid in ranked["full"]}
        row = {
            "requisition_id": req.id, "title": req.title, "pool_size": len(grades),
            "relevant_in_pool": sum(g >= 1 for g in grades.values()),
            "from_cache": hit,
        }
        for variant, ids in ranked.items():
            row[f"{variant}_ndcg5"] = round(ndcg_at_k(ids, grades, 5), 4)
            row[f"{variant}_p3"] = round(precision_at_k(ids, grades, 3), 4)
        row["candidates"] = [
            {
                "candidate_id": cid, "name": pool[cid].name,
                "grade": grades[cid], "reason": cache[_cache_key(req.id, cid)]["reason"],
                **{f"rank_{v}": ids.index(cid) + 1 for v, ids in ranked.items()},
            }
            for cid in ranked["full"]
        ]
        items.append(row)

    mean = {
        v: {
            "ndcg5": _mean([i[f"{v}_ndcg5"] for i in items]),
            "p3": _mean([i[f"{v}_p3"] for i in items]),
        }
        for v in VARIANTS
    }
    metrics = {
        "requisitions": len(items),
        "pool_sizes": {i["requisition_id"]: i["pool_size"] for i in items},
        "mean": mean,
        "judge_calls_this_run": judged_now,
        "tie_break": "score ties sort by candidate name, so a variant that scores many ties is "
        "ranked alphabetically within the tie",
    }
    body = {
        "task_model": "claude-sonnet-5-5", "judge_model": JUDGE_MODEL,
        "prompt_version": RANKING_PROMPT_VERSION, "metrics": metrics, "items": items,
        "cost_usd": round(judge_cost(attempts), 4), "tokens": token_totals(attempts),
    }
    return write_results(out_dir, "ranking", now_utc(), body), body
