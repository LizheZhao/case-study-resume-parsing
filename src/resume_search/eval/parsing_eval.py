from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from resume_search.eval.judge import (
    JUDGE_MODEL,
    PARSING_PROMPT_VERSION,
    judge_cost,
    judge_parsing,
    token_totals,
)
from resume_search.eval.metrics import agreement, field_accuracy
from resume_search.eval.results import now_utc, write_results
from resume_search.parse.client import Attempt, default_client
from resume_search.store.db import connect, get_candidate

DEFAULT_GOLD = "evals/gold/parsing_human.jsonl"
DEFAULT_LOCATION_GOLD = "evals/gold/location_human.jsonl"


def load_gold(path: str | Path) -> dict[str, dict]:
    p = Path(path)
    if not p.is_file():
        return {}
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return {r["candidate_id"]: r for r in rows}


def load_text(conn, text_dir: Path, candidate_id: str, source_file: str | None) -> tuple[str, str]:
    """The full extracted text when exports/text has it, else the FTS body (contact stripped)."""
    if source_file:
        f = text_dir / f"{Path(source_file).stem}.txt"
        if f.is_file():
            return f.read_text(encoding="utf-8"), "text_file"
    row = conn.execute(
        "SELECT body FROM candidate_fts WHERE candidate_id = ?", (candidate_id,)
    ).fetchone()
    return (row["body"] if row else ""), "fts_body"


def location_check(records: dict[str, tuple[str, str | None]], gold: dict[str, dict]) -> dict | None:
    """Exact match of derived current_country against the location gold; no judge involved.
    records maps candidate_id to (name, derived current_country)."""
    rows = [(cid, *records[cid], g["current_country"]) for cid, g in gold.items() if cid in records]
    if not rows:
        return None
    misses = [{"name": n, "derived": got, "gold": want} for _, n, got, want in rows if got != want]
    return {"n": len(rows), "correct": len(rows) - len(misses),
            "accuracy": round(1 - len(misses) / len(rows), 4), "mismatches": misses}


def _summary(grades: list[dict]) -> dict:
    acc = field_accuracy(grades)["overall"]
    return {
        "accuracy": round(acc["accuracy"], 4), "soft_accuracy": round(acc["soft_accuracy"], 4),
        "fields": acc["total"], "hallucinated": acc["hallucinated"], "missing": acc["missing"],
    }


def run_parsing_eval(
    db_path: str | Path,
    out_dir: str | Path,
    client=None,
    only: list[str] | None = None,
    gold_path: str | Path = DEFAULT_GOLD,
    location_gold_path: str | Path = DEFAULT_LOCATION_GOLD,
    text_dir: str | Path | None = None,
    workers: int = 4,
) -> tuple[Path, dict]:
    client = client or default_client()
    conn = connect(db_path)
    text_dir = Path(text_dir) if text_dir else Path(db_path).resolve().parent / "text"
    rows = conn.execute(
        "SELECT candidate_id, name, source_file FROM candidates ORDER BY name"
    ).fetchall()
    rows = [r for r in rows if not only or r["candidate_id"] in only]
    gold = load_gold(gold_path)

    inputs = []
    countries: dict[str, tuple[str, str | None]] = {}
    for r in rows:
        record = get_candidate(conn, r["candidate_id"])
        text, source = load_text(conn, text_dir, r["candidate_id"], r["source_file"])
        inputs.append((r["candidate_id"], r["name"], record.extracted, text, source))
        countries[r["candidate_id"]] = (r["name"], record.derived.current_country)

    def one(args) -> tuple[dict, list[Attempt]]:
        cid, name, parsed, text, source = args
        attempts: list[Attempt] = []
        grades = [g.model_dump() for g in judge_parsing(text, parsed, client, cid, attempts).grades]
        item = {"candidate_id": cid, "name": name, "text_source": source, **_summary(grades),
                "grades": grades}
        if cid in gold:
            item["human_agreement"] = agreement(gold[cid]["grades"], grades)
        return item, attempts

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, inputs))
    items = [i for i, _ in results]
    attempts = [a for _, atts in results for a in atts]

    all_grades = [g for i in items for g in i["grades"]]
    per_field = {
        k: {n: round(v, 4) if isinstance(v, float) else v for n, v in d.items()}
        for k, d in field_accuracy(all_grades).items()
    }
    gold_items = [i for i in items if "human_agreement" in i]

    def keyed(cid: str, grades: list[dict]) -> list[dict]:
        return [{**g, "field": f"{cid}:{g['field']}"} for g in grades]

    overall = agreement(
        [g for i in gold_items for g in keyed(i["candidate_id"], gold[i["candidate_id"]]["grades"])],
        [g for i in gold_items for g in keyed(i["candidate_id"], i["grades"])],
    ) if gold_items else None

    run_at = now_utc()
    metrics = {
        "candidates": len(items),
        **_summary(all_grades),
        "per_field": per_field,
        "agreement": {
            "gold_status": sorted({gold[i["candidate_id"]].get("status", "final") for i in gold_items}),
            "candidates": [i["name"] for i in gold_items],
            "overall": overall,
            "per_candidate": {i["name"]: i["human_agreement"] for i in gold_items},
        },
        "location": location_check(countries, load_gold(location_gold_path)),
    }
    body = {
        "task_model": "claude-sonnet-5-5", "judge_model": JUDGE_MODEL,
        "prompt_version": PARSING_PROMPT_VERSION, "metrics": metrics, "items": items,
        "cost_usd": round(judge_cost(attempts), 4), "tokens": token_totals(attempts),
    }
    return write_results(out_dir, "parsing", run_at, body), body
