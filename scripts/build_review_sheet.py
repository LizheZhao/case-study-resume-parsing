"""Rebuild evals/gold/REVIEW.md from the gold files, the parsed records and the latest parsing
eval run, so the sheet always describes the parse the gold was graded against.

    python scripts/build_review_sheet.py [--db exports/candidates.sqlite]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from resume_search.eval.parsing_eval import DEFAULT_GOLD, DEFAULT_LOCATION_GOLD, load_gold
from resume_search.store.db import connect, get_candidate

OUT = Path("evals/gold/REVIEW.md")
RESULTS = Path("evals/results")


def _cut(s: str, n: int = 160) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _dates(e: dict) -> str:
    if not e["start"]:
        return "undated"
    stints = [f"{s}→{t or 'now'}" for s, t in zip(e["start"], e["end"], strict=True)]
    return ", ".join(stints)


def _value(x: dict, field: str) -> str:
    if m := re.fullmatch(r"experience\[(\d+)\]", field):
        e = x["experience"][int(m[1])]
        return (f"{e['employer']} / {e['title'] or '(no title)'} / {_dates(e)} / "
                f"{e['employment_type']} / {e['relevance']}")
    match field:
        case "name" | "search_summary":
            return _cut(x[field])
        case "education":
            return "; ".join(
                f"{e['institution']}: {e['degree']} ({e['degree_level']}"
                + (f", minor {', '.join(e['minor_fields'])}" if e.get("minor_fields") else "") + ")"
                for e in x["education"]
            )
        case "experience":
            return f"{len(x['experience'])} roles"
        case "investment_approach":
            a = x[field]
            return f"{a['label']} — \"{_cut(a['evidence']['quote'], 100)}\""
        case _:
            return _cut(json.dumps(x[field], ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="exports/candidates.sqlite")
    args = ap.parse_args()
    conn = connect(args.db)
    gold = load_gold(DEFAULT_GOLD)
    run = max(RESULTS.glob("parsing-*.json"))
    judged = {i["candidate_id"]: {g["field"]: g["grade"] for g in i["grades"]}
              for i in json.loads(run.read_text(encoding="utf-8"))["items"]}

    versions = sorted({g.get("prompt_version", "?") for g in gold.values()})
    out = [
        f"# Gold review sheet ({len(gold)} calibration resumes, parse {'/'.join(versions)})",
        "",
        "Grades: correct / partial / wrong / hallucinated / missing. The **Human** column is the "
        "gold in `parsing_human.jsonl`; edit that file, then rerun "
        "`python scripts/build_review_sheet.py`. The Judge column is from "
        f"`{run.name}`; ⚠ marks a disagreement.",
    ]
    for cid, g in gold.items():
        x = get_candidate(conn, cid).extracted.model_dump(mode="json")
        out += ["", f"## {g['name']}  (`{cid}`)", "",
                "| Field | Parsed value | Human | Judge | Reason |", "|---|---|---|---|---|"]
        for row in g["grades"]:
            f, human = row["field"], row["grade"]
            judge = judged.get(cid, {}).get(f, "—")
            flag = " ⚠" if judge not in (human, "—") else ""
            out.append(f"| {f} | {_cell(_value(x, f))} | **{human}**{flag} | {judge} | "
                       f"{_cell(row['reason'])} |")

    loc = load_gold(DEFAULT_LOCATION_GOLD)
    if loc:
        out += ["", "## Location (all candidates)", "",
                "Gold in `location_human.jsonl`; derived is `current_country` in the database.", "",
                "| Candidate | Gold | Derived | Basis |", "|---|---|---|---|"]
        for cid, g in loc.items():
            rec = get_candidate(conn, cid)
            got = rec.derived.current_country if rec else None
            flag = " ⚠" if got != g["current_country"] else ""
            out.append(f"| {g['name']} | **{g['current_country']}** | {got}{flag} | {_cell(g['basis'])} |")

    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT} from {run.name}")


if __name__ == "__main__":
    main()
