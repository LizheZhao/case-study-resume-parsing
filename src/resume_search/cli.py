from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from resume_search.extract import ExtractedText, extract
from resume_search.ingest.pipeline import attempt_cost, build_meta, ingest, rederive
from resume_search.parse import ParseOk, parse_resume
from resume_search.parse.client import ParseResult
from resume_search.search import (
    Criterion,
    Requisition,
    Weights,
    parse_requisition,
    rerank_llm,
    search,
)
from resume_search.search.rank import SearchResult
from resume_search.store.db import connect, list_candidates

SUPPORTED = {".docx", ".pdf"}
DEFAULT_DB = "exports/candidates.sqlite"


def _supported(paths: list[str]) -> list[str]:
    """A shell glob like data/* also matches non-resume files; skip them loudly."""
    for p in paths:
        if Path(p).suffix.lower() not in SUPPORTED:
            print(f"skip\t{Path(p).name}\tunsupported suffix", file=sys.stderr)
    return [p for p in paths if Path(p).suffix.lower() in SUPPORTED]


def _quality(e: ExtractedText) -> dict:
    return asdict(e.quality)


def _cmd_extract(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in _supported(args.paths):
        e = extract(path)
        (out / f"{Path(path).stem}.txt").write_text(e.text, encoding="utf-8")
        print(
            json.dumps(
                {
                    "file": e.source_file,
                    "pages": e.page_count,
                    "chars": len(e.text),
                    "extraction_path": e.extraction_path.value,
                    **_quality(e),
                }
            )
        )
    return 0


def _parse_one(path: str, out: Path) -> tuple[ExtractedText, ParseResult]:
    e = extract(path)
    result = parse_resume(e)
    doc: dict = {"meta": build_meta(e, result), "extracted": None}
    if isinstance(result, ParseOk):
        doc["extracted"] = result.record.model_dump(mode="json")
    else:
        doc["error"] = result.last_error
    (out / f"{e.candidate_id}.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return e, result


def _cmd_parse(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    total_cost = 0.0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [(p, pool.submit(_parse_one, p, out)) for p in _supported(args.paths)]
        for path, fut in futures:
            name = Path(path).name
            try:
                _, result = fut.result()
            except Exception as exc:  # noqa: BLE001  one bad file must not stop the batch
                failed += 1
                print(f"{name}\tERROR\t{type(exc).__name__}: {exc}")
                continue
            tokens_in = sum(
                a.input_tokens + a.cache_read_tokens + a.cache_creation_tokens
                for a in result.attempts
            )
            tokens_out = sum(a.output_tokens for a in result.attempts)
            latency = sum(a.latency_ms for a in result.attempts)
            cost = sum(attempt_cost(a) for a in result.attempts)
            total_cost += cost
            failed += result.parse_status == "failed"
            v = result.verification if isinstance(result, ParseOk) else None
            checks = (
                f"checks={v.checks_run} dis={v.disagreements} regen={v.regenerated} "
                f"unres={len(v.unresolved)}"
                if v
                else "checks=- dis=- regen=- unres=-"
            )
            print(
                f"{name}\t{result.parse_status}\t{result.attempts[-1].model}\t"
                f"in={tokens_in} out={tokens_out}\t{latency}ms\t"
                f"attempts={len(result.attempts)}\t{checks}\t${cost:.4f}"
            )
    print(f"total cost ${total_cost:.4f}")
    return 1 if failed else 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    if args.reparse_derive_only:
        print(f"rederived {rederive(args.db)} candidates")
        return 0
    jobs = ingest(_supported(args.paths), args.db, force=args.force, workers=args.workers)
    total = 0.0
    for j in jobs:
        total += j.cost_usd or 0.0
        print(
            f"{j.source_file}\t{j.status.value}\tin={j.input_tokens or 0} out={j.output_tokens or 0}"
            f"\t{j.latency_ms or 0}ms\t${j.cost_usd or 0:.4f}\t{j.error or ''}"
        )
    print(f"total cost ${total:.4f}")
    return 1 if any(j.status.value == "failed" for j in jobs) else 0


def _cmd_rederive(args: argparse.Namespace) -> int:
    print(f"rederived {rederive(args.db)} candidates")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    needle = args.candidate.lower()
    hits = [
        r
        for r in list_candidates(connect(args.db))
        if r.meta["candidate_id"].startswith(needle) or needle in r.extracted.name.lower()
    ]
    if not hits:
        print(f"no candidate matches '{args.candidate}'", file=sys.stderr)
        return 1
    for r in hits:
        print(f"{r.extracted.name} ({r.meta['candidate_id']})")
        print(json.dumps(r.derived.model_dump(mode="json"), indent=2, ensure_ascii=False))
    return 0


def _years(text: str) -> tuple[float | None, float | None]:
    lo, sep, hi = text.partition("-") if "-" in text else (text.rstrip("+"), "", "")
    if not sep and not text.endswith("+"):
        return float(lo), float(lo)
    return (float(lo) if lo else None), (float(hi) if hi else None)


def _build_requisition(args: argparse.Namespace, attempts: list) -> Requisition:
    jd = Path(args.jd_file).read_text(encoding="utf-8") if args.jd_file else args.jd
    req = parse_requisition(jd, attempts=attempts) if jd else Requisition()
    if jd:
        print("Parsed requisition:", file=sys.stderr if args.json else sys.stdout)
        print(req.model_dump_json(indent=2, exclude_none=True), file=sys.stderr if args.json else sys.stdout)
    update: dict = {}
    if args.location:
        update["location"] = Criterion[str](value=args.location)
    if args.scope:
        update["location_scope"] = args.scope
    if args.approach:
        update["approach"] = Criterion[str](value=args.approach)
    if args.sector:
        update["sectors"] = Criterion[list[str]](value=args.sector)
    if args.years:
        update["years_min"], update["years_max"] = _years(args.years)
        update["years_mode"] = "must"
    if args.skill:
        update["skills_preferred"] = [*req.skills_preferred, *args.skill]
    if args.must_skill:
        update["skills_required"] = [*req.skills_required, *args.must_skill]
    if args.cfa:
        update["certifications"] = Criterion[list[str]](value=["CFA"], mode="must")
    if jd and not update:
        return req
    return Requisition.model_validate({**req.model_dump(), **{k: v for k, v in update.items()}})


def _row(rank: int, r: SearchResult) -> str:
    b = r.breakdown
    return (
        f"{rank:>2}  {r.name:<24} final={b.final:.3f} rule={b.rule:.3f} kw={b.keyword:.3f} "
        f"emb={b.embedding:.3f}  {r.seniority_band or '-':<9} {r.current_country or '-'}"
    )


def _chips_line(r: SearchResult) -> str:
    chips = "  ".join(
        f"{'✓' if c.status == 'matched' else '✗'} {c.label}" + (f" [{c.detail}]" if c.detail else "")
        for c in r.chips
    )
    return f"    {chips}" + (f"\n    kw: {', '.join(r.keywords)}" if r.keywords else "")


def _cmd_search(args: argparse.Namespace) -> int:
    attempts: list = []
    try:
        req = _build_requisition(args, attempts)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    results = search(connect(args.db), req, Weights.load(), top_k=args.top)
    reasons: dict[str, str] = {}
    if args.rerank and results:
        order = rerank_llm(req, results[:10], attempts=attempts)
        reasons = dict(order)
        by_id = {r.candidate_id: r for r in results}
        results = [by_id[i] for i, _ in order] + results[10:]
    if attempts:
        cost = sum(attempt_cost(a) for a in attempts)
        tokens = sum(a.input_tokens + a.output_tokens for a in attempts)
        print(f"api calls={len(attempts)} tokens={tokens} cost=${cost:.4f}", file=sys.stderr)
    if args.json:
        print(json.dumps(
            [{**asdict(r), "reason": reasons.get(r.candidate_id)} for r in results],
            indent=2, ensure_ascii=False, default=str,
        ))
        return 0
    if not results:
        print("no candidates match")
    for i, r in enumerate(results, 1):
        print(_row(i, r))
        print(_chips_line(r))
        if r.candidate_id in reasons:
            print(f"    reason: {reasons[r.candidate_id]}")
    return 0


def _cmd_eval_run(args: argparse.Namespace) -> int:
    from resume_search.eval.parsing_eval import run_parsing_eval
    from resume_search.eval.ranking_eval import run_ranking_eval
    from resume_search.eval.report import FORMATTERS

    only = [x for x in args.only.split(",") if x] if args.only else None
    suites = ["parsing", "ranking"] if args.suite == "all" else [args.suite]
    total = 0.0
    for suite in suites:
        if suite == "parsing":
            path, body = run_parsing_eval(args.db, args.out, only=only)
        else:
            path, body = run_ranking_eval(args.db, args.out)
        total += body["cost_usd"]
        print(FORMATTERS[suite](body))
        print(f"wrote {path}\n")
    print(f"total judge cost ${total:.4f}")
    return 0


def _cmd_eval_show(args: argparse.Namespace) -> int:
    from resume_search.eval.report import FORMATTERS
    from resume_search.eval.results import SUITES, latest_result

    shown = 0
    for suite in [args.suite] if args.suite else SUITES:
        if found := latest_result(args.out, suite):
            print(f"{found[0]}\n{FORMATTERS[suite](found[1])}\n")
            shown += 1
    if not shown:
        print(f"no eval results in {args.out}", file=sys.stderr)
    return 0 if shown else 1


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="resume-search")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ext = sub.add_parser("extract", help="extract text and print quality metrics")
    p_ext.add_argument("paths", nargs="+")
    p_ext.add_argument("--out", default="exports/text")
    p_ext.set_defaults(func=_cmd_extract)
    p_par = sub.add_parser("parse", help="extract and parse resumes with the LLM")
    p_par.add_argument("paths", nargs="+")
    p_par.add_argument("--out", default="exports/parsed")
    p_par.add_argument("--workers", type=int, default=4)
    p_par.set_defaults(func=_cmd_parse)
    p_ing = sub.add_parser("ingest", help="extract, parse, derive and index resumes into SQLite")
    p_ing.add_argument("paths", nargs="*")
    p_ing.add_argument("--db", default=DEFAULT_DB)
    p_ing.add_argument("--force", action="store_true")
    p_ing.add_argument("--workers", type=int, default=4)
    p_ing.add_argument("--reparse-derive-only", action="store_true")
    p_ing.set_defaults(func=_cmd_ingest)
    p_red = sub.add_parser("rederive", help="rebuild derived fields and vectors without API calls")
    p_red.add_argument("--db", default=DEFAULT_DB)
    p_red.set_defaults(func=_cmd_rederive)
    p_show = sub.add_parser("show", help="print a candidate's derived fields")
    p_show.add_argument("candidate", help="candidate_id prefix or name substring")
    p_show.add_argument("--db", default=DEFAULT_DB)
    p_show.set_defaults(func=_cmd_show)
    p_s = sub.add_parser("search", help="rank ingested candidates against a requisition")
    p_s.add_argument("--jd")
    p_s.add_argument("--jd-file")
    p_s.add_argument("--location")
    p_s.add_argument("--scope", choices=["current", "worked", "either"])
    p_s.add_argument("--approach")
    p_s.add_argument("--sector", action="append")
    p_s.add_argument("--years", help="MIN-MAX, MIN+ or a single number; always a must filter")
    p_s.add_argument("--skill", action="append", help="preferred skill")
    p_s.add_argument("--must-skill", action="append")
    p_s.add_argument("--cfa", action="store_true", help="require the CFA")
    p_s.add_argument("--top", type=int, default=20)
    p_s.add_argument("--rerank", action="store_true")
    p_s.add_argument("--db", default=DEFAULT_DB)
    p_s.add_argument("--json", action="store_true")
    p_s.set_defaults(func=_cmd_search)
    p_ev = sub.add_parser("eval", help="LLM-judged parsing and ranking evals")
    ev = p_ev.add_subparsers(dest="eval_cmd", required=True)
    p_er = ev.add_parser("run", help="run a suite and write evals/results/<suite>-<timestamp>.json")
    p_er.add_argument("--suite", choices=["parsing", "ranking", "all"], default="all")
    p_er.add_argument("--db", default=DEFAULT_DB)
    p_er.add_argument("--out", default="evals/results")
    p_er.add_argument("--only", help="comma-separated candidate_ids (parsing suite)")
    p_er.set_defaults(func=_cmd_eval_run)
    p_es = ev.add_parser("show", help="print the latest metrics table per suite")
    p_es.add_argument("--suite", choices=["parsing", "ranking"])
    p_es.add_argument("--out", default="evals/results")
    p_es.set_defaults(func=_cmd_eval_show)
    args = parser.parse_args(argv)
    sys.exit(args.func(args))
