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
    args = parser.parse_args(argv)
    sys.exit(args.func(args))
