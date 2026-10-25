from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from resume_search.extract import ExtractedText, extract
from resume_search.parse import ParseOk, parse_resume
from resume_search.parse.client import OPUS, Attempt, ParseResult, verification_dict
from resume_search.parse.prompt import PROMPT_VERSION

SCHEMA_VERSION = 1


SUPPORTED = {".docx", ".pdf"}

# USD per million tokens: (input, output). Cache reads are $0.20 for both models; cache writes
# are billed at 1.25x the input price.
PRICES = {OPUS: (4.0, 20.0)}
SONNET_PRICE = (2.0, 10.0)
CACHE_READ_PRICE = 0.20


def attempt_cost(a: Attempt) -> float:
    p_in, p_out = PRICES.get(a.model, SONNET_PRICE)
    return (
        a.input_tokens * p_in
        + a.cache_creation_tokens * p_in * 1.25
        + a.cache_read_tokens * CACHE_READ_PRICE
        + a.output_tokens * p_out
    ) / 1e6


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


def _meta(e: ExtractedText, result: ParseResult) -> dict:
    return {
        "candidate_id": e.candidate_id,
        "source_file": e.source_file,
        "file_sha256": e.file_sha256,
        "file_bytes": e.file_bytes,
        "page_count": e.page_count,
        "parsed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "extraction_path": e.extraction_path.value,
        "llm_model": result.attempts[-1].model,
        "prompt_version": PROMPT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "parse_status": result.parse_status,
        "attempts": [asdict(a) for a in result.attempts],
        "verification": verification_dict(result.verification) if isinstance(result, ParseOk) else None,
        "text_quality": _quality(e),
    }


def _parse_one(path: str, out: Path) -> tuple[ExtractedText, ParseResult]:
    e = extract(path)
    result = parse_resume(e)
    doc: dict = {"meta": _meta(e, result), "extracted": None}
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
    args = parser.parse_args(argv)
    sys.exit(args.func(args))
