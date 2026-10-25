"""ingest(): sha256 -> skip check -> extract -> parse (worker threads) -> derive -> index (main
thread). SQLite connections stay on the main thread, so workers only return results."""

from __future__ import annotations

import hashlib
import json
import queue
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from resume_search.derive import derive
from resume_search.derive.models import SCHEMA_VERSION, CandidateRecord
from resume_search.extract import ExtractedText, extract
from resume_search.ingest.jobs import Job, JobResult, JobStatus, advance, already_ok
from resume_search.parse import ParseOk, parse_resume
from resume_search.parse.client import OPUS, Attempt, ParseResult, verification_dict
from resume_search.parse.models import ParsedResume
from resume_search.parse.prompt import PROMPT_VERSION
from resume_search.search.vectors import Embedder, candidate_vectors, default_embedder
from resume_search.store.db import connect, list_candidates, load_taxonomy, upsert_candidate
from resume_search.store.export import export_csv, export_json
from resume_search.store.fts import fts_body
from resume_search.taxonomy import Taxonomy, load_taxonomy_data

S = JobStatus

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


def build_meta(e: ExtractedText, result: ParseResult) -> dict:
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
        "text_quality": asdict(e.quality),
    }


def _accounting(result: ParseResult, latency_ms: int) -> dict:
    atts = result.attempts
    return {
        "attempts_json": json.dumps([asdict(a) for a in atts]),
        "input_tokens": sum(a.input_tokens + a.cache_read_tokens + a.cache_creation_tokens for a in atts),
        "output_tokens": sum(a.output_tokens for a in atts),
        "cost_usd": sum(attempt_cost(a) for a in atts),
        "latency_ms": latency_ms,
    }


def _work(i: int, path: str, events: queue.Queue) -> tuple[ExtractedText, ParseResult, int]:
    started = time.monotonic()
    events.put((i, S.extracting))
    e = extract(path)
    events.put((i, S.parsing))
    result = parse_resume(e)
    return e, result, round((time.monotonic() - started) * 1000)


def _index(
    conn, job: Job, e: ExtractedText, result: ParseOk, tax: Taxonomy, embedder: Embedder
) -> None:
    meta = build_meta(e, result)
    derived = derive(result.record, meta, tax)
    advance(conn, job, S.indexing)
    record = CandidateRecord(meta=meta, extracted=result.record, derived=derived)
    upsert_candidate(conn, record, fts_body(e.text), candidate_vectors(record, embedder))
    advance(conn, job, S.ok, candidate_id=e.candidate_id)


def ingest(
    paths: Sequence[str | Path],
    db_path: str | Path,
    *,
    force: bool = False,
    workers: int = 4,
    export_dir: str | Path | None = None,
    embedder: Embedder | None = None,
) -> list[JobResult]:
    conn = connect(db_path)
    load_taxonomy(conn)
    tax = load_taxonomy_data()
    embedder = embedder or default_embedder()
    events: queue.Queue = queue.Queue()
    jobs: list[Job] = []
    pending: dict[Future, int] = {}

    def drain() -> None:
        while True:
            try:
                i, status = events.get_nowait()
            except queue.Empty:
                return
            advance(conn, jobs[i], status)

    def fail(job: Job, exc: BaseException | str, **fields) -> None:
        msg = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
        advance(conn, job, S.failed, error=msg, **fields)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, path in enumerate(map(str, paths)):
            try:
                sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            except OSError as exc:
                job = Job("", Path(path).name, PROMPT_VERSION)
                jobs.append(job)
                advance(conn, job, S.queued)
                fail(job, exc)
                continue
            job = Job(sha, Path(path).name, PROMPT_VERSION, candidate_id=sha[:12])
            jobs.append(job)
            advance(conn, job, S.queued)
            if not force and already_ok(conn, sha, PROMPT_VERSION):
                advance(conn, job, S.skipped)
                continue
            pending[pool.submit(_work, i, path, events)] = i

        while pending:
            done, _ = wait(pending, timeout=0.2, return_when="FIRST_COMPLETED")
            drain()
            for fut in done:
                job = jobs[pending.pop(fut)]
                try:
                    e, result, latency = fut.result()
                    acct = _accounting(result, latency)
                    if not isinstance(result, ParseOk):
                        fail(job, result.last_error, **acct)
                        continue
                    advance(conn, job, S.deriving, **acct)
                    _index(conn, job, e, result, tax, embedder)
                except Exception as exc:  # noqa: BLE001  one bad file must not stop the batch
                    fail(job, exc)

    out = Path(export_dir) if export_dir else Path(db_path).parent
    export_json(conn, out / "candidates.json")
    export_csv(conn, out / "candidates.csv")
    conn.close()
    return jobs


def rederive(
    db_path: str | Path, *, export_dir: str | Path | None = None, embedder: Embedder | None = None
) -> int:
    """Rebuild derived fields and vectors for every candidate from record_json, keeping the
    stored FTS body (the source text is not kept). No API calls."""
    conn = connect(db_path)
    load_taxonomy(conn)
    tax = load_taxonomy_data()
    embedder = embedder or default_embedder()
    records = list_candidates(conn)
    for r in records:
        cid = r.meta["candidate_id"]
        meta = {**r.meta, "schema_version": SCHEMA_VERSION}
        parsed = ParsedResume.model_validate(r.extracted.model_dump())
        new = CandidateRecord(meta=meta, extracted=parsed, derived=derive(parsed, meta, tax))
        row = conn.execute("SELECT body FROM candidate_fts WHERE candidate_id = ?", (cid,)).fetchone()
        upsert_candidate(conn, new, row["body"] if row else "", candidate_vectors(new, embedder))
    out = Path(export_dir) if export_dir else Path(db_path).parent
    export_json(conn, out / "candidates.json")
    export_csv(conn, out / "candidates.csv")
    conn.close()
    return len(records)
