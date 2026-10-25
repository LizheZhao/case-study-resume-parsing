"""Per-file job row and its state machine. Every status change goes through advance()."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


class JobStatus(str, Enum):
    queued = "queued"
    extracting = "extracting"
    parsing = "parsing"
    deriving = "deriving"
    indexing = "indexing"
    ok = "ok"
    failed = "failed"
    skipped = "skipped"


S = JobStatus
TERMINAL = frozenset({S.ok, S.failed, S.skipped})
_NEXT = {
    S.queued: {S.extracting, S.skipped, S.failed},
    S.extracting: {S.parsing, S.failed},
    S.parsing: {S.deriving, S.failed},
    S.deriving: {S.indexing, S.failed},
    S.indexing: {S.ok, S.failed},
}

_COLUMNS = (
    "attempts_json", "input_tokens", "output_tokens", "cost_usd", "latency_ms", "error",
    "finished_at",
)


@dataclass
class Job:
    file_sha256: str
    source_file: str
    prompt_version: str
    status: JobStatus = S.queued
    job_id: int | None = None
    candidate_id: str | None = None
    attempts_json: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    error: str | None = None
    queued_at: str | None = None
    finished_at: str | None = None


JobResult = Job


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def advance(conn: sqlite3.Connection, job: Job, status: JobStatus, **fields) -> Job:
    """Move the job to `status`, apply `fields`, and write the row. A job with no row yet is
    inserted. Terminal statuses stamp finished_at."""
    if job.job_id is not None and status not in _NEXT.get(job.status, set()):
        raise ValueError(f"illegal job transition {job.status.value} -> {status.value}")
    for k, v in fields.items():
        if k not in _COLUMNS and k != "candidate_id":
            raise TypeError(f"unknown job field '{k}'")
        setattr(job, k, v)
    job.status = status
    if status in TERMINAL:
        job.finished_at = _now()
    with conn:
        if job.job_id is None:
            job.queued_at = _now()
            cur = conn.execute(
                "INSERT INTO parse_jobs (file_sha256, source_file, prompt_version, status, queued_at)"
                " VALUES (?,?,?,?,?)",
                (job.file_sha256, job.source_file, job.prompt_version, status.value, job.queued_at),
            )
            job.job_id = cur.lastrowid
        sets = ", ".join(f"{c} = ?" for c in ("status", *_COLUMNS))
        conn.execute(
            f"UPDATE parse_jobs SET {sets} WHERE job_id = ?",
            (status.value, *(getattr(job, c) for c in _COLUMNS), job.job_id),
        )
    return job


def already_ok(conn: sqlite3.Connection, file_sha256: str, prompt_version: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM parse_jobs WHERE file_sha256 = ? AND prompt_version = ? AND status = 'ok'",
            (file_sha256, prompt_version),
        ).fetchone()
        is not None
    )
