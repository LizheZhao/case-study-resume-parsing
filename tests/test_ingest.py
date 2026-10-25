import sqlite3

import pytest
from builders import resume

from resume_search.extract.types import ExtractedText, ExtractionPath, SourceFormat, TextQuality
from resume_search.ingest import pipeline
from resume_search.ingest.jobs import Job, JobStatus, advance
from resume_search.parse import ParseOk
from resume_search.parse.client import Attempt, ParseFailed


def _extracted(path: str, text: str = "Test Person\nBuilt a factor model.") -> ExtractedText:
    import hashlib
    from pathlib import Path

    data = Path(path).read_bytes()
    return ExtractedText(
        source_file=Path(path).name, file_sha256=hashlib.sha256(data).hexdigest(),
        file_bytes=len(data), format=SourceFormat.docx, page_count=1, text=text,
        quality=TextQuality(0.0, 0.0, 0, 1000.0, 0.0), extraction_path=ExtractionPath.text,
    )


ATTEMPT = Attempt("claude-sonnet-5-5", 1000, 1_000_000, 100_000, None, call="profile")


@pytest.fixture
def files(tmp_path):
    paths = []
    for name in ("a.docx", "b.docx"):
        p = tmp_path / name
        p.write_bytes(name.encode())
        paths.append(str(p))
    return paths


@pytest.fixture
def calls(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(pipeline, "extract", lambda path: _extracted(path))

    def fake_parse(e):
        seen.append(e.source_file)
        if e.source_file == "a.docx":
            raise RuntimeError("boom")
        return ParseOk(resume(name="Test Person"), [ATTEMPT])

    monkeypatch.setattr(pipeline, "parse_resume", fake_parse)
    return seen


def _statuses(jobs):
    return {j.source_file: j.status for j in jobs}


def test_second_run_skips_and_force_reruns(tmp_path, files, calls):
    db = tmp_path / "t.sqlite"
    b = [files[1]]
    first = pipeline.ingest(b, db)
    assert _statuses(first) == {"b.docx": JobStatus.ok}
    assert first[0].cost_usd == pytest.approx(2.0 + 1.0)
    second = pipeline.ingest(b, db)
    assert _statuses(second) == {"b.docx": JobStatus.skipped}
    assert calls == ["b.docx"]
    third = pipeline.ingest(b, db, force=True)
    assert _statuses(third) == {"b.docx": JobStatus.ok}
    assert calls == ["b.docx", "b.docx"]
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 1
    assert (tmp_path / "candidates.json").exists()


def test_failure_does_not_stop_batch(tmp_path, files, calls):
    jobs = pipeline.ingest(files, tmp_path / "t.sqlite", workers=2)
    st = _statuses(jobs)
    assert st == {"a.docx": JobStatus.failed, "b.docx": JobStatus.ok}
    assert "boom" in next(j for j in jobs if j.source_file == "a.docx").error
    conn = sqlite3.connect(tmp_path / "t.sqlite")
    assert conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 1


def test_parse_failed_result_marks_failed_with_cost(tmp_path, files, monkeypatch):
    monkeypatch.setattr(pipeline, "extract", lambda path: _extracted(path))
    monkeypatch.setattr(pipeline, "parse_resume", lambda e: ParseFailed([ATTEMPT], "bad schema"))
    (job,) = pipeline.ingest([files[1]], tmp_path / "t.sqlite")
    assert job.status is JobStatus.failed
    assert job.error == "bad schema"
    assert job.cost_usd > 0


def test_illegal_transition_rejected(tmp_path):
    conn = pipeline.connect(tmp_path / "t.sqlite")
    job = advance(conn, Job("sha", "f.docx", "v2"), JobStatus.queued)
    with pytest.raises(ValueError):
        advance(conn, job, JobStatus.ok)
