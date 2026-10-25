"""Results files: evals/results/<suite>-<UTC timestamp>.json."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

SUITES = ("parsing", "ranking")
_NAME = re.compile(r"^(parsing|ranking)-\d{8}T\d{6}Z\.json$")


def now_utc() -> datetime:
    return datetime.now(UTC)


def write_results(out_dir: str | Path, suite: str, run_at: datetime, body: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{suite}-{run_at.strftime('%Y%m%dT%H%M%SZ')}.json"
    doc = {"suite": suite, "run_at": run_at.isoformat(timespec="seconds"), **body}
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def result_files(out_dir: str | Path, suite: str) -> list[Path]:
    out = Path(out_dir)
    if not out.is_dir():
        return []
    return sorted(
        p for p in out.iterdir() if (m := _NAME.match(p.name)) and m.group(1) == suite
    )


def latest_result(out_dir: str | Path, suite: str) -> tuple[Path, dict] | None:
    for path in reversed(result_files(out_dir, suite)):
        try:
            return path, json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None
