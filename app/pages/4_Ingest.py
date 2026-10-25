import threading
from datetime import datetime
from pathlib import Path

import common
import pandas as pd
import streamlit as st

from resume_search.ingest import pipeline
from resume_search.ingest.jobs import JobStatus

st.set_page_config(page_title="Ingest", layout="wide")
st.title("Ingest")

db_path = common.db_path()
ss = st.session_state

st.markdown(
    "Job states: `queued` -> `extracting` -> `parsing` -> `deriving` -> `indexing` -> `ok`. "
    "Any running state can end in `failed`. A `queued` job ends as `skipped` when the same file "
    "and prompt version already parsed ok."
)
st.caption("States: " + ", ".join(s.value for s in JobStatus))


def _run(holder: dict, paths: list[str], force: bool, workers: int) -> None:
    try:
        holder["results"] = pipeline.ingest(paths, db_path, force=force, workers=workers)
    except Exception as exc:  # noqa: BLE001  reported on the page
        holder["error"] = f"{type(exc).__name__}: {exc}"


def start(paths: list[str], force: bool, workers: int) -> None:
    holder: dict = {"results": None, "error": None}
    holder["thread"] = threading.Thread(
        target=_run, args=(holder, paths, force, workers), daemon=True
    )
    holder["thread"].start()
    ss["_ingest"] = holder


holder = ss.get("_ingest")
running = holder is not None and holder["thread"].is_alive()

uploads = st.file_uploader("Resumes", type=["docx", "pdf"], accept_multiple_files=True)
c1, c2 = st.columns(2)
force = c1.checkbox("Force re-parse (ignore the file hash skip)")
workers = c2.slider("Workers", 1, 8, 4)
if st.button("Ingest", type="primary", disabled=running or not uploads):
    folder = common.uploads_dir()
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in uploads:
        p = folder / Path(f.name).name
        p.write_bytes(f.getvalue())
        paths.append(str(p))
    start(paths, force, workers)
    st.rerun()


def failed_files() -> list[str]:
    rows = common.conn().execute(
        "SELECT DISTINCT source_file FROM parse_jobs WHERE status = 'failed' AND file_sha256 NOT IN "
        "(SELECT file_sha256 FROM parse_jobs WHERE status = 'ok')"
    ).fetchall()
    found = []
    for r in rows:
        for folder in (common.uploads_dir(), Path("data")):
            if (folder / r["source_file"]).is_file():
                found.append(str(folder / r["source_file"]))
                break
    return found


c1, c2 = st.columns(2)
retry = failed_files()
if c1.button("Re-run failed", disabled=running or not retry, help=f"{len(retry)} files found on disk"):
    start(retry, True, workers)
    st.rerun()
if c2.button("Rederive all (no API)", disabled=running):
    with st.spinner("Rederiving"):
        n = pipeline.rederive(db_path)
    common.clear_caches()
    st.success(f"Rederived {n} candidates")


def jobs_frame() -> pd.DataFrame:
    return pd.read_sql(
        "SELECT job_id, source_file, status, input_tokens, output_tokens, cost_usd, latency_ms, "
        "error, queued_at, finished_at FROM parse_jobs ORDER BY job_id DESC", common.conn(),
    )


def metrics(df: pd.DataFrame) -> None:
    ok = df[df["status"] == "ok"]
    c = st.columns(4)
    c[0].metric("Parsed ok", len(ok))
    c[1].metric("Failed", int((df["status"] == "failed").sum()))
    c[2].metric("Cumulative cost (USD)", f"{df['cost_usd'].fillna(0).sum():.4f}")
    rate = "n/a"
    if len(ok):
        t0 = pd.to_datetime(ok["queued_at"]).min()
        t1 = pd.to_datetime(ok["finished_at"]).max()
        minutes = (t1 - t0).total_seconds() / 60
        if minutes > 0:
            rate = f"{len(ok) / minutes:.1f}"
    c[3].metric("Throughput (resumes/min)", rate)
    if len(ok):
        st.caption(f"Mean latency per resume: {ok['latency_ms'].mean() / 1000:.1f}s")


def panel() -> None:
    df = jobs_frame()
    metrics(df)
    st.caption(f"Updated {datetime.now():%H:%M:%S}")
    st.dataframe(df, hide_index=True)
    if holder is not None and not holder["thread"].is_alive():
        st.rerun()


st.button("Refresh")
if running:
    st.info("Ingest running. The table refreshes every 2 seconds.")
    st.fragment(run_every=2)(panel)()
else:
    if holder is not None and not holder.get("handled"):
        holder["handled"] = True
        common.clear_caches()
    if holder is not None and holder["error"]:
        st.error(holder["error"])
    elif holder is not None and holder["results"] is not None:
        st.success(
            "Last run: " + ", ".join(f"{j.source_file} {j.status.value}" for j in holder["results"])
        )
    df = jobs_frame()
    metrics(df)
    st.dataframe(df, hide_index=True)
