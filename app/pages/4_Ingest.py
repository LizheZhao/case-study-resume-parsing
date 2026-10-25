import threading
from datetime import datetime
from pathlib import Path

import common
import pandas as pd
import streamlit as st

from resume_search.ingest import pipeline

st.set_page_config(page_title="Ingest", layout="wide")
st.title("Ingest")

db_path = common.db_path()
ss = st.session_state

STATUS_HELP = (
    "queued → extracting → parsing → deriving → indexing → ok. Any running step can end in "
    "failed. skipped means the same file and prompt version already parsed ok."
)
STATUS_COLUMN = {"status": st.column_config.TextColumn("status", help=STATUS_HELP)}


def _run(holder: dict, paths: list[str], force: bool, workers: int) -> None:
    try:
        holder["results"] = pipeline.ingest(paths, db_path, force=force, workers=workers)
    except Exception as exc:  # noqa: BLE001  reported on the page
        holder["error"] = f"{type(exc).__name__}: {exc}"


def start(paths: list[str], force: bool, workers: int) -> None:
    last_id = common.conn().execute("SELECT COALESCE(MAX(job_id), 0) FROM parse_jobs").fetchone()[0]
    holder: dict = {"results": None, "error": None, "after": last_id, "files": [Path(p).name for p in paths]}
    holder["thread"] = threading.Thread(
        target=_run, args=(holder, paths, force, workers), daemon=True
    )
    holder["thread"].start()
    ss["_ingest"] = holder


holder = ss.get("_ingest")
running = holder is not None and holder["thread"].is_alive()

uploads = st.file_uploader("Resumes", type=["docx", "pdf"], accept_multiple_files=True)
with st.expander("Advanced"):
    c1, c2 = st.columns(2)
    force = c1.checkbox("Force re-parse (ignore the file hash skip)")
    workers = c2.slider("Workers", 1, 8, 4)

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


actions = st.container(horizontal=True)
retry = failed_files()
if actions.button("Ingest", type="primary", disabled=running or not uploads):
    folder = common.uploads_dir()
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in uploads:
        p = folder / Path(f.name).name
        p.write_bytes(f.getvalue())
        paths.append(str(p))
    start(paths, force, workers)
    st.rerun()
if actions.button("Re-run failed", disabled=running or not retry, help=f"{len(retry)} files found on disk"):
    start(retry, True, workers)
    st.rerun()
if actions.button("Rederive all (no API)", disabled=running):
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


def this_run(df: pd.DataFrame) -> None:
    """One row per file in the latest run from this page, its job's current state."""
    if holder is None:
        return
    jobs = df[df["job_id"] > holder["after"]].drop_duplicates("source_file")
    rows = pd.DataFrame({"source_file": holder["files"]}).merge(jobs, on="source_file", how="left")
    rows["status"] = rows["status"].fillna("waiting")
    rows["seconds"] = (rows["latency_ms"] / 1000).round(1)
    st.subheader("This run")
    st.dataframe(rows[["source_file", "status", "seconds", "error"]], hide_index=True,
                 column_config=STATUS_COLUMN)


def history(df: pd.DataFrame) -> None:
    with st.expander(f"Job history ({len(df)} jobs)"):
        everything = st.toggle("Show every job", help="Off shows only the latest job for each file.")
        shown = df if everything else df.drop_duplicates("source_file")
        st.dataframe(shown, hide_index=True, column_config=STATUS_COLUMN)


def panel() -> None:
    df = jobs_frame()
    this_run(df)
    metrics(df)
    st.caption(f"Updated {datetime.now():%H:%M:%S}")
    if holder is not None and not holder["thread"].is_alive():
        st.rerun()


if running:
    st.info("Ingest is running. The table refreshes itself every 2 seconds.")
    st.fragment(run_every=2)(panel)()
else:
    if holder is not None and not holder.get("handled"):
        holder["handled"] = True
        common.clear_caches()
    if holder is not None and holder["error"]:
        st.error(holder["error"])
    df = jobs_frame()
    this_run(df)
    st.subheader("Totals")
    metrics(df)
    history(df)
