import json
from pathlib import Path

import common
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Evals", layout="wide")
st.title("Evals and quality")

RESULTS = Path("evals/results")

st.header("Eval results")
files = sorted(RESULTS.glob("*.json")) if RESULTS.is_dir() else []
if not files:
    st.info(
        "No eval results yet. When the harness runs it writes evals/results/<suite>-<timestamp>.json "
        "and this page will show the latest file per suite: a metrics table and a per-item table "
        "(parsing grades, ranking nDCG@5 and P@3, judge versus human agreement)."
    )
else:
    latest: dict[str, Path] = {}
    for f in files:
        latest[f.stem.rsplit("-", 1)[0] if "-" in f.stem else f.stem] = f
    for suite, f in latest.items():
        st.subheader(f"{suite} ({f.name})")
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Cannot read {f.name}: {exc}")
            continue
        body = data if isinstance(data, dict) else {"items": data}
        scalars = {k: v for k, v in body.items() if isinstance(v, (int, float, str, bool))}
        metrics = body.get("metrics") if isinstance(body.get("metrics"), dict) else scalars
        if metrics:
            st.dataframe(pd.DataFrame([metrics]).T.rename(columns={0: "value"}))
        items = next(
            (v for k, v in body.items() if k != "metrics" and isinstance(v, list) and v), None
        )
        if items:
            st.dataframe(pd.json_normalize(items), hide_index=True)
        with st.expander("Raw JSON"):
            st.json(data)

st.header("Per-candidate parse quality")
rows = []
for r in common.conn().execute("SELECT candidate_id, name, record_json FROM candidates ORDER BY name"):
    meta = json.loads(r["record_json"])["meta"]
    ver = meta.get("verification") or {}
    tq = meta.get("text_quality") or {}
    rows.append({
        "candidate": r["name"], "parse_status": meta.get("parse_status"),
        "checks_run": ver.get("checks_run"), "disagreements": ver.get("disagreements"),
        "regenerated": ver.get("regenerated"), "unresolved": len(ver.get("unresolved", [])),
        "unresolved_items": "; ".join(str(u) for u in ver.get("unresolved", [])),
        "unknown_ratio_before": tq.get("unknown_ratio_before"),
        "unknown_ratio_after": tq.get("unknown_ratio_after"), "repairs": tq.get("repairs"),
        "chars_per_page": tq.get("chars_per_page"),
        "image_page_coverage": tq.get("max_image_page_coverage"),
        "extraction": meta.get("extraction_path"),
    })
if rows:
    df = pd.DataFrame(rows)
    st.metric("Candidates with unresolved verification items", int((df["unresolved"] > 0).sum()))
    st.dataframe(df, hide_index=True)
else:
    st.info("No candidates yet.")
