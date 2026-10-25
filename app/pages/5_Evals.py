import json
from pathlib import Path

import common
import pandas as pd
import streamlit as st

from resume_search.eval.results import SUITES, latest_result

st.set_page_config(page_title="Evals", layout="wide")
st.title("Evals and quality")

RESULTS = Path("evals/results")

VARIANTS = {
    "full": "Full blend", "rule_only": "Rule only", "keyword_only": "Keyword only",
    "embedding_only": "Embedding only",
}


def run_caption(path: Path, doc: dict) -> str:
    return (
        f"{path.name}. Run {doc.get('run_at')}. Task model {doc.get('task_model')}, judge "
        f"{doc.get('judge_model')}, judge prompt {doc.get('prompt_version')}. Cost "
        f"${doc.get('cost_usd', 0):.4f}, tokens {doc.get('tokens')}."
    )


def show_parsing(doc: dict) -> None:
    m = doc["metrics"]
    cols = st.columns(4)
    cols[0].metric("Candidates", m["candidates"])
    cols[1].metric("Field accuracy", f"{m['accuracy']:.0%}")
    cols[2].metric("Soft accuracy", f"{m['soft_accuracy']:.0%}")
    cols[3].metric("Hallucinated / missing", f"{m['hallucinated']} / {m['missing']}")
    st.markdown("**Per-field accuracy.** Accuracy is correct over total; soft counts partial as half.")
    st.dataframe(
        pd.DataFrame(m["per_field"]).T.rename_axis("field").reset_index(),
        hide_index=True, column_config={
            "accuracy": st.column_config.ProgressColumn("accuracy", min_value=0, max_value=1),
            "soft_accuracy": st.column_config.ProgressColumn("soft_accuracy", min_value=0, max_value=1),
        },
    )
    ag = m.get("agreement") or {}
    if ag.get("overall"):
        o = ag["overall"]
        st.markdown(
            f"**Judge versus human.** {o['n']} fields on {', '.join(ag['candidates'])} "
            f"(gold status: {', '.join(ag['gold_status'])}). Exact agreement {o['exact']:.0%}, "
            f"within one level {o['within_one_level']:.0%}."
        )
        st.dataframe(
            pd.DataFrame(
                [{"candidate": n, **{k: v for k, v in a.items() if k != "disagreements"}}
                 for n, a in ag["per_candidate"].items()]
            ),
            hide_index=True,
        )
        rows = [{"candidate": n, **d} for n, a in ag["per_candidate"].items() for d in a["disagreements"]]
        if rows:
            st.markdown("**Disagreements.**")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("No human grades found for these candidates (evals/gold/parsing_human.jsonl).")
    st.markdown("**Per candidate.** The judge's grade and reason for every field.")
    for item in sorted(doc["items"], key=lambda i: i["accuracy"]):
        label = f"{item['name']}: {item['accuracy']:.0%} correct, {item['soft_accuracy']:.0%} soft"
        with st.expander(label):
            st.dataframe(pd.DataFrame(item["grades"]), hide_index=True)


def show_ranking(doc: dict) -> None:
    m = doc["metrics"]
    items = doc["items"]
    st.markdown("**nDCG@5 and P@3 by requisition.** Same judged pool and grades for every variant.")
    rows = [
        {"requisition": i["title"], "pool": i["pool_size"], "relevant": i["relevant_in_pool"],
         **{f"{label} nDCG@5": i[f"{v}_ndcg5"] for v, label in VARIANTS.items()},
         **{f"{label} P@3": i[f"{v}_p3"] for v, label in VARIANTS.items()}}
        for i in items
    ]
    rows.append({
        "requisition": "Mean", "pool": None, "relevant": None,
        **{f"{label} nDCG@5": m["mean"][v]["ndcg5"] for v, label in VARIANTS.items()},
        **{f"{label} P@3": m["mean"][v]["p3"] for v, label in VARIANTS.items()},
    })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(m.get("tie_break", ""))
    for i in items:
        with st.expander(f"{i['title']}: pool {i['pool_size']}, full nDCG@5 {i['full_ndcg5']:.2f}"):
            st.dataframe(
                pd.DataFrame(sorted(i["candidates"], key=lambda c: c["rank_full"])), hide_index=True
            )


SHOW = {"parsing": show_parsing, "ranking": show_ranking}

st.header("Eval results")
found = {s: latest_result(RESULTS, s) for s in SUITES}
if not any(found.values()):
    st.info(
        "No eval results yet. Run `uv run resume-search eval run --suite all`; it writes "
        "evals/results/<suite>-<timestamp>.json and this page shows the latest file per suite."
    )
for suite, hit in found.items():
    if hit is None:
        continue
    path, doc = hit
    st.subheader(suite.capitalize())
    st.caption(run_caption(path, doc))
    try:
        SHOW[suite](doc)
    except (KeyError, TypeError) as exc:
        st.error(f"{path.name} does not match the expected format: {exc!r}")
    with st.expander("Raw JSON"):
        st.json(doc)

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
