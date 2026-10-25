from datetime import date

import common
import pandas as pd
import plotly.express as px
import streamlit as st

from resume_search.store.db import get_candidate

st.set_page_config(page_title="Candidate", layout="wide")
st.title("Candidate")

db = common.conn()
rows = db.execute("SELECT candidate_id, name FROM candidates ORDER BY name").fetchall()
if not rows:
    st.info("No candidates in the database yet. Ingest some resumes first.")
    st.stop()
names = {r["candidate_id"]: r["name"] for r in rows}
ids = list(names)
wanted = st.query_params.get("candidate_id")
index = ids.index(wanted) if wanted in names else 0
# The id disambiguates only names that repeat.
repeated = {n for n in names.values() if list(names.values()).count(n) > 1}
cid = st.selectbox(
    "Candidate", ids, index=index,
    format_func=lambda i: f"{names[i]} ({i})" if names[i] in repeated else names[i],
)
st.query_params["candidate_id"] = cid

rec = get_candidate(db, cid)
ex, d, meta = rec.extracted, rec.derived, rec.meta

st.header(ex.name)
st.write(d.search_summary)
c = st.columns(4)
c[0].metric("Location", common.location_label(d.current_country, d.current_region))
c[1].metric("Seniority", d.seniority_band or common.UNKNOWN)
c[2].metric("Relevant years", f"{d.months_full_time_relevant / 12:.1f}",
            help="Full-time roles in investing or something close to it. This is what sets the seniority band.")
c[3].metric("Internship years", f"{d.months_internship / 12:.1f}")
c = st.columns(4)
c[0].metric("Core years", f"{d.months_full_time_core / 12:.1f}",
            help="Full-time roles in investing proper. A subset of the relevant years.")
c[1].metric("Approach", d.investment_approach)
c[2].metric("Highest degree", d.highest_degree or "n/a")
c[3].metric("CFA", (d.cfa_level or "yes") if d.has_cfa else "no")
st.caption(f"Approach evidence: {d.approach_evidence}")

st.subheader("Experience")
exp = pd.read_sql(
    "SELECT employer, title, employment_type AS type, relevance, firm_type, start, end, "
    "is_current, months, country FROM experience WHERE candidate_id = ? ORDER BY idx",
    db, params=(cid,),
)
sectors_by_role = [", ".join(s.label for s in e.sectors) for e in ex.experience]
exp["sectors"] = sectors_by_role[: len(exp)]
st.dataframe(exp, hide_index=True)


def _date(v: str | None, end: bool = False) -> pd.Timestamp | None:
    if not v:
        return None
    if len(v) == 4:
        v = f"{v}-12-31" if end else f"{v}-01-01"
    else:
        v = f"{v}-01"
        if end:
            return pd.Timestamp(v) + pd.offsets.MonthEnd(0)
    return pd.Timestamp(v)


gantt = []
for role in ex.experience:
    for start, end in zip(role.start, role.end, strict=True):
        finish = pd.Timestamp(date.today()) if end is None else _date(end, end=True)
        gantt.append({
            "role": f"{role.title} @ {role.employer}", "start": _date(start), "end": finish,
            "relevance": role.relevance.value, "type": role.employment_type.value,
        })
if gantt:
    fig = px.timeline(
        pd.DataFrame(gantt), x_start="start", x_end="end", y="role", color="relevance",
        hover_data=["type"],
    )
    fig.update_yaxes(autorange="reversed", title=None)
    st.plotly_chart(fig)
else:
    st.caption("None of the roles carry dates.")

st.subheader("Education")
edu = pd.read_sql(
    "SELECT institution, degree, degree_level, field, minor_fields, start, end, country FROM education "
    "WHERE candidate_id = ? ORDER BY idx", db, params=(cid,),
)
st.dataframe(edu, hide_index=True)

st.subheader("Skills")
skills = pd.read_sql(
    "SELECT skill, source FROM candidate_skills WHERE candidate_id = ? ORDER BY skill", db,
    params=(cid,),
)
left, right = st.columns(2)
left.markdown("**LLM** (source llm or both)")
left.write(", ".join(skills[skills["source"].isin(["llm", "both"])]["skill"]) or "none")
right.markdown("**Taxonomy matched** (source ruler or both)")
right.write(", ".join(skills[skills["source"].isin(["ruler", "both"])]["skill"]) or "none")
raw_skills = ex.skills
st.caption(
    "As written. Programming: " + (", ".join(raw_skills.programming) or "none")
    + ". Tools: " + (", ".join(raw_skills.tools) or "none")
    + ". Domain: " + (", ".join(raw_skills.domain) or "none") + "."
)

c = st.columns(2)
c[0].subheader("Certifications")
c[0].write(", ".join(ex.certifications) or "none")
c[1].subheader("Languages")
c[1].write(", ".join(f"{lang.name} ({lang.level or 'n/a'})" for lang in ex.languages) or "none")

st.subheader("Sector evidence")
if d.sectors:
    for s in d.sectors:
        st.markdown(f"**{s.label}**: \"{s.evidence}\"")
else:
    st.write("none")

st.subheader("Verification")
ver = meta.get("verification") or {}
tq = meta.get("text_quality") or {}
c = st.columns(4)
c[0].metric("Checks run", ver.get("checks_run", "n/a"))
c[1].metric("Disagreements", ver.get("disagreements", "n/a"))
c[2].metric("Regenerated", ver.get("regenerated", "n/a"))
c[3].metric("Unresolved", len(ver.get("unresolved", [])))
if ver.get("unresolved"):
    st.warning("Still disagree after regeneration (kept the LLM value): " + "; ".join(str(u) for u in ver["unresolved"]))
if ver.get("details"):
    with st.expander("Disagreement details"):
        st.json(ver["details"])
st.caption(
    f"Model {meta.get('llm_model')}, prompt {meta.get('prompt_version')}, "
    f"status {meta.get('parse_status')}, extraction {meta.get('extraction_path')}, "
    f"unknown-word ratio {tq.get('unknown_ratio_after', 'n/a')}"
)

with st.expander("Raw resume text (contact block removed)"):
    row = db.execute("SELECT body FROM candidate_fts WHERE candidate_id = ?", (cid,)).fetchone()
    st.text(row["body"] if row else "No text stored.")
