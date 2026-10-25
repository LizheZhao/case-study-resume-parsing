import common
import pandas as pd
import streamlit as st

from resume_search.search import Requisition, parse_requisition, rerank_llm
from resume_search.search.requisition import APPROACHES, BANDS, DEGREES

st.set_page_config(page_title="Search", layout="wide")
st.title("Search")

common.init_state()
tax = common.get_taxonomy()

# A parse request set by the button is applied here, before any widget is instantiated.
if st.session_state.pop("_parse_pending", False):
    try:
        with st.spinner("Parsing requisition"):
            parsed = parse_requisition(st.session_state["jd_text"])
        st.session_state.update(common.state_from_requisition(parsed))
        st.session_state["_parsed"] = parsed
        st.session_state.pop("_parse_error", None)
    except Exception as exc:  # noqa: BLE001  surfaced to the user below
        st.session_state["_parse_error"] = f"{type(exc).__name__}: {exc}"

ss = st.session_state
locations = common.location_options()

with st.sidebar:
    st.header("Filters")
    st.selectbox("Location", list(locations), key="f_location")
    st.radio("Location applies to", ["current", "worked", "either"], key="f_scope", horizontal=True)
    st.selectbox("Approach", [common.ANY, *APPROACHES], key="f_approach")
    st.multiselect("Sectors", tax["sectors"], key="f_sectors")
    st.multiselect("Markets covered", tax["markets"], key="f_markets")
    st.slider("Years of relevant experience", 0, common.YEARS_MAX, key="f_years")
    st.checkbox("Years is a hard filter", key="f_years_hard")
    st.multiselect("Seniority bands", list(BANDS), key="f_bands")
    st.selectbox("Degree (minimum)", [common.ANY, *DEGREES], key="f_degree")
    st.checkbox("CFA", key="f_cfa")
    st.text_input("Other certifications (comma-separated)", key="f_certs")
    st.text_input("Required skills (comma-separated)", key="f_skills_req")
    st.text_input("Preferred skills (comma-separated)", key="f_skills_pref")
    st.text_input("Languages (comma-separated)", key="f_languages")
    st.multiselect("Asset classes", tax["asset_classes"], key="f_assets")
    options = common.set_criteria(ss)
    ss["f_must"] = [f for f in ss["f_must"] if f in options]
    st.multiselect(
        "Treat as must", options, key="f_must", format_func=lambda f: common.MODE_LABELS[f],
        help="Must criteria prune candidates. The rest only score. Required skills are always must.",
    )

with st.expander("Advanced: ranking weights"):
    base = common.get_weights()
    rule_w, blend_w = {}, {}
    cols = st.columns(5)
    for i, (k, v) in enumerate(base.rule.items()):
        rule_w[k] = cols[i % 5].number_input(f"rule: {k}", 0.0, 20.0, float(v), 0.5, key=f"w_rule_{k}")
    cols = st.columns(3)
    for i, name in enumerate(("blend_rule", "blend_keyword", "blend_embedding")):
        blend_w[name] = cols[i].number_input(
            name.replace("_", " "), 0.0, 1.0, float(getattr(base, name)), 0.05, key=f"w_{name}"
        )
    cols = st.columns(4)
    adv = {
        "location_current": cols[0].number_input("location: current", 0.0, 1.0, base.location_current, 0.1),
        "location_worked": cols[1].number_input("location: worked", 0.0, 1.0, base.location_worked, 0.1),
        "location_region_factor": cols[2].number_input(
            "location: region factor", 0.0, 1.0, base.location_region_factor, 0.1
        ),
        "years_decay": cols[3].number_input("years decay (y)", 0.1, 10.0, base.years_decay, 0.5),
    }
weights = {**common.weights_to_dict(base), "rule": rule_w, **blend_w, **adv}

st.text_area(
    "Job description or keywords", key="jd_text", height=160,
    help="Always feeds the keyword and embedding scores. Parse it to pre-fill the filters.",
)
c1, c2, c3 = st.columns([1, 1, 3])
c1.button(
    "Parse requisition", on_click=lambda: ss.update(_parse_pending=True),
    disabled=not ss["jd_text"].strip(), help="One API call",
)
do_search = c2.button("Search", type="primary")
top_k = c3.number_input("Top", 1, 50, 20, label_visibility="collapsed")

if ss.get("_parse_error"):
    st.error(ss["_parse_error"])
if ss.get("_parsed") is not None:
    st.caption("Parsed requisition. Edit it with the sidebar filters.")
    st.dataframe(pd.DataFrame(common.requisition_rows(ss["_parsed"])), hide_index=True)

try:
    req = common.requisition_from_state(ss)
except ValueError as exc:
    st.error(f"Invalid requisition: {exc}")
    st.stop()

if do_search:
    ss["_last"] = {"req": common.req_cache_key(req), "weights": weights, "top": int(top_k)}
last = ss.get("_last")
if last is None:
    st.info("Set filters or paste a job description, then press Search.")
    st.stop()

results = common.run_search(last["req"], last["weights"], common.db_path(), last["top"])
searched = Requisition.model_validate_json(last["req"])
if searched.is_empty():
    st.caption("Empty requisition: every candidate matches.")
st.subheader(f"{len(results)} candidates")

reasons: dict[str, str] = {}
if st.toggle("Rerank top 10 with LLM", value=False, help="One API call"):
    key = f"{last['req']}|{[r.candidate_id for r in results[:10]]}"
    cache = ss.setdefault("_rerank", {})
    if key not in cache:
        try:
            with st.spinner("Reranking"):
                cache[key] = rerank_llm(searched, results[:10])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Rerank failed: {type(exc).__name__}: {exc}")
    if key in cache:
        reasons = dict(cache[key])
        by_id = {r.candidate_id: r for r in results}
        results = [by_id[i] for i, _ in cache[key]] + results[10:]

for rank, r in enumerate(results, 1):
    with st.container(border=True):
        head, score = st.columns([4, 2])
        with head:
            st.page_link(
                "pages/2_Candidate.py", label=f"{rank}. {r.name}",
                query_params={"candidate_id": r.candidate_id},
            )
            st.caption(
                f"{r.seniority_band or 'n/a'} | {r.current_country or '?'} / {r.current_region or '?'}"
            )
            st.write(r.search_summary)
        with score:
            b = r.breakdown
            st.progress(min(max(b.final, 0.0), 1.0), text=f"final {b.final:.3f}")
            st.dataframe(
                pd.DataFrame([{"rule": b.rule, "keyword": b.keyword, "embedding": b.embedding}]),
                hide_index=True,
            )
        if r.candidate_id in reasons:
            st.markdown(f"**Rerank reason.** {reasons[r.candidate_id]}")
        common.chips(r)
        if r.keywords:
            st.caption("Keyword hits: " + ", ".join(r.keywords))
        with st.expander("Rule score breakdown"):
            st.dataframe(
                pd.DataFrame(r.breakdown.rule_terms, columns=["term", "weight", "match"]),
                hide_index=True,
            )
