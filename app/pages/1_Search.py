import common
import pandas as pd
import streamlit as st

from resume_search.search import Requisition, parse_requisition, rerank_llm
from resume_search.search.requisition import APPROACHES, BANDS, DEGREES

PAGE_SIZE = 20
RERANK_TOP = 10
MUST_HELP = "Must removes candidates who don't match. Otherwise the filter only affects the score."
SCOPES = {"either": "Current or past", "current": "Current", "worked": "Past roles"}

st.set_page_config(page_title="Search", layout="wide")
st.title("Search")

common.init_state()
tax = common.get_taxonomy()
picks = common.picker_options(common.db_path())

# A parse request set by the button is applied here, before any widget is instantiated.
if st.session_state.pop("_parse_pending", False):
    try:
        with st.spinner("Reading the job description"):
            parsed = parse_requisition(st.session_state["jd_text"])
        st.session_state.update(common.state_from_requisition(parsed))
        st.session_state["_parsed"] = True
        st.session_state.pop("_parse_error", None)
    except Exception as exc:  # noqa: BLE001  surfaced to the user below
        st.session_state["_parse_error"] = f"{type(exc).__name__}: {exc}"

ss = st.session_state

st.text_area(
    "Job description or keywords", key="jd_text", height=140,
    placeholder="Senior healthcare analyst in APAC, fundamental, CFA preferred",
    help="Drives the keyword and embedding scores. Press the button below to have it fill in the filters too.",
)
st.button(
    "Fill filters from description", on_click=lambda: ss.update(_parse_pending=True),
    disabled=not ss["jd_text"].strip(), help="Makes one API call.",
)
if ss.get("_parse_error"):
    st.error(f"Could not make sense of that job description. {ss['_parse_error']}")
elif ss.get("_parsed"):
    st.caption("Filters filled in from the description. Change any of them below before you search.")


def options_with(base: list[str], key: str) -> list[str]:
    """Keep values already in state (typed earlier or set by a parse) selectable."""
    return base + [v for v in ss[key] if v not in base]


def labeled(label: str, field: str | None, help: str | None = None) -> None:
    """A filter's label row, with its Must toggle when the criterion has a mode."""
    row = st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center")
    row.markdown(label, help=help, width="content")
    if field:
        row.toggle("Must", key=f"m_{field}", help=MUST_HELP, width="content")


def picker(label: str, key: str, options: list[str], field: str | None, **kw) -> None:
    labeled(label, field, kw.pop("help", None))
    st.multiselect(label, options_with(options, key), key=key, label_visibility="collapsed", **kw)


# Countries outside the selected regions drop out before anything reads them.
choices = common.country_choices(ss["f_regions"])
ss["f_countries"] = [i for i in ss["f_countries"] if i in choices]
try:
    req = common.requisition_from_state(ss)
except ValueError as exc:
    req = None
    st.error(f"These filters clash: {exc}")
n_set = 0 if req is None else sum(
    getattr(req, f) is not None for f in common.MODE_FIELDS
) + (req.years_min is not None or req.years_max is not None) + bool(req.skills_required or req.skills_preferred)

with st.expander(f"Filters ({n_set} set)" if n_set else "Filters", expanded=True):
    st.markdown("**Location**")
    c = st.columns(3)
    with c[0]:
        picker("Regions", "f_regions", tax["regions"], "location",
               help="Pick countries inside a region to narrow it down. Leave them empty and the whole region counts.")
    with c[1]:
        labeled("Countries", None)
        st.multiselect("Countries", choices, key="f_countries", format_func=common.country_option,
                       label_visibility="collapsed",
                       placeholder="Any in the selected regions" if ss["f_regions"] else "Any country")
    with c[2]:
        labeled("Applies to", None)
        st.selectbox("Applies to", list(SCOPES), key="f_scope", format_func=SCOPES.get,
                     label_visibility="collapsed")

    st.markdown("**Experience**")
    c = st.columns(2)
    with c[0]:
        picker("Seniority", "f_bands", list(BANDS), "seniority_bands")
    with c[1]:
        labeled("Years of relevant experience", "years")
        st.slider("Years of relevant experience", 0, common.YEARS_MAX, key="f_years",
                  label_visibility="collapsed", format="%d y")

    st.markdown("**Focus**")
    c = st.columns(2)
    with c[0]:
        labeled("Approach", "approach")
        st.selectbox("Approach", [common.ANY, *APPROACHES], key="f_approach", label_visibility="collapsed")
        picker("Markets covered", "f_markets", tax["markets"], "markets_covered",
               format_func=common.market_option)
    with c[1]:
        picker("Sectors", "f_sectors", tax["sectors"], "sectors")
        picker("Asset classes", "f_assets", tax["asset_classes"], "asset_classes")

    st.markdown("**Qualifications**")
    c = st.columns(3)
    with c[0]:
        labeled("Degree", "degree_min")
        st.selectbox("Degree", [common.ANY, *DEGREES], key="f_degree", label_visibility="collapsed",
                     format_func=lambda d: d if d == common.ANY else f"{d} or higher")
    with c[1]:
        picker("Certifications", "f_certs", picks["certs"], "certifications", accept_new_options=True)
    with c[2]:
        picker("Languages", "f_languages", picks["languages"], "languages", accept_new_options=True)

    st.markdown("**Skills**")
    c = st.columns(2)
    with c[0]:
        picker("Required skills", "f_skills_req", picks["skills"], None, accept_new_options=True,
               help="A required skill removes anyone without it. Preferred skills only add to the score.")
    with c[1]:
        picker("Preferred skills", "f_skills_pref", picks["skills"], None, accept_new_options=True)

go, opt = st.columns([1, 3], vertical_alignment="center")
if go.button("Search", type="primary", disabled=req is None):
    n_all = common.conn().execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    ss["_last"] = {"req": common.req_cache_key(req), "top": max(n_all, 1)}
    ss["_page"] = 0
rerank_on = opt.toggle(
    f"Rerank the top {RERANK_TOP} with an LLM", key="rerank",
    help="Optional. Sends the top candidates and the requisition to Sonnet, which reorders them "
         "and gives a one-line reason each. One API call per search; off means the blended score "
         "decides the order.",
)
last = ss.get("_last")
if last is None:
    st.info("Set a few filters or paste a job description, then hit Search.")
    st.stop()

results = common.run_search(last["req"], common.db_path(), last["top"])
searched = Requisition.model_validate_json(last["req"])
if searched.is_empty():
    st.caption("Nothing to filter on yet, so this is everyone.")

# With the toggle on, the LLM reorders the top of the list. A failure keeps the plain ranking.
reasons: dict[str, str] = {}
key = f"{last['req']}|{[r.candidate_id for r in results[:RERANK_TOP]]}"
cache = ss.setdefault("_rerank", {})
if rerank_on and key not in cache and results:
    try:
        with st.spinner("Reranking the top candidates"):
            cache[key] = rerank_llm(searched, results[:RERANK_TOP])
    except Exception:  # noqa: BLE001  ranking stays as scored
        cache[key] = None
if rerank_on and results and cache.get(key) is None:
    st.caption("The rerank call failed, so this is the blended order. It needs ANTHROPIC_API_KEY.")
if rerank_on and cache.get(key):
    reasons = dict(cache[key])
    by_id = {r.candidate_id: r for r in results}
    results = [by_id[i] for i, _ in cache[key]] + results[RERANK_TOP:]

n = len(results)
pages = max(1, -(-n // PAGE_SIZE))
page = min(ss.get("_page", 0), pages - 1)
start = page * PAGE_SIZE
shown = results[start:start + PAGE_SIZE]
st.subheader(f"{n} candidate{'' if n == 1 else 's'}")
if pages > 1:
    st.caption(f"Showing {start + 1}–{start + len(shown)}")

for rank, r in enumerate(shown, start + 1):
    with st.container(border=True):
        head, score = st.columns([4, 2])
        with head:
            st.page_link(
                "pages/2_Candidate.py", label=f"{rank}. {r.name}",
                query_params={"candidate_id": r.candidate_id},
            )
            st.caption(common.profile_line(r.seniority_band, r.current_country, r.current_region))
            st.write(r.search_summary)
        with score:
            b = r.breakdown
            st.progress(min(max(b.final, 0.0), 1.0), text=f"Match score {b.final:.2f}")
        if r.candidate_id in reasons:
            st.markdown(f"**Why ranked here.** {reasons[r.candidate_id]}")
        common.chips(r)
        if r.keywords:
            st.caption("Keyword hits: " + ", ".join(r.keywords))
        with st.expander("Score details"):
            st.caption("Match score is a blend of the three: rule, keyword, embedding.")
            st.dataframe(
                pd.DataFrame([{"rule": b.rule, "keyword": b.keyword, "embedding": b.embedding}]),
                hide_index=True,
            )
            st.dataframe(
                pd.DataFrame(b.rule_terms, columns=["term", "weight", "match"]), hide_index=True,
            )

if pages > 1:
    prev, label, nxt = st.columns([1, 2, 1])
    prev.button("Previous", disabled=page == 0, on_click=lambda: ss.update(_page=page - 1),
                width="stretch")
    label.markdown(f"<div style='text-align:center'>Page {page + 1} of {pages}</div>",
                   unsafe_allow_html=True)
    nxt.button("Next", disabled=page >= pages - 1, on_click=lambda: ss.update(_page=page + 1),
               width="stretch")
