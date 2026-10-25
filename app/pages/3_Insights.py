import json

import common
import pandas as pd
import plotly.express as px
import streamlit as st

from resume_search.parse.models import Approach
from resume_search.search.requisition import BANDS, DEGREES

st.set_page_config(page_title="Insights", layout="wide")
st.title("Insights")

db = common.conn()
tax = common.get_taxonomy()
# A fixed category list with fewer values than this is a pie, when each candidate counts once.
PIE_MAX = 10


def q(sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, db)


cands = q(
    "SELECT candidate_id, current_country, current_region, seniority_band, investment_approach, "
    "highest_degree, record_json FROM candidates"
)
if cands.empty:
    st.info("Nothing to chart yet. Ingest some resumes first.")
    st.stop()
st.caption(f"{len(cands)} candidates")


def counts(values, categories: list[str] | None = None) -> pd.Series:
    """Counts in category order, zero where the data has none, then any value outside the list.
    A missing value counts as Unknown."""
    s = pd.Series(list(values), dtype=object).fillna(common.UNKNOWN).value_counts()
    if categories is None:
        return s
    extra = [v for v in s.index if v not in categories]
    return s.reindex(list(categories) + extra, fill_value=0)


def ordered_colors(n: int) -> list[str]:
    """Light to dark blues for an ordered scale."""
    blues = px.colors.sequential.Blues[2:]
    return [blues[round(i * (len(blues) - 1) / max(n - 1, 1))] for i in range(n)]


def chart(s: pd.Series, title: str, unit: str = "candidates", pie: bool = False,
          ordered: bool = False) -> None:
    df = s.rename_axis("label").reset_index(name="n")
    if pie and len(df) < PIE_MAX:
        df["label"] = [f"{lab} ({n})" for lab, n in zip(df["label"], df["n"], strict=True)]
        colors = ordered_colors(len(df)) if ordered else px.colors.qualitative.Safe
        fig = px.pie(df, names="label", values="n", title=title, color_discrete_sequence=colors,
                     category_orders={"label": list(df["label"])})
        fig.update_traces(
            sort=False, texttemplate=["%{percent}" if n else "" for n in df["n"]],
            hovertemplate=f"%{{label}}: %{{value}} {unit}<extra></extra>",
        )
    else:
        fig = px.bar(df, x="n", y="label", orientation="h", title=title)
        fig.update_yaxes(autorange="reversed", title=None)
        fig.update_xaxes(title=unit.capitalize(), dtick=1 if df["n"].max() <= 10 else None)
        fig.update_traces(hovertemplate=f"%{{y}}: %{{x}} {unit}<extra></extra>")
        fig.update_layout(height=max(260, 32 * len(df) + 110))
    st.plotly_chart(fig)


def latest_full_time_role(record: dict) -> dict | None:
    """The current full-time role with the latest start, else the latest by start; first listed
    when nothing is dated."""
    roles = [(i, r) for i, r in enumerate(record["extracted"]["experience"])
             if r["employment_type"] == "full_time"]
    if not roles:
        return None

    def key(item):
        i, r = item
        return (r["is_current"], max(r["start"]) if r["start"] else "", -i)

    return max(roles, key=key)[1]


tag_to_core = {k.lower(): v for k, v in common.tx._load("sectors")["tag_to_core"].items()}
recent_sectors: list[str] = []
recent_firms: list[str | None] = []
for raw in cands["record_json"]:
    role = latest_full_time_role(json.loads(raw))
    recent_firms.append(role["firm_type"] if role else None)
    if role:
        found = [s["label"] for s in role["sectors"]]
        found += [tag_to_core[t.strip().lower()] for t in role["other_sector_tags"] if t.strip().lower() in tag_to_core]
        recent_sectors += list(dict.fromkeys(found))

profile, location, expertise = st.tabs(["Profile", "Location", "Expertise"])

with profile:
    a, b = st.columns(2)
    with a:
        chart(counts(recent_sectors, tax["sectors"]), "Sectors, most recent full-time role (a role can list several)")
    with b:
        chart(counts(cands["seniority_band"], list(BANDS)), "Seniority", pie=True, ordered=True)
    a, b = st.columns(2)
    with a:
        chart(counts(cands["investment_approach"], [x.value for x in Approach]), "Investment approach", pie=True)
    with b:
        chart(counts(cands["highest_degree"].fillna("None"), list(DEGREES)), "Highest degree", pie=True, ordered=True)

with location:
    a, b = st.columns(2)
    with a:
        chart(counts(cands["current_region"], tax["regions"]), "Region, current", pie=True)
    with b:
        chart(counts(cands["current_country"].map(common.country_code, na_action="ignore")), "Country, current")
    approaches = [x.value for x in Approach]
    regions = tax["regions"] + ([common.UNKNOWN] if cands["current_region"].isna().any() else [])
    cross = (
        cands.assign(region=cands["current_region"].fillna(common.UNKNOWN))
        .pivot_table(index="investment_approach", columns="region", values="candidate_id", aggfunc="count")
        .reindex(index=approaches, columns=regions).fillna(0).astype(int)
    )
    fig = px.imshow(cross, text_auto=True, title="Approach by region", aspect="auto",
                    labels={"x": "Region", "y": "Approach", "color": "Candidates"})
    fig.update_traces(hovertemplate="%{y}, %{x}: %{z} candidates<extra></extra>")
    st.plotly_chart(fig)

with expertise:
    a, b = st.columns(2)
    with a:
        skills = q("SELECT skill, COUNT(DISTINCT candidate_id) AS n FROM candidate_skills "
                   "GROUP BY 1 ORDER BY n DESC, skill LIMIT 20")
        chart(skills.set_index("skill")["n"], "Top 20 skills")
    with b:
        chart(counts(recent_firms, tax["firm_types"]), "Firm type, most recent full-time role")
    assets = q("SELECT DISTINCT candidate_id, asset_class FROM candidate_assets")["asset_class"]
    chart(counts(assets, tax["asset_classes"]), "Asset classes (a candidate can count in several)")
