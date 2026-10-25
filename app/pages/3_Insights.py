import common
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Insights", layout="wide")
st.title("Insights")

db = common.conn()


def q(sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, db)


n = q("SELECT COUNT(*) AS n FROM candidates")["n"][0]
if n == 0:
    st.info("No candidates yet. Ingest resumes first.")
    st.stop()
st.caption(f"{n} candidates")


def bar(df: pd.DataFrame, x: str, title: str, horizontal: bool = False):
    if df.empty:
        st.caption(f"{title}: no data")
        return
    fig = px.bar(df, x="n" if horizontal else x, y=x if horizontal else "n", title=title)
    if horizontal:
        fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig)


a, b = st.columns(2)
with a:
    bar(q("SELECT COALESCE(current_region,'?') AS region, COUNT(*) AS n FROM candidates GROUP BY 1"),
        "region", "Candidates by region")
with b:
    bar(q("SELECT COALESCE(current_country,'?') AS country, COUNT(*) AS n FROM candidates "
          "GROUP BY 1 ORDER BY n DESC"), "country", "Candidates by country")
a, b = st.columns(2)
with a:
    bar(q("SELECT investment_approach AS approach, COUNT(*) AS n FROM candidates GROUP BY 1"),
        "approach", "Investment approach")
with b:
    bar(q("SELECT seniority_band AS band, COUNT(*) AS n FROM candidates GROUP BY 1"),
        "band", "Seniority band")
a, b = st.columns(2)
with a:
    bar(q("SELECT sector, COUNT(DISTINCT candidate_id) AS n FROM candidate_sectors "
          "GROUP BY 1 ORDER BY n DESC"), "sector", "Sectors (candidates can count in several)")
with b:
    bar(q("SELECT COALESCE(highest_degree,'none') AS degree, COUNT(*) AS n FROM candidates GROUP BY 1"),
        "degree", "Highest degree")
a, b = st.columns(2)
with a:
    bar(q("SELECT skill, COUNT(DISTINCT candidate_id) AS n FROM candidate_skills "
          "GROUP BY 1 ORDER BY n DESC, skill LIMIT 20"), "skill", "Top skills", horizontal=True)
with b:
    bar(q("SELECT firm_type, COUNT(*) AS n FROM experience GROUP BY 1 ORDER BY n DESC"),
        "firm_type", "Firm types (roles)")

cross = q(
    "SELECT COALESCE(current_region,'?') AS region, investment_approach AS approach, COUNT(*) AS n "
    "FROM candidates GROUP BY 1, 2"
).pivot(index="approach", columns="region", values="n").fillna(0)
st.plotly_chart(px.imshow(cross, text_auto=True, title="Approach by region", aspect="auto"))
