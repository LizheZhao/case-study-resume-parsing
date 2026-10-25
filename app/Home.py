import common
import streamlit as st

PAGES = (
    ("pages/1_Search.py", "Search", ":material/search:", "Find and rank candidates for a role"),
    ("pages/2_Candidate.py", "Candidate", ":material/person:", "Read one candidate's full profile"),
    ("pages/3_Insights.py", "Insights", ":material/pie_chart:", "See how the candidate pool is distributed"),
    ("pages/4_Ingest.py", "Ingest", ":material/upload_file:", "Add new DOCX or PDF resumes"),
    ("pages/5_Evals.py", "Evals", ":material/fact_check:", "Check parsing and ranking quality"),
)

st.set_page_config(page_title="Resume Search", layout="wide")
st.title("Candidate resume search")
st.write(
    "Find hedge fund candidates for a role. Each resume is read into a structured profile, so "
    "you can filter by location, seniority, sector and skills, then rank candidates by how well "
    "they fit a job description."
)

n_cand = common.conn().execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
st.metric("Candidates", n_cand)

st.subheader("Go to a page")
with st.container(border=True):
    for path, label, icon, purpose in PAGES:
        link, text = st.columns([1, 3], vertical_alignment="center")
        link.page_link(path, label=label, icon=icon)
        text.caption(purpose)

st.caption(f"Database: {common.db_path()} (set RESUME_DB to change)")
