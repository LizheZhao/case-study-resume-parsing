import common
import streamlit as st

st.set_page_config(page_title="Resume Search", layout="wide")
st.title("Candidate resume search")
st.write(
    "Search hedge fund candidates parsed from DOCX and PDF resumes. Each resume is read by an LLM into "
    "a structured record, normalized in code, stored in SQLite with a keyword index and "
    "embeddings, and ranked against a requisition by a rule score, BM25, and embedding "
    "similarity. Ingest new resumes, inspect any candidate, and see how the pool is distributed."
)

path = common.db_path()
st.caption(f"Database: {path} (set RESUME_DB to change)")
db = common.conn()
n_cand = db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
n_jobs = db.execute("SELECT COUNT(*) FROM parse_jobs").fetchone()[0]
c1, c2 = st.columns(2)
c1.metric("Candidates", n_cand)
c2.metric("Parse jobs", n_jobs)

st.page_link("pages/1_Search.py", label="Search candidates")
st.page_link("pages/2_Candidate.py", label="Candidate detail")
st.page_link("pages/3_Insights.py", label="Insights")
st.page_link("pages/4_Ingest.py", label="Ingest resumes")
st.page_link("pages/5_Evals.py", label="Evals and quality")
