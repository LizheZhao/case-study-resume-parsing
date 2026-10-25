"""Generate 2025_ds_case_study_resume_platform.ipynb from the files in this repository.

The notebook writes the package out with %%writefile cells whose bodies are read from disk, so it
cannot drift from src/. Usage:

    uv run python scripts/build_notebook.py              # write the notebook
    uv run python scripts/build_notebook.py --execute    # write it, then run it with nbconvert
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

sys.path.insert(0, str(Path(__file__).resolve().parent))
import notebook_text as text

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "2025_ds_case_study_resume_platform.ipynb"
BRIEF = ROOT / "data" / "2025_ds_case_study_resume_platform.ipynb"
KERNEL = "case-study"
TIMEOUT = 600
# Paths whose uncommitted changes would be baked into the notebook. pyproject.toml is checked
# separately because the dev extras it carries are needed by the notebook itself.
GUARDED = ("src", "app", "taxonomy", "evals/requisitions.jsonl")

# One sentence per file: the module table and the header above each %%writefile cell.
DESCRIPTIONS = {
    "src/resume_search/cli.py": "The `resume-search` command line: extract, parse, ingest, rederive, show, search, and eval.",
    "src/resume_search/taxonomy.py": "Loads the YAML files in `taxonomy/` into one plain-data view of countries, sectors, firm types, asset classes, and skill synonyms.",
    "src/resume_search/extract/__init__.py": "`extract(path)` turns a DOCX or PDF into an `ExtractedText` with repaired text, quality numbers, and the extraction path.",
    "src/resume_search/extract/types.py": "The `ExtractedText` and `TextQuality` data shapes, including the rule that decides when the vision path is needed.",
    "src/resume_search/extract/docx.py": "Reads a DOCX body in document order and writes each table row as one line with cells joined by ` | `.",
    "src/resume_search/extract/pdf.py": "Reads PDF text per page with PyMuPDF, measures image coverage, and renders pages to PNG for the vision path.",
    "src/resume_search/extract/repair.py": "Replaces ligature glyphs that PDF fonts emit (such as `Ɵ` for `ti`) and drops private-use characters.",
    "src/resume_search/extract/quality.py": "Computes the share of unknown words in a text and fixes close misspellings caused by bad glyphs.",
    "src/resume_search/parse/__init__.py": "Public surface of the parser: `parse_resume`, `ParseOk`, `ParseFailed`, and `ParsedResume`.",
    "src/resume_search/parse/models.py": "The Pydantic models for LLM output, with validators for dates, ISO-2 countries, and verbatim evidence quotes.",
    "src/resume_search/parse/prompt.py": "The versioned system prompts and the user-message builders, rendered from the taxonomy files.",
    "src/resume_search/parse/client.py": "Runs the profile, experience, and assessment calls with constrained decoding, cross-verification, targeted regeneration, and the Sonnet-Sonnet-Opus retry ladder.",
    "src/resume_search/parse/verify.py": "A registry of checks that compare the LLM's fields against deterministic text parsing and name the field to regenerate on disagreement.",
    "src/resume_search/derive/__init__.py": "`derive()` builds the `derived` layer from a parsed resume and the taxonomy, in pure code.",
    "src/resume_search/derive/models.py": "The `DerivedFields` and `CandidateRecord` data shapes and the schema version.",
    "src/resume_search/derive/experience.py": "Month arithmetic over roles: merges overlapping ranges, falls back to stated durations, and maps years to a seniority band.",
    "src/resume_search/derive/location.py": "Derives the current country, the countries worked in, and the markets covered.",
    "src/resume_search/derive/skills.py": "Normalizes skills to canonical tokens and decides CFA status.",
    "src/resume_search/store/schema.sql": "The SQLite DDL: candidate and facet tables, the FTS5 index, embeddings, `parse_jobs`, and taxonomy mirrors.",
    "src/resume_search/store/db.py": "Opens the database, loads taxonomy tables, and writes or reads one candidate atomically.",
    "src/resume_search/store/fts.py": "Builds the keyword-index body by removing email, URL, and phone lines from the resume text.",
    "src/resume_search/store/export.py": "Writes `candidates.json` and `candidates.csv` from the database.",
    "src/resume_search/search/__init__.py": "Public surface of the search package.",
    "src/resume_search/search/requisition.py": "The `Requisition` object with must and preferred criteria, and the LLM step that turns a pasted JD into one.",
    "src/resume_search/search/rules.py": "Prunes candidates by must criteria in SQL and scores preferred criteria, producing the explanation chips.",
    "src/resume_search/search/keyword.py": "BM25 scoring over the FTS5 index and the list of query terms each candidate matched.",
    "src/resume_search/search/vectors.py": "Embeds text with `BAAI/bge-small-en-v1.5` and scores candidates by cosine similarity over summary and role vectors.",
    "src/resume_search/search/weights.py": "Loads rule weights and the blend weights from `taxonomy/weights.yaml`.",
    "src/resume_search/search/rank.py": "`search()` combines the rule, BM25, and embedding scores into the final order, and `rerank_llm` optionally reorders the top 10.",
    "src/resume_search/search/llm.py": "A shared structured-output call with one retry, used by the requisition parser and the reranker.",
    "src/resume_search/ingest/__init__.py": "Package marker for the ingest pipeline.",
    "src/resume_search/ingest/jobs.py": "The `parse_jobs` row and its state machine, and the idempotency check on file hash and prompt version.",
    "src/resume_search/ingest/pipeline.py": "`ingest()` runs extract, parse, derive, and index for many files with worker threads, and `rederive()` rebuilds derived data without the API.",
    "src/resume_search/eval/__init__.py": "Package marker for the evaluation harness.",
    "src/resume_search/eval/models.py": "Data shapes for judge output and evaluation requisitions.",
    "src/resume_search/eval/judge.py": "The Opus judge prompts and calls for the parsing and ranking evals, and their cost accounting.",
    "src/resume_search/eval/metrics.py": "Pure scoring functions: per-field accuracy, nDCG@k, precision@k, and judge-human agreement.",
    "src/resume_search/eval/parsing_eval.py": "Runs the parsing suite: judges every stored record against its resume text and compares with the hand grades.",
    "src/resume_search/eval/ranking_eval.py": "Runs the ranking suite: ranks each requisition with four score variants and grades the pool once.",
    "src/resume_search/eval/report.py": "Formats suite results as plain-text tables.",
    "src/resume_search/eval/results.py": "Reads and writes the timestamped result files in `evals/results/`.",
    "app/Home.py": "Landing page with database counts and links to the other pages.",
    "app/common.py": "Shared Streamlit helpers: cached resources, the widget-to-`Requisition` mapping, and chip rendering.",
    "app/pages/1_Search.py": "Filter sidebar, JD paste box, must and preferred toggles, ranked result cards with chips, and optional LLM rerank.",
    "app/pages/2_Candidate.py": "One candidate in full: derived facets, role timeline, evidence quotes, skills, and parse metadata.",
    "app/pages/3_Insights.py": "Distributions of region, country, approach, sector, seniority, degree, skills, and firm type.",
    "app/pages/4_Ingest.py": "Multi-file upload that calls `ingest()` and shows live job status, throughput, and cost.",
    "app/pages/5_Evals.py": "Shows the latest parsing and ranking results and the judge-human disagreements.",
}


def collect_files() -> dict[str, list[Path]]:
    """Files to embed, grouped by notebook section, as paths relative to ROOT."""
    rel = lambda paths: sorted(p.relative_to(ROOT) for p in paths)
    pkg = ROOT / "src" / "resume_search"
    return {
        "Taxonomy": rel((ROOT / "taxonomy").glob("*.yaml")),
        "Project configuration": [Path("pyproject.toml")],
        "Package: src/resume_search": rel([*pkg.rglob("*.py"), *pkg.rglob("*.sql")]),
        "Streamlit app: app": rel((ROOT / "app").rglob("*.py")),
        "Evaluation requisitions": [Path("evals/requisitions.jsonl")],
    }


def check_clean(allow_dirty: bool) -> None:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *GUARDED],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if out and not allow_dirty:
        sys.exit(
            "refusing to build: uncommitted changes would be baked into the notebook "
            f"(commit them or pass --allow-dirty):\n{out}"
        )
    pyproject = subprocess.run(
        ["git", "status", "--porcelain", "--", "pyproject.toml"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if pyproject:
        print(f"warning: pyproject.toml has uncommitted changes and is embedded as is ({pyproject})")


def md(source: str):
    return new_markdown_cell(source.strip("\n"))


def code(source: str):
    return new_code_cell(source.strip("\n"))


def writefile_cell(rel: Path):
    body = (ROOT / rel).read_text(encoding="utf-8")
    # writefile appends one newline to the cell body, so drop exactly one to round-trip the file.
    body = body.removesuffix("\n")
    return new_code_cell(f"%%writefile {rel.as_posix()}\n{body}")


def module_rows(files: dict[str, list[Path]]) -> str:
    rows = []
    for rel in (p for group in files.values() for p in group):
        key = rel.as_posix()
        if key in DESCRIPTIONS:
            rows.append(f"| `{key.removeprefix('src/resume_search/')}` | {DESCRIPTIONS[key]} |")
    return "\n".join(rows)


def check_descriptions(files: dict[str, list[Path]]) -> list[Path]:
    """Every non-empty source file needs a sentence. Returns the empty files, which get no cell."""
    empty = []
    for group in files.values():
        for rel in group:
            key = rel.as_posix()
            if not key.startswith(("src/", "app/")):
                continue
            if (ROOT / rel).stat().st_size == 0:
                empty.append(rel)
            elif key not in DESCRIPTIONS:
                sys.exit(f"no description for {key}; add it to DESCRIPTIONS in {Path(__file__).name}")
    return empty


SETUP_DIRS = '''\
from pathlib import Path

# %%writefile does not create directories, so make them first; empty package markers are touched.
for d in {dirs}:
    Path(d).mkdir(parents=True, exist_ok=True)
for f in {empty}:
    Path(f).touch()
print(f"ready: {{len({dirs})}} directories, {{len({empty})}} empty package markers")
'''

INSTALL = '''\
import os
import subprocess

CMD = ["uv", "sync", "--extra", "dev"]   # with pip instead: pip install -e ".[dev]"
if os.environ.get("INSTALL") == "1":
    subprocess.run(CMD, check=True)
else:
    print("not installing; set INSTALL=1 to run:", " ".join(CMD))
'''

DEMO_SETUP = '''\
import json
import os

os.environ.setdefault("TQDM_DISABLE", "1")  # keep model-loading progress bars out of the stored output
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import sqlite3
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 70)

ROOT = Path.cwd()
DATA = ROOT / "data"
DB = ROOT / "exports" / "candidates.sqlite"


def rebuild_db_from_export(db_path: Path = DB) -> int:
    """Rebuild the SQLite file from exports/candidates.json and the resume files.
    Uses the stored records, local code, and the local embedding model. No API call."""
    from resume_search.derive.models import CandidateRecord
    from resume_search.extract import extract
    from resume_search.search.vectors import candidate_vectors, default_embedder
    from resume_search.store.db import connect, load_taxonomy, upsert_candidate
    from resume_search.store.fts import fts_body

    records = [
        CandidateRecord.model_validate(r)
        for r in json.loads((ROOT / "exports" / "candidates.json").read_text(encoding="utf-8"))
    ]
    conn = connect(db_path)
    load_taxonomy(conn)
    for r in records:
        body = fts_body(extract(DATA / r.meta["source_file"]).text)
        upsert_candidate(conn, r, body, candidate_vectors(r, default_embedder()))
    conn.close()
    return len(records)


if not DB.exists():
    print(f"{DB.name} missing; rebuilt {rebuild_db_from_export()} candidates from exports/candidates.json")
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
print("candidates in database:", conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
'''

DEMO_EXTRACT = '''\
from resume_search.extract import extract

for name in ["Chen Li (Alex).docx", "Omar El-Hassan 202405.pdf"]:
    e = extract(DATA / name)
    print(f"=== {e.source_file}  ({e.format.value}, {e.page_count} page(s), {len(e.text)} chars, path={e.extraction_path.value})")
    print("\\n".join(e.text.splitlines()[:20]))
    print()
    print(pd.Series(vars(e.quality), name="text_quality").to_frame().T.to_string(index=False))
    print()
'''

DEMO_REPAIR = '''\
from resume_search.extract.pdf import extract_pdf
from resume_search.extract.repair import repair_text

raw, *_ = extract_pdf(DATA / "Omar El-Hassan 202405.pdf")
fixed, repairs = repair_text(raw)
e = extract(DATA / "Omar El-Hassan 202405.pdf")
q = e.quality
print(f"unknown-word ratio  before repair: {q.unknown_ratio_before:.1%}   after repair and fuzzy fix: {q.unknown_ratio_after:.1%}")
print(f"glyph repairs applied: {q.repairs}")
bad = [line for line in raw.splitlines() if "\\u019f" in line][:3]
for line in bad:
    print("raw  :", line)
    print("fixed:", repair_text(line)[0])
'''

DEMO_RECORD = '''\
parsed = {}
for f in (ROOT / "exports" / "parsed").glob("*.json"):
    doc = json.loads(f.read_text(encoding="utf-8"))
    parsed[doc["meta"]["source_file"]] = doc
chen = parsed["Chen Li (Alex).docx"]

print("--- extracted record (LLM output, validated) ---")
print(json.dumps(chen["extracted"], indent=2, ensure_ascii=False))
'''

DEMO_VERIFY = '''\
v = chen["meta"]["verification"]
print("--- verification block ---")
print(f"checks run: {v['checks_run']}   disagreements: {v['disagreements']}   "
      f"regenerated: {v['regenerated']}   unresolved: {len(v['unresolved'])}")
for d in v["details"]:
    print(f"\\nfield: {d['field']}")
    print("  LLM value :", json.dumps(d["llm_value"], ensure_ascii=False)[:300], "...")
    print("  text value:", d["text_value"])
print()
print(pd.DataFrame(chen["meta"]["attempts"])[["call", "model", "latency_ms", "input_tokens", "output_tokens", "cache_read_tokens", "error"]])
'''

DEMO_TABLE = '''\
derived = pd.read_sql(
    """
    SELECT name, current_country AS country, current_region AS region, seniority_band AS band,
           ROUND(months_full_time_relevant / 12.0, 1) AS yrs_relevant,
           ROUND(months_full_time_core / 12.0, 1) AS yrs_core,
           investment_approach AS approach, highest_degree AS degree, has_cfa AS cfa, parse_status
    FROM candidates ORDER BY name
    """,
    conn,
)
facets = {
    "sectors": ("candidate_sectors", "sector"),
    "assets": ("candidate_assets", "asset_class"),
    "worked": ("candidate_countries", "country"),
}
for col, (table, field) in facets.items():
    extra = " AND kind = 'worked'" if col == "worked" else ""
    s = pd.read_sql(
        f"SELECT c.name, GROUP_CONCAT(DISTINCT t.{field}) AS v FROM candidates c "
        f"JOIN {table} t USING (candidate_id) WHERE 1=1{extra} GROUP BY c.name", conn,
    ).set_index("name")["v"]
    derived[col] = derived["name"].map(s)
derived
'''

DEMO_SEARCH = '''\
from resume_search.search import Criterion, Requisition, Weights, search

weights = Weights.load()
SEARCHES = {
    "Fundamental healthcare in the US": Requisition(
        free_text="fundamental healthcare equity analyst, US pharmaceuticals and biotech",
        location=Criterion[str](value="US"),
        approach=Criterion[str](value="Fundamental"),
        sectors=Criterion[list[str]](value=["Healthcare"]),
    ),
    "Systematic, 5 years or less, Python preferred": Requisition(
        free_text="systematic quantitative researcher, backtesting, factor models",
        approach=Criterion[str](value="Systematic"),
        years_max=5,
        skills_preferred=["python"],
    ),
    "Senior, CFA required": Requisition(
        seniority_bands=Criterion[list[str]](value=["Senior", "Principal"], mode="must"),
        certifications=Criterion[list[str]](value=["CFA"], mode="must"),
    ),
}
for title, req in SEARCHES.items():
    results = search(conn, req, weights, top_k=5)
    print(f"=== {title}  ({len(results)} after pruning)")
    for i, r in enumerate(results, 1):
        b = r.breakdown
        chips = "  ".join(("+" if c.status.value == "matched" else "-") + c.label for c in r.chips)
        print(f"{i}. {r.name:<22} final={b.final:.3f} rule={b.rule:.3f} kw={b.keyword:.3f} emb={b.embedding:.3f}   {chips}")
    print()
'''

DEMO_INSIGHTS = '''\
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "plotly_mimetype"  # JSON payload only: no 3 MB plotly.js bundle per figure


def q(sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, conn)


by_region = q("SELECT COALESCE(current_region, '?') AS region, investment_approach AS approach, COUNT(*) AS n "
              "FROM candidates GROUP BY 1, 2")
skills = q("SELECT skill, COUNT(DISTINCT candidate_id) AS n FROM candidate_skills "
           "GROUP BY 1 ORDER BY n DESC, skill LIMIT 12")
print("Candidates by region and approach")
print(by_region.pivot(index="region", columns="approach", values="n").fillna(0).astype(int))
print("\\nTop skills (candidates)")
print(skills.to_string(index=False))
px.bar(by_region, x="region", y="n", color="approach", title="Candidates by region and approach").show()
fig = px.bar(skills, x="n", y="skill", orientation="h", title="Top skills")
fig.update_yaxes(autorange="reversed")
fig.show()
'''

DEMO_EVALS = '''\
from resume_search.eval.results import latest_result

p_path, parsing = latest_result("evals/results", "parsing")
r_path, ranking = latest_result("evals/results", "ranking")
pm = parsing["metrics"]
print(f"parsing: {p_path.name}  judge={parsing['judge_model']}  task={parsing['task_model']}  "
      f"fields={pm['fields']}  accuracy={pm['accuracy']:.1%}  soft={pm['soft_accuracy']:.1%}  "
      f"hallucinated={pm['hallucinated']}  missing={pm['missing']}  cost=${parsing['cost_usd']:.2f}")
per_field = pd.DataFrame(pm["per_field"]).T[["total", "correct", "partial", "wrong", "hallucinated", "missing", "accuracy"]]
per_field.round(3)
'''

DEMO_AGREEMENT = '''\
ag = pm["agreement"]
print(f"judge vs human, gold status {ag['gold_status']}: {ag['overall']['n']} fields, "
      f"exact {ag['overall']['exact']:.1%}, within one level {ag['overall']['within_one_level']:.1%}")
pd.DataFrame(ag["per_candidate"]).T[["n", "exact", "within_one_level"]]
'''

DEMO_RANKING = '''\
rm = ranking["metrics"]
print(f"ranking: {r_path.name}  judge={ranking['judge_model']}  pools={rm['pool_sizes']}")
cols = [f"{v}_{k}" for v in ["full", "rule_only", "keyword_only", "embedding_only"] for k in ["ndcg5", "p3"]]
table = pd.DataFrame(ranking["items"]).set_index("requisition_id")[["pool_size", *cols]]
table.loc["MEAN"] = [None] + [rm["mean"][c.rsplit("_", 1)[0]][c.rsplit("_", 1)[1]] for c in cols]
table.round(3)
'''

LLM_PARSE = '''\
if os.environ.get("RUN_LLM") == "1":
    from resume_search.extract import extract
    from resume_search.ingest.pipeline import attempt_cost
    from resume_search.parse import ParseOk, parse_resume

    result = parse_resume(extract(DATA / "Chen Li (Alex).docx"))
    attempts = pd.DataFrame([vars(a) for a in result.attempts])
    attempts["cost_usd"] = [attempt_cost(a) for a in result.attempts]
    print("status:", result.parse_status)
    print(attempts[["call", "model", "latency_ms", "input_tokens", "output_tokens", "cache_read_tokens", "error", "cost_usd"]])
    print(f"total cost ${attempts['cost_usd'].sum():.4f}")
    if isinstance(result, ParseOk):
        v = result.verification
        print(f"verification: checks={v.checks_run} disagreements={v.disagreements} regenerated={v.regenerated} unresolved={len(v.unresolved)}")
        print(result.record.search_summary)
else:
    print("set RUN_LLM=1 to run (about $0.08)")
'''

LLM_JD = '''\
if os.environ.get("RUN_LLM") == "1":
    from resume_search.ingest.pipeline import attempt_cost
    from resume_search.search import parse_requisition

    jd = json.loads((ROOT / "evals" / "requisitions.jsonl").read_text(encoding="utf-8").splitlines()[0])["free_text"]
    attempts = []
    req = parse_requisition(jd, attempts=attempts)
    print(req.model_dump_json(indent=2, exclude_none=True))
    print(f"api calls={len(attempts)}  cost=${sum(attempt_cost(a) for a in attempts):.4f}")
else:
    print("set RUN_LLM=1 to run (about $0.01)")
'''


def build() -> nbformat.NotebookNode:
    files = collect_files()
    empty = check_descriptions(files)
    brief = nbformat.read(BRIEF, as_version=4).cells[0]
    assert brief.cell_type == "markdown"

    cells = [
        new_markdown_cell(brief.source),
        md(text.HOW_TO_READ),
        *(md(section) for section in text.SOLUTION_SUMMARY),
        md(text.ARCHITECTURE.replace("{MODULE_ROWS}", module_rows(files))),
    ]

    dirs = sorted({p.parent.as_posix() for g in files.values() for p in g if p.parent != Path(".")})
    cells.append(md("## Code\n\nThese cells write the repository's source files back to disk. Each header names the file; the line under it says what the file is for."))
    cells.append(code(SETUP_DIRS.format(
        dirs=repr(dirs), empty=repr([p.as_posix() for p in empty])
    )))
    for section, paths in files.items():
        cells.append(md(f"### {section}"))
        for rel in paths:
            if rel in empty:
                continue
            key = rel.as_posix()
            blurb = DESCRIPTIONS.get(key)
            cells.append(md(f"#### `{key}`" + (f"\n\n{blurb}" if blurb else "")))
            cells.append(writefile_cell(rel))

    cells += [md(text.INSTALL_INTRO), code(INSTALL), md(text.DEMO_INTRO), code(DEMO_SETUP)]
    cells += [
        md("### Extraction\n\nOne DOCX and one PDF, first 20 lines of extracted text plus the text-quality numbers. The PDF is Omar El-Hassan's, picked because its font emits ligature glyphs."),
        code(DEMO_EXTRACT),
        md("#### Ligature repair\n\nThat font encodes `ti` as `Ɵ` (U+019F), so the raw text is full of words like `QuanƟtaƟve`. The repair map plus a fuzzy fix take the unknown-word ratio from 17.8% down to 2.6%."),
        code(DEMO_REPAIR),
        md("### A stored record\n\nChen Li's parse result from `exports/parsed/`: what the LLM returned once it passed validation, then the verification block that held it up against deterministic text parsing."),
        code(DEMO_RECORD),
        code(DEMO_VERIFY),
        md("### The derived table\n\nOne row per candidate, straight from the database. Years, band, country, and CFA status are all computed in code from the extracted record. The LLM is never asked for them."),
        code(DEMO_TABLE),
        md("### Search\n\nThree requisitions built by hand, no JD parsing, no API call. `final` is the blend, `rule` the preferred-criteria score, `kw` BM25, `emb` the embedding score, each scaled 0 to 1 within the pruned set. `+` and `-` mark chips that matched and did not. No free text means no `kw` and no `emb`. And when a requisition is all must criteria, as in the third search, every survivor scores 1.0 and the tie goes to name order."),
        code(DEMO_SEARCH),
        md("### Insights\n\nTwo of the charts from the Insights page. The tables print the same numbers, so the cell still reads where plots do not render."),
        code(DEMO_INSIGHTS),
        md("### Evaluation metrics\n\nRead from the latest files in `evals/results/`. Parsing first, per-field accuracy as graded by Opus."),
        code(DEMO_EVALS),
        md("The judge against my hand grades on the three calibration resumes."),
        code(DEMO_AGREEMENT),
        md("Ranking: nDCG@5 and P@3 per requisition, full blend next to the three single-signal ablations."),
        code(DEMO_RANKING),
        md(text.LLM_INTRO),
        md("#### Parse one resume end to end"),
        code(LLM_PARSE),
        md("#### Parse a job description into a requisition"),
        code(LLM_JD),
        md(text.RESULTS),
        md(text.SCALABILITY),
        md(text.LAYOUT_AND_RUN),
    ]

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": KERNEL, "display_name": "case-study (3.12)", "language": "python"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
    return nb


def execute(files: dict[str, list[Path]]) -> None:
    # The %%writefile cells rewrite the repository's files. IPython trims trailing blank lines,
    # so snapshot the files, and restore any that differ only in trailing whitespace.
    snapshot = {rel: (ROOT / rel).read_bytes() for group in files.values() for rel in group}
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
        f"--ExecutePreprocessor.timeout={TIMEOUT}", f"--ExecutePreprocessor.kernel_name={KERNEL}",
        str(NOTEBOOK),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    for rel, before in snapshot.items():
        after = (ROOT / rel).read_bytes()
        if after == before:
            continue
        if after.rstrip() != before.rstrip():
            sys.exit(f"{rel} changed during execution; the notebook no longer matches the repository")
        (ROOT / rel).write_bytes(before)
        print(f"restored trailing whitespace in {rel}")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    errors = [
        (i, o.get("ename"))
        for i, c in enumerate(nb.cells) if c.cell_type == "code"
        for o in c.get("outputs", []) if o.output_type == "error"
    ]
    if errors:
        sys.exit(f"cells with errors: {errors}")
    # Library warnings on stderr (tqdm, torch, HF Hub) name the local machine and say nothing
    # about the case study, so they do not ship in the notebook.
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = [o for o in c.outputs if not (o.output_type == "stream" and o.name == "stderr")]
    nbformat.write(nb, NOTEBOOK)
    print(f"executed {sum(c.cell_type == 'code' for c in nb.cells)} code cells without error")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--execute", action="store_true", help="run the notebook with nbconvert and store outputs")
    parser.add_argument("--allow-dirty", action="store_true", help="build even with uncommitted changes in src, app, taxonomy")
    args = parser.parse_args()
    check_clean(args.allow_dirty)
    nb = build()
    nbformat.validate(nb)
    nbformat.write(nb, NOTEBOOK)
    print(f"wrote {NOTEBOOK.name}: {len(nb.cells)} cells")
    if args.execute:
        execute(collect_files())


if __name__ == "__main__":
    main()
