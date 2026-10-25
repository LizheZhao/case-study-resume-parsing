"""Prose cells for the submission notebook. Kept apart from build_notebook.py so the generator
reads as logic and this file reads as the document."""

HOW_TO_READ = """\
---

# Candidate resume search platform: solution

## How to read this notebook

The code does not live in this notebook. It lives in `src/resume_search/` (the package) and `app/` (the Streamlit app). The cells in the "Code" sections below write those files out with `%%writefile`, one cell per file, generated from the repository by `scripts/build_notebook.py`. Run the notebook top to bottom and the package is recreated on disk.

The sections run in this order.

1. Architecture overview and a table of modules.
2. Code, one cell per file: taxonomy, project configuration, package, Streamlit app, evaluation requisitions.
3. Install the package.
4. Demos that run without an API key: extraction, a stored record, the candidate table, searches, insights, eval metrics.
5. Two demos that call the Anthropic API. They are off by default and run only when `RUN_LLM=1`.
6. Results, then scalability and additional features, then the repository layout and how to run everything.

**Streamlit app:** run locally with `uv run streamlit run app/Home.py`; public URL: _to be added at submission_

**Exports:** `exports/candidates.json` (full records) and `exports/candidates.csv` (the `candidates` table plus facet columns). Both are rewritten at the end of every ingest.

**Requirements.** Python 3.12 and [uv](https://docs.astral.sh/uv/). Parsing, search, and the evals call the Anthropic API and read `ANTHROPIC_API_KEY` from the environment or from a `.env` file. The demos in this notebook need no key.
"""

ARCHITECTURE = """\
## Architecture overview

A resume goes through five stages. Each stage reads the output of the one before it and nothing else, so a stage can be re-run alone.

```
 data/*.docx, *.pdf
        |
        v   extract    python-docx and PyMuPDF; ligature repair; unknown-word ratio;
        |              vision trigger (not hit by any of the 10 samples)
        v
 ExtractedText
        |
        v   parse      3 Sonnet calls: profile || experience, then assessment
        |              cross-check against deterministic text parsing, targeted regeneration
        |              retry ladder: Sonnet -> Sonnet + validation errors -> Opus
        v
 ParsedResume          (literal, evidence-quoted fields)
        |
        v   derive     pure code: months of experience, seniority band, countries,
        |              regions, normalized skills, CFA status. No API call.
        v
 SQLite                candidates + facet tables + FTS5 body + float32 embeddings
        |              + parse_jobs (ingest state) + taxonomy mirrors
        v
 search                requisition -> prune (must criteria, SQL)
        |                          -> rule score (preferred criteria)
        |                          -> BM25 (FTS5) and embedding cosine
        |                          -> final = 0.5 rule + 0.2 BM25 + 0.3 embedding
        v
 Streamlit             Search, Candidate, Insights, Ingest, Evals
```

Three layers describe every record. `extracted` holds what the resume says, with a verbatim evidence quote for every judgement. `derived` holds values computed in code from `extracted` and the taxonomy. The `search` layer is the index: the FTS5 body and the embeddings. The full design record, with the decisions and their reasons, is in `docs/design.md`.

### Modules

| Module | What it does |
|---|---|
{MODULE_ROWS}
"""

RESULTS = """\
## Results

Numbers come from the latest files in `evals/results/`. The judge is Claude Opus 5.5 and the task model is Claude Sonnet 5.5. The tables above are read from those files, so they match what follows.

**Parsing.** The judge graded 143 fields across the 10 resumes. 86.7% were `correct` and 93.4% were `correct` or `partial`. It found 0 hallucinated and 0 missing fields. `name`, `contact`, `certifications`, and `languages` scored 100%. `search_summary` scored 40%: the judge marks a summary `partial` when it attaches a skill or figure to the wrong role, which is a real fault in the summary but not a wrong fact about the candidate. Judging cost $0.66 for the run.

**Judge against human.** Three resumes (Chen Li, Omar El-Hassan, Zara Al-Rashid) were graded by hand. On their 42 fields the judge matched the human grade exactly 92.9% of the time and within one grade level 100% of the time. The three disagreements are one `skills`, one `investment_approach`, and one `experience` field.

**Ranking.** Six requisitions, each graded 0/1/2 per candidate by the judge. Mean nDCG@5 / P@3 for the four variants:

| Variant | nDCG@5 | P@3 |
|---|---|---|
| full blend (0.5 rule, 0.2 BM25, 0.3 embedding) | 0.969 | 0.722 |
| rule only | 0.958 | 0.722 |
| BM25 only | 0.968 | 0.778 |
| embedding only | 0.927 | 0.722 |

**Cost.** The 10 parses cost $0.7535 in total, $0.075 per resume (three calls plus any targeted regeneration), and took 28.5 seconds per resume on average.

### Caveats

- **The human gold is a draft.** The three hand-graded resumes were graded by one person and the grades are marked `draft`. Judge-human agreement of 92.9% is agreement with that draft.
- **The pools are tiny.** Each requisition prunes the 10 candidates down to between 3 and 9 before ranking. In a pool of 3, nDCG@5 is close to 1 whatever the order is.
- **The judge varies between runs.** Two parsing runs on the same data scored 85.3% and 86.7%. The judge prompt also changed between them (`pj1` to `pj2`), so the gap mixes judge variance with a prompt change, and one run per setting is not enough to separate the two.
- **The blend cannot be tuned on this data.** The four variants differ by less than one swapped pair in a small pool, so the data cannot say whether 0.5/0.2/0.3 is better than any other split. The weights come from the design, not from the eval.
- **Ties sort alphabetically.** A variant that scores many candidates equally is ranked by name inside each tie, which can help or hurt a variant by chance.
"""

SCALABILITY = """\
## Scalability and additional features

This section covers what the code is built to support, what would change at volume, and what is not built. The reasons behind each choice are in `docs/design.md` (sections 3.4, 6, 7, 9, 10, and 13).

### Other input formats

Only DOCX and PDF are implemented. The route for each other format:

| Tier | Formats | Route | Notes |
|---|---|---|---|
| 1 | docx, pdf | python-docx, PyMuPDF | implemented |
| 1 | txt, md, html | read the file directly | not built, a few lines each |
| 2 | doc, rtf, odt, pages | LibreOffice headless to docx or pdf, then tier 1 | adds a system dependency |
| 3 | png, jpg, scanned pdf | vision model, or Tesseract OCR | OCR is cheaper at volume and worse on layout |
| 4 | LinkedIn JSON, ATS XML | map straight into `extracted` | skips the LLM entirely |

### Vision fallback

`extract` decides per file whether the text layer is usable. It switches to the vision path when a PDF page has fewer than 400 characters, when an image covers more than half a page, or when the unknown-word ratio after repair is above 10%. The vision path renders the pages to PNG at 150 dpi and sends them as image blocks with the same schemas. The path is written but none of the 10 sample files triggers it, so it has not been exercised on real input.

### Postgres and pgvector

The DDL in `store/schema.sql` moves to Postgres with three changes. FTS5 becomes a `tsvector` column with a GIN index. The `embeddings` table becomes a `pgvector` column with an HNSW index, which replaces the in-memory brute-force cosine in `VectorIndex`. The `parse_jobs` table becomes the work queue. SQLite is enough for this demo. The first limits at volume are the in-memory vector scan and the single writer (a guess from the code, not measured).

### Queue-based ingest and the `parse_jobs` contract

`ingest()` already treats each file as an independent job. A job has a `file_sha256`, a `prompt_version`, and a status that moves only forward through `queued`, `extracting`, `parsing`, `deriving`, `indexing`, and one of `ok`, `failed`, or `skipped`. The idempotency key is `(file_sha256, prompt_version)`: a second ingest of the same bytes under the same prompt is skipped. Each file is written in one transaction, so a crash leaves either the old candidate or the new one.

To scale out, the thread pool is replaced by a queue (SQS, Cloud Tasks, or a Postgres `SKIP LOCKED` table). Producers insert `queued` rows. Stateless workers claim a row, run extract and parse, and write the result. The state machine and the idempotency key do not change.

The arithmetic matters here. The measured latency is 28.5 seconds per resume. With the 4 workers `ingest` uses, that is about 8 resumes a minute. A bulk load of 100,000 resumes at that rate takes about 8 days, so it needs about 100 concurrent workers (about 8 hours) and an API rate limit sized for them. The design called for a token-bucket rate limiter on API calls. It is not built: the code relies on 4 workers and the SDK's retries (`max_retries=4`), and a `RateLimitError` that survives the retries is recorded as that attempt's error and moves the retry ladder to its next rung.

### Re-parse and re-derive are different operations

A **prompt-version bump** changes what the LLM returns, so every resume must be parsed again and costs API money. `ingest --force` does that, and a changed `PROMPT_VERSION` makes the skip check miss. A **schema-version bump** changes only how `derived` is computed from `extracted`, so `resume-search rederive` rebuilds the derived fields, the facet tables, and the vectors from the stored `record_json` with no API call. Splitting the layers is what makes the second case free.

### Term matching beyond the synonym dictionary

Skills are matched today by the synonym dictionary in `taxonomy/skills.yaml` plus a rapidfuzz fallback, and the cross-check in `parse/verify.py` compares the LLM's skills against the same dictionary. Two further layers are designed and not built:

- **spaCy EntityRuler cross-check.** Phrase patterns compiled from the taxonomy (skills, tools, firms, sectors, certifications, locations, phone prefixes) run over the raw text at ingest. The spans would highlight evidence on the Candidate page and give a second, independent skill and country guess. The LLM keeps priority for display and disagreements are counted into `quality`.
- **Knowledge-graph query expansion.** A networkx graph built from the taxonomy YAML (synonym to canonical, skill to category, firm to firm type, approach to related terms) would expand a query at search time, so "systematic" also finds "stat arb", "factor", and "quant".

### A small classifier as a cheap pre-filter

At volume it is wasteful to send every resume to an LLM to learn its investment approach. A TF-IDF plus logistic-regression classifier, trained on the LLM's own labels for `investment_approach` and sector, could label most resumes for almost nothing and send only low-confidence ones to the LLM. A confusion matrix against the LLM labels would show where it is safe. It is not built, and 10 resumes are too few to train one.

### PII

`email`, `phone`, and `location_text` live only in their own `candidates` columns. The FTS body is the resume text with email, URL, and phone lines removed, and the embedding inputs are the `search_summary` and experience bullets. A validator rejects a `search_summary` that contains an `@` or a run of seven or more digits. Two gaps remain. The full resume text, contact block included, goes to the Anthropic API during parsing, which is a data-processing decision to confirm before real candidates are loaded. And the read-time redaction switch in the design (mask the three columns for users who should not see them) is not built.

### Cost per resume at volume

The measured cost is $0.075 per resume (10 resumes, $0.7535 in total, at the per-token prices in `ingest/pipeline.py`). That is about $75 per 1,000 resumes and about $7,500 per 100,000 at list prices. Prompt caching is already on for the system prompts. The batch API, which is billed at a discount and suits an overnight bulk load, and the small-classifier pre-filter are the two levers not yet used. Judging costs extra: $0.66 for a parsing eval and $0.17 for a ranking eval.

### What nondeterminism means for re-ingest

An LLM does not return the same JSON twice. The skip check means a re-ingest of the same file under the same prompt version does not call the model, so the stored record stays stable. Nondeterminism enters only when a re-parse is forced or the prompt version changes. The eval harness is the control for that case. Run `resume-search eval run --suite parsing` before and after the change, compare per-field accuracy, and read the judge-human agreement to see whether the judge itself moved. A prompt change that lowers a field's accuracy or adds hallucinations shows up before it reaches the index.

### Known limitations

- **The JD years range is a hard filter.** A pasted JD that says "4-8 years" prunes candidates outside it (`years_mode` defaults to `must`). A candidate with 8.5 years never appears.
- **Ties sort alphabetically.** Equal final scores are ordered by candidate name, so the order inside a tie carries no information.
- **Country is null when the LLM leaves a role's location blank.** The role then does not count toward `current_country` or `countries_worked`, and location criteria miss the candidate.
- **A completed CFA Level III does not count as a charter.** `has_cfa` is true only for a charterholder or a bare "CFA" entry, so a candidate who passed Level III and has not yet been awarded the charter is excluded by a CFA filter.
- **The eval is small.** Ten resumes, six requisitions, pools of 3 to 9, and a draft gold set. It catches regressions. It does not rank two close designs.
"""

LAYOUT_AND_RUN = """\
## Repository layout

```
case-study-millennium/
  2025_ds_case_study_resume_platform.ipynb   this notebook (generated)
  README.md                                  quick start and reference
  pyproject.toml                             uv project, Python 3.12
  docs/design.md                             architecture record
  scripts/build_notebook.py                  generates and executes this notebook
  taxonomy/                                  YAML: countries, sectors, firm types, asset classes, skills, weights
  src/resume_search/
    extract/ parse/ derive/ store/ search/ ingest/ eval/
    taxonomy.py cli.py
  app/                                       Home.py and pages/1_Search.py ... 5_Evals.py
  evals/                                     requisitions.jsonl, gold/, results/
  exports/                                   candidates.json, candidates.csv, candidates.sqlite, parsed/, text/
  data/                                      the 10 sample resumes (input only)
  tests/                                     pytest suite
```

## How to run

```
uv sync --extra dev
echo "ANTHROPIC_API_KEY=..." > .env

uv run resume-search ingest data/*.docx data/*.pdf     # parse, derive, index, export
uv run resume-search search --location US --approach Fundamental --sector Healthcare
uv run resume-search show "Chen Li"
uv run resume-search eval run --suite all              # LLM-judged evals, about $0.83
uv run resume-search eval show

uv run streamlit run app/Home.py                       # the app
RUN_LLM=1 uv run python scripts/build_notebook.py --execute   # rebuild this notebook, with the LLM demos
```

`resume-search` subcommands: `extract`, `parse`, `ingest`, `rederive`, `show`, `search`, `eval run`, `eval show`. The README lists every flag.
"""

INSTALL_INTRO = """\
## Install the package

The cell below installs the package you just wrote out. It prints the command by default so that a run of this notebook does not change your environment. Set `INSTALL=1` to run it.
"""

DEMO_INTRO = """\
## Demos

The cells below use the sample resumes in `data/`, the stored parse results in `exports/parsed/`, and the SQLite database `exports/candidates.sqlite`. None of them calls the API. If the database file is missing (it is git-ignored), the first demo cell rebuilds it from `exports/candidates.json` and the resume files, using only local code and the local embedding model.
"""

LLM_INTRO = """\
### Live LLM demos

These two cells call the Anthropic API and cost money. They run only when the environment variable `RUN_LLM` is `1`. Otherwise they print what they would do.
"""
