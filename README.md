# Candidate resume search platform

Resumes in, ranked shortlist out. Claude reads each DOCX or PDF into a structured record, SQLite stores it, and a requisition ranks candidates with a rule score, BM25, and embedding similarity. A Streamlit app sits on top: search, candidate detail, charts, ingest, eval results. Built for the Millennium BD data science case study, against the 10 sample resumes in `data/`.

**Live app:** [https://lizhezhao.github.io/case-study-resume-parsing](https://lizhezhao.github.io/case-study-resume-parsing)

**Notebook:** [open on GitHub](https://github.com/LizheZhao/case-study-resume-parsing/blob/main/2025_ds_case_study_resume_platform.ipynb) or [in nbviewer](https://nbviewer.org/github/LizheZhao/case-study-resume-parsing/blob/main/2025_ds_case_study_resume_platform.ipynb) if GitHub's preview times out on the size.

The submission notebook, `2025_ds_case_study_resume_platform.ipynb`, writes out every source file, runs the demos that need no API key, and talks through scale. If you want the reasoning behind a design choice, it is in [`docs/design.md`](docs/design.md).

## Quick start

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

1. Install the dependencies, including the dev tools.

   ```
   uv sync --extra dev
   ```

2. Add your API key. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

3. Parse and index the sample resumes. This one calls the API; expect about $0.78 for the 10 files.

   ```
   uv run resume-search ingest data/*.docx data/*.pdf
   ```

   You get `exports/candidates.sqlite`, `exports/candidates.json`, and `exports/candidates.csv`. Run it again and it skips every file it has already parsed.

4. Start the app.

   ```
   uv run streamlit run app/Home.py
   ```

   The app reads `exports/candidates.sqlite`. Set `RESUME_DB` to use another file.

The SQLite file is git-ignored. Parsing is the only step that needs a key. Search, the notebook demos, and the app all run without one, except for three spots that call the API on purpose: the JD parser, the rerank toggle, and the Ingest page.

## CLI reference

`uv run resume-search <command>`. Every command that takes `--db` defaults to `exports/candidates.sqlite`.

| Command | What it does | Options |
|---|---|---|
| `extract PATH...` | Extracts text, prints quality metrics as JSON, and writes `<stem>.txt`. No API call. | `--out` (default `exports/text`) |
| `parse PATH...` | Extracts and parses with the LLM, and writes one JSON file per resume. Does not touch the database. | `--out` (default `exports/parsed`), `--workers` (default 4) |
| `ingest [PATH...]` | Extracts, parses, derives, indexes, and exports. Skips files already parsed under the current prompt version. | `--db`, `--force` (re-parse), `--workers`, `--reparse-derive-only` |
| `rederive` | Rebuilds derived fields, facet tables, and vectors from the stored records. No API call. | `--db` |
| `show CANDIDATE` | Prints a candidate's derived fields. `CANDIDATE` is an id prefix or a name substring. | `--db` |
| `search` | Ranks candidates against a requisition built from a JD, from flags, or both. | see below |
| `eval run` | Runs the LLM-judged suites and writes `evals/results/<suite>-<timestamp>.json`. | `--suite parsing\|ranking\|all`, `--db`, `--out`, `--only` (candidate ids, parsing only) |
| `eval show` | Prints the latest metrics table for each suite. | `--suite`, `--out` |

`search` options:

| Option | Meaning |
|---|---|
| `--jd TEXT`, `--jd-file PATH` | Parse a job description into a requisition (one API call). Flags below override the parsed values. |
| `--location`, `--scope current\|worked\|either` | ISO-2 country or region, and which of the candidate's locations to match. |
| `--approach`, `--sector` | `Fundamental`, `Systematic`, or `Both`. `--sector` can repeat. |
| `--years MIN-MAX` | A single number, `MIN-MAX`, or `MIN+`. Always a must filter. |
| `--skill`, `--must-skill` | Preferred and required skills. Both can repeat. |
| `--cfa` | Require the CFA charter. |
| `--top N`, `--rerank`, `--json` | Result count (default 20), Sonnet rerank of the top 10, JSON output. |

## Architecture

Five stages. Each one can be re-run on its own.

1. **Extract.** `python-docx` and PyMuPDF give plain text in reading order. A repair map fixes ligature glyphs; an unknown-word ratio flags text that is still bad. A PDF with too little text, a big image, or too many unknown words goes down the vision path instead.
2. **Parse.** Three constrained-decoding calls (profile and experience in parallel, then assessment) come back as Pydantic models. A registry of deterministic checks holds the result up against the text, and any disagreement triggers one targeted regeneration. A call that fails climbs the ladder: Sonnet, Sonnet with the validation errors, Opus.
3. **Derive.** Months of experience, seniority band, countries, regions, normalized skills, CFA status. All code, none of it asked of the LLM.
4. **Store.** SQLite with scalar columns for filters, the full record as JSON, facet tables, an FTS5 keyword index built from the text minus the contact block, and float32 embeddings.
5. **Search.** Must criteria prune in SQL. Preferred criteria become a rule score. BM25 and embedding cosine score the free text. Final = 0.5 rule + 0.2 BM25 + 0.3 embedding.

Schema, prompt, cross-verification table, and the scale-up notes are all in `docs/design.md`.

## Measured results

Numbers from `evals/results/` (Sonnet 5.5 does the work, Opus 5.5 grades it) and from the `parse_jobs` table.

| Measure | Value |
|---|---|
| Parsing accuracy | 97.9% of 143 fields `correct`, 99.0% `correct` or `partial`, 0 hallucinated, 0 missing |
| Weakest field | `skills`, 90%: one skill taken from a paper title |
| Judge against human | 95.2% exact, 100% within one grade level, 42 fields on three hand-graded resumes |
| Location | `current_country` matches the hand gold for 10 of 10 candidates |
| Cost per resume | $0.078 (10 resumes, $0.7797 total), three calls plus any regeneration at list prices |
| Time per resume | 28.4 seconds on average with 4 workers |
| Ranking, mean nDCG@5 | full blend 0.969, rule only 0.958, BM25 only 0.968, embedding only 0.927, over six requisitions with pools of 3 to 9 after pruning |
| Ranking, mean P@3 | full blend 0.722, rule only 0.722, BM25 only 0.778, embedding only 0.722 |

A parsing eval costs about $0.65 and a ranking eval about $0.17.

## Repository layout

```
2025_ds_case_study_resume_platform.ipynb   submission notebook (generated)
pyproject.toml                             uv project, Python 3.12
docs/design.md                             architecture record
scripts/build_notebook.py                  generates and runs the notebook
scripts/notebook_text.py                   the notebook's prose cells
taxonomy/                                  YAML: countries, sectors, firm types, asset classes, skills, weights
src/resume_search/
  extract/                                 DOCX and PDF to text, repair, quality
  parse/                                   models, prompts, LLM client, cross-verification
  derive/                                  experience, location, skills
  store/                                   schema.sql, database, FTS body, exports
  search/                                  requisition, rules, BM25, vectors, ranking
  ingest/                                  pipeline and the parse_jobs state machine
  eval/                                    judge, metrics, parsing and ranking suites
  taxonomy.py, cli.py
app/                                       Home.py, common.py, pages/1_Search.py to 5_Evals.py
evals/                                     requisitions.jsonl, gold/, results/
exports/                                   candidates.json, candidates.csv, parsed/, text/
data/                                      the 10 sample resumes and the case-study brief
tests/                                     pytest suite
```

## Regenerate the notebook

```
uv run python scripts/build_notebook.py --execute
```

The script reads the files listed above straight from disk and turns them into `%%writefile` cells, which is why the notebook can never drift from the code. It refuses to run while `src/`, `app/`, `taxonomy/`, or `evals/requisitions.jsonl` has uncommitted changes; `--allow-dirty` overrides that.

With `--execute` it also runs the notebook through `jupyter nbconvert` on the `case-study` kernel (600-second timeout) and keeps the outputs. The two cells that call the API print a skip message unless `RUN_LLM=1` is set. Register the kernel once:

```
uv run python -m ipykernel install --user --name case-study --display-name "case-study (3.12)"
```

## Run the evals

```
uv run resume-search eval run --suite all
uv run resume-search eval show
```

`eval run` calls the Opus judge; both suites together come to about $0.83. My hand grades for three resumes are in `evals/gold/parsing_human.jsonl`, the six requisitions in `evals/requisitions.jsonl`. Ranking grades are cached in `evals/results/ranking-grades-cache.json`, so a re-run only grades requisitions or candidates that actually changed.

## Run the tests

```
uv run pytest -q
uv run ruff check src tests app scripts
```

## Next features

- **spaCy EntityRuler cross-check.** A second, independent read on skills and country straight off the raw text. Today skills are only cross-checked against the synonym dictionary.
- **Knowledge-graph query expansion.** So a search for "systematic" also pulls up "stat arb" and "factor".
- **Small classifier.** TF-IDF plus logistic regression trained on the LLM's own labels, so most resumes never need the LLM at volume.
- **Rate limiter and queue.** A token bucket in front of the API client, and a real work queue where the 4-thread pool is now.
- **Other file formats.** Readers for txt, md, and html, plus a LibreOffice route for doc, rtf, and odt. DOCX and PDF work today, with the vision path covering image-only PDFs.
- **PII redaction at read time.** A per-role switch that masks the three contact columns. They are already out of the keyword index and the embeddings.
