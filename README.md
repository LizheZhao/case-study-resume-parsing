# Candidate resume search platform

This project reads resumes (DOCX and PDF) with Claude, stores them as structured records in SQLite, and ranks candidates against a job requisition with a rule score, BM25, and embedding similarity. A Streamlit app wraps search, candidate detail, charts, ingest, and the eval results. It was built as the Millennium BD data science case study, with 10 sample resumes in `data/`.

The submission notebook, `2025_ds_case_study_resume_platform.ipynb`, writes out all the code, runs demos that need no API key, and discusses scale. The reasons behind each design choice are in [`docs/design.md`](docs/design.md).

## Quick start

You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

1. Install the dependencies, including the dev tools.

   ```
   uv sync --extra dev
   ```

2. Add your API key. Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

3. Parse and index the sample resumes. This calls the API and costs about $0.75 for the 10 files.

   ```
   uv run resume-search ingest data/*.docx data/*.pdf
   ```

   The command writes `exports/candidates.sqlite`, `exports/candidates.json`, and `exports/candidates.csv`. A second run skips every file it has already parsed.

4. Start the app.

   ```
   uv run streamlit run app/Home.py
   ```

   The app reads `exports/candidates.sqlite`. Set `RESUME_DB` to use another file.

The SQLite file is git-ignored. Parsing is the only step that needs an API key. Search, the demos in the notebook, and the Streamlit app run without one, apart from the JD parser, the rerank toggle, and the Ingest page.

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

A resume passes through five stages, and each stage can be re-run alone.

1. **Extract.** `python-docx` and PyMuPDF produce plain text in reading order. A repair map fixes ligature glyphs, and an unknown-word ratio flags bad text. A PDF with too little text, a large image, or too many unknown words goes to the vision path.
2. **Parse.** Three constrained-decoding calls (profile and experience in parallel, then assessment) return Pydantic models. A registry of deterministic checks compares the result with the text, and a disagreement triggers one targeted regeneration. A failed call moves up a ladder: Sonnet, Sonnet with the validation errors, Opus.
3. **Derive.** Code, not the LLM, computes months of experience, seniority band, countries, regions, normalized skills, and CFA status.
4. **Store.** SQLite holds scalar columns for filters, the full record as JSON, facet tables, an FTS5 keyword index with the contact block removed, and float32 embeddings.
5. **Search.** Must criteria prune in SQL. Preferred criteria give a rule score. BM25 and embedding cosine score the free text. The final score is 0.5 rule + 0.2 BM25 + 0.3 embedding.

`docs/design.md` has the schema, the prompt, the cross-verification table, and the scale-up notes.

## Measured results

The numbers come from `evals/results/` (Claude Sonnet 5.5 as the task model, Claude Opus 5.5 as the judge) and from the `parse_jobs` table.

| Measure | Value | Caveat |
|---|---|---|
| Parsing accuracy | 86.7% of 143 fields `correct`, 93.4% `correct` or `partial`, 0 hallucinated, 0 missing | One judge run. Two runs differed by 1.4 points, and the judge prompt changed between them. |
| Weakest field | `search_summary`, 40% | The judge marks a summary `partial` when it attaches a fact to the wrong role. |
| Judge against human | 92.9% exact, 100% within one grade level, 42 fields | Three resumes, one grader, and the grades are marked `draft`. |
| Cost per resume | $0.075 (10 resumes, $0.7535 total) | Three calls plus any regeneration, at list prices. |
| Time per resume | 28.5 seconds on average | Measured with 4 workers. |
| Ranking, mean nDCG@5 | full blend 0.969, rule only 0.958, BM25 only 0.968, embedding only 0.927 | Six requisitions, pools of 3 to 9 after pruning. |
| Ranking, mean P@3 | full blend 0.722, rule only 0.722, BM25 only 0.778, embedding only 0.722 | The variants differ by less than one swapped pair, so the data cannot tune the blend. |

A parsing eval costs about $0.66 and a ranking eval about $0.17.

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

The script reads the files listed above from disk and writes `%%writefile` cells, so the notebook cannot differ from the code. It refuses to run when `src/`, `app/`, `taxonomy/`, or `evals/requisitions.jsonl` has uncommitted changes. Pass `--allow-dirty` to override that.

With `--execute`, the script runs the notebook with `jupyter nbconvert` on the `case-study` kernel (a 600-second timeout) and stores the outputs. The two cells that call the API print a skip message unless `RUN_LLM=1` is set. Register the kernel once with:

```
uv run python -m ipykernel install --user --name case-study --display-name "case-study (3.12)"
```

## Run the evals

```
uv run resume-search eval run --suite all
uv run resume-search eval show
```

`eval run` calls the Opus judge and costs about $0.83 for both suites. Hand grades for three resumes are in `evals/gold/parsing_human.jsonl`. The six requisitions are in `evals/requisitions.jsonl`. Judge grades for the ranking suite are cached in `evals/results/ranking-grades-cache.json`, so a re-run grades only requisitions or candidates that changed.

## Run the tests

```
uv run pytest -q
uv run ruff check src tests app scripts
```

## What is not built

- **spaCy EntityRuler cross-check.** Skills are cross-checked against the synonym dictionary only.
- **Knowledge-graph query expansion.** A search for "systematic" does not expand to "stat arb" or "factor".
- **Small classifier.** There is no pre-classifier that would spare the LLM at volume.
- **Rate limiter and queue.** Ingest uses 4 worker threads and the SDK's retries. There is no token bucket and no external queue.
- **Other file formats.** Only DOCX and PDF are read. The vision path exists but no sample file triggers it.
- **PII redaction at read time.** Contact fields are kept out of the keyword index and the embeddings, but the app shows them to every user.
- **Deployment.** The app runs locally. The public URL is to be added at submission.
