# Candidate Resume Search Platform — Design

Status: agreed in design review, 2026-10-24. No code written yet.
Scope: Millennium BD case study. 10 sample resumes (8 DOCX, 2 PDF), ~9 hour budget,
deliverables are the notebook, a Streamlit app link, JSON/CSV exports, and a
scalability discussion.

## 1. Decisions at a glance

| Area | Decision |
|---|---|
| Runtime | Python 3.12, uv, venv prompt `case-study` |
| Extraction | Local text first: python-docx (body in document order, tables as pipe rows), PyMuPDF for PDF. Glyph repair + unknown-word check on every file. Vision path only when triggers fire. |
| LLM | Anthropic API, structured outputs from the Pydantic schema, validate, retry once on Sonnet, second retry on Opus |
| Record | Three layers: `extracted` (literal), `derived` (normalized, computed in code where possible), `search` (index-only) |
| Taxonomy | YAML in `taxonomy/`, loaded into SQLite tables at startup. Country → region is a table, never asked of the LLM. Sectors = core enum + free tags. |
| Storage | SQLite: scalar columns for filters, JSON column for full record, child tables, FTS5 for keywords, float32 blobs for embeddings |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers, local CPU, 384-dim. Two granularities: summary + each experience entry. |
| Search | Requisition object (from widgets or from a pasted JD parsed by the LLM). Must criteria prune, preferred criteria score. Final = 0.5 rule + 0.2 BM25 + 0.3 embedding. Optional LLM rerank toggle. |
| Terms | Synonym dictionary + rapidfuzz first. spaCy EntityRuler cross-check (skills, country) second. networkx knowledge graph for query expansion if time. |
| Eval | Opus as judge when Sonnet does the task. Parsing: per-field grades, 3 resumes hand-calibrated. Ranking: 6 requisitions × 10 candidates, nDCG@5, P@3. |
| Pipeline | One `ingest()` used by CLI and Streamlit. Idempotent on file hash + prompt version. `parse_jobs` table. Worker pool with rate limiter. |
| UI | 5 pages: Search, Candidate, Insights, Ingest, Evals. Layout polish is last. |
| PII | Contact fields in own columns, excluded from FTS text and embeddings |
| Deferred | Deployment target and GitHub Actions, decided at submission time |

## 2. Repo layout

```
case-study-millennium/
  pyproject.toml            uv-managed, Python 3.12
  notebook.ipynb            %%writefile cells for src/, demo cells, app link, discussion
  docs/design.md            this file
  taxonomy/                 countries.yaml sectors.yaml firm_types.yaml asset_classes.yaml
                            skills.yaml locations.yaml weights.yaml
  src/resume_search/
    extract/   docx.py pdf.py repair.py quality.py   -> ExtractedText
    parse/     models.py prompt.py client.py         -> ParsedResume (LLM)
    derive/    experience.py location.py skills.py summary.py -> DerivedFields
    store/     schema.sql db.py export.py            -> SQLite, JSON/CSV
    search/    requisition.py rules.py keyword.py vectors.py rank.py
    ingest/    pipeline.py jobs.py
    eval/      judge.py parsing_eval.py ranking_eval.py
  app/         Home.py pages/1_Search.py 2_Candidate.py 3_Insights.py 4_Ingest.py 5_Evals.py
  evals/       parsing_cases.jsonl requisitions.jsonl results/
  exports/     candidates.json candidates.csv
  data/        the 10 resumes (input only)
  tests/
```

## 3. Extraction

### 3.1 Text representation sent to the LLM

Plain text in reading order. No markdown, no section detection, no stop-word
removal.

- DOCX: iterate `document.element.body` children in order; paragraphs emit their
  text, tables emit one line per row with cells joined by ` | `. This keeps
  Viktor Sharat's header (which lives in a table) at the top.
- PDF: PyMuPDF `page.get_text("text")` per page, pages joined by `\n\f\n`.
- Normalize: NFKC, collapse runs of spaces/tabs to one space, collapse 3+ blank
  lines to 2, strip trailing whitespace. Curly quotes left alone (NFKC handles
  the common ones).

### 3.2 Repair and quality check (runs on every file)

1. Repair map applied to the text: `Ɵ→ti`, `ﬀ→ff`, `ﬃ→ffi`, `ﬁ→fi`, `ﬂ→fl`,
   `ﬄ→ffl`; drop private-use glyphs U+E000–U+F8FF. The map lives in
   `extract/repair.py` and is extended as new fonts show up.
2. Tokenize on whitespace/punctuation. `unknown_ratio` = tokens not in
   (English wordlist ∪ taxonomy terms ∪ tokens containing a digit) / tokens.
   For tokens that contained a repaired or non-ASCII character and are still
   unknown, try `rapidfuzz.process.extractOne` against the wordlist; accept at
   score ≥ 90.
3. Record `text_quality = {unknown_ratio_before, unknown_ratio_after, repairs: n}`.

### 3.3 Vision trigger

Any of: text < 400 chars per page; an image bbox covers > 50% of a page;
`unknown_ratio_after` > 0.10. Then render pages to PNG at 150 dpi and send them
as image blocks with the same tool schema. `meta.extraction_path` is `text` or
`vision`. None of the 10 sample files trigger it; it is tested once by rendering
Ryan Patel's PDF to images.

### 3.4 Other formats (discussion section, code only for tier 1 + vision)

| Tier | Formats | Route | Notes |
|---|---|---|---|
| 1 | docx, pdf, txt, md, html | native libraries | implemented |
| 2 | doc, rtf, odt, pages | LibreOffice headless → docx/pdf, then tier 1 | adds a system dependency |
| 3 | png, jpg, scanned pdf | vision model (implemented) or Tesseract OCR | OCR is cheaper at volume, worse on layout |
| 4 | LinkedIn JSON, ATS XML | direct mapping into `extracted` | skips the LLM entirely |

## 4. Record schema

### 4.1 Pydantic models (LLM output = `ParsedResume`)

```python
from enum import Enum
from pydantic import BaseModel, Field

class Relevance(str, Enum):
    core = "core"            # investment research, trading, PM, quant, risk at a financial firm
    adjacent = "adjacent"    # consulting, corporate finance, data science elsewhere
    unrelated = "unrelated"  # teaching, student clubs, retail, etc.

class EmploymentType(str, Enum):
    full_time = "full_time"; internship = "internship"; part_time = "part_time"; academic = "academic"

class FirmType(str, Enum):      # from taxonomy/firm_types.yaml
    hedge_fund = "Hedge fund"; asset_manager = "Asset manager"; sell_side_bank = "Sell-side bank"
    sell_side_broker = "Sell-side broker"; pe_vc = "PE/VC"; consulting = "Consulting"
    corporate = "Corporate"; academic = "Academic"; other = "Other"

class AssetClass(str, Enum):
    equities = "Equities"; fixed_income = "Fixed Income"; credit = "Credit"; fx = "FX"
    commodities = "Commodities"; derivatives = "Derivatives"; multi_asset = "Multi-asset"

class Sector(str, Enum):        # core enum from the brief; free tags go in other_sector_tags
    technology = "Technology"; healthcare = "Healthcare"; financials = "Financials"
    energy = "Energy"; industrials = "Industrials"; consumer = "Consumer"
    credit = "Credit"; macro = "Macro"; other = "Other"

class Approach(str, Enum):
    fundamental = "Fundamental"; systematic = "Systematic"; both = "Both"; unclear = "Unclear"

class Evidence(BaseModel):
    quote: str = Field(max_length=200, description="Verbatim span from the resume")

class Labeled(BaseModel):
    label: str
    evidence: Evidence

class Contact(BaseModel):
    email: str | None; phone: str | None; location_text: str | None

class Education(BaseModel):
    institution: str
    degree: str | None            # as written, e.g. "MSc", "BBA", "PhD"
    degree_level: str | None      # Bachelors | Masters | MBA | PhD | Other
    field: str | None
    start: str | None             # YYYY-MM, YYYY if month unknown
    end: str | None
    country: str | None           # ISO-2 of the institution if stated or unambiguous

class Experience(BaseModel):
    employer: str
    title: str
    location_text: str | None
    country: str | None           # ISO-2 where the role was based
    start: list[str]              # one YYYY-MM or YYYY per stint; [] when the resume gives no dates
    end: list[str | None]         # parallel to start; last entry null iff is_current
    is_current: bool
    employment_type: EmploymentType
    relevance: Relevance
    firm_type: FirmType
    asset_classes: list[AssetClass] = []
    sectors: list[Labeled] = []           # label from Sector enum, with evidence
    other_sector_tags: list[str] = []     # free text, e.g. "Specialty Chemicals"
    markets_covered: list[Labeled] = []   # ISO-2 or region label, ONLY when stated
    duration_months: int | None   # from duration text, used when dates are absent
    bullets: list[str]

class Skills(BaseModel):
    programming: list[str] = []
    tools: list[str] = []
    domain: list[str] = []        # "DCF", "factor models", "options pricing"

class Language(BaseModel):
    name: str; level: str | None

class ParsedResume(BaseModel):
    name: str
    contact: Contact
    education: list[Education]
    experience: list[Experience]
    skills: Skills
    certifications: list[str] = []   # "CFA", "CFA Level II candidate", "FRM"
    languages: list[Language] = []
    investment_approach: Labeled      # label from Approach, with evidence
    search_summary: str = Field(min_length=200, max_length=700)
```

Validators (Pydantic, run before accepting a response):
- dates match `^\d{4}(-\d{2})?$`. `Experience.start` and `.end` are parallel lists, one entry
  per stint of the role; a normal role has one-element lists. Equal length; both empty when
  the resume gives no dates (duration-only roles), otherwise at least one stint. With
  `is_current` the last `end` is null; every other `end` is non-null. Each `start ≤` its
  `end`; stints are chronological with no overlap between stints of one role. A one-month
  stint written as a single month ("Sep-'13") is `start = end = 2013-09` and counts as one
  month. Zara Al-Rashid's J.P. Morgan line "Sep-'13, Dec-'13 & Nov-'14" is three such stints.
- `country` values in the ISO-2 set from `taxonomy/countries.yaml`.
- `sectors[].label` in Sector enum; `markets_covered[].label` in ISO-2 ∪ region names.
- `search_summary` contains no `@` and no 7+ digit run (contact leak check).
- every `Evidence.quote` is a substring of the resume text after whitespace
  normalization; a failing quote is a validation error and triggers the retry.

### 4.2 Derived fields (computed in code, `derive/`)

| Field | Rule |
|---|---|
| `current_country` | country of the `is_current` role with the latest start; else latest role; else gazetteer on `contact.location_text` |
| `current_region` | lookup `taxonomy/countries.yaml` |
| `countries_worked` | ordered unique list of `(country, months)` over full-time roles |
| `regions_worked` | mapped from above |
| `markets_covered` | union over roles |
| `months_full_time_relevant` | full_time roles with relevance ∈ {core, adjacent}; a role's months are the sum over its stints of (end or parse date) − start, a stint with start = end counting one month; the ranges of all stints of all roles are merged before summing; `YYYY`-only dates use June; roles with empty date lists use `duration_months` |
| `months_full_time_core` | same, core only |
| `months_internship` | internship roles, any relevance, kept separate, never added to the above |
| `seniority_band` | from `months_full_time_relevant`: Intern/Student (<6), Junior (6–24), Mid (24–60), Senior (60–120), Principal (>120) |
| `investment_approach` | from LLM `Labeled`, as parsed (the full-time-relevant-role rule is enforced in the assessment prompt and the verify registry, section 5.5) |
| `sectors` | union of role sector labels, with the first evidence quote each |
| `other_sector_tags` | union, lowercased, deduped |
| `asset_classes` | union |
| `firm_types` | union |
| `highest_degree` | max over `degree_level` in order Bachelors < Masters < MBA < PhD |
| `degree_fields` | list |
| `skills_normalized` | each skill lowercased → `taxonomy/skills.yaml` synonym map → canonical token; rapidfuzz ≥ 90 fallback; union with EntityRuler spans once that step exists |
| `has_cfa` | any certification matches `^CFA( charter(holder)?)?$` ; `cfa_level` for candidates |
| `search_summary` | from LLM, stored as is |
| `quality` | `text_quality` + `skill_disagreement_count` + `country_disagreement` once cross-checks exist |

### 4.3 `search_summary` contract (in the prompt)

60–90 words, slot order fixed:

1. seniority band and approach, with relevant years in parentheses
2. asset classes and sectors
3. markets covered (only if stated)
4. current or most recent employer, firm type in parentheses, and title
5. two or three standout skills, tools or achievements, with numbers when present
6. highest degree and institution, certifications

No contact information. No evaluative adjectives.

Example: "Junior systematic equity analyst (2.3y relevant). Greater China
technology and healthcare equities. Most recently Investment Analyst at Bank of
China (sell-side bank), Hong Kong; built a multi-factor model (Sharpe 1.8+) and
an automated market surveillance system. Python, R, SQL, Bloomberg, WIND. MSc
Financial Technology, University of Hong Kong."

### 4.4 Meta

```
candidate_id   sha256(file bytes)[:12]
source_file, file_sha256, file_bytes, page_count
parsed_at, extractor, extraction_path (text|vision)
llm_model, prompt_version, schema_version
parse_status   ok | ok_retry_sonnet | ok_retry_opus | failed
attempts       [{model, latency_ms, input_tokens, output_tokens, error}]
text_quality   {unknown_ratio_before, unknown_ratio_after, repairs}
```

## 5. LLM call contract

### 5.1 Function signature

```python
def parse_resume(text: str, meta: SourceMeta, *, images: list[bytes] | None = None) -> ParseResult
# ParseResult = ParseOk(record: ParsedResume, attempts: list[Attempt]) | ParseFailed(attempts, last_error)
```

Attempt ladder: Sonnet → (validation or API error) → Sonnet with the validation
errors appended to the user message → Opus with the same. Three attempts max.
Each attempt records model, tokens, latency, error. Output is constrained with
structured outputs (`client.messages.parse(..., output_format=ParsedResume)`),
not a forced tool call: Sonnet 5.5 and Opus 5.5 reject `tool_choice` of type
`tool`/`any`. Model IDs are `claude-sonnet-5-5` and `claude-opus-5-5`.

### 5.2 System prompt (v1, rendered from taxonomy files)

```
You extract structured data from a single candidate resume for a hedge fund
business-development team. Return only the JSON object matching the schema.

Rules
1. Copy facts; never infer. If a value is not stated, use null or an empty list.
2. Dates: YYYY-MM when the month is given, YYYY otherwise. "Present", "till
   now", "since X" mean is_current=true and the last end is null. start and
   end are parallel lists, one entry per stint. Separate periods under one
   role are separate stints: "Sep-'13, Dec-'13 & Nov-'14" is start
   [2013-09, 2013-12, 2014-11], end [2013-09, 2013-12, 2014-11]. A single
   month ("Sep-'13") is one stint with start = end = 2013-09. Fix obvious
   typos in month names.
3. Country: ISO-2 code of where the role or school was based, from the
   location on that line or the contact block. Do not output regions.
4. markets_covered: fill ONLY when the resume states what markets or
   geographies the person covered (e.g. "Indian healthcare", "Greater China
   markets"). Do not infer from employer location. Give the quote.
5. Sectors: choose from {Technology, Healthcare, Financials, Energy,
   Industrials, Consumer, Credit, Macro, Other}; put narrower names
   ("Specialty Chemicals", "MedTech") in other_sector_tags. Give a quote for
   each sector label.
6. relevance: core = investment research, trading, portfolio management,
   quant research/development, risk, at a financial firm; adjacent =
   consulting, corporate finance, data or software work outside finance;
   unrelated = everything else (teaching, clubs, retail).
7. employment_type: internship when the title or text says intern, summer
   analyst, or the role is under 6 months while studying.
8. firm_type: {Hedge fund, Asset manager, Sell-side bank, Sell-side broker,
   PE/VC, Consulting, Corporate, Academic, Other}. Use your knowledge of
   the firm when the resume does not say.
9. investment_approach: Systematic when the work is model-, factor-, or
   code-driven trading or research; Fundamental when it is company analysis,
   valuation, coverage; Both when clearly both; Unclear otherwise. Quote the
   deciding span. Fundamental, Systematic or Both only when the quote comes
   from a full_time role with relevance core or adjacent; internships,
   part-time, academic and unrelated roles do not count, and the label is then
   Unclear (the quote still points at the best available span).
10. Every evidence quote must be a verbatim span of the resume text, at most
    200 characters.
11. search_summary: 60–90 words, slots in this order: (1) seniority band and
    approach with relevant years, (2) asset classes and sectors, (3) markets
    covered if stated, (4) most recent employer with firm type in
    parentheses and title, (5) two or three concrete skills, tools or
    achievements with numbers, (6) highest degree and institution,
    certifications. No contact details, no adjectives like "passionate".
12. Keep bullets verbatim, one string each, bullet characters removed.
```

### 5.3 User message

```
<source filename="{filename}" pages="{page_count}" extraction="{text|vision}"/>
<resume>
{text}
</resume>
```

On retry, append:

```
<validation_errors>
{pydantic error list, one per line}
</validation_errors>
Fix only these and return the full tool call again.
```

### 5.4 Three calls per resume (replaces the single call)

The full `ParsedResume` schema exceeds the structured-output grammar limit, so
parsing is three smaller calls, each with constrained decoding
(`client.messages.parse(..., output_format=Model)`):

| Call | Input | Output model | Runs |
|---|---|---|---|
| profile | source filename + resume text | `Profile`: name, contact, education, skills, certifications, languages | parallel with experience |
| experience | resume text | `ExperienceList`: list of `Experience` | parallel with profile |
| assessment | verified Profile + verified ExperienceList as JSON, plus resume text for quotes | `Assessment`: investment_approach (Labeled), search_summary | after cross-verification |

The filename goes to the profile call because names, certifications and sector
hints live there ("Michael Rodriguez, CFA.docx",
"Priya Nakamura_sellside_healthcare_RLTM.docx"). Prompt rule: the filename
corroborates; it is never the sole source for anything except the name.

Display-only fields are dropped from the schema: `summary_text`, `awards`,
`education.honors`, `education.gpa`, `contact.linkedin`.

Each `Experience` gains `duration_months: int | None`, filled from duration
text ("8 years 10 months" → 106, "3 Months" → 3, "8 weeks" → 2) when the
resume gives a duration. The derive layer uses dates when both are present
and `duration_months` otherwise. Viktor Sharat is the live case: seven roles,
no dates, durations only.

The retry ladder (Sonnet, Sonnet with errors, Opus with errors) applies per
call. `ParsedResume` is assembled from the three outputs and is the record
stored; downstream code is unchanged.

### 5.5 Cross-verification between the LLM and deterministic text parsing

A registry of checks, one row per field, in `parse/verify.py`. Each row names
a deterministic extractor over the repaired text (and filename), a comparator,
and the regeneration target. The registry is data; adding a check is adding a
row.

| Field | Deterministic source | Comparator | On disagreement, regenerate |
|---|---|---|---|
| `name` | filename stem minus suffixes like `Resume`, `CV`, `CFA`, dates; first non-empty text line | token-set ratio ≥ 80 against either | `name` |
| `contact.email` | email regex | exact, case-insensitive | `contact` |
| `contact.phone` | phone regex (`+` and 7+ digits with separators) | digits-only equality | `contact` |
| `certifications` | regex for CFA, FRM, CAIA, CPA, ACCA (text + filename), "Level I/II/III candidate" | set equality on canonical cert tokens | `certifications` |
| `skills` | taxonomy term matcher over text (`skills.yaml` variants, word-boundary, case-insensitive) | every matched taxonomy term must appear in the LLM list after normalization; LLM extras are allowed | `skills` |
| `experience[i].start/end` | every `YYYY-MM` or `YYYY` the LLM emits, for every stint, must correspond to a month-year or year token in the text (month names, `'22`, `Sept.2017`, etc.) | all stint dates grounded or not | that role's dates |
| `experience[i].duration_months` | duration regex (`N years M months`, `N months`, `N weeks`) near the role's employer line; and, when the role has stints, the stint sum computed in code | equality within 1 month | that role |
| `experience` count | number of distinct employer lines matched from `known_firms` plus title-line patterns | LLM must have ≥ that many roles | `experience` (whole list) |
| `education[i].degree_level` | degree keyword regex (MBA, MSc/MS/Master, BSc/BA/BBA/Bachelor, PhD, PGDM, Diplôme d'Ingénieur → Masters) | mapped level equals LLM level | that education entry |
| `contact` country / current role country | phone-prefix lookup and location-alias gazetteer from `countries.yaml` | equality | `contact` or that role |
| `investment_approach` | which role's title, employer or bullets contain the evidence quote, after the same whitespace and quote folding as `check_evidence` | label is `Unclear`, or the located role is full_time with relevance core or adjacent | the assessment call (runs after it) |

Policy:

1. Run all checks after the profile and experience calls return.
2. For each disagreement, make one targeted regeneration call with a schema
   containing only that field (or that list item), the resume text, the LLM's
   previous value, and the deterministic finding stated as evidence
   ("The text contains 'CFA' at: '...'; your certifications list was empty.").
3. Re-run that field's check. If it still disagrees, keep the LLM value,
   record the pair in `quality.unresolved` (field, llm_value, text_value), and
   count it. The Evals page lists unresolved items per resume.
4. The assessment call receives the verified profile and experience JSON, not
   the raw LLM output, so the summary is written from checked facts.

The `investment_approach` row needs the assessment, so it runs after that call,
with one regeneration of the assessment and then `quality.unresolved` as usual.

`meta` gains `verification: {checks_run, disagreements, regenerated,
unresolved: [...]}`.

## 6. SQLite schema

```sql
CREATE TABLE candidates (
  candidate_id TEXT PRIMARY KEY, name TEXT NOT NULL,
  email TEXT, phone TEXT, location_text TEXT,            -- PII columns, never indexed
  current_country TEXT, current_region TEXT,
  months_full_time_relevant INTEGER, months_full_time_core INTEGER, months_internship INTEGER,
  seniority_band TEXT, investment_approach TEXT, approach_evidence TEXT,
  highest_degree TEXT, has_cfa INTEGER, cfa_level TEXT,
  search_summary TEXT NOT NULL,
  record_json TEXT NOT NULL,                              -- full extracted+derived+meta
  prompt_version TEXT, schema_version INTEGER, parse_status TEXT, parsed_at TEXT,
  file_sha256 TEXT UNIQUE, source_file TEXT
);
CREATE TABLE experience (
  candidate_id TEXT REFERENCES candidates, idx INTEGER,
  employer TEXT, title TEXT, country TEXT,
  start TEXT, end TEXT,                  -- first stint start, last stint end (ordering, display)
  is_current INTEGER,
  employment_type TEXT, relevance TEXT, firm_type TEXT,
  months INTEGER,                        -- sum over stints, or duration_months when undated
  stints_json TEXT,                      -- {"start": [...], "end": [...]}
  PRIMARY KEY (candidate_id, idx)
);
CREATE TABLE education (candidate_id TEXT, idx INTEGER, institution TEXT, degree TEXT,
  degree_level TEXT, field TEXT, start TEXT, end TEXT, country TEXT, PRIMARY KEY (candidate_id, idx));
-- many-to-many facets, one row per (candidate, value)
CREATE TABLE candidate_sectors   (candidate_id TEXT, sector TEXT, evidence TEXT);
CREATE TABLE candidate_tags      (candidate_id TEXT, tag TEXT);
CREATE TABLE candidate_assets    (candidate_id TEXT, asset_class TEXT);
CREATE TABLE candidate_skills    (candidate_id TEXT, skill TEXT, source TEXT);   -- llm | ruler | both
CREATE TABLE candidate_countries (candidate_id TEXT, country TEXT, months INTEGER, kind TEXT); -- worked | covered
CREATE TABLE candidate_certs     (candidate_id TEXT, cert TEXT);
CREATE TABLE candidate_languages (candidate_id TEXT, language TEXT, level TEXT);
-- keyword index: resume text minus the contact block
CREATE VIRTUAL TABLE candidate_fts USING fts5(candidate_id UNINDEXED, body, tokenize='porter unicode61');
-- vectors: float32 little-endian blobs
CREATE TABLE embeddings (candidate_id TEXT, kind TEXT, idx INTEGER, model TEXT, dim INTEGER,
  vec BLOB, PRIMARY KEY (candidate_id, kind, idx));      -- kind: summary | role
-- ingest tracking
CREATE TABLE parse_jobs (
  job_id INTEGER PRIMARY KEY, file_sha256 TEXT, source_file TEXT, prompt_version TEXT,
  status TEXT,          -- queued | extracting | parsing | deriving | indexing | ok | failed | skipped
  attempts_json TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
  latency_ms INTEGER, error TEXT, queued_at TEXT, finished_at TEXT
);
-- taxonomy mirrors, loaded from YAML at startup
CREATE TABLE taxonomy_country (iso2 TEXT PRIMARY KEY, name TEXT, region TEXT);
CREATE TABLE taxonomy_sector (name TEXT PRIMARY KEY, parent TEXT);
CREATE TABLE taxonomy_skill_synonym (variant TEXT PRIMARY KEY, canonical TEXT, category TEXT);
CREATE TABLE ranking_weights (criterion TEXT PRIMARY KEY, weight REAL);
CREATE INDEX idx_cand_country ON candidates(current_country);
CREATE INDEX idx_cand_months ON candidates(months_full_time_relevant);
CREATE INDEX idx_exp_country ON experience(country);
```

Exports: `exports/candidates.json` (list of full records) and
`exports/candidates.csv` (the `candidates` table plus `;`-joined facet
columns), regenerated at the end of every ingest.

Scale-up note for the discussion: same DDL moves to Postgres; FTS5 → `tsvector`;
`embeddings` → `pgvector` with an HNSW index; `parse_jobs` → a queue.

## 7. Ingest pipeline

```
ingest(paths, *, force=False) -> list[JobResult]
```

State machine per file (status column in `parse_jobs`):

```
queued ─► extracting ─► parsing ─► deriving ─► indexing ─► ok
   │           │            │           │           │
   └► skipped  └► failed    └► failed   └► failed   └► failed
   (same sha256 + prompt_version already ok, and not force)
```

- `extracting`: read file, text repair, quality check, vision trigger decision.
- `parsing`: `parse_resume` ladder; token and cost accounting per attempt.
- `deriving`: compute §4.2; run EntityRuler cross-checks when available.
- `indexing`: upsert all tables in one transaction; FTS body; embeddings
  (summary + one per experience entry).
- After the batch: regenerate exports, `st.cache_data.clear()` when called
  from Streamlit.

Concurrency: `ThreadPoolExecutor(max_workers=4)` over files with a token-bucket
rate limiter on API calls. Each file is its own transaction, so a failure never
leaves a half-written candidate. Idempotency key is `(file_sha256,
prompt_version)`. A prompt-version bump re-parses; a schema-version bump that
only changes `derived` re-runs `deriving` + `indexing` from `record_json`
without an API call.

Streamlit Ingest page: uploader (multi-file) → calls `ingest` → shows
`parse_jobs` table live (status, attempts, tokens, cost, latency), throughput
(resumes/min), and cumulative cost. This page is the scalability demonstration.

## 8. Search and ranking

### 8.1 Requisition object

```python
class Criterion(BaseModel):
    value: Any
    mode: Literal["must", "preferred"] = "preferred"

class Requisition(BaseModel):
    free_text: str | None                      # pasted JD or keywords
    location: Criterion | None                 # ISO-2 or region
    location_scope: Literal["current", "worked", "either"] = "either"
    approach: Criterion | None
    sectors: Criterion | None                  # list
    markets_covered: Criterion | None          # list
    years_min: float | None; years_max: float | None; years_mode: Literal["must","preferred"] = "must"
    seniority_bands: Criterion | None
    skills_required: list[str] = []            # each a must
    skills_preferred: list[str] = []
    degree_min: Criterion | None
    certifications: Criterion | None
    languages: Criterion | None
    asset_classes: Criterion | None
```

Two ways to fill it: the filter widgets, or paste a JD → `parse_requisition()`
(same tool-use pattern, Sonnet) pre-fills the widgets → user edits → search.
The free text always feeds BM25 and the embedding query regardless.

### 8.2 Pipeline

1. **Prune** with SQL on every `must` criterion (and `years_mode=must`).
2. **Rule score** over `preferred` criteria, weights from `taxonomy/weights.yaml`:

   | criterion | weight | match rule |
   |---|---|---|
   | approach | 3 | exact, or `Both` matches either |
   | sector | 3 | any overlap; fraction of requested sectors matched |
   | location | 2 | see 8.3 |
   | markets_covered | 2 | any overlap |
   | years in range | 2 | 1 inside range, linear decay to 0 at ±2y outside |
   | each preferred skill | 1 | canonical token match, or rapidfuzz ≥ 90 |
   | degree | 1 | ≥ requested level |
   | certification | 1 | present |
   | asset class | 1 | any overlap |

   `rule = Σ w_i m_i / Σ w_i` over criteria the user actually set.
3. **Keyword score**: FTS5 BM25 on `free_text` (plus required and preferred
   skills), min-max scaled over the pruned set.
4. **Embedding score**: cosine(query embedding of `free_text`, candidate
   vectors); candidate score = max over its summary and role vectors; min-max
   scaled over the pruned set.
5. **Final** `= 0.5·rule + 0.2·keyword + 0.3·embedding`. When `free_text` is
   empty, keyword and embedding are dropped and rule is renormalized.
6. **Optional rerank** (toggle, off by default): top 10 to Sonnet with the
   requisition and each candidate's `search_summary` + derived facets → new
   order plus a one-line reason each.

### 8.3 Location scoring

- `location_scope = current`: match on `current_country`/`current_region` only.
- `location_scope = worked`: match on `countries_worked`.
- `location_scope = either` (the default when the user picks a place without
  saying which): current match scores 1.0; worked-only match scores 0.5;
  region-level match where country was requested scores half of either. The
  chip says which one matched ("Currently in UK" vs "Worked in UK, 4y").

### 8.4 Explanation chips

Each result card shows one chip per criterion the user set and per keyword:
green = matched (with the matching value, e.g. "Healthcare ✓", "Python ✓",
"Worked in GB 4.0y"), orange = not matched. Hovering a sector chip shows the
evidence quote.

## 9. Term matching layers

1. **Synonym dictionary + rapidfuzz** (`taxonomy/skills.yaml`): `variant →
   canonical, category`. Applied in `derive/skills.py` and to requisition
   skills. First pass.
2. **spaCy EntityRuler** (`en_core_web_sm` + phrase patterns compiled from the
   taxonomy: skills, tools, firms, sectors, certifications, locations, phone
   prefixes). Runs on raw text at ingest. Outputs: spans for highlighting on
   the Candidate page; deterministic skill set and country guess for
   cross-checks. LLM wins for display; union feeds `skills_normalized`;
   disagreements counted into `quality`. Second pass.
3. **Knowledge graph** (networkx, built from the taxonomy YAML): edges
   `synonym→canonical`, `skill→category`, `firm→firm_type`, `firm→sector`,
   `approach→related_term`. Used at query time to expand `free_text` and
   skills (search "systematic" also hits "stat arb", "factor", "quant").
   Third pass, if time allows.
4. **Small classifier** (scalability section only): TF-IDF + logistic
   regression trained on LLM labels for `investment_approach` and sectors, with
   a confusion matrix against the LLM. Shown as a cheap pre-classifier at
   volume, not used in the live ranking.

## 10. Evaluation

Judge model is always a different, stronger model than the task model (Sonnet
task → Opus judge).

### 10.1 Parsing eval (`evals/parsing_cases.jsonl`, one line per resume)

Judge input: repaired resume text + `extracted` JSON. Judge output per
top-level field (name, contact, education, experience, skills, certifications,
languages, investment_approach, search_summary) and per experience entry:
`correct | partial | wrong | hallucinated | missing` with a one-line reason.
Metrics: per-field accuracy (correct / total), hallucination count, missing
count. Three resumes (Chen Li, Zara Al-Rashid, Omar El-Hassan: clean DOCX,
table-heavy DOCX, quirky PDF) are hand-graded by the author; judge–human
agreement is reported and the judge prompt is revised until agreement ≥ 90% on
those three.

### 10.2 Ranking eval (`evals/requisitions.jsonl`)

Six requisitions drafted by the LLM from the taxonomy, edited by the author, to
cover: US fundamental healthcare, APAC systematic equities, Europe credit,
India sell-side healthcare (markets_covered), junior quant developer, senior
generalist L/S. For each requisition × 10 candidates, the judge grades
relevance 0/1/2 from the requisition and the candidate's `derived` + summary.
Metrics: nDCG@5, Precision@3, per requisition and mean. Also reported: the same
metrics for rule-only, keyword-only, embedding-only, so the weights in 8.2 can
be tuned against the data rather than guessed.

Runner: `python -m resume_search.eval run --suite parsing|ranking`, pure
scoring functions, results to `evals/results/{suite}_{timestamp}.json`,
latest shown on the Evals page.

## 11. Streamlit pages

| Page | Content |
|---|---|
| Search | filter sidebar, JD paste box with "Parse requisition" button, must/preferred toggles, result cards with chips, score breakdown expander, rerank toggle, advanced expander for weights |
| Candidate | header with derived facets, timeline of roles with relevance and type badges, evidence quotes, skills (LLM vs ruler), raw text toggle with highlighted spans |
| Insights | distributions: region and country, approach, sectors, seniority band, top skills, firm types, degree levels; cross-tab approach × region |
| Ingest | uploader, parse_jobs table, throughput and cost, re-run failed, force re-parse |
| Evals | latest parsing and ranking results, judge–human agreement, disagreement drill-down |

Caching: `st.cache_resource` for the DB connection, embedding model, spaCy
pipeline, loaded taxonomy; `st.cache_data` for query results keyed on the
Requisition hash; both cleared after ingest.

## 12. PII

`email`, `phone`, `location_text` live only in their `candidates` columns. The
FTS body is the resume text with the contact block removed (first 8 lines that
match email/phone/URL patterns, plus any later line matching them). Embedding
inputs are `search_summary` and experience bullets, neither of which may
contain contact data (validator in 4.1). A future redaction switch masks the
three columns at read time.

## 13. Open and deferred

- Deployment target and CI (GitHub Actions for eval + ingest): decided at
  submission. Local-first until then.
- Knowledge graph and small classifier: only after Search, Ingest and Evals
  work end to end.
- Visual polish of all pages: last.
