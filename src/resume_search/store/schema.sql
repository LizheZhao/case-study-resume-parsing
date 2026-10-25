CREATE TABLE IF NOT EXISTS candidates (
  candidate_id TEXT PRIMARY KEY, name TEXT NOT NULL,
  email TEXT, phone TEXT, location_text TEXT,
  current_country TEXT, current_region TEXT,
  months_full_time_relevant INTEGER, months_full_time_core INTEGER, months_internship INTEGER,
  seniority_band TEXT, investment_approach TEXT, approach_evidence TEXT,
  highest_degree TEXT, has_cfa INTEGER, cfa_level TEXT,
  search_summary TEXT NOT NULL,
  record_json TEXT NOT NULL,
  prompt_version TEXT, schema_version INTEGER, parse_status TEXT, parsed_at TEXT,
  file_sha256 TEXT UNIQUE, source_file TEXT
);
CREATE TABLE IF NOT EXISTS experience (
  candidate_id TEXT REFERENCES candidates, idx INTEGER,
  employer TEXT, title TEXT, country TEXT, start TEXT, end TEXT, is_current INTEGER,
  employment_type TEXT, relevance TEXT, firm_type TEXT, months INTEGER, stints_json TEXT,
  PRIMARY KEY (candidate_id, idx)
);
CREATE TABLE IF NOT EXISTS education (candidate_id TEXT, idx INTEGER, institution TEXT, degree TEXT,
  degree_level TEXT, field TEXT, start TEXT, end TEXT, country TEXT, minor_fields TEXT,
  PRIMARY KEY (candidate_id, idx));
CREATE TABLE IF NOT EXISTS candidate_sectors   (candidate_id TEXT, sector TEXT, evidence TEXT);
CREATE TABLE IF NOT EXISTS candidate_tags      (candidate_id TEXT, tag TEXT);
CREATE TABLE IF NOT EXISTS candidate_assets    (candidate_id TEXT, asset_class TEXT);
CREATE TABLE IF NOT EXISTS candidate_skills    (candidate_id TEXT, skill TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS candidate_countries (candidate_id TEXT, country TEXT, months INTEGER, kind TEXT);
CREATE TABLE IF NOT EXISTS candidate_certs     (candidate_id TEXT, cert TEXT);
CREATE TABLE IF NOT EXISTS candidate_languages (candidate_id TEXT, language TEXT, level TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS candidate_fts USING fts5(candidate_id UNINDEXED, body, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS embeddings (candidate_id TEXT, kind TEXT, idx INTEGER, model TEXT, dim INTEGER,
  vec BLOB, PRIMARY KEY (candidate_id, kind, idx));
CREATE TABLE IF NOT EXISTS parse_jobs (
  job_id INTEGER PRIMARY KEY, file_sha256 TEXT, source_file TEXT, prompt_version TEXT,
  status TEXT,
  attempts_json TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL,
  latency_ms INTEGER, error TEXT, queued_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS taxonomy_country (iso2 TEXT PRIMARY KEY, name TEXT, region TEXT);
CREATE TABLE IF NOT EXISTS taxonomy_sector (name TEXT PRIMARY KEY, parent TEXT);
CREATE TABLE IF NOT EXISTS taxonomy_skill_synonym (variant TEXT PRIMARY KEY, canonical TEXT, category TEXT);
CREATE TABLE IF NOT EXISTS ranking_weights (criterion TEXT PRIMARY KEY, weight REAL);
CREATE INDEX IF NOT EXISTS idx_cand_country ON candidates(current_country);
CREATE INDEX IF NOT EXISTS idx_cand_months ON candidates(months_full_time_relevant);
CREATE INDEX IF NOT EXISTS idx_exp_country ON experience(country);
CREATE INDEX IF NOT EXISTS idx_jobs_key ON parse_jobs(file_sha256, prompt_version);
