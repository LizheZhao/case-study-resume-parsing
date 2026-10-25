from __future__ import annotations

import sqlite3

from resume_search import taxonomy as tx
from resume_search.search.requisition import Requisition

STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "our", "that", "the", "their", "this", "to", "we", "will", "with", "you", "your", "who", "they", "than", "then", "there", "these", "those", "been", "being", "into", "over", "per", "such", "can", "may", "must", "not", "also", "other", "about", "across", "within"]
)
MAX_TOKENS = 60


def keyword_query(req: Requisition) -> str:
    """Empty when there is no free text: keyword scoring is dropped, not run on skills alone."""
    text = (req.free_text or "").strip()
    return " ".join([text, *req.skills_required, *req.skills_preferred]) if text else ""


def query_tokens(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for tok in tx.tokenize(text):
        low = tok.lower()
        if len(low) > 1 and low not in STOPWORDS:
            seen.setdefault(low)
    return list(seen)[:MAX_TOKENS]


def _quote(token: str) -> str:
    return '"' + token.replace('"', '""') + '"'


def keyword_scores(
    conn: sqlite3.Connection, candidate_ids: list[str], query_text: str
) -> dict[str, float]:
    """BM25 relevance (higher is better) for the candidates that match at least one token.
    FTS5's bm25() is negative with lower meaning better, so it is negated here."""
    tokens = query_tokens(query_text)
    if not tokens or not candidate_ids:
        return {}
    ph = ",".join("?" * len(candidate_ids))
    rows = conn.execute(
        f"SELECT candidate_id, -bm25(candidate_fts) AS score FROM candidate_fts "
        f"WHERE candidate_fts MATCH ? AND candidate_id IN ({ph})",
        [" OR ".join(map(_quote, tokens)), *candidate_ids],
    ).fetchall()
    return {r["candidate_id"]: r["score"] for r in rows}


def keyword_matches(
    conn: sqlite3.Connection, candidate_ids: list[str], query_text: str
) -> dict[str, list[str]]:
    """Which query tokens occur in each candidate's text (stemmed like the index)."""
    out: dict[str, list[str]] = {cid: [] for cid in candidate_ids}
    if not candidate_ids:
        return out
    ph = ",".join("?" * len(candidate_ids))
    for tok in query_tokens(query_text):
        rows = conn.execute(
            f"SELECT candidate_id FROM candidate_fts WHERE candidate_fts MATCH ? "
            f"AND candidate_id IN ({ph})",
            [_quote(tok), *candidate_ids],
        )
        for r in rows:
            out[r["candidate_id"]].append(tok)
    return out
