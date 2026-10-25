from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from pydantic import BaseModel

from resume_search.search.keyword import keyword_matches, keyword_query, keyword_scores
from resume_search.search.llm import structured_call
from resume_search.search.requisition import Requisition
from resume_search.search.rules import Chip, ChipStatus, load_candidates, prune_sql, rule_score
from resume_search.search.vectors import VectorIndex, default_embedder
from resume_search.search.weights import Weights


@dataclass(frozen=True)
class ScoreBreakdown:
    rule: float
    keyword: float
    embedding: float
    final: float
    rule_terms: list[tuple[str, float, float]]


@dataclass(frozen=True)
class SearchResult:
    candidate_id: str
    name: str
    search_summary: str
    seniority_band: str | None
    current_country: str | None
    current_region: str | None
    breakdown: ScoreBreakdown
    chips: list[Chip]
    keywords: list[str]


def _minmax(raw: dict[str, float], ids: list[str], missing: float) -> dict[str, float]:
    """Scale to 0..1 over the pruned set. When nothing separates the candidates, everyone with
    a signal gets 1.0 and everyone without gets 0.0."""
    vals = {i: raw.get(i, missing) for i in ids}
    lo, hi = min(vals.values(), default=0.0), max(vals.values(), default=0.0)
    if hi == lo:
        return {i: 1.0 if hi > 0 and i in raw else 0.0 for i in ids}
    return {i: (v - lo) / (hi - lo) for i, v in vals.items()}


def search(
    conn: sqlite3.Connection,
    req: Requisition,
    weights: Weights,
    embedder=None,
    index: VectorIndex | None = None,
    top_k: int = 20,
) -> list[SearchResult]:
    ids = [r[0] for r in conn.execute(*prune_sql(req))]
    cands = load_candidates(conn, ids)
    ids = [c.candidate_id for c in cands]
    rules = {c.candidate_id: rule_score(req, c, weights) for c in cands}
    has_rule = any(terms for _, terms, _ in rules.values())
    query = keyword_query(req)
    has_text = bool((req.free_text or "").strip())

    kw = emb = {i: 0.0 for i in ids}
    if has_text:
        kw = _minmax(keyword_scores(conn, ids, query), ids, 0.0)
        qvec = (embedder or default_embedder()).embed([req.free_text.strip()])[0]
        raw = (index or VectorIndex.load(conn)).scores(qvec)
        raw = {i: raw[i] for i in ids if i in raw}
        emb = _minmax(raw, ids, min(raw.values(), default=0.0))

    parts = []
    if has_rule:
        parts.append((weights.blend_rule, "rule"))
    if has_text:
        parts += [(weights.blend_keyword, "kw"), (weights.blend_embedding, "emb")]

    def final(cid: str) -> float:
        if not parts:
            return 1.0
        vals = {"rule": rules[cid][0], "kw": kw[cid], "emb": emb[cid]}
        return sum(w * vals[k] for w, k in parts) / sum(w for w, _ in parts)

    scored = sorted(((final(c.candidate_id), c) for c in cands), key=lambda t: (-t[0], t[1].name))
    top = scored[:top_k]
    hits = keyword_matches(conn, [c.candidate_id for _, c in top], query) if has_text else {}
    return [
        SearchResult(
            candidate_id=c.candidate_id, name=c.name, search_summary=c.search_summary,
            seniority_band=c.seniority_band, current_country=c.current_country,
            current_region=c.current_region,
            breakdown=ScoreBreakdown(
                rule=rules[c.candidate_id][0], keyword=kw[c.candidate_id],
                embedding=emb[c.candidate_id], final=f, rule_terms=rules[c.candidate_id][1],
            ),
            chips=rules[c.candidate_id][2],
            keywords=hits.get(c.candidate_id, []),
        )
        for f, c in top
    ]


class _Ranked(BaseModel):
    candidate_id: str
    reason: str


class _Reranking(BaseModel):
    order: list[_Ranked]


_RERANK_SYSTEM = """\
You re-rank shortlisted hedge fund candidates against a search requisition. The reply is
constrained to the output schema.
- Return every candidate exactly once, best fit first. Use only the candidate_id values given.
- reason is one sentence naming the facts from the summary or criteria that decided the
  position. Do not use facts that are not in the input.
- Must criteria are hard requirements. Preferred criteria and the free text are soft.
"""


def _chip_line(c: Chip) -> str:
    mark = "met" if c.status is ChipStatus.matched else "not met"
    return f"{c.label} ({mark}{': ' + c.detail if c.detail else ''})"


def rerank_llm(
    req: Requisition, results: list[SearchResult], client=None, attempts: list | None = None
) -> list[tuple[str, str]]:
    blocks = [
        f'<candidate id="{r.candidate_id}" band="{r.seniority_band}" '
        f'country="{r.current_country}">\n{r.search_summary}\n'
        f"Criteria: {'; '.join(_chip_line(c) for c in r.chips) or 'none'}\n</candidate>"
        for r in results
    ]
    user = (
        f"<requisition>\n{req.model_dump_json(exclude_none=True)}\n</requisition>\n"
        + "\n".join(blocks)
    )
    want = {r.candidate_id for r in results}

    def validate(out: _Reranking) -> list[str]:
        got = [x.candidate_id for x in out.order]
        errors = [f"unknown candidate_id {i}" for i in got if i not in want]
        errors += [f"missing candidate_id {i}" for i in want - set(got)]
        errors += [f"duplicate candidate_id {i}" for i in {i for i in got if got.count(i) > 1}]
        return errors

    out = structured_call(
        client, _RERANK_SYSTEM, user, _Reranking, validate=validate, attempts=attempts,
        call="rerank",
    )
    return [(x.candidate_id, x.reason) for x in out.order]
