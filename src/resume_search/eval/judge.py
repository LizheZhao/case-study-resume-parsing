"""Opus judges for the parsing and ranking evals (design section 10). The task model is Sonnet."""

from __future__ import annotations

import json
from collections.abc import Sequence

from resume_search.eval.models import (
    EvalRequisition,
    ParsingGrades,
    ParsingJudgement,
    RankingGrades,
    RankingJudgement,
    expected_fields,
)
from resume_search.parse.client import OPUS, Attempt
from resume_search.parse.models import ParsedResume
from resume_search.search.llm import structured_call

PARSING_PROMPT_VERSION = "pj3"
RANKING_PROMPT_VERSION = "rj1"
JUDGE_MODEL = OPUS
JUDGE_MAX_TOKENS = 12000

# USD per million tokens on Opus
PRICE_IN, PRICE_OUT, PRICE_CACHE_READ, PRICE_CACHE_WRITE = 4.0, 20.0, 0.40, 5.00


def judge_cost(attempts: Sequence[Attempt]) -> float:
    return sum(
        a.input_tokens * PRICE_IN
        + a.output_tokens * PRICE_OUT
        + a.cache_read_tokens * PRICE_CACHE_READ
        + a.cache_creation_tokens * PRICE_CACHE_WRITE
        for a in attempts
    ) / 1e6


def token_totals(attempts: Sequence[Attempt]) -> dict[str, int]:
    return {
        "input": sum(a.input_tokens for a in attempts),
        "output": sum(a.output_tokens for a in attempts),
        "cache_read": sum(a.cache_read_tokens for a in attempts),
        "cache_write": sum(a.cache_creation_tokens for a in attempts),
        "calls": len(attempts),
    }


_PARSING_SYSTEM = """\
You audit a resume parser. You get the resume text and the JSON the parser extracted from it.
Grade each field of the JSON by comparing it to the resume text. The reply is constrained to
the output schema.

Grades
- correct: the JSON value is fully supported by the text and nothing material is left out.
- partial: mostly right, with one wrong or unsupported detail (a wrong date, a wrong title, a
  list with one extra or one missing item, a misclassified type). Name the detail in reason.
- wrong: the JSON contradicts the text, or the value is mostly unsupported.
- hallucinated: the JSON holds a value the text does not support at all, where the text has
  nothing on the subject.
- missing: the text has the information and the JSON leaves it out or null.
When a field has both an unsupported item and an omission, pick the worse grade and name both.
A field the text is silent on, with an empty or null value in the JSON, is correct.

Fields to grade, each exactly once, with the field name spelled as given
- name, contact (email, phone, location_text), education (every entry, dates, degree, level,
  field), experience (the list as a whole: every role present, none invented, none merged or
  split wrongly), skills (programming, tools, domain), certifications, languages,
  investment_approach, search_summary.
- experience[i] for each role in order, i from 0. Judge employer, title, start and end dates,
  is_current, employment_type, relevance and country together. A role whose dates are wrong
  is not correct.
- Your list of field names is given in the user message. Use exactly those.

Checks you must do
- Every date: compare each start and end in education and experience to the text, month and
  year. In experience, start and end are parallel lists with one entry per stint of the role
  (a role listed as "Sep-'13, Dec-'13 & Nov-'14" has three one-month stints). Check that every
  stint is present, none is merged into a span, and an empty pair of lists is right only when
  the text gives the role no dates. Check is_current against "present", "till now" or "since".
- Every evidence quote: investment_approach.evidence.quote and every sectors and
  markets_covered evidence quote must appear in the text and must support the label it is
  attached to. A quote that does not support its label makes that role or the approach
  partial or wrong.
- search_summary: every factual claim (employer, title, tools, numbers, degrees) must be in
  the text or in the JSON. A claim in neither is unsupported. Flag a summary that links a
  tool or result to a role where the text does not. The opening sentence carries a seniority
  band (Intern/Student, Junior, Mid, Senior, Principal) and an "N.Ny relevant" figure that
  code computed from the JSON dates and roles, and they are not in the text by design. Do not
  mark the summary down for either one being absent from the text. Do mark it down when the
  figure rests on a date or role you found wrong, and say which.
- skills: items must be skills the text lists or clearly uses. Course names, awards and
  interests are not skills.
- A field the text states ambiguously or inconsistently is graded on the most reasonable
  reading. Say so in reason.

reason is one specific sentence. Name the value you checked and what the text says. For a
correct grade, say what you verified. Do not use facts that are in neither input.
"""


def _parsing_user(text: str, parsed: ParsedResume, fields: list[str]) -> str:
    return (
        f"<resume_text>\n{text}\n</resume_text>\n"
        f"<parsed_json>\n{parsed.model_dump_json(indent=1)}\n</parsed_json>\n"
        f"<fields_to_grade>\n{chr(10).join(fields)}\n</fields_to_grade>"
    )


def judge_parsing(
    text: str,
    parsed_record: ParsedResume,
    client,
    candidate_id: str = "",
    attempts: list[Attempt] | None = None,
) -> ParsingJudgement:
    fields = expected_fields(len(parsed_record.experience))

    def validate(out: ParsingGrades) -> list[str]:
        got = [r.field for r in out.grades]
        errors = [f"missing field {f}" for f in fields if f not in got]
        errors += [f"unknown field {f}" for f in dict.fromkeys(got) if f not in fields]
        errors += [f"duplicate field {f}" for f in dict.fromkeys(got) if got.count(f) > 1]
        return errors

    out = structured_call(
        client, _PARSING_SYSTEM, _parsing_user(text, parsed_record, fields), ParsingGrades,
        validate=validate, attempts=attempts, call="judge_parsing", model=JUDGE_MODEL,
        max_tokens=JUDGE_MAX_TOKENS,
    )
    order = {f: i for i, f in enumerate(fields)}
    return ParsingJudgement(
        candidate_id=candidate_id, grades=sorted(out.grades, key=lambda r: order[r.field])
    )


_RANKING_SYSTEM = """\
You judge how well each shortlisted hedge fund candidate fits a search requisition. The reply
is constrained to the output schema.

Grades
- 2: strong fit. Meets the must criteria and most of the preferred ones, with experience that
  matches the role.
- 1: partially meets. Misses a preferred criterion or two, or is a near fit in seniority,
  sector or approach, and the gap could be closed in an interview.
- 0: not relevant. Misses a must criterion, or the experience does not match the role.

Rules
- Must criteria are hard requirements. Preferred criteria and the free text are soft.
- Judge the candidate against the requisition on its own. Do not compare candidates to each
  other, and do not try to spread the grades. All candidates may get 0, or all 2.
- Use only the facts in the candidate block. Do not infer facts about a firm or person.
- Return every candidate exactly once. Use only the candidate_id values given.
- reason is one sentence naming the facts that decided the grade.
"""


def _facets(c: dict) -> str:
    skip = {"candidate_id", "name", "search_summary"}
    facets = {k: v for k, v in c.items() if k not in skip and v not in (None, [], "")}
    return json.dumps(facets, ensure_ascii=False)


def ranking_candidate_payload(c: dict) -> str:
    return (
        f'<candidate id="{c["candidate_id"]}" name="{c["name"]}">\n'
        f"{c['search_summary']}\nFacets: {_facets(c)}\n</candidate>"
    )


def judge_ranking(
    requisition: EvalRequisition,
    candidates: list[dict],
    client,
    attempts: list[Attempt] | None = None,
) -> RankingJudgement:
    """One call per requisition covering every candidate in the pool."""
    user = (
        f"<requisition title=\"{requisition.title}\">\n"
        f"{requisition.requisition.model_dump_json(exclude_none=True, indent=1)}\n</requisition>\n"
        + "\n".join(ranking_candidate_payload(c) for c in candidates)
    )
    want = [c["candidate_id"] for c in candidates]

    def validate(out: RankingGrades) -> list[str]:
        got = [r.candidate_id for r in out.grades]
        errors = [f"unknown candidate_id {i}" for i in dict.fromkeys(got) if i not in want]
        errors += [f"missing candidate_id {i}" for i in want if i not in got]
        errors += [f"duplicate candidate_id {i}" for i in dict.fromkeys(got) if got.count(i) > 1]
        return errors

    out = structured_call(
        client, _RANKING_SYSTEM, user, RankingGrades, validate=validate, attempts=attempts,
        call="judge_ranking", model=JUDGE_MODEL, max_tokens=JUDGE_MAX_TOKENS,
    )
    return RankingJudgement(requisition_id=requisition.id, grades=out.grades)
