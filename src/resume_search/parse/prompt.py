from __future__ import annotations

import json
from html import escape

from resume_search import taxonomy
from resume_search.extract.types import ExtractedText

PROMPT_VERSION = "v1"

_RULES = """\
You extract structured data from a single candidate resume for a hedge fund
business-development team. Return only the JSON object matching the schema.

Rules
1. Copy facts; never infer. If a value is not stated, use null or an empty list.
2. Dates: YYYY-MM when the month is given, YYYY otherwise. "Present", "till
   now", "since X" mean is_current=true and end=null. Fix obvious typos in
   month names.
3. Country: ISO-2 code of where the role or school was based, from the
   location on that line or the contact block. Do not output regions.
   Allowed codes: {iso2}.
4. markets_covered: fill ONLY when the resume states what markets or
   geographies the person covered (e.g. "Indian healthcare", "Greater China
   markets"). Do not infer from employer location. Give the quote. The label
   is an ISO-2 code or one of: {market_labels}.
5. Sectors: choose from {{{sectors}}}; put narrower names
   ("Specialty Chemicals", "MedTech") in other_sector_tags. Give a quote for
   each sector label.
6. relevance: core = investment research, trading, portfolio management,
   quant research/development, risk, at a financial firm; adjacent =
   consulting, corporate finance, data or software work outside finance;
   unrelated = everything else (teaching, clubs, retail).
7. employment_type: internship when the title or text says intern, summer
   analyst, or the role is under 6 months while studying.
8. firm_type: {{{firm_types}}}. Use your knowledge of
   the firm when the resume does not say.
9. investment_approach: Systematic when the work is model-, factor-, or
   code-driven trading or research; Fundamental when it is company analysis,
   valuation, coverage; Both when clearly both; Unclear otherwise. Quote the
   deciding span. The label is one of {{{approaches}}}.
10. Every evidence quote must be a verbatim span of the resume text, at most
    200 characters.
11. search_summary: 60-90 words, slots in this order: (1) seniority band and
    approach with relevant years, (2) asset classes and sectors, (3) markets
    covered if stated, (4) most recent employer with firm type in
    parentheses and title, (5) two or three concrete skills, tools or
    achievements with numbers, (6) highest degree and institution,
    certifications. No contact details, no adjectives like "passionate".
12. Keep bullets verbatim, one string each, bullet characters removed.
"""


def _strip_titles(node):
    if isinstance(node, dict):
        return {k: _strip_titles(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_strip_titles(v) for v in node]
    return node


def _render() -> str:
    from resume_search.parse.models import Approach, ParsedResume

    # The API rejects this schema as a constrained-decoding grammar (too large), so the
    # model gets it as text and the client validates the reply with Pydantic.
    schema = json.dumps(_strip_titles(ParsedResume.model_json_schema()), separators=(",", ":"))
    rules = _RULES.format(
        iso2=", ".join(sorted(taxonomy.iso2_codes())),
        market_labels=", ".join(sorted(taxonomy.market_labels())),
        sectors=", ".join(taxonomy.sectors()),
        firm_types=", ".join(taxonomy.firm_types()),
        approaches=", ".join(a.value for a in Approach),
    )
    return (
        f"{rules}\nOutput\nReply with one JSON object that validates against this JSON Schema, "
        f"with no prose and no code fences. Include every property; use null or [] when "
        f"a value is absent.\n<schema>\n{schema}\n</schema>\n"
    )


SYSTEM_PROMPT = _render()


def user_message(extracted: ExtractedText) -> str:
    return (
        f'<source filename="{escape(extracted.source_file, quote=True)}" '
        f'pages="{extracted.page_count}" extraction="{extracted.extraction_path.value}"/>\n'
        f"<resume>\n{extracted.text}\n</resume>"
    )


def retry_suffix(errors: list[str]) -> str:
    return (
        "\n<validation_errors>\n"
        + "\n".join(errors)
        + "\n</validation_errors>\n"
        + "Fix only these and return the full JSON object again."
    )
