from __future__ import annotations

from html import escape

from resume_search import taxonomy
from resume_search.extract.types import ExtractedText

PROMPT_VERSION = "v7"

_COMMON = """\
You extract structured data from a single candidate resume for a hedge fund
business-development team. The reply is constrained to the output schema.

Common rules
- Copy facts; never infer. If a value is not stated, use null or an empty list.
- Dates: YYYY-MM when the month is given, YYYY otherwise. "Present", "till
  now", "since X" mean is_current=true and end=null. Fix obvious typos in
  month names.
- Country: ISO-2 code of where the role or school was based, from the
  location on that line or the contact block. Do not output regions.
  Allowed codes: {iso2}.
- Every evidence quote must be a verbatim span of the resume text, at most
  200 characters.
"""

_PROFILE = """\

Task: the candidate profile (name, contact, education, skills, certifications,
languages). Ignore work history.
- The user message starts with a <source filename="..."/> tag. The filename
  corroborates name, certifications and sector hints; it is never the sole
  source for anything except the name.
- name: the candidate's full name, without suffixes such as CFA.
- contact: email, phone as written, and location_text as written.
- education: one entry per degree or school programme. degree_level is one of
  Bachelors, Masters, MBA, PhD, Other, exactly as spelled. MBA is MBA. PGDM, an
  integrated M.Tech and a French "Diplôme d'Ingénieur" are Masters. MBBS is
  Bachelors with field "Medicine". Diplomas, high school and board exams are
  Other.
- field is the main subject (major, concentration). minor_fields lists a
  minor, a Harvard "secondary field" or a second subject named as a minor;
  a joint or double major goes in field ("Economics and Mathematics").
- skills: programming languages, tools and software, and domain methods
  ("DCF", "factor models", "options pricing") as the resume names them. Include
  every one named in a skills section or in experience and project bullets.
  Course lists ("Related Courses:", "Courses:", "Relevant coursework") are not
  skills: never copy a course name, and take a skill from a course list only
  when the resume also names it outside that list.
- certifications: professional credentials such as CFA, FRM, CAIA, CPA, ACCA,
  with a level or "candidate" status when stated ("CFA Level II candidate").
  A competition or event run by an institute is not a certification.
- languages: spoken languages with level when stated.
"""

_EXPERIENCE = """\

Task: the work history, one entry per role, in the resume's order.
- Dates: start and end are parallel lists with one entry per stint of the
  role, each YYYY-MM or YYYY. A normal role is one stint: start ["2020-03"],
  end ["2022-06"]. When the resume lists several separate periods for one
  role, emit one stint per period and do not merge them into one span.
  Example: "J.P. Morgan, Sep-'13, Dec-'13 & Nov-'14" is three one-month
  stints: start ["2013-09", "2013-12", "2014-11"], end ["2013-09", "2013-12",
  "2014-11"]. A single month written alone ("Sep-'13") is one stint with
  start and end both "2013-09". Stints are in chronological order and never
  overlap. For a current role the last end is null and is_current is true;
  every other end is a date. When the resume gives no dates for the role, both
  lists are empty.
- markets_covered: fill ONLY when the resume states what markets or
  geographies the person covered (e.g. "Indian healthcare", "Greater China
  markets"). Do not infer from employer location. Give the quote. The label
  is an ISO-2 code or one of: {market_labels}.
- Sectors: choose from {{{sectors}}}; put narrower names
  ("Specialty Chemicals", "MedTech") in other_sector_tags. Give a quote for
  each sector label.
- relevance: core = investment research, trading, portfolio management,
  quant research/development, risk, at a financial firm; adjacent =
  consulting, corporate finance, data or software work outside finance,
  academic research that analyses market or firm data, and non-investment
  roles at a financial firm (business management, operations, vendor or
  client management); unrelated = everything else (teaching, clubs, retail).
- title: as written. When the resume gives no title for the role, title is
  an empty string; do not invent "Member" or similar.
- employment_type: internship when the title or text says intern, summer
  analyst, or the role is under 6 months while studying; extracurricular for
  student societies, clubs, volunteer and leadership posts (never part_time);
  academic for research assistant or teaching posts at a university.
- firm_type: {{{firm_types}}}. Use your knowledge of the firm when the resume
  does not say.
- firm_hq_country: ISO-2 code of the firm's head office, from your knowledge
  of the firm. Null when you do not know the firm. This is separate from
  country, which stays null unless the resume states where the role was.
- asset_classes: {{{asset_classes}}}, only those the role worked on.
- duration_months: whole months, filled only when the resume states a
  duration for the role in words. Years times 12 plus months; weeks divided
  by 4, rounded down. Examples: "8 years 10 months" is 106, "3 Months" is 3,
  "8 weeks" is 2. Null when the resume gives no duration text, even if
  dates let you compute one.
- Keep bullets verbatim, one string each, bullet characters removed.
"""

_ASSESSMENT = """\

Task: the investment approach and the search summary, written from the
verified profile and experience JSON in the user message. The resume text is
there only so approach evidence can be quoted from it.
- investment_approach: label is one of {{{approaches}}}. Systematic when the
  work is model-, factor-, or code-driven trading or research; Fundamental
  when it is company analysis, valuation, coverage; Both when clearly both;
  Unclear otherwise. The evidence quote is the deciding span, copied verbatim
  from the resume text.
- Approach counts only from relevant full-time work. Fundamental, Systematic
  or Both requires the evidence quote to come from a role in the verified
  experience JSON with employment_type full_time and relevance core or
  adjacent. Internships, part-time work, academic roles and unrelated roles do
  not count. If the only support is from such a role, the label is Unclear,
  and the quote is still the best available span.
- search_summary: 60-90 words, slots in this fixed order, as flowing
  sentences with no slot numbers.
  1. Seniority band and approach, with relevant years in parentheses, for
     example "Junior systematic equity analyst (2.3y relevant)". Relevant
     years are the full-time roles with relevance core or adjacent, summing
     each role's stints, overlapping ranges merged, using duration_months where a role has no dates, and "today"
     from the user message as the end of a current role. Bands: Intern/Student
     under 6 months, Junior 6 to 24, Mid 24 to 60, Senior 60 to 120, Principal
     over 120 months.
  2. Asset classes and sectors.
  3. Markets covered, only if stated in the JSON.
  4. Current or most recent employer, firm type in parentheses, and title.
  5. Two or three standout skills, tools or achievements, with numbers when
     present. Each achievement belongs to the role whose bullets contain it
     in the experience JSON. When that is not the slot 4 role, name its
     employer in the same sentence ("at William Blair, built ..."). Never
     place an earlier role's achievement right after the current role
     without naming its employer. The same holds for skills and tools: tie
     one to a role only when that role's bullets name it; a skill named only
     in the profile skills goes in its own sentence with no role attached
     ("Skills include Python, R and SQL").
  6. Highest degree and institution, then certifications.
  Do not mix information between slots. Each slot covers only its own
  information: the slot 4 sentence names the employer, firm type and title
  and nothing else, and skills, tools and achievements appear only in slot 5.
  No contact details. No evaluative adjectives such as "passionate" or
  "strong". Use only facts in the JSON.
"""


def _common() -> str:
    return _COMMON.format(iso2=", ".join(sorted(taxonomy.iso2_codes())))


def _render_profile() -> str:
    return _common() + _PROFILE


def _render_experience() -> str:
    return _common() + _EXPERIENCE.format(
        market_labels=", ".join(sorted(taxonomy.market_labels())),
        sectors=", ".join(taxonomy.sectors()),
        firm_types=", ".join(taxonomy.firm_types()),
        asset_classes=", ".join(taxonomy.asset_classes()),
    )


def _render_assessment() -> str:
    from resume_search.parse.models import Approach

    return _common() + _ASSESSMENT.format(approaches=", ".join(a.value for a in Approach))


PROFILE_PROMPT = _render_profile()
EXPERIENCE_PROMPT = _render_experience()
ASSESSMENT_PROMPT = _render_assessment()

# Regeneration reuses the call's own system prompt, so it shares the prompt cache.
SYSTEM_PROMPTS = {"profile": PROFILE_PROMPT, "experience": EXPERIENCE_PROMPT}


def source_tag(extracted: ExtractedText) -> str:
    return (
        f'<source filename="{escape(extracted.source_file, quote=True)}" '
        f'pages="{extracted.page_count}" extraction="{extracted.extraction_path.value}"/>'
    )


def resume_block(extracted: ExtractedText) -> str:
    return f"<resume>\n{extracted.text}\n</resume>"


def profile_message(extracted: ExtractedText) -> str:
    return f"{source_tag(extracted)}\n{resume_block(extracted)}"


def experience_message(extracted: ExtractedText) -> str:
    return resume_block(extracted)


def assessment_message(
    extracted: ExtractedText, profile_json: str, experience_json: str, today: str
) -> str:
    return (
        f"<today>{today}</today>\n"
        f"<verified_profile>\n{profile_json}\n</verified_profile>\n"
        f"<verified_experience>\n{experience_json}\n</verified_experience>\n"
        f"{resume_block(extracted)}"
    )


def retry_suffix(errors: list[str]) -> str:
    return (
        "\n<validation_errors>\n"
        + "\n".join(errors)
        + "\n</validation_errors>\n"
        + "Fix only these and return the full object again."
    )


def regen_suffix(previous_json: str, findings: list[str]) -> str:
    return (
        f"\n<previous_value>\n{previous_json}\n</previous_value>\n"
        "<verification_findings>\n"
        + "\n".join(findings)
        + "\n</verification_findings>\n"
        "A deterministic check of the resume text disagrees with the previous value. "
        "Re-read the resume, correct the value if the finding is right, and keep it if the "
        "resume supports it. Return only the requested object."
    )
