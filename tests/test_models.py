import pytest
from pydantic import ValidationError

from resume_search.parse.models import ParsedResume, check_evidence

TEXT = (
    "Jane Doe\nInvestment Analyst, Acme Capital, London, 2020-03 to Present\n"
    "Covered European healthcare equities with a factor model."
)
SUMMARY = (
    "Junior systematic equity analyst (4.0y relevant). European healthcare equities. "
    "Most recently Investment Analyst at Acme Capital (hedge fund), London; built a factor "
    "model covering 40 stocks. Python, SQL. BSc Economics, University of London, CFA candidate."
)


def record(**exp_overrides) -> dict:
    exp = {
        "employer": "Acme Capital",
        "title": "Investment Analyst",
        "location_text": "London",
        "country": "GB",
        "start": "2020-03",
        "end": None,
        "is_current": True,
        "employment_type": "full_time",
        "relevance": "core",
        "firm_type": "Hedge fund",
        "asset_classes": ["Equities"],
        "sectors": [{"label": "Healthcare", "evidence": {"quote": "European healthcare equities"}}],
        "other_sector_tags": [],
        "markets_covered": [
            {"label": "Europe", "evidence": {"quote": "European healthcare equities"}}
        ],
        "bullets": ["Covered European healthcare equities with a factor model."],
    } | exp_overrides
    return {
        "name": "Jane Doe",
        "contact": {"email": None, "phone": None, "location_text": "London", "linkedin": None},
        "summary_text": None,
        "education": [
            {
                "institution": "University of London", "degree": "BSc",
                "degree_level": "Bachelors", "field": "Economics", "start": "2015",
                "end": "2018", "gpa": None, "honors": [], "country": "GB",
            }
        ],
        "experience": [exp],
        "skills": {"programming": ["Python"], "tools": [], "domain": []},
        "investment_approach": {"label": "Systematic", "evidence": {"quote": "a factor model"}},
        "search_summary": SUMMARY,
    }


def test_valid_record_validates_and_evidence_passes():
    parsed = ParsedResume.model_validate(record())
    assert check_evidence(parsed, TEXT) == []


def test_end_set_while_current_fails():
    with pytest.raises(ValidationError, match="end must be null"):
        ParsedResume.model_validate(record(end="2022-01"))


def test_finished_role_without_dates_is_allowed():
    parsed = ParsedResume.model_validate(
        record(start=None, end=None, is_current=False)
    )
    assert parsed.experience[0].end is None


def test_bad_country_fails():
    with pytest.raises(ValidationError, match="ISO-2"):
        ParsedResume.model_validate(record(country="XX"))


def test_start_after_end_fails():
    with pytest.raises(ValidationError, match="after end"):
        ParsedResume.model_validate(record(start="2022-05", end="2021-01", is_current=False))


def test_contact_leak_in_summary_fails():
    bad = record() | {"search_summary": SUMMARY + " Call 12345678."}
    with pytest.raises(ValidationError, match="7\\+ digits"):
        ParsedResume.model_validate(bad)


def test_check_evidence_flags_quote_missing_from_text():
    data = record(
        sectors=[{"label": "Healthcare", "evidence": {"quote": "Asian biotech coverage"}}]
    )
    errors = check_evidence(ParsedResume.model_validate(data), TEXT)
    assert len(errors) == 1
    assert "Asian biotech coverage" in errors[0]


def test_check_evidence_ignores_whitespace_differences():
    data = record()
    data["investment_approach"]["evidence"]["quote"] = "a   factor\nmodel"
    assert check_evidence(ParsedResume.model_validate(data), TEXT) == []
