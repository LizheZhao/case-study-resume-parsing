from datetime import date

import pytest
from pydantic import ValidationError

from resume_search.parse.models import (
    Assessment,
    ExperienceList,
    ParsedResume,
    Profile,
    check_evidence,
)

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
        "start": ["2020-03"],
        "end": [None],
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
        "duration_months": None,
        "bullets": ["Covered European healthcare equities with a factor model."],
    } | exp_overrides
    return {
        "name": "Jane Doe",
        "contact": {"email": None, "phone": None, "location_text": "London"},
        "education": [
            {
                "institution": "University of London", "degree": "BSc",
                "degree_level": "Bachelors", "field": "Economics", "start": "2015",
                "end": "2018", "country": "GB",
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
    with pytest.raises(ValidationError, match="last end must be null"):
        ParsedResume.model_validate(record(end=["2022-01"]))


def test_current_role_last_end_is_null():
    parsed = ParsedResume.model_validate(record())
    assert parsed.experience[0].end == [None]
    assert parsed.experience[0].last_end is None


def test_finished_role_with_null_end_fails():
    with pytest.raises(ValidationError, match=r"end\[0\] is null"):
        ParsedResume.model_validate(record(end=[None], is_current=False))


def test_finished_role_without_dates_is_allowed():
    parsed = ParsedResume.model_validate(
        record(start=[], end=[], is_current=False)
    )
    assert parsed.experience[0].end == []
    assert parsed.experience[0].first_start is None


def test_unequal_start_and_end_lists_fail():
    with pytest.raises(ValidationError, match="parallel lists"):
        ParsedResume.model_validate(record(start=["2020-03", "2021-01"], end=[None]))


STINTS = {"start": ["2013-09", "2013-12", "2014-11"], "end": ["2013-09", "2013-12", "2014-11"],
          "is_current": False}


def test_discrete_single_month_stints_validate():
    exp = ParsedResume.model_validate(record(**STINTS)).experience[0]
    assert (exp.first_start, exp.last_end) == ("2013-09", "2014-11")


def test_three_one_month_stints_sum_to_three_months():
    exp = ParsedResume.model_validate(record(**STINTS)).experience[0]
    assert exp.months(date(2026, 10, 25)) == 3


def test_overlapping_stints_in_one_role_fail():
    with pytest.raises(ValidationError, match="overlaps"):
        ParsedResume.model_validate(
            record(start=["2013-09", "2013-11"], end=["2013-12", "2014-02"], is_current=False)
        )


def test_out_of_order_stints_fail():
    with pytest.raises(ValidationError, match="overlaps or precedes"):
        ParsedResume.model_validate(
            record(start=["2014-11", "2013-09"], end=["2014-11", "2013-09"], is_current=False)
        )


def test_a_repeated_single_month_stint_overlaps():
    with pytest.raises(ValidationError, match="overlaps"):
        ParsedResume.model_validate(
            record(start=["2013-09", "2013-09"], end=["2013-09", "2013-09"], is_current=False)
        )


def test_bad_country_fails():
    with pytest.raises(ValidationError, match="ISO-2"):
        ParsedResume.model_validate(record(country="XX"))


def test_start_after_end_fails():
    with pytest.raises(ValidationError, match="after end"):
        ParsedResume.model_validate(record(start=["2022-05"], end=["2021-01"], is_current=False))


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


def test_record_splits_into_the_three_call_models():
    data = record(duration_months=14)
    parsed = ParsedResume.model_validate(data)
    assert parsed.experience[0].duration_months == 14
    profile = Profile.model_validate({k: data[k] for k in Profile.model_fields if k in data})
    exps = ExperienceList.model_validate({"experience": data["experience"]})
    assessment = Assessment.model_validate(
        {k: data[k] for k in ("investment_approach", "search_summary")}
    )
    assert parsed.model_dump() == {
        **profile.model_dump(), **exps.model_dump(), **assessment.model_dump()
    }


def test_dropped_display_fields_are_not_in_the_schema():
    fields = set(ParsedResume.model_fields)
    assert not fields & {"summary_text", "awards"}
    assert "linkedin" not in str(ParsedResume.model_json_schema())
    assert list(ParsedResume.model_fields)[:2] == ["name", "contact"]


def test_assessment_rejects_unknown_approach_label():
    bad = record()["investment_approach"] | {"label": "Quant"}
    with pytest.raises(ValidationError, match="investment_approach label"):
        Assessment.model_validate({"investment_approach": bad, "search_summary": SUMMARY})


def test_extracurricular_is_a_valid_employment_type():
    from resume_search.parse.models import EmploymentType

    assert EmploymentType("extracurricular") is EmploymentType.extracurricular
