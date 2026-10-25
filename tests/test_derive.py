from datetime import date

import pytest
from builders import TODAY_META, resume, role

from resume_search.derive import derive
from resume_search.derive.experience import seniority_band
from resume_search.taxonomy import load_taxonomy_data

TAX = load_taxonomy_data()


def d(parsed, **kw):
    return derive(parsed, TODAY_META, TAX, **kw)


def test_overlapping_ranges_merge():
    r = resume([
        role(start="2020-01", end="2021-01"),
        role(start="2020-07", end="2021-07"),
        role(start="2023-01", end="2023-07"),
    ])
    assert d(r).months_full_time_relevant == 18 + 6


def test_year_only_dates_use_june():
    r = resume([role(start="2016", end="2019")])
    assert d(r).months_full_time_relevant == 36
    r = resume([role(start="2016", end="2019-12")])
    assert d(r).months_full_time_relevant == 42


def test_current_role_ends_today():
    r = resume([role(start="2025-10", end=None, is_current=True)])
    assert d(r, today=date(2026, 10, 25)).months_full_time_relevant == 12


def test_current_role_uses_parse_date_from_meta():
    r = resume([role(start="2025-10", end=None, is_current=True)])
    assert d(r).months_full_time_relevant == 12


def test_duration_months_when_no_dates():
    r = resume([
        role(start=None, end=None, duration_months=106),
        role(start=None, end=None, duration_months=10, relevance="adjacent"),
        role(start=None, end=None, duration_months=7, relevance="unrelated"),
    ])
    out = d(r)
    assert out.months_full_time_relevant == 116
    assert out.months_full_time_core == 106


def test_internship_months_stay_separate():
    r = resume([
        role(start="2021-06", end="2023-06"),
        role(start="2019-12", end="2021-05", employment_type="internship"),
        role(start="2018-01", end="2019-01", employment_type="part_time"),
        role(start="2017-01", end="2018-01", employment_type="academic", relevance="adjacent"),
    ])
    out = d(r)
    assert out.months_full_time_relevant == 24
    assert out.months_internship == 17


@pytest.mark.parametrize(
    ("months", "band"),
    [(0, "Intern/Student"), (5, "Intern/Student"), (6, "Junior"), (24, "Junior"),
     (25, "Mid"), (60, "Mid"), (61, "Senior"), (120, "Senior"), (121, "Principal")],
)
def test_seniority_band_boundaries(months, band):
    assert seniority_band(months) == band


def test_current_country_prefers_current_role_with_latest_start():
    r = resume([
        role(country="GB", start="2024-01", end=None, is_current=True),
        role(country="US", start="2022-01", end=None, is_current=True),
        role(country="FR", start="2025-06", end="2025-09"),
    ])
    out = d(r)
    assert (out.current_country, out.current_region) == ("GB", "Europe")


def test_current_country_falls_back_to_contact_when_roles_have_none():
    r = resume(
        [role(country=None)],
        contact={"email": None, "phone": None, "location_text": "Mumbai, India"},
    )
    assert d(r).current_country == "IN"


def test_countries_worked_counts_full_time_only():
    r = resume([
        role(country="GB", start="2021-01", end="2022-01"),
        role(country="US", start="2019-01", end="2020-01"),
        role(country="US", start="2018-01", end="2018-07", employment_type="internship"),
    ])
    out = d(r)
    assert [(c.country, c.months) for c in out.countries_worked] == [("GB", 12), ("US", 12)]
    assert out.regions_worked == ["Europe", "US"]


def test_markets_covered_expands_labels():
    r = resume([role(markets_covered=[{"label": "Greater China", "evidence": {"quote": "q"}}])])
    out = d(r)
    assert {"Greater China", "CN", "HK", "TW", "APAC"} <= set(out.markets_covered)


def test_tag_promotes_to_core_sector():
    r = resume([
        role(
            sectors=[{"label": "Technology", "evidence": {"quote": "software"}}],
            other_sector_tags=["Biotech", "specialty chemicals", "Biotech"],
        )
    ])
    out = d(r)
    by_label = {s.label: s.evidence for s in out.sectors}
    assert by_label["Technology"] == "software"
    assert by_label["Healthcare"] == "biotech"
    assert by_label["Industrials"] == "specialty chemicals"
    assert out.other_sector_tags == ["biotech", "specialty chemicals"]


def test_skills_normalized_synonyms_fuzzy_and_dedupe():
    r = resume(skills={
        "programming": ["Python", "py", "C++"],
        "tools": ["Bloomberg Terminal", "Bloomberg", "Bloomberg Terminall", "Obscure Tool"],
        "domain": [],
    })
    assert d(r).skills_normalized == ["bloomberg", "cpp", "obscure tool", "python"]


@pytest.mark.parametrize(
    ("certs", "has", "level"),
    [
        (["CFA"], True, None),
        (["CFA Charterholder"], True, None),
        (["CFA Level II candidate"], False, "II"),
        (["Chartered Financial Analyst Level 3 (CFA, USA)"], False, "III"),
        (["FRM", "FINRA Series 7"], False, None),
    ],
)
def test_cfa(certs, has, level):
    out = d(resume(certifications=certs))
    assert (out.has_cfa, out.cfa_level) == (has, level)


def test_highest_degree_order():
    edu = lambda lv: {"institution": "U", "degree": None, "degree_level": lv, "field": "F",
                      "start": None, "end": None, "country": None}
    out = d(resume(education=[edu("Masters"), edu("MBA"), edu("Bachelors"), edu("Other")]))
    assert out.highest_degree == "MBA"
    assert d(resume(education=[edu("Other")])).highest_degree == "Other"
    assert d(resume(education=[])).highest_degree is None


def test_compound_skill_splits_into_both_tokens():
    from resume_search.derive.skills import split_compound
    from resume_search.taxonomy import load_taxonomy_data

    tax = load_taxonomy_data()
    assert split_compound("C# / .NET", tax) == ["C#", ".NET"]
    assert split_compound("long/short", tax) == ["long/short"]


def test_stints_across_roles_still_merge_overlaps():
    r = resume([
        role(start=["2013-09", "2014-01"], end=["2013-12", "2014-06"]),
        role(start="2013-11", end="2014-03"),
    ])
    assert d(r).months_full_time_relevant == 9


def test_three_one_month_stints_count_three_months():
    r = resume([role(start=["2013-09", "2013-12", "2014-11"], end=["2013-09", "2013-12", "2014-11"])])
    assert d(r).months_full_time_relevant == 3
