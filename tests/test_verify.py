from types import SimpleNamespace

import pytest

from resume_search.extract.types import (
    ExtractedText,
    ExtractionPath,
    SourceFormat,
    TextQuality,
)
from resume_search.parse.client import _State, _verify_and_repair
from resume_search.parse.models import CertificationsOnly, ExperienceList, Profile
from resume_search.parse.verify import (
    _compare_name,
    _extract_certs,
    _extract_name,
    _extract_phone_country,
    duration_months,
    grounded_dates,
    verify,
)


@pytest.mark.parametrize(
    ("text", "months"),
    [
        ("Analyst, Acme, Sept.2017 - Present", {"2017-09"}),
        ("Intern at Foo May'22", {"2022-05"}),
        ("Associate Oct-'20 to Jun 2021", {"2020-10", "2021-06"}),
        ("Joined on Mar 15, 2019", {"2019-03"}),
        ("Period 03/2020 - 2021-07", {"2020-03", "2021-07"}),
        ("Janvier 2018 puis mai 2019", {"2018-01", "2019-05"}),
    ],
)
def test_grounded_month_tokens(text, months):
    assert months <= grounded_dates(text)[0]


def test_year_only_token_grounds_year_but_not_month():
    months, years = grounded_dates("Associate, 2019 - 2021")
    assert {"2019", "2021"} <= years
    assert "2019-06" not in months


def test_prose_word_before_year_is_not_a_month():
    assert grounded_dates("Remarketing 2019")[0] == set()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("8 years 10 months", 106),
        ("3 Months", 3),
        ("8 weeks", 2),
        ("2 yrs", 24),
        ("no duration here", None),
    ],
)
def test_duration_regex(text, expected):
    assert duration_months(text) == expected


def test_certifications_found_in_filename_only():
    found = _extract_certs("Michael Rodriguez\nAnalyst", "Michael Rodriguez, CFA.docx")
    assert found.value == ["CFA"]


def test_cfa_institute_event_is_not_a_certification():
    found = _extract_certs("Finalist, CFA Institute Research Challenge", "x.docx")
    assert found.value == []


def test_name_from_filename_matches_llm_name():
    cands = _extract_name("RLTM profile\n", "Priya Nakamura_sellside_healthcare_RLTM.docx").value
    assert _compare_name("Priya Nakamura", cands)
    assert not _compare_name("Someone Else", cands)


def test_phone_prefix_maps_to_hong_kong():
    assert _extract_phone_country("Tel: +852 5123 4567", "x.docx").value == ["HK"]
    assert _extract_phone_country("Tel: 5123 4567", "x.docx").value == []


PROFILE = {
    "name": "Jane Doe",
    "contact": {"email": "jane@acme.com", "phone": None, "location_text": "London"},
    "education": [],
    "skills": {"programming": ["Python"], "tools": [], "domain": []},
    "certifications": [],
    "languages": [],
}


def _exp(**kw) -> dict:
    base = {
        "employer": "Acme Capital", "title": "Analyst", "location_text": None, "country": "GB",
        "start": "2020-03", "end": None, "is_current": True, "employment_type": "full_time",
        "relevance": "core", "firm_type": "Hedge fund", "duration_months": None, "bullets": [],
    }
    return base | kw


def test_verify_reports_ungrounded_date_and_missing_cert():
    text = "Jane Doe CFA\njane@acme.com\nAnalyst, Acme Capital, Mar 2020 - Present. Python"
    profile = Profile.model_validate(PROFILE)
    exps = ExperienceList.model_validate({"experience": [_exp(start="2018-07")]})
    report = verify(profile, exps, text, "Jane Doe.docx")
    fields = {d.field for d in report.disagreements}
    assert fields == {"experience[0].start", "certifications"}


def test_verify_passes_clean_record():
    text = "Jane Doe\njane@acme.com\nAnalyst, Acme Capital, Mar 2020 - Present. Python"
    profile = Profile.model_validate(PROFILE)
    exps = ExperienceList.model_validate({"experience": [_exp()]})
    report = verify(profile, exps, text, "Jane Doe.docx")
    assert report.disagreements == []
    assert report.checks_run > 0


class _FakeClient:
    """Stands in for anthropic.Anthropic; every parse call returns `answer` for the asked model."""

    def __init__(self, answer):
        usage = SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        self.calls = 0
        self.messages = SimpleNamespace(parse=lambda **kw: self._parse(usage, answer, **kw))

    def _parse(self, usage, answer, **kw):
        self.calls += 1
        return SimpleNamespace(usage=usage, parsed_output=answer(kw["output_format"]),
                               stop_reason="end_turn")


def _extracted(text, filename="Jane Doe.docx"):
    return ExtractedText(filename, "0" * 64, 1, SourceFormat.docx, 1, text,
                         TextQuality(0, 0, 0, 1000, 0), ExtractionPath.text)


def test_regeneration_that_fixes_the_field_replaces_it():
    text = "Jane Doe CFA\njane@acme.com\nAnalyst, Acme Capital, Mar 2020 - Present. Python"
    state = _State(Profile.model_validate(PROFILE),
                   ExperienceList.model_validate({"experience": [_exp()]}).experience)
    client = _FakeClient(lambda model: CertificationsOnly(certifications=["CFA Charterholder"]))
    fixed, summary, attempts = _verify_and_repair(client, _extracted(text), state)
    assert fixed.profile.certifications == ["CFA Charterholder"]
    assert (summary.disagreements, summary.regenerated, summary.unresolved) == (1, 1, [])
    assert [a.call for a in attempts] == ["regen:certifications"]


def test_regeneration_that_does_not_fix_keeps_llm_value_and_records_unresolved():
    text = "Jane Doe CFA\njane@acme.com\nAnalyst, Acme Capital, Mar 2020 - Present. Python"
    state = _State(Profile.model_validate(PROFILE),
                   ExperienceList.model_validate({"experience": [_exp()]}).experience)
    client = _FakeClient(lambda model: CertificationsOnly(certifications=[]))
    kept, summary, _ = _verify_and_repair(client, _extracted(text), state)
    assert kept.profile.certifications == []
    assert [u["field"] for u in summary.unresolved] == ["certifications"]
    assert client.calls == 1


def test_dropped_duration_only_role_is_flagged():
    text = (
        "Organization | Axis | Duration | 2 months\n"
        "Concurrent Project (8 weeks): Reliance Infrastructure Ltd.\n"
        "Responsibilities: tracked 38 companies within 3 months of joining"
    )
    profile = Profile.model_validate(PROFILE)
    one = _exp(employer="Axis", start=None, is_current=False, duration_months=2)
    two = _exp(employer="Reliance Infrastructure Ltd.", start=None, is_current=False,
               duration_months=2)
    missing = verify(profile, ExperienceList.model_validate({"experience": [one]}), text, "x.docx")
    assert "experience" in {d.field for d in missing.disagreements}
    both = verify(profile, ExperienceList.model_validate({"experience": [one, two]}), text, "x.docx")
    assert "experience" not in {d.field for d in both.disagreements}
