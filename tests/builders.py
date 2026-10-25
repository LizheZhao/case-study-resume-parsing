from __future__ import annotations

from resume_search.derive import derive
from resume_search.derive.models import CandidateRecord
from resume_search.parse.models import ParsedResume
from resume_search.taxonomy import load_taxonomy_data

SUMMARY = (
    "Mid systematic equity analyst (4.0y relevant). Technology and healthcare equities. "
    "Currently Quantitative Analyst at Example Capital (hedge fund), New York; built a "
    "multi-factor model and a pricing library in C++ and Python. Bloomberg, SQL. "
    "MSc Financial Engineering, Example University. CFA charterholder."
)
TODAY_META = {"parsed_at": "2026-10-25T00:00:00+00:00"}


def _as_list(v):
    """A scalar date (or None, meaning no dates) becomes the one-stint list the model wants."""
    return v if isinstance(v, list) else ([] if v is None else [v])


def role(**kw) -> dict:
    """`start` and `end` may be scalars: one stint, or none when start is None."""
    start = _as_list(kw.pop("start", "2020-01"))
    end = kw.pop("end", "2021-01")
    end = end if isinstance(end, list) else ([end] if start else [])
    base = {
        "employer": "Acme", "title": "Analyst", "location_text": None, "country": "US",
        "start": start, "end": end, "is_current": False,
        "employment_type": "full_time", "relevance": "core", "firm_type": "Hedge fund",
        "asset_classes": [], "sectors": [], "other_sector_tags": [], "markets_covered": [],
        "duration_months": None, "bullets": ["Built models."],
    }
    return {**base, **kw}


def resume(experience: list[dict] | None = None, **kw) -> ParsedResume:
    data = {
        "name": "Test Person",
        "contact": {"email": "t@example.com", "phone": "+1 212 555 0100", "location_text": None},
        "education": [],
        "skills": {"programming": [], "tools": [], "domain": []},
        "certifications": [],
        "languages": [],
        "experience": experience if experience is not None else [role()],
        "investment_approach": {"label": "Systematic", "evidence": {"quote": "factor model"}},
        "search_summary": SUMMARY,
    }
    return ParsedResume.model_validate({**data, **kw})


def record(parsed: ParsedResume, candidate_id: str = "abc123def456", **meta) -> CandidateRecord:
    m = {
        "candidate_id": candidate_id, "source_file": "x.docx", "file_sha256": candidate_id * 4,
        "prompt_version": "v3", "schema_version": 2, "parse_status": "ok", **TODAY_META, **meta,
    }
    return CandidateRecord(meta=m, extracted=parsed, derived=derive(parsed, m, load_taxonomy_data()))
