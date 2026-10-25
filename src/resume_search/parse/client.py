from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import ClassVar

import anthropic
import pydantic

from resume_search.derive.skills import tidy_skills
from resume_search.extract.types import ExtractedText, ExtractionPath
from resume_search.parse import prompt, verify
from resume_search.parse.models import (
    Assessment,
    CertificationsOnly,
    ContactOnly,
    EducationItem,
    Experience,
    ExperienceItem,
    ExperienceList,
    NameOnly,
    ParsedResume,
    Profile,
    SkillsOnly,
    check_evidence,
    check_experience_evidence,
)
from resume_search.taxonomy import load_taxonomy_data

SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
# (model, append the previous attempt's validation errors)
LADDER = ((SONNET, False), (SONNET, True), (OPUS, True))
MAX_TOKENS = 16000
MAIN_CALLS = ("profile", "experience", "assessment")


@dataclass(frozen=True)
class Attempt:
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    error: str | None
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    call: str = ""


@dataclass(frozen=True)
class Verification:
    checks_run: int = 0
    disagreements: int = 0
    regenerated: int = 0
    unresolved: list[dict] = field(default_factory=list)
    details: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ParseOk:
    record: ParsedResume
    attempts: list[Attempt]
    verification: Verification = field(default_factory=Verification)

    @property
    def parse_status(self) -> str:
        main = [a for a in self.attempts if a.call in MAIN_CALLS]
        if any(a.model == OPUS for a in main):
            return "ok_retry_opus"
        return "ok" if len(main) == len(MAIN_CALLS) else "ok_retry_sonnet"


@dataclass(frozen=True)
class ParseFailed:
    attempts: list[Attempt]
    last_error: str

    parse_status: ClassVar[str] = "failed"


ParseResult = ParseOk | ParseFailed


def _validation_lines(exc: pydantic.ValidationError) -> list[str]:
    return [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]


def _content(extracted: ExtractedText, message: str) -> list[dict]:
    blocks: list[dict] = []
    if extracted.extraction_path is ExtractionPath.vision:
        blocks += [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.standard_b64encode(png).decode("ascii"),
                },
            }
            for png in extracted.page_images_png
        ]
    blocks.append({"type": "text", "text": message})
    return blocks


def default_client() -> anthropic.Anthropic:
    # Keys that are not workspace-scoped need the workspace id header; the SDK does not
    # read ANTHROPIC_WORKSPACE_ID itself.
    headers = {}
    if workspace := os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = workspace
    return anthropic.Anthropic(max_retries=4, default_headers=headers)


def _attempt(
    client, model: str, system: str, content: list[dict], output_format: type[pydantic.BaseModel]
):
    return client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
        output_format=output_format,
    )


def _run_ladder[T: pydantic.BaseModel](
    client,
    call: str,
    system: str,
    extracted: ExtractedText,
    message: str,
    output_format: type[T],
    validate: Callable[[T], list[str]],
    ladder=LADDER,
) -> tuple[T | None, list[Attempt]]:
    """One call on the Sonnet, Sonnet-with-errors, Opus-with-errors ladder."""
    attempts: list[Attempt] = []
    prev_errors: list[str] = []
    for model, with_errors in ladder:
        suffix = prompt.retry_suffix(prev_errors) if with_errors and prev_errors else ""
        started = time.monotonic()
        usage = None
        out: T | None = None
        errors: list[str] = []
        try:
            response = _attempt(
                client, model, system, _content(extracted, message + suffix), output_format
            )
            usage = response.usage
            out = response.parsed_output
            if out is None:
                errors = [f"no parsed output (stop_reason={response.stop_reason})"]
            else:
                errors = validate(out)
        except pydantic.ValidationError as exc:
            errors = _validation_lines(exc)
        except anthropic.RateLimitError as exc:
            errors = [f"RateLimitError: {exc}"]
        except anthropic.APIStatusError as exc:
            errors = [f"APIStatusError {exc.status_code}: {exc.message}"]
        except anthropic.APIConnectionError as exc:
            errors = [f"APIConnectionError: {exc}"]
        attempts.append(
            Attempt(
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
                error="; ".join(errors) if errors else None,
                cache_read_tokens=(usage.cache_read_input_tokens or 0) if usage else 0,
                cache_creation_tokens=(usage.cache_creation_input_tokens or 0) if usage else 0,
                call=call,
            )
        )
        if not errors and out is not None:
            return out, attempts
        # An API failure carries nothing the model could fix, so the next rung gets no suffix.
        prev_errors = [] if usage is None else errors
    return None, attempts


# Regeneration ---------------------------------------------------------------------------------

_REGEN_MODEL: dict[str, type[pydantic.BaseModel]] = {
    "name": NameOnly,
    "contact": ContactOnly,
    "certifications": CertificationsOnly,
    "skills": SkillsOnly,
    "experience_item": ExperienceItem,
    "education_item": EducationItem,
    "experience": ExperienceList,
}
_REGEN_SYSTEM = {
    "experience_item": "experience",
    "experience": "experience",
}


@dataclass
class _State:
    profile: Profile
    experience: list[Experience]


def _previous_json(state: _State, target: str, index: int | None) -> str:
    match target:
        case "name":
            return json.dumps({"name": state.profile.name})
        case "contact":
            return state.profile.model_dump_json(include={"contact"})
        case "certifications":
            return state.profile.model_dump_json(include={"certifications"})
        case "skills":
            return state.profile.model_dump_json(include={"skills"})
        case "experience_item":
            return state.experience[index].model_dump_json()
        case "education_item":
            return state.profile.education[index].model_dump_json()
        case _:
            return ExperienceList(experience=state.experience).model_dump_json()


def _apply(state: _State, target: str, index: int | None, out: pydantic.BaseModel) -> _State:
    match target:
        case "name" | "contact" | "certifications" | "skills":
            update = {name: getattr(out, name) for name in type(out).model_fields}
            return _State(state.profile.model_copy(update=update), state.experience)
        case "experience_item":
            exps = list(state.experience)
            exps[index] = out
            return _State(state.profile, exps)
        case "education_item":
            edu = list(state.profile.education)
            edu[index] = out
            return _State(state.profile.model_copy(update={"education": edu}), state.experience)
        case _:
            return _State(state.profile, list(out.experience))


def _finding(d: verify.Disagreement) -> str:
    where = f' at: "{d.evidence_snippet}"' if d.evidence_snippet else ""
    return (
        f"{d.field}: the resume text deterministically yields {json.dumps(d.text_value)}"
        f"{where}; your value was {json.dumps(d.llm_value)}."
    )


def _regenerate(
    client,
    extracted: ExtractedText,
    state: _State,
    target: str,
    index: int | None,
    group: list[verify.Disagreement],
) -> tuple[_State | None, list[Attempt]]:
    system_key = _REGEN_SYSTEM.get(target, "profile")
    message = (
        prompt.profile_message(extracted)
        if system_key == "profile"
        else prompt.experience_message(extracted)
    ) + prompt.regen_suffix(
        _previous_json(state, target, index), [_finding(d) for d in group]
    )
    model_cls = _REGEN_MODEL[target]

    def validate(out):
        if target == "experience_item":
            return check_experience_evidence([out], extracted.text)
        if target == "experience":
            return check_experience_evidence(out.experience, extracted.text)
        return []

    out, attempts = _run_ladder(
        client, f"regen:{target}", prompt.SYSTEM_PROMPTS[system_key], extracted, message,
        model_cls, validate, ladder=LADDER[:1],
    )
    return (_apply(state, target, index, out) if out is not None else None), attempts


def _verify_and_repair(
    client, extracted: ExtractedText, state: _State
) -> tuple[_State, Verification, list[Attempt]]:
    text, filename = extracted.text, extracted.source_file
    report = verify.verify(state.profile, state.experience, text, filename)
    groups: dict[tuple[str, int | None], list[verify.Disagreement]] = {}
    for d in report.disagreements:
        if d.regen_target == "experience_item" and d.index is None:
            continue
        groups.setdefault((d.regen_target, d.index), []).append(d)
    attempts: list[Attempt] = []
    unresolved: list[dict] = []
    regenerated = 0
    for (target, index), group in groups.items():
        regenerated += 1
        candidate, regen_attempts = _regenerate(client, extracted, state, target, index, group)
        attempts += regen_attempts
        fields = frozenset(d.field for d in group)
        remaining = group
        if candidate is not None:
            recheck = verify.verify(
                candidate.profile, candidate.experience, text, filename, only=fields
            )
            # Keep a regeneration only when it fixed something; otherwise keep the LLM value.
            if len(recheck.disagreements) < len(group):
                state, remaining = candidate, recheck.disagreements
        unresolved += [
            {"field": d.field, "llm_value": d.llm_value, "text_value": d.text_value}
            for d in remaining
        ]
    details = [
        {"field": d.field, "llm_value": d.llm_value, "text_value": d.text_value}
        for d in report.disagreements
    ]
    summary = Verification(
        checks_run=report.checks_run,
        disagreements=len(report.disagreements),
        regenerated=regenerated,
        unresolved=unresolved,
        details=details,
    )
    return state, summary, attempts


def _assemble(state: _State, assessment: Assessment) -> ParsedResume:
    return ParsedResume.model_validate(
        {
            **state.profile.model_dump(),
            "experience": [e.model_dump() for e in state.experience],
            **assessment.model_dump(),
        }
    )


def parse_resume(
    extracted: ExtractedText, client: anthropic.Anthropic | None = None
) -> ParseResult:
    client = client or default_client()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_profile = pool.submit(
            _run_ladder, client, "profile", prompt.PROFILE_PROMPT, extracted,
            prompt.profile_message(extracted), Profile, lambda _: [],
        )
        f_exp = pool.submit(
            _run_ladder, client, "experience", prompt.EXPERIENCE_PROMPT, extracted,
            prompt.experience_message(extracted), ExperienceList,
            lambda out: check_experience_evidence(out.experience, extracted.text),
        )
        profile, profile_attempts = f_profile.result()
        exp_list, exp_attempts = f_exp.result()
    attempts = profile_attempts + exp_attempts
    if profile is None or exp_list is None:
        failed, last = ("profile", profile_attempts) if profile is None else ("experience", exp_attempts)
        return ParseFailed(attempts, f"{failed} call: {last[-1].error or 'unknown error'}")

    state, verification, regen_attempts = _verify_and_repair(
        client, extracted, _State(profile, exp_list.experience)
    )
    attempts += regen_attempts
    skills = tidy_skills(state.profile.skills, load_taxonomy_data())
    state = replace(state, profile=state.profile.model_copy(update={"skills": skills}))

    message = prompt.assessment_message(
        extracted,
        state.profile.model_dump_json(),
        ExperienceList(experience=state.experience).model_dump_json(),
        date.today().isoformat(),
    )

    def validate(out: Assessment) -> list[str]:
        return check_evidence(_assemble(state, out), extracted.text)

    assessment, assess_attempts = _run_ladder(
        client, "assessment", prompt.ASSESSMENT_PROMPT, extracted, message, Assessment, validate
    )
    attempts += assess_attempts
    if assessment is None:
        return ParseFailed(attempts, f"assessment call: {attempts[-1].error or 'unknown error'}")
    assessment, verification, approach_attempts = _verify_approach(
        client, extracted, state, assessment, message, verification, validate
    )
    attempts += approach_attempts
    return ParseOk(_assemble(state, assessment), attempts, verification)


def _approach_report(state: _State, assessment: Assessment, extracted: ExtractedText):
    return verify.verify(
        state.profile, state.experience, extracted.text, extracted.source_file,
        only=frozenset({"investment_approach"}), approach=assessment.investment_approach,
    )


def _verify_approach(
    client,
    extracted: ExtractedText,
    state: _State,
    assessment: Assessment,
    message: str,
    verification: Verification,
    validate: Callable[[Assessment], list[str]],
) -> tuple[Assessment, Verification, list[Attempt]]:
    """The approach row needs the assessment, so it runs after that call: one regeneration of the
    assessment on disagreement, then unresolved as usual."""
    report = _approach_report(state, assessment, extracted)
    verification = replace(verification, checks_run=verification.checks_run + report.checks_run)
    if not report.disagreements:
        return assessment, verification, []
    group = report.disagreements
    regen_message = message + prompt.regen_suffix(
        assessment.model_dump_json(include={"investment_approach"}), [_finding(d) for d in group]
    )
    candidate, attempts = _run_ladder(
        client, "regen:assessment", prompt.ASSESSMENT_PROMPT, extracted, regen_message,
        Assessment, validate, ladder=LADDER[:1],
    )
    remaining = group
    if candidate is not None:
        recheck = _approach_report(state, candidate, extracted)
        if len(recheck.disagreements) < len(group):
            assessment, remaining = candidate, recheck.disagreements
    verification = replace(
        verification,
        disagreements=verification.disagreements + len(group),
        regenerated=verification.regenerated + 1,
        unresolved=verification.unresolved + [
            {"field": d.field, "llm_value": d.llm_value, "text_value": d.text_value}
            for d in remaining
        ],
        details=verification.details + [
            {"field": d.field, "llm_value": d.llm_value, "text_value": d.text_value}
            for d in group
        ],
    )
    return assessment, verification, attempts


def verification_dict(v: Verification) -> dict:
    return asdict(v)
