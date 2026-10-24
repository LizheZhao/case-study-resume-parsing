from __future__ import annotations

import base64
import os
import re
import time
from dataclasses import dataclass
from typing import ClassVar

import anthropic
import pydantic

from resume_search.extract.types import ExtractedText, ExtractionPath
from resume_search.parse.models import ParsedResume, check_evidence
from resume_search.parse.prompt import SYSTEM_PROMPT, retry_suffix, user_message

SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
# (model, append the previous attempt's validation errors)
LADDER = ((SONNET, False), (SONNET, True), (OPUS, True))
MAX_TOKENS = 16000
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass(frozen=True)
class Attempt:
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    error: str | None
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass(frozen=True)
class ParseOk:
    record: ParsedResume
    attempts: list[Attempt]

    @property
    def parse_status(self) -> str:
        last = self.attempts[-1]
        if len(self.attempts) == 1:
            return "ok"
        return "ok_retry_opus" if last.model == OPUS else "ok_retry_sonnet"


@dataclass(frozen=True)
class ParseFailed:
    attempts: list[Attempt]
    last_error: str

    parse_status: ClassVar[str] = "failed"


ParseResult = ParseOk | ParseFailed


def _validation_lines(exc: pydantic.ValidationError) -> list[str]:
    return [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]


def _content(extracted: ExtractedText, suffix: str) -> list[dict]:
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
    blocks.append({"type": "text", "text": user_message(extracted) + suffix})
    return blocks


def _attempt(client, model: str, content: list[dict]):
    return client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": content}],
    )


def default_client() -> anthropic.Anthropic:
    # Keys that are not workspace-scoped need the workspace id header; the SDK does not
    # read ANTHROPIC_WORKSPACE_ID itself.
    headers = {}
    if workspace := os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        headers["anthropic-workspace-id"] = workspace
    return anthropic.Anthropic(max_retries=4, default_headers=headers)


def parse_resume(
    extracted: ExtractedText, client: anthropic.Anthropic | None = None
) -> ParseResult:
    client = client or default_client()
    attempts: list[Attempt] = []
    prev_errors: list[str] = []
    for model, with_errors in LADDER:
        suffix = retry_suffix(prev_errors) if with_errors and prev_errors else ""
        started = time.monotonic()
        usage = None
        record: ParsedResume | None = None
        errors: list[str] = []
        raw = ""
        try:
            response = _attempt(client, model, _content(extracted, suffix))
            usage = response.usage
            raw = "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason == "max_tokens" or not raw:
                errors = [f"no complete output (stop_reason={response.stop_reason})"]
            else:
                record = ParsedResume.model_validate_json(_FENCE.sub("", raw.strip()))
                errors = check_evidence(record, extracted.text)
        except pydantic.ValidationError as exc:
            errors = _validation_lines(exc)
            if any(e["type"] == "json_invalid" for e in exc.errors()):
                errors.append(f"reply began: {raw[:200]!r}")
        except anthropic.RateLimitError as exc:
            errors = [f"RateLimitError: {exc}"]
        except anthropic.APIStatusError as exc:
            errors = [f"APIStatusError {exc.status_code}: {exc.message}"]
        except anthropic.APIConnectionError as exc:
            errors = [f"APIConnectionError: {exc}"]
        api_failed = usage is None
        attempts.append(
            Attempt(
                model=model,
                latency_ms=int((time.monotonic() - started) * 1000),
                input_tokens=usage.input_tokens if usage else 0,
                output_tokens=usage.output_tokens if usage else 0,
                error="; ".join(errors) if errors else None,
                cache_read_tokens=(usage.cache_read_input_tokens or 0) if usage else 0,
                cache_creation_tokens=(usage.cache_creation_input_tokens or 0) if usage else 0,
            )
        )
        if not errors and record is not None:
            return ParseOk(record, attempts)
        # An API failure carries nothing the model could fix, so the next rung gets no suffix.
        prev_errors = [] if api_failed else errors
    return ParseFailed(attempts, attempts[-1].error or "unknown error")
