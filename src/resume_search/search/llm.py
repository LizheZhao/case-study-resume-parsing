from __future__ import annotations

import time
from collections.abc import Callable

import pydantic

from resume_search.parse import prompt
from resume_search.parse.client import SONNET, Attempt, _validation_lines, default_client

MAX_TOKENS = 4000


def structured_call[T: pydantic.BaseModel](
    client,
    system: str,
    user: str,
    output_format: type[T],
    *,
    validate: Callable[[T], list[str]] = lambda _: [],
    attempts: list[Attempt] | None = None,
    call: str = "",
) -> T:
    """One structured-output Sonnet call, retried once with the validation errors appended."""
    client = client or default_client()
    suffix = ""
    errors: list[str] = []
    for _ in range(2):
        started = time.monotonic()
        usage = None
        out: T | None = None
        try:
            response = client.messages.parse(
                model=SONNET,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user + suffix}],
                output_format=output_format,
            )
            usage = response.usage
            out = response.parsed_output
            errors = (
                validate(out) if out is not None
                else [f"no parsed output (stop_reason={response.stop_reason})"]
            )
        except pydantic.ValidationError as exc:
            errors = _validation_lines(exc)
        if attempts is not None:
            attempts.append(
                Attempt(
                    model=SONNET,
                    latency_ms=int((time.monotonic() - started) * 1000),
                    input_tokens=usage.input_tokens if usage else 0,
                    output_tokens=usage.output_tokens if usage else 0,
                    error="; ".join(errors) or None,
                    call=call,
                )
            )
        if not errors and out is not None:
            return out
        suffix = prompt.retry_suffix(errors)
    raise ValueError(f"{call} failed after retry: {'; '.join(errors)}")
