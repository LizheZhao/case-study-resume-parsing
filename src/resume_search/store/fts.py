from __future__ import annotations

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"https?://|www\.|linkedin\.com|github\.com", re.IGNORECASE)
# Anywhere in the text only an explicit phone shape counts; a bare digit run would also match
# date ranges like "2018-01 - 2020-03".
_PHONE_STRICT = re.compile(
    r"\+\d[\d\s().-]{7,}\d|\(\d{3}\)\s*\d{3}[-.\s]\d{4}|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
)
_PHONE_LOOSE = re.compile(r"\d[\d\s().-]{7,}\d")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
HEADER_LINES = 8


def fts_body(text: str) -> str:
    """Resume text without the contact block: email, URL and phone lines are dropped. The first
    lines also get a looser phone test, unless they carry a year."""
    kept = []
    for i, line in enumerate(text.splitlines()):
        if _EMAIL.search(line) or _URL.search(line) or _PHONE_STRICT.search(line):
            continue
        if i < HEADER_LINES and _PHONE_LOOSE.search(line) and not _YEAR.search(line):
            continue
        kept.append(line)
    return "\n".join(kept)
