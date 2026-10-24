from __future__ import annotations

import re

from rapidfuzz import fuzz, process
from wordfreq import zipf_frequency

from resume_search.taxonomy import taxonomy_terms, tokenize

__all__ = ["fuzzy_fix_tokens", "taxonomy_terms", "unknown_ratio"]

_DIGIT = re.compile(r"\d")
_FUZZY_CUTOFF = 90
_MIN_LEN = 3


def _is_known(token: str, extra_known: set[str] | frozenset[str]) -> bool:
    low = token.lower()
    return (
        bool(_DIGIT.search(token))
        or low in extra_known
        or zipf_frequency(low, "en") > 0
    )


def unknown_ratio(text: str, extra_known: set[str] | frozenset[str]) -> float:
    tokens = [t for t in tokenize(text) if len(t) >= _MIN_LEN]
    if not tokens:
        return 0.0
    return sum(not _is_known(t, extra_known) for t in tokens) / len(tokens)


def fuzzy_fix_tokens(text: str, extra_known: set[str] | frozenset[str]) -> tuple[str, int]:
    """Substitute unknown tokens carrying a non-ASCII char with a close known token.

    Candidates are the document's own known tokens plus extra_known. Only non-ASCII
    tokens are tried: repair already handled the known glyph maps, so what is left with
    a stray glyph is what a fuzzy match can plausibly recover.
    """
    tokens = [t for t in tokenize(text) if len(t) >= _MIN_LEN]
    known = {t.lower() for t in tokens if _is_known(t, extra_known) and t.isascii()}
    candidates = list(known | set(extra_known))
    fixes = 0

    def sub(m: re.Match[str]) -> str:
        nonlocal fixes
        tok = m.group(0)
        if len(tok) < _MIN_LEN or tok.isascii() or _is_known(tok, extra_known):
            return tok
        hit = process.extractOne(
            tok.lower(), candidates, scorer=fuzz.ratio, score_cutoff=_FUZZY_CUTOFF
        )
        if hit is None:
            return tok
        fixes += 1
        fixed = hit[0]
        return fixed.capitalize() if tok[0].isupper() else fixed

    out = re.sub(r"[^\W_]+(?:['’][^\W_]+)?", sub, text)
    return out, fixes
