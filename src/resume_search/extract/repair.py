from __future__ import annotations

import re
import unicodedata

REPAIR_MAP: dict[str, str] = {
    "Ɵ": "ti",  # U+019F, how the "ti" ligature glyph decodes in some PDF fonts
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ƞ": "tf",  # U+019E, the "tf" ligature in the same fonts
}
_PRIVATE_USE = re.compile("[-]")
_SPACES = re.compile(r"[ \t]+")
_BLANKS = re.compile(r"\n{4,}")


def repair_text(text: str) -> tuple[str, int]:
    count = 0
    for bad, good in REPAIR_MAP.items():
        count += text.count(bad)
        text = text.replace(bad, good)
    text, n = _PRIVATE_USE.subn("", text)
    count += n
    # NFKC would also expand ligatures, but the map runs first so repairs are counted
    text = unicodedata.normalize("NFKC", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n\n", text), count
