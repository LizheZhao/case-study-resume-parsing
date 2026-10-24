from __future__ import annotations

import hashlib
from pathlib import Path

from resume_search.extract.docx import extract_docx
from resume_search.extract.pdf import extract_pdf, render_pages
from resume_search.extract.quality import fuzzy_fix_tokens, taxonomy_terms, unknown_ratio
from resume_search.extract.repair import repair_text
from resume_search.extract.types import (
    ExtractedText,
    ExtractionPath,
    SourceFormat,
    TextQuality,
)

__all__ = ["ExtractedText", "TextQuality", "extract"]


def extract(path: str | Path) -> ExtractedText:
    path = Path(path)
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in ("docx", "pdf"):
        raise ValueError(f"unsupported file type '.{suffix}' for {path.name}; expected .docx or .pdf")
    data = path.read_bytes()
    extra = taxonomy_terms()

    coverage, pngs = 0.0, []
    if suffix == "docx":
        raw, pages = extract_docx(path)
    else:
        raw, pages, coverage, pngs = extract_pdf(path)

    before = unknown_ratio(raw, extra)
    text, repairs = repair_text(raw)
    text, fixes = fuzzy_fix_tokens(text, extra)
    quality = TextQuality(
        unknown_ratio_before=before,
        unknown_ratio_after=unknown_ratio(text, extra),
        repairs=repairs + fixes,
        chars_per_page=len(text) / max(pages, 1),
        max_image_page_coverage=coverage,
    )
    vision = suffix == "pdf" and quality.needs_vision()
    if vision and not pngs:
        pngs = render_pages(path)
    return ExtractedText(
        source_file=path.name,
        file_sha256=hashlib.sha256(data).hexdigest(),
        file_bytes=len(data),
        format=SourceFormat(suffix),
        page_count=pages,
        text=text,
        quality=quality,
        extraction_path=ExtractionPath.vision if vision else ExtractionPath.text,
        page_images_png=tuple(pngs) if vision else (),
    )
