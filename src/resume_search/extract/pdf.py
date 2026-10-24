from __future__ import annotations

from pathlib import Path

import pymupdf

from resume_search.extract.types import TextQuality

RENDER_DPI = 150
PAGE_SEP = "\n\f\n"


def _image_coverage(page: pymupdf.Page) -> float:
    area = page.rect.width * page.rect.height
    if area <= 0:
        return 0.0
    covered = sum(
        (r & page.rect).get_area()
        for info in page.get_image_info()
        if not (r := pymupdf.Rect(info["bbox"])).is_empty
    )
    return min(covered / area, 1.0)


def render_pages(path: str | Path, dpi: int = RENDER_DPI) -> list[bytes]:
    with pymupdf.open(str(path)) as doc:
        return [page.get_pixmap(dpi=dpi).tobytes("png") for page in doc]


def extract_pdf(path: str | Path) -> tuple[str, int, float, list[bytes]]:
    """Pages are rendered here only for the triggers knowable from the raw PDF. The
    unknown-word trigger needs repaired text, so the caller renders via render_pages."""
    with pymupdf.open(str(path)) as doc:
        texts = [page.get_text("text") for page in doc]
        coverage = max((_image_coverage(p) for p in doc), default=0.0)
        n = doc.page_count
    text = PAGE_SEP.join(texts)
    sparse = TextQuality(0.0, 0.0, 0, len(text) / max(n, 1), coverage).needs_vision()
    return text, n, coverage, render_pages(path) if sparse else []
