"""Data shape produced by extraction and consumed by parsing.

One ExtractedText per source file. Every downstream step (LLM parse, quality
metrics, FTS body, vision fallback) reads from this object, never from the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SourceFormat(str, Enum):
    docx = "docx"
    pdf = "pdf"
    txt = "txt"
    md = "md"
    html = "html"


class ExtractionPath(str, Enum):
    text = "text"
    vision = "vision"


@dataclass(frozen=True)
class TextQuality:
    unknown_ratio_before: float
    unknown_ratio_after: float
    repairs: int
    chars_per_page: float
    max_image_page_coverage: float

    def needs_vision(self) -> bool:
        return (
            self.chars_per_page < 400
            or self.max_image_page_coverage > 0.5
            or self.unknown_ratio_after > 0.10
        )


@dataclass(frozen=True)
class ExtractedText:
    source_file: str
    file_sha256: str
    file_bytes: int
    format: SourceFormat
    page_count: int
    text: str
    quality: TextQuality
    extraction_path: ExtractionPath
    page_images_png: tuple[bytes, ...] = field(default=())

    @property
    def candidate_id(self) -> str:
        return self.file_sha256[:12]
