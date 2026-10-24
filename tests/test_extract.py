from pathlib import Path

import pytest

from resume_search.extract import extract
from resume_search.extract.types import ExtractionPath

DATA = Path(__file__).resolve().parents[1] / "data"
FILES = sorted([*DATA.glob("*.docx"), *DATA.glob("*.pdf")])


def test_sample_set_is_ten_files():
    assert len(FILES) == 10


@pytest.fixture(scope="module", params=FILES, ids=lambda p: p.name)
def extracted(request):
    return extract(request.param)


def test_extracts_without_vision_and_cleaner_than_raw(extracted):
    assert extracted.text.strip()
    assert extracted.quality.unknown_ratio_after <= extracted.quality.unknown_ratio_before
    assert not extracted.quality.needs_vision()
    assert extracted.extraction_path is ExtractionPath.text
    assert extracted.page_images_png == ()


def test_viktor_header_comes_first():
    assert extract(DATA / "Viktor Sharat.docx").text.startswith("Viktor Sharat")


def test_omar_ligatures_repaired():
    e = extract(DATA / "Omar El-Hassan 202405.pdf")
    assert "Quantitative" in e.text
    assert "Ɵ" not in e.text
    assert e.quality.repairs > 0
    assert e.quality.unknown_ratio_before > e.quality.unknown_ratio_after


def test_unsupported_suffix_is_rejected(tmp_path):
    f = tmp_path / "cv.rtf"
    f.write_text("x")
    with pytest.raises(ValueError, match="unsupported"):
        extract(f)
