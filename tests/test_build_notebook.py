import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_notebook as bn


@pytest.fixture(scope="module")
def nb():
    return bn.build()


def test_first_cell_is_the_brief_verbatim(nb):
    import nbformat

    brief = nbformat.read(bn.BRIEF, as_version=4).cells[0]
    assert nb.cells[0].cell_type == "markdown"
    assert nb.cells[0].source == brief.source


def test_kernel_is_case_study(nb):
    assert nb.metadata["kernelspec"]["name"] == "case-study"


def test_every_embedded_file_matches_disk(nb):
    cells = {
        c.source.split("\n", 1)[0].removeprefix("%%writefile "): c.source.split("\n", 1)[1]
        for c in nb.cells
        if c.cell_type == "code" and c.source.startswith("%%writefile ")
    }
    expected = [p for group in bn.collect_files().values() for p in group if (ROOT / p).stat().st_size]
    assert sorted(cells) == sorted(p.as_posix() for p in expected)
    for path in expected:
        assert cells[path.as_posix()] + "\n" == (ROOT / path).read_text(encoding="utf-8")


def test_llm_cells_are_guarded(nb):
    demos = [c.source for c in nb.cells if c.cell_type == "code" and not c.source.startswith("%%")]
    llm = [src for src in demos if "parse_resume(" in src or "parse_requisition(" in src]
    assert len(llm) == 2
    assert all('os.environ.get("RUN_LLM") == "1"' in src for src in llm)
