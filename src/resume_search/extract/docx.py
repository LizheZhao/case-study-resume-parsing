from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

_MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


def _row_line(row) -> str:
    cells: list[str] = []
    prev_tc = None
    for cell in row.cells:
        # python-docx yields the same cell object for every grid column a merged cell spans
        if cell._tc is prev_tc:
            continue
        prev_tc = cell._tc
        text = " ".join(p.text.strip() for p in cell.paragraphs if p.text.strip())
        if text and (not cells or cells[-1] != text):
            cells.append(text)
    return " | ".join(cells)


def _textbox_lines(doc) -> list[str]:
    """Floating text boxes (page furniture such as the name banner) come first.

    Word stores each box twice, as mc:Choice and a VML mc:Fallback, so the fallback copy
    is skipped. python-docx's paragraph.text never descends into them.
    """
    lines: list[str] = []
    for box in doc.element.body.iter(qn("w:txbxContent")):
        if any(a.tag == _MC_FALLBACK for a in box.iterancestors()):
            continue
        for p in box.iterchildren(qn("w:p")):
            text = Paragraph(p, doc).text.strip()
            if text and text != "-":
                lines.append(text)
    return lines


def extract_docx(path: str | Path) -> tuple[str, int]:
    doc = Document(str(path))
    lines = _textbox_lines(doc)
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            lines.append(Paragraph(child, doc).text)
        elif child.tag == qn("w:tbl"):
            lines.extend(line for line in map(_row_line, Table(child, doc).rows) if line)
    return "\n".join(lines), 1
