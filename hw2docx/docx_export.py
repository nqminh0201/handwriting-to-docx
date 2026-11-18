"""Turn recognized text into an editable .docx.

Handwritten lines wrap at the page edge, so single line breaks are merged into one paragraph;
blank lines start a new paragraph. Words flagged as low-confidence are highlighted in yellow so
the user can review them quickly in Word.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.enum.text import WD_COLOR_INDEX


@dataclass
class Segment:
    text: str
    low_conf: bool = False


def text_to_paragraphs(text: str) -> list[str]:
    """Split on blank lines; within a paragraph, join wrapped lines (de-hyphenating line breaks)."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        merged = lines[0]
        for nxt in lines[1:]:
            if merged.endswith("-") and nxt[:1].islower():
                merged = merged[:-1] + nxt          # "recog-" + "nition" -> "recognition"
            else:
                merged = f"{merged} {nxt}"
        paragraphs.append(merged)
    return paragraphs


def build_docx(paragraphs: list[str] | list[list[Segment]], out_path: Path,
               title: str | None = None) -> Path:
    """Write paragraphs (plain strings or lists of Segments) to a .docx file."""
    doc = Document()
    if title:
        doc.add_heading(title, level=1)
    for para in paragraphs:
        p = doc.add_paragraph()
        segments = [Segment(para)] if isinstance(para, str) else para
        for i, seg in enumerate(segments):
            run = p.add_run(seg.text + (" " if i < len(segments) - 1 else ""))
            if seg.low_conf:
                run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    return out_path


def text_to_docx(text: str, out_path: Path, title: str | None = None) -> Path:
    return build_docx(text_to_paragraphs(text), out_path, title=title)
