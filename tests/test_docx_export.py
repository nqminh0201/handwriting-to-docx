from docx import Document

from hw2docx.docx_export import Segment, build_docx, text_to_docx, text_to_paragraphs


def test_paragraphs_merge_wrapped_lines_and_dehyphenate():
    text = "Xin chào các\nbạn.\n\nThis is recog-\nnition of text."
    assert text_to_paragraphs(text) == ["Xin chào các bạn.", "This is recognition of text."]


def test_docx_roundtrip_keeps_vietnamese_and_highlight(tmp_path):
    out = tmp_path / "a.docx"
    build_docx([[Segment("Xin"), Segment("chào", low_conf=True)]], out, title="Bản nháp")
    doc = Document(str(out))
    texts = [p.text for p in doc.paragraphs]
    assert "Bản nháp" in texts and "Xin chào" in texts
    runs = doc.paragraphs[-1].runs
    assert runs[1].font.highlight_color is not None and runs[0].font.highlight_color is None

    out2 = text_to_docx("Một đoạn\ntiếng Việt.", tmp_path / "b.docx")
    assert Document(str(out2)).paragraphs[0].text == "Một đoạn tiếng Việt."
