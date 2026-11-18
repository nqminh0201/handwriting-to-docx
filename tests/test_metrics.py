from hw2docx.metrics import cer, edit_distance, report, strip_diacritics, wer


def test_edit_distance_basic():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance(["a", "b"], ["a", "c"]) == 1


def test_cer_wer_and_empty_reference():
    assert cer("hello", "hello") == 0.0
    assert abs(cer("hello", "hallo") - 0.2) < 1e-9
    assert abs(wer("the cat sat", "the dog sat") - 1 / 3) < 1e-9
    assert cer("", "") == 0.0 and cer("", "x") == 1.0


def test_unicode_normalization_does_not_inflate_errors():
    composed = "Việt Nam"
    decomposed = "Vie\u0323\u0302t Nam"      # same text with combining marks
    assert cer(composed, decomposed) == 0.0


def test_strip_diacritics_vietnamese():
    assert strip_diacritics("Đường Nguyễn Huệ, Việt Nam") == "Duong Nguyen Hue, Viet Nam"


def test_report_separates_diacritic_errors():
    r = report("Việt Nam", "Viet Nam")       # only marks are wrong
    assert r["cer"] > 0 and r["cer_no_diacritics"] == 0.0
    assert abs(r["diacritic_cer"] - r["cer"]) < 1e-9
