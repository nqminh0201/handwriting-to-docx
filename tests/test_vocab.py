from hw2docx.vocab import BLANK_ID, Vocab


def test_vocab_reserves_zero_for_blank_and_roundtrips_vietnamese():
    v = Vocab.from_texts(["Xin chào Việt Nam!", "Đường Nguyễn Huệ 123"])
    assert v.id_to_char[BLANK_ID] == "<blank>"
    ids = v.encode("Xin chào Việt Nam!")
    assert BLANK_ID not in ids
    assert v.decode(ids) == "Xin chào Việt Nam!"


def test_vocab_normalizes_nfc_before_encoding():
    v = Vocab.from_texts(["Việt"])
    decomposed = "Vie\u0323\u0302t"
    assert v.encode(decomposed) == v.encode("Việt")


def test_ctc_greedy_collapses_repeats_and_drops_blanks():
    v = Vocab(["a", "b", "c"])
    a, b, c = v.char_to_id["a"], v.char_to_id["b"], v.char_to_id["c"]
    # "abbc": repeated 'b' separated by blank becomes "bb", adjacent repeats collapse
    seq = [0, a, a, 0, b, b, 0, b, c, c, 0]
    assert v.decode_ctc_greedy(seq) == "abbc"


def test_vocab_save_and_load_roundtrip(tmp_path):
    v = Vocab.from_texts(["Tiếng Việt & English"])
    path = tmp_path / "vocab.json"
    v.save(path)
    loaded = Vocab.load(path)
    assert loaded.id_to_char == v.id_to_char
