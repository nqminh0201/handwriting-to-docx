"""Handwriting-recognition metrics: CER, WER and a Vietnamese diacritic-aware breakdown."""

import unicodedata


def normalize(text: str) -> str:
    """NFC-normalize and collapse whitespace so the same glyphs compare equal."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def strip_diacritics(text: str) -> str:
    """Remove tone/vowel marks (a, a-circumflex, a-grave... -> a; d-stroke -> d)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))


def edit_distance(a: list | str, b: list | str) -> int:
    """Levenshtein distance (insert/delete/substitute), O(len(a)*len(b)) time, O(len(b)) memory."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def report(reference: str, hypothesis: str) -> dict[str, float]:
    """CER/WER plus the same metrics ignoring diacritics.

    `diacritic_cer` = cer - cer_no_diacritics is roughly the share of character errors that are
    only wrong/missing marks, which is the dominant failure mode for Vietnamese handwriting.
    """
    c = cer(reference, hypothesis)
    c_plain = cer(strip_diacritics(reference), strip_diacritics(hypothesis))
    return {
        "cer": c,
        "wer": wer(reference, hypothesis),
        "cer_no_diacritics": c_plain,
        "diacritic_cer": max(0.0, c - c_plain),
    }
