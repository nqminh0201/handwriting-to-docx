"""Character vocabulary for CTC-based handwriting recognition (Vietnamese + English).

Index 0 is always reserved for the CTC blank token (`torch.nn.CTCLoss(blank=0)`).
All text is Unicode NFC-normalized so equivalent Vietnamese glyphs map to one ID.
"""

import json
import unicodedata
from pathlib import Path

BLANK_ID = 0
BLANK_TOKEN = "<blank>"


def normalize_label(text: str) -> str:
    """NFC-normalize and collapse internal whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).split())


class Vocab:
    def __init__(self, chars: list[str]):
        unique = []
        seen = set()
        for ch in chars:
            if ch != BLANK_TOKEN and ch not in seen:
                seen.add(ch)
                unique.append(ch)
        self.id_to_char: list[str] = [BLANK_TOKEN, *unique]
        self.char_to_id: dict[str, int] = {ch: i for i, ch in enumerate(self.id_to_char)}

    @classmethod
    def from_texts(cls, texts: list[str]) -> "Vocab":
        chars = sorted({ch for t in texts for ch in normalize_label(t)})
        return cls(chars)

    def __len__(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs, skipping characters not in the vocabulary."""
        norm = normalize_label(text)
        return [self.char_to_id[ch] for ch in norm if ch in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        """Convert plain token IDs (without CTC collapsing) back to text."""
        return "".join(
            self.id_to_char[i]
            for i in ids
            if 0 < i < len(self.id_to_char)
        )

    def decode_ctc_greedy(self, ids: list[int]) -> str:
        """Standard CTC greedy decoding: collapse consecutive duplicates, then drop blanks."""
        out: list[str] = []
        prev = None
        for i in ids:
            if i != prev and 0 < i < len(self.id_to_char):
                out.append(self.id_to_char[i])
            prev = i
        return "".join(out)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"chars": self.id_to_char[1:]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])
