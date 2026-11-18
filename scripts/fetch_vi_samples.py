"""Build Vietnamese paragraph test images from UIT-HWDB (line-level, CC-BY-4.0).

Source: Hugging Face mirror `blue7012/UIT_HWDB` (license: cc-by-4.0; credit the original UIT-HWDB authors).
Each dataset row is one handwritten line image + its transcription. Consecutive lines of the same
writer are stacked vertically on a white page, so the ground truth is the lines joined by "\n".

NOTE: these pages are *clean* real handwriting (no phone-camera skew/shadows). Use them for
recognition accuracy; degrade them synthetically to test preprocessing.

    uv run python scripts/fetch_vi_samples.py
"""

import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

UA = {"User-Agent": "handwriting-to-docx/0.1 (learning project)"}
DATASET = "blue7012/UIT_HWDB"
OUT = Path(__file__).resolve().parent.parent / "data" / "samples"

# (name, split, offset, number of lines)
PAGES = [
    ("vi_uithwdb_w250", "test", 0, 8),    # writer 250: newspaper paragraph
    ("vi_uithwdb_w1", "train", 3, 8),     # writer 1: textbook paragraph (skips the odd 'Đông.' row)
]


def get(url: str, timeout: int = 90) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def fetch_rows(split: str, offset: int, length: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": split,
                                "offset": offset, "length": length})
    return json.loads(get("https://datasets-server.huggingface.co/rows?" + q))["rows"]


def compose(lines: list[Image.Image], margin: int = 80, gap: int = 30) -> Image.Image:
    width = max(im.width for im in lines) + 2 * margin
    height = sum(im.height for im in lines) + gap * (len(lines) - 1) + 2 * margin
    page = Image.new("RGB", (width, height), "white")
    y = margin
    for im in lines:
        page.paste(im.convert("RGB"), (margin, y))
        y += im.height + gap
    return page


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notes = []
    for name, split, offset, n in PAGES:
        rows = fetch_rows(split, offset, n)
        writers = {r["row"]["writer_id"] for r in rows}
        if len(writers) != 1:
            print(f"! {name}: rows span several writers {writers}, skipping")
            continue
        images = []
        for r in rows:
            images.append(Image.open(io.BytesIO(get(r["row"]["image"]["src"]))))
            time.sleep(0.5)  # be polite to the server
        compose(images).save(OUT / f"{name}.png")
        (OUT / f"{name}.txt").write_text("\n".join(r["row"]["text"] for r in rows), encoding="utf-8")
        notes.append(f"- {name}.png | CC-BY-4.0 | UIT-HWDB via https://huggingface.co/datasets/{DATASET} "
                     f"({split} split, writer {writers.pop()}, rows {offset}-{offset + n - 1}, lines stacked)")
        print("saved", name, f"{len(images)} lines")
    with open(OUT / "SOURCES.md", "a", encoding="utf-8") as f:
        f.write("\n".join(notes) + "\n")


if __name__ == "__main__":
    main()
