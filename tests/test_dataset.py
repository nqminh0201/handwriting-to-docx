import json

import cv2
import numpy as np

from hw2docx.dataset import LineDataset, WIDTH_PAD_MULTIPLE, collate_lines
from hw2docx.vocab import Vocab


def _write_dummy_line(path, width: int, height: int = 80):
    img = np.full((height, width), 255, dtype=np.uint8)
    cv2.line(img, (10, height // 2), (width - 10, height // 2), 0, 3)
    cv2.imwrite(str(path), img)


def test_line_dataset_resizes_and_collates_variable_widths(tmp_path):
    _write_dummy_line(tmp_path / "a.png", width=200, height=80)
    _write_dummy_line(tmp_path / "b.png", width=400, height=80)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join([
            json.dumps({"image": "a.png", "text": "Xin chào"}, ensure_ascii=False),
            json.dumps({"image": "b.png", "text": "Việt Nam"}, ensure_ascii=False),
        ]),
        encoding="utf-8",
    )
    vocab = Vocab.from_texts(["Xin chào", "Việt Nam"])
    ds = LineDataset(manifest, vocab, img_height=64)
    assert len(ds) == 2

    batch = collate_lines([ds[0], ds[1]])
    imgs = batch["images"]
    assert imgs.shape[0] == 2 and imgs.shape[1] == 1 and imgs.shape[2] == 64
    assert imgs.shape[3] % WIDTH_PAD_MULTIPLE == 0
    # Right-padded region of the shorter line must be 0.0 (background)
    short_w = int(batch["widths"][0])
    assert float(imgs[0, 0, :, short_w:].abs().max()) == 0.0
    # Target lengths match concatenated targets
    assert int(batch["target_lengths"].sum()) == len(batch["targets"])
