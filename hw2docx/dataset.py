"""PyTorch Dataset and batch collation for handwritten text lines (CRNN + CTC).

Each line image is resized to a fixed height (default 64px) while keeping its aspect ratio,
then inverted to float32 in [0, 1] where 0.0 = white background and 1.0 = dark ink.
That way, right-padding variable-width lines with 0.0 pads cleanly with background.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .vocab import Vocab, normalize_label

DEFAULT_IMG_HEIGHT = 64
DEFAULT_MAX_WIDTH = 1024
WIDTH_PAD_MULTIPLE = 8  # keeps CNN downsampling widths aligned


def load_line_tensor(
    image_path: Path,
    img_height: int = DEFAULT_IMG_HEIGHT,
    max_width: int = DEFAULT_MAX_WIDTH,
) -> torch.Tensor:
    """Load a line image as a float32 tensor of shape (1, img_height, W) in [0, 1]."""
    raw = np.fromfile(str(image_path), dtype=np.uint8)
    gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return array_to_line_tensor(gray, img_height=img_height, max_width=max_width)


def array_to_line_tensor(
    gray: np.ndarray,
    img_height: int = DEFAULT_IMG_HEIGHT,
    max_width: int = DEFAULT_MAX_WIDTH,
) -> torch.Tensor:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    new_w = max(8, min(max_width, int(round(w * (img_height / h)))))
    resized = cv2.resize(gray, (new_w, img_height), interpolation=cv2.INTER_AREA)
    # Invert: paper -> 0.0, ink -> 1.0
    inv = (255.0 - resized.astype(np.float32)) / 255.0
    return torch.from_numpy(inv).unsqueeze(0)


@dataclass
class LineSample:
    image: torch.Tensor      # (1, H, W)
    target: torch.Tensor     # (L,) int64
    text: str


class LineDataset(Dataset):
    """Reads a JSONL manifest where each line has {"image": "<rel_path>", "text": "..."}."""

    def __init__(
        self,
        manifest_path: Path,
        vocab: Vocab,
        img_height: int = DEFAULT_IMG_HEIGHT,
        max_width: int = DEFAULT_MAX_WIDTH,
    ):
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.vocab = vocab
        self.img_height = img_height
        self.max_width = max_width
        self.rows: list[dict] = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = normalize_label(row["text"])
            if text and vocab.encode(text):
                self.rows.append({"image": row["image"], "text": text})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> LineSample:
        row = self.rows[idx]
        img = load_line_tensor(self.root / row["image"], self.img_height, self.max_width)
        ids = self.vocab.encode(row["text"])
        return LineSample(
            image=img,
            target=torch.tensor(ids, dtype=torch.long),
            text=row["text"],
        )


def collate_lines(batch: list[LineSample]) -> dict:
    """Pad variable-width line images on the right and pack targets for CTCLoss."""
    max_w = max(s.image.shape[2] for s in batch)
    padded_w = ((max_w + WIDTH_PAD_MULTIPLE - 1) // WIDTH_PAD_MULTIPLE) * WIDTH_PAD_MULTIPLE
    h = batch[0].image.shape[1]

    images = torch.zeros((len(batch), 1, h, padded_w), dtype=torch.float32)
    widths = torch.zeros(len(batch), dtype=torch.long)
    for i, s in enumerate(batch):
        w = s.image.shape[2]
        images[i, :, :, :w] = s.image
        widths[i] = w

    targets = torch.cat([s.target for s in batch])
    target_lengths = torch.tensor([len(s.target) for s in batch], dtype=torch.long)
    return {
        "images": images,
        "widths": widths,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": [s.text for s in batch],
    }
