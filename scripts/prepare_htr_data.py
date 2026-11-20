r"""Download Vietnamese handwritten line images + labels into data/htr/ (no pyarrow required).

Uses Hugging Face's Datasets Server HTTP API on `HenryBui/vietnamese_handwriting_line_ocr`
(7,028 train lines, 201 test lines). Already-downloaded images are skipped so interrupted runs
resume cleanly.

Usage:
    .\.venv\Scripts\python.exe scripts\prepare_htr_data.py --train-limit 500 --test-limit 100
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from hw2docx.vocab import Vocab, normalize_label

DATASET = "HenryBui/vietnamese_handwriting_line_ocr"
API = "https://datasets-server.huggingface.co/rows"
UA = {"User-Agent": "handwriting-to-docx/0.1 (learning project)"}
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "htr"


def fetch_bytes(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_split(split: str, limit: int, out_dir: Path, page_size: int = 50) -> list[dict]:
    img_dir = out_dir / split
    img_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    offset = 0
    while offset < limit:
        length = min(page_size, limit - offset)
        q = urllib.parse.urlencode(
            {"dataset": DATASET, "config": "default", "split": split, "offset": offset, "length": length}
        )
        data = json.loads(fetch_bytes(f"{API}?{q}"))
        batch = data.get("rows", [])
        if not batch:
            break
        for item in batch:
            idx = item["row_idx"]
            row = item["row"]
            text = normalize_label(row.get("content", ""))
            if not text:
                continue
            rel = f"{split}/{idx:05d}.png"
            dest = out_dir / rel
            if not dest.exists():
                dest.write_bytes(fetch_bytes(row["image"]["src"]))
            rows.append({"image": rel, "text": text})
        offset += len(batch)
        print(f"  {split}: {len(rows)}/{limit} lines ready", flush=True)
        time.sleep(0.3)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-limit", type=int, default=500, help="total train+val lines to download")
    ap.add_argument("--test-limit", type=int, default=100, help="test lines to download")
    ap.add_argument("--val-ratio", type=float, default=0.1, help="fraction of train split held out for val")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {DATASET} -> {args.out}")
    train_all = fetch_split("train", args.train_limit, args.out)
    test_rows = fetch_split("test", args.test_limit, args.out)

    # Deterministic tail split so contiguous lines from the same writer are less likely to leak
    n_val = max(1, int(round(len(train_all) * args.val_ratio)))
    train_rows = train_all[:-n_val]
    val_rows = train_all[-n_val:]

    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "val.jsonl", val_rows)
    write_jsonl(args.out / "test.jsonl", test_rows)

    vocab = Vocab.from_texts([r["text"] for r in train_rows])
    vocab.save(args.out / "vocab.json")
    print(
        f"Done: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}, "
        f"vocab_size={len(vocab)} (incl. blank)"
    )


if __name__ == "__main__":
    main()
