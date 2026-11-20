r"""Train or overfit-sanity-check the CRNN+CTC handwriting recognizer.

Sanity check (Phase 5 -- memorize 16 real lines to verify gradients, CTC alignment & decoder):
    .\.venv\Scripts\python.exe scripts\train_crnn.py --overfit 16 --epochs 80 --lr 2e-3

Full training (Phase 6):
    .\.venv\Scripts\python.exe scripts\train_crnn.py --epochs 40 --batch-size 16
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from hw2docx.crnn import CRNN, count_parameters, ctc_decode_batch, save_checkpoint
from hw2docx.dataset import LineDataset, collate_lines
from hw2docx.metrics import cer, wer
from hw2docx.vocab import BLANK_ID, Vocab


@torch.no_grad()
def evaluate(model: CRNN, loader: DataLoader, vocab: Vocab, criterion: nn.CTCLoss, device: str) -> dict:
    model.eval()
    total_loss, n_batches = 0.0, 0
    cers, wers = [], []
    samples: list[tuple[str, str]] = []
    for batch in loader:
        imgs = batch["images"].to(device)
        targets = batch["targets"].to(device)
        tgt_lens = batch["target_lengths"].to(device)
        in_lens = CRNN.output_lengths(batch["widths"]).to(device)

        log_probs = model(imgs)
        loss = criterion(log_probs, targets, in_lens, tgt_lens)
        total_loss += float(loss.item())
        n_batches += 1

        preds = ctc_decode_batch(log_probs.cpu(), in_lens.cpu(), vocab)
        for ref, hyp in zip(batch["texts"], preds):
            cers.append(cer(ref, hyp))
            wers.append(wer(ref, hyp))
            if len(samples) < 3:
                samples.append((ref, hyp))
    return {
        "loss": total_loss / max(n_batches, 1),
        "cer": sum(cers) / max(len(cers), 1),
        "wer": sum(wers) / max(len(wers), 1),
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/htr"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--overfit", type=int, default=0, help="if >0, train & eval on first N train lines only")
    ap.add_argument("--save", type=Path, default=Path("checkpoints/crnn_best.pt"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = Vocab.load(args.data_dir / "vocab.json")
    train_ds = LineDataset(args.data_dir / "train.jsonl", vocab)
    if args.overfit > 0:
        n = min(args.overfit, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        val_ds = train_ds
        print(f"Overfit sanity check on {n} lines (device={device}, vocab={len(vocab)})")
    else:
        val_ds = LineDataset(args.data_dir / "val.jsonl", vocab)
        print(f"Training on {len(train_ds)} lines, val={len(val_ds)} (device={device}, vocab={len(vocab)})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_lines)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_lines)

    model = CRNN(vocab_size=len(vocab), dropout=0.0 if args.overfit > 0 else 0.2).to(device)
    print(f"CRNN parameters: {count_parameters(model):,}")

    criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4 if args.overfit == 0 else 0.0)
    scheduler = (
        None
        if args.overfit > 0
        else torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(1, len(train_loader))
        )
    )

    best_cer = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss, n_batches = 0.0, 0
        for batch in train_loader:
            imgs = batch["images"].to(device)
            targets = batch["targets"].to(device)
            tgt_lens = batch["target_lengths"].to(device)
            in_lens = CRNN.output_lengths(batch["widths"]).to(device)

            optimizer.zero_grad(set_to_none=True)
            log_probs = model(imgs)
            loss = criterion(log_probs, targets, in_lens, tgt_lens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_loss += float(loss.item())
            n_batches += 1

        elapsed = time.perf_counter() - t0
        should_eval = (epoch == 1) or (epoch == args.epochs) or (args.overfit == 0) or (epoch % 10 == 0)
        if should_eval:
            stats = evaluate(model, val_loader, vocab, criterion, device)
            print(
                f"epoch {epoch:3d}/{args.epochs} | train_loss={train_loss/max(n_batches,1):.3f} "
                f"| val_loss={stats['loss']:.3f} | CER={stats['cer']:.3f} | WER={stats['wer']:.3f} "
                f"| {elapsed:.1f}s",
                flush=True,
            )
            if stats["cer"] < best_cer and args.overfit == 0:
                best_cer = stats["cer"]
                save_checkpoint(args.save, model, vocab, epoch, {"val_cer": best_cer, "val_wer": stats["wer"]})
            if epoch == args.epochs:
                print("\nSample predictions:")
                for ref, hyp in stats["samples"]:
                    print(f"  REF: {ref}")
                    print(f"  HYP: {hyp}")
                    print()
        elif epoch % 5 == 0:
            print(f"epoch {epoch:3d}/{args.epochs} | train_loss={train_loss/max(n_batches,1):.3f} | {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
