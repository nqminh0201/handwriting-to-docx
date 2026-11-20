"""CRNN (CNN + BiLSTM + CTC) architecture for handwritten text-line recognition.

Input:  (B, 1, 64, W)  float32 in [0, 1] (0 = background, 1 = ink)
Output: (T, B, V)      log-probabilities per horizontal time step, where T = W // 4
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocab

HORIZONTAL_DOWNSAMPLE = 4


class CRNN(nn.Module):
    def __init__(self, vocab_size: int, rnn_hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.rnn_hidden = rnn_hidden
        self.dropout = dropout

        # Height: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # Width:   W -> W/2 -> W/4 -> W/4 -> W/4 -> W/4 -> W/4
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 256, kernel_size=(2, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=rnn_hidden,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Linear(rnn_hidden * 2, vocab_size)

    @staticmethod
    def output_lengths(widths: torch.Tensor) -> torch.Tensor:
        return torch.clamp(widths // HORIZONTAL_DOWNSAMPLE, min=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(images)            # (B, 256, 1, T)
        feat = feat.squeeze(2)             # (B, 256, T)
        seq = feat.permute(2, 0, 1)        # (T, B, 256)
        seq, _ = self.rnn(seq)             # (T, B, 2*rnn_hidden)
        logits = self.head(seq)            # (T, B, vocab_size)
        return F.log_softmax(logits, dim=-1)


def ctc_decode_batch(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    vocab: Vocab,
) -> list[str]:
    """Greedy CTC decode for a batch of log_probs of shape (T, B, V)."""
    best = log_probs.argmax(dim=-1)  # (T, B)
    texts: list[str] = []
    for b in range(best.shape[1]):
        t_len = int(input_lengths[b])
        texts.append(vocab.decode_ctc_greedy(best[:t_len, b].tolist()))
    return texts


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    path: Path,
    model: CRNN,
    vocab: Vocab,
    epoch: int,
    metrics: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_chars": vocab.id_to_char[1:],
            "rnn_hidden": model.rnn_hidden,
            "dropout": model.dropout,
            "epoch": epoch,
            "metrics": metrics or {},
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[CRNN, Vocab, dict]:
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    vocab = Vocab(ckpt["vocab_chars"])
    model = CRNN(
        vocab_size=len(vocab),
        rnn_hidden=ckpt.get("rnn_hidden", 128),
        dropout=ckpt.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, ckpt.get("metrics", {})
