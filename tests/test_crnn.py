import torch
import torch.nn as nn

from hw2docx.crnn import (
    CRNN,
    HORIZONTAL_DOWNSAMPLE,
    count_parameters,
    ctc_decode_batch,
    load_checkpoint,
    save_checkpoint,
)
from hw2docx.vocab import BLANK_ID, Vocab


def test_crnn_output_shape_and_log_probabilities():
    vocab = Vocab.from_texts(["Xin chào Việt Nam"])
    model = CRNN(vocab_size=len(vocab))
    imgs = torch.zeros(2, 1, 64, 160)
    out = model(imgs)
    assert out.shape == (160 // HORIZONTAL_DOWNSAMPLE, 2, len(vocab))
    probs_sum = out.exp().sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)
    assert count_parameters(model) > 100_000


def test_crnn_ctc_loss_backward_produces_finite_gradients():
    vocab = Vocab.from_texts(["Xin chào", "Việt Nam"])
    model = CRNN(vocab_size=len(vocab))
    imgs = torch.rand(2, 1, 64, 160)
    widths = torch.tensor([120, 160])
    t1, t2 = vocab.encode("Xin chào"), vocab.encode("Việt Nam")
    targets = torch.tensor(t1 + t2, dtype=torch.long)
    tgt_lens = torch.tensor([len(t1), len(t2)], dtype=torch.long)

    log_probs = model(imgs)
    in_lens = CRNN.output_lengths(widths)
    loss = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)(log_probs, targets, in_lens, tgt_lens)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()

    preds = ctc_decode_batch(log_probs.detach(), in_lens, vocab)
    assert len(preds) == 2 and all(isinstance(p, str) for p in preds)


def test_checkpoint_save_and_load_roundtrip(tmp_path):
    vocab = Vocab.from_texts(["abc Việt"])
    model = CRNN(vocab_size=len(vocab), rnn_hidden=32, dropout=0.0)
    model.eval()
    ckpt_path = tmp_path / "crnn.pt"
    save_checkpoint(ckpt_path, model, vocab, epoch=3, metrics={"val_cer": 0.12})

    loaded, loaded_vocab, metrics = load_checkpoint(ckpt_path)
    assert loaded_vocab.id_to_char == vocab.id_to_char
    assert metrics["val_cer"] == 0.12
    x = torch.rand(1, 1, 64, 96)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))
