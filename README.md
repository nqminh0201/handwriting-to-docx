# Handwriting → DOCX

Photograph a handwritten paragraph (Vietnamese and/or English) and get an editable `.docx`.

## Pipeline

```
photo → preprocess (EXIF fix, resize, flatten lighting, deskew)
      → recognize (swappable `Recognizer`)
      → paragraphs (merge wrapped lines, de-hyphenate)
      → .docx (low-confidence words highlighted, planned)
```

## Milestones

| # | Goal | Status |
|---|------|--------|
| 1 | End-to-end baseline with a VLM (Ollama): photo → `.docx` | code written, not yet run on real photos |
| 2 | Eval set (50–100 photos + ground truth) and CER/WER harness | metrics done, data pending |
| 3 | Better preprocessing + text-line segmentation, measured ablation | deskew/lighting hardened (see findings); line segmentation pending |
| 4 | Fine-tune a dedicated recognizer (TrOCR / VietOCR / QLoRA VLM) and compare with milestone 1 | — |
| 5 | Post-correction (diacritics/spelling) + confidence highlighting | — |
| 6 | Streamlit UI (upload → preview → edit → download) and write-up | — |

## Layout

```
hw2docx/        config, preprocess, recognize, pipeline, docx_export, metrics, evaluate, cli
tests/          offline tests (no model or network needed)
data/eval/      <name>.jpg + <name>.txt (ground truth)   -- git-ignored
data/samples/   your own photos                           -- git-ignored
outputs/        generated .docx and eval reports          -- git-ignored
```

## Usage

```powershell
uv sync
uv run pytest                                   # offline tests
uv run python -m hw2docx.cli preview photo.jpg  # inspect preprocessing result
uv run python -m hw2docx.cli convert photo.jpg  # needs Ollama + a vision model
uv run python -m hw2docx.cli eval               # CER/WER over data/eval
```

The baseline needs a running Ollama server with a vision model (default `qwen2.5vl:7b`,
override with `HW2DOCX_VLM`).

## Metrics

CER and WER on NFC-normalized text, plus the same metrics with diacritics stripped. Their gap
(`diacritic_cer`) shows how much of the error is only wrong or missing Vietnamese marks.

## Findings so far (preprocessing, 5 public samples)

- Phone-style photos on yellowed paper: illumination flattening + deskew work well.
- Deskew levels the handwriting *baseline*, so a perfectly straight page can still be rotated by
  1-2 degrees if the writer's lines slope. That is intended (recognizers want level lines), but the
  estimate cannot tell page skew from writer slope.
- Photos that only *contain* some handwriting (portrait with a signature) are detected as
  "not a page" (ink fraction > 10%) and passed through as plain grayscale. The margin is thin
  (12.3% vs 4-7% for pages); thresholds in `config.py` come from 5 images and must be re-tuned on the
  eval set.
- UIT-HWDB labels are the *printed text writers were asked to copy*, not an exact transcription of
  what they wrote ("&" written as "và", paraphrased words). Ground truth for evaluation must be
  verified by hand before computing CER.
