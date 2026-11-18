"""Central configuration. Override via environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
EVAL_DIR = ROOT / "data" / "eval"
OUTPUT_DIR = ROOT / "outputs"

# Baseline recognizer: a vision-language model served by Ollama (milestone 1).
# NOTE: confirm the exact tag with `ollama list` / the Ollama library before pulling.
VLM_MODEL = os.getenv("HW2DOCX_VLM", "qwen2.5vl:7b")
VLM_NUM_CTX = int(os.getenv("HW2DOCX_NUM_CTX", "4096"))

# Preprocessing
MAX_SIDE = 2000          # downscale very large phone photos
DESKEW_RANGE_DEG = 10.0  # search window for page rotation
DESKEW_STEP_DEG = 0.5
DESKEW_MIN_ANGLE = 0.25      # ignore rotations smaller than this (degrees)

# Thresholds below were chosen from 5 sample images (see README); re-tune on the eval set.
DESKEW_MIN_GAIN = 0.03       # rotate only if text-line "peakiness" improves by >= 3% vs 0 degrees
DESKEW_MIN_PEAK_RATIO = 1.6  # best-angle score / median score; text pages 2.2-11, a photo 1.25
PAGE_MAX_INK_FRACTION = 0.10 # above this the image is not a page of text (pages: 4-7%, photo: 12%)
