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
