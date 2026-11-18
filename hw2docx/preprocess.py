"""Image preprocessing for phone photos of handwriting.

Design rule: only change the image when we are confident it is a page of text. Photos of
people/objects that merely contain some handwriting are passed through as plain grayscale instead of
being "corrected" into something worse.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config


@dataclass
class SkewEstimate:
    angle: float        # degrees (cv2 convention) to rotate by to level the text lines
    gain: float         # relative improvement of the line-profile score vs. no rotation
    peak_ratio: float   # best score / median score over all tested angles (confidence)


@dataclass
class PreprocessInfo:
    ink_fraction: float
    page_like: bool
    skew: SkewEstimate | None = None
    rotated_by: float = 0.0
    notes: list[str] = field(default_factory=list)


def load_image(path: Path) -> np.ndarray:
    """Load as BGR, honouring EXIF orientation (phone photos are often stored rotated)."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def limit_size(img: np.ndarray, max_side: int = config.MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by a heavily blurred copy to flatten shadows and uneven lighting."""
    k = max(31, (min(gray.shape) // 8) | 1)
    background = cv2.medianBlur(cv2.GaussianBlur(gray, (k, k), 0), 5)
    flat = cv2.divide(gray, background, scale=255)
    return cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)


def ink_fraction(flat_gray: np.ndarray) -> float:
    """Share of pixels darker than their local neighbourhood (a page of text: a few percent)."""
    ink = cv2.adaptiveThreshold(flat_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 41, 15)
    return float((ink > 0).mean())


def _rotate(img: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def analyze_skew(gray: np.ndarray, search: float = config.DESKEW_RANGE_DEG,
                 step: float = config.DESKEW_STEP_DEG) -> SkewEstimate:
    """Projection-profile skew estimate.

    When text lines are horizontal, row sums of ink pixels are most "peaky" (maximum variance).
    Note this levels the handwriting *baseline*, which is also what line recognizers want.
    """
    small = cv2.resize(gray, (800, max(1, int(gray.shape[0] * 800 / gray.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    angles = np.arange(-search, search + step / 2, step)
    scores = np.array([np.var(_rotate(ink, float(a), border=0).sum(axis=1)) for a in angles])
    best = int(np.argmax(scores))
    zero = float(scores[int(np.argmin(np.abs(angles)))])
    median = float(np.median(scores))
    return SkewEstimate(
        angle=float(angles[best]),
        gain=float(scores[best] / zero - 1) if zero > 0 else 0.0,
        peak_ratio=float(scores[best] / median) if median > 0 else 0.0,
    )


def estimate_skew_angle(gray: np.ndarray, search: float = config.DESKEW_RANGE_DEG,
                        step: float = config.DESKEW_STEP_DEG) -> float:
    return analyze_skew(gray, search, step).angle


def deskew(gray: np.ndarray) -> tuple[np.ndarray, SkewEstimate, str | None]:
    """Rotate only when the estimate is large enough, improves the score, and is confident."""
    est = analyze_skew(gray)
    if abs(est.angle) < config.DESKEW_MIN_ANGLE:
        return gray, est, None
    if est.gain < config.DESKEW_MIN_GAIN:
        return gray, est, f"skip deskew: gain {est.gain:.1%} < {config.DESKEW_MIN_GAIN:.0%} (noise)"
    if est.peak_ratio < config.DESKEW_MIN_PEAK_RATIO:
        return gray, est, f"skip deskew: low confidence (peak ratio {est.peak_ratio:.2f})"
    return _rotate(gray, est.angle), est, None


def preprocess_array(img: np.ndarray, binarize: bool = False) -> tuple[np.ndarray, PreprocessInfo]:
    gray = to_gray(limit_size(img))
    flat = normalize_illumination(gray)
    frac = ink_fraction(flat)
    if frac > config.PAGE_MAX_INK_FRACTION:
        info = PreprocessInfo(ink_fraction=frac, page_like=False,
                              notes=[f"not a page of text (ink {frac:.1%}): returned plain grayscale"])
        return gray, info

    info = PreprocessInfo(ink_fraction=frac, page_like=True)
    out, info.skew, note = deskew(flat)
    if note:
        info.notes.append(note)
    elif info.skew.angle and out is not flat:
        info.rotated_by = info.skew.angle
    if binarize:
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 41, 15)
    return out, info


def preprocess_with_info(path: Path, binarize: bool = False) -> tuple[np.ndarray, PreprocessInfo]:
    return preprocess_array(load_image(path), binarize=binarize)


def preprocess(path: Path, binarize: bool = False) -> np.ndarray:
    """Full pipeline: load -> resize -> flatten lighting -> deskew (-> optional binarize)."""
    return preprocess_with_info(path, binarize=binarize)[0]
