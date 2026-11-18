"""Image preprocessing for phone photos of handwriting."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import config


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


def _rotate(img: np.ndarray, angle: float, border: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def estimate_skew_angle(gray: np.ndarray, search: float = config.DESKEW_RANGE_DEG,
                        step: float = config.DESKEW_STEP_DEG) -> float:
    """Angle (degrees, cv2 convention) to rotate by so text lines become horizontal.

    Projection-profile method: when lines are horizontal, row sums of ink pixels are most
    "peaky", i.e. have maximum variance.
    """
    small = cv2.resize(gray, (800, max(1, int(gray.shape[0] * 800 / gray.shape[1]))),
                       interpolation=cv2.INTER_AREA)
    ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-search, search + step / 2, step):
        score = float(np.var(_rotate(ink, float(angle), border=0).sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    return best_angle


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    angle = estimate_skew_angle(to_gray(img))
    if abs(angle) < 0.25:
        return img, 0.0
    return _rotate(img, angle, border=255 if img.ndim == 2 else (255, 255, 255)), angle


def preprocess(path: Path, binarize: bool = False) -> np.ndarray:
    """Full pipeline: load -> resize -> flatten lighting -> deskew (-> optional binarize)."""
    gray = normalize_illumination(to_gray(limit_size(load_image(path))))
    gray, _ = deskew(gray)
    if binarize:
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 41, 15)
    return gray
