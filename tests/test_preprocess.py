import cv2
import numpy as np

from hw2docx.preprocess import _rotate, estimate_skew_angle, normalize_illumination


def _synthetic_page() -> np.ndarray:
    img = np.full((600, 900), 255, np.uint8)
    for i in range(8):
        cv2.putText(img, "handwriting recognition test line", (40, 70 + i * 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3, cv2.LINE_AA)
    return img


def test_deskew_recovers_rotation():
    page = _synthetic_page()
    skewed = _rotate(page, 6.0)
    angle = estimate_skew_angle(skewed)
    assert abs(angle + 6.0) <= 1.0
    assert abs(estimate_skew_angle(page)) <= 1.0


def test_illumination_normalization_flattens_gradient():
    page = _synthetic_page().astype(np.float32)
    gradient = np.linspace(0.45, 1.0, page.shape[1], dtype=np.float32)[None, :]
    shaded = (page * gradient).astype(np.uint8)
    flat = normalize_illumination(shaded)
    left, right = flat[:, :150].mean(), flat[:, -150:].mean()
    assert abs(float(left) - float(right)) < abs(float(shaded[:, :150].mean()) - float(shaded[:, -150:].mean()))
