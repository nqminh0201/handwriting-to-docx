import cv2
import numpy as np

from hw2docx import config
from hw2docx.preprocess import (_rotate, analyze_skew, deskew, estimate_skew_angle,
                                ink_fraction, normalize_illumination, preprocess_array)


def _synthetic_page() -> np.ndarray:
    img = np.full((600, 900), 255, np.uint8)
    for i in range(8):
        cv2.putText(img, "handwriting recognition test line", (40, 70 + i * 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3, cv2.LINE_AA)
    return img


def _photo_like() -> np.ndarray:
    """Dense texture with no line structure: stands in for a photo of people/objects."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (600, 900), dtype=np.uint8)


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


def test_straight_page_is_left_alone():
    """Regression: a straight page must come out of deskew() unchanged."""
    page = _synthetic_page()
    out, est, _ = deskew(page)
    assert abs(est.angle) < config.DESKEW_MIN_ANGLE
    assert np.array_equal(out, page)


def test_skewed_page_is_rotated_back():
    skewed = _rotate(_synthetic_page(), 6.0)
    out, est, note = deskew(skewed)
    assert note is None and abs(est.angle + 6.0) <= 1.0
    assert abs(estimate_skew_angle(out)) <= 1.0


def test_page_has_confident_peak_and_low_ink_fraction():
    est = analyze_skew(_rotate(_synthetic_page(), 4.0))
    assert est.peak_ratio >= config.DESKEW_MIN_PEAK_RATIO and est.gain > config.DESKEW_MIN_GAIN
    assert ink_fraction(normalize_illumination(_synthetic_page())) < config.PAGE_MAX_INK_FRACTION


def test_non_page_image_is_passed_through_untouched():
    """Regression: a photo with some handwriting must not be rotated or contrast-burned."""
    photo = _photo_like()
    out, info = preprocess_array(photo)
    assert info.page_like is False and info.rotated_by == 0.0
    assert np.array_equal(out, photo)          # plain grayscale, no illumination "fix"
    assert any("not a page" in n for n in info.notes)


def test_preprocess_array_reports_rotation_for_skewed_page():
    skewed_bgr = cv2.cvtColor(_rotate(_synthetic_page(), 5.0), cv2.COLOR_GRAY2BGR)
    out, info = preprocess_array(skewed_bgr)
    assert info.page_like and abs(info.rotated_by + 5.0) <= 1.0
    assert out.shape == skewed_bgr.shape[:2]
