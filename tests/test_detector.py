"""
tests/test_detector.py
----------------------
Unit tests for ChangeDetector.
"""

from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import ChangeDetector
from src.config import AppConfig, DetectorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bgr(h: int = 100, w: int = 100, color: tuple = (100, 100, 100)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChangeDetector:

    def test_identical_images_have_zero_change(self) -> None:
        img = make_bgr()
        detector = ChangeDetector()
        result = detector.compare(img, img.copy(), method="combined")
        assert result["score"] == pytest.approx(1.0, abs=0.01)
        assert result["change_percent"] == pytest.approx(0.0, abs=1.0)

    def test_completely_different_images_have_high_change(self) -> None:
        img_a = make_bgr(color=(0, 0, 0))
        img_b = make_bgr(color=(255, 255, 255))
        detector = ChangeDetector()
        result = detector.compare(img_a, img_b, method="pixel")
        assert result["change_percent"] > 50.0

    def test_partial_change_detected(self) -> None:
        img_a = make_bgr(200, 200, (50, 50, 50))
        img_b = img_a.copy()
        img_b[50:150, 50:150] = (200, 200, 200)

        cfg = AppConfig(detector=DetectorConfig(threshold=20, min_contour_area=100, blur_kernel=5))
        detector = ChangeDetector(app_config=cfg)
        result = detector.compare(img_a, img_b, method="combined")
        assert result["change_percent"] > 0
        assert len(result["bboxes"]) >= 1

    def test_result_keys(self) -> None:
        img = make_bgr()
        detector = ChangeDetector()
        result = detector.compare(img, img.copy())
        expected_keys = {"diff_image", "mask", "contours", "bboxes",
                         "change_percent", "score", "method"}
        assert expected_keys <= set(result.keys())

    def test_all_methods_run(self) -> None:
        img_a = make_bgr(color=(30, 30, 30))
        img_b = make_bgr(color=(180, 180, 180))
        detector = ChangeDetector()
        for method in ["pixel", "ssim", "combined"]:
            result = detector.compare(img_a, img_b, method=method)
            assert result["method"] == method

    def test_invalid_method_raises(self) -> None:
        img = make_bgr()
        detector = ChangeDetector()
        with pytest.raises(ValueError, match="Unknown method"):
            detector.compare(img, img.copy(), method="invalid")

    def test_mismatched_sizes_handled(self) -> None:
        img_a = make_bgr(100, 100)
        img_b = make_bgr(200, 150)
        detector = ChangeDetector()
        result = detector.compare(img_a, img_b)
        assert "score" in result

    def test_score_is_float(self) -> None:
        img = make_bgr()
        result = ChangeDetector().compare(img, img.copy())
        assert isinstance(result["score"], float)

    def test_change_percent_is_float(self) -> None:
        img = make_bgr()
        result = ChangeDetector().compare(img, img.copy())
        assert isinstance(result["change_percent"], float)

    def test_non_array_input_raises(self) -> None:
        detector = ChangeDetector()
        with pytest.raises(ValueError, match="Expected numpy array"):
            detector.compare("not_an_array", make_bgr())  # type: ignore

    def test_empty_image_raises(self) -> None:
        detector = ChangeDetector()
        empty = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        with pytest.raises(ValueError, match="empty"):
            detector.compare(empty, make_bgr())
