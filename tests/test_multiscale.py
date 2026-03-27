"""
tests/test_multiscale.py
------------------------
Unit tests for MultiScaleDetector.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.multiscale import MultiScaleDetector
from src.config import MultiscaleConfig


def make_bgr(h: int = 200, w: int = 200, color: tuple = (100, 100, 100)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


class TestMultiScaleDetector:

    def test_identical_images_no_change(self) -> None:
        img = make_bgr()
        detector = MultiScaleDetector()
        result = detector.detect(img, img.copy())
        assert result["change_percent"] == pytest.approx(0.0, abs=1.0)
        assert len(result["bboxes"]) == 0

    def test_different_images_detect_change(self) -> None:
        img_a = make_bgr(200, 200, (50, 50, 50))
        img_b = img_a.copy()
        # Large change region
        img_b[50:150, 50:150] = (220, 220, 220)

        cfg = MultiscaleConfig(threshold=20, min_contour_area=100, consistency=2)
        detector = MultiScaleDetector(config=cfg)
        result = detector.detect(img_a, img_b)
        assert result["change_percent"] > 0
        assert len(result["bboxes"]) >= 1

    def test_result_keys(self) -> None:
        img = make_bgr()
        detector = MultiScaleDetector()
        result = detector.detect(img, img.copy())
        expected = {"mask", "bboxes", "contours", "level_masks", "change_percent"}
        assert expected <= set(result.keys())

    def test_level_masks_count(self) -> None:
        img = make_bgr()
        cfg = MultiscaleConfig(num_levels=3)
        detector = MultiScaleDetector(config=cfg)
        result = detector.detect(img, img.copy())
        assert len(result["level_masks"]) == 3

    def test_valid_mask_applied(self) -> None:
        img_a = make_bgr(200, 200, (0, 0, 0))
        img_b = make_bgr(200, 200, (255, 255, 255))

        # Valid mask covers only center
        valid_mask = np.zeros((200, 200), dtype=np.uint8)
        valid_mask[50:150, 50:150] = 255

        cfg = MultiscaleConfig(threshold=20, min_contour_area=100, consistency=1)
        detector = MultiScaleDetector(config=cfg)
        result = detector.detect(img_a, img_b, valid_mask)

        # Change should be limited to the valid mask area
        change_mask = result["mask"]
        # Nothing should be detected outside valid_mask
        outside_valid = change_mask.copy()
        outside_valid[50:150, 50:150] = 0
        assert np.sum(outside_valid > 0) == 0

    def test_legacy_kwargs_init(self) -> None:
        detector = MultiScaleDetector(num_levels=5, threshold=30)
        assert detector._cfg.num_levels == 5
        assert detector._cfg.threshold == 30
