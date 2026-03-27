"""
tests/test_aligner.py
---------------------
Unit tests for ImageAligner.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aligner import ImageAligner, AlignmentResult
from src.config import AlignmentConfig


def make_bgr(h: int = 200, w: int = 200) -> np.ndarray:
    """Create a BGR image with some features (not just flat colour)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Add some shapes for feature detection
    cv2.rectangle(img, (30, 30), (80, 80), (255, 0, 0), -1)
    cv2.circle(img, (150, 150), 30, (0, 255, 0), -1)
    cv2.line(img, (10, 10), (190, 190), (0, 0, 255), 2)
    cv2.putText(img, "TEST", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


class TestImageAligner:

    def test_identical_images_align(self) -> None:
        img = make_bgr()
        aligner = ImageAligner()
        result = aligner.align(img, img.copy())
        # Identical images should align successfully (or fail gracefully)
        assert isinstance(result, AlignmentResult)
        assert result.warped.shape == img.shape

    def test_alignment_result_fields(self) -> None:
        img = make_bgr()
        aligner = ImageAligner()
        result = aligner.align(img, img.copy())
        assert hasattr(result, "warped")
        assert hasattr(result, "homography")
        assert hasattr(result, "valid_mask")
        assert hasattr(result, "confidence")
        assert hasattr(result, "success")

    def test_fail_result_on_blank_images(self) -> None:
        blank = np.zeros((100, 100, 3), dtype=np.uint8)
        aligner = ImageAligner()
        result = aligner.align(blank, blank.copy())
        # Blank images have no features → should fail gracefully
        assert result.success is False
        assert result.confidence == 0.0

    def test_config_init(self) -> None:
        cfg = AlignmentConfig(feature_type="orb", max_features=1000)
        aligner = ImageAligner(config=cfg)
        assert aligner._cfg.feature_type == "orb"
        assert aligner._cfg.max_features == 1000

    def test_legacy_kwargs_init(self) -> None:
        aligner = ImageAligner(feature_type="orb", max_features=2000)
        assert aligner._cfg.feature_type == "orb"
        assert aligner._cfg.max_features == 2000

    def test_draw_matches(self) -> None:
        img = make_bgr()
        aligner = ImageAligner()
        vis = aligner.draw_matches(img, img.copy())
        assert isinstance(vis, np.ndarray)
        assert vis.ndim == 3

    def test_homography_shape(self) -> None:
        img = make_bgr()
        aligner = ImageAligner()
        result = aligner.align(img, img.copy())
        assert result.homography.shape == (3, 3)

    def test_valid_mask_is_binary(self) -> None:
        img = make_bgr()
        aligner = ImageAligner()
        result = aligner.align(img, img.copy())
        unique = set(np.unique(result.valid_mask))
        assert unique <= {0, 255}
