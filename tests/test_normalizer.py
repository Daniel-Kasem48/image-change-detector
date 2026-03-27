"""
tests/test_normalizer.py
------------------------
Unit tests for PhotoNormalizer.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.normalizer import PhotoNormalizer
from src.config import NormalizationConfig


def make_bgr(h: int = 100, w: int = 100, color: tuple = (100, 100, 100)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


class TestPhotoNormalizer:

    def test_normalize_pair_combined(self) -> None:
        img_a = make_bgr(color=(100, 100, 100))
        img_b = make_bgr(color=(150, 150, 150))
        normalizer = PhotoNormalizer()
        a_out, b_out = normalizer.normalize_pair(img_a, img_b)
        assert a_out.shape == img_a.shape
        assert b_out.shape == img_b.shape

    def test_normalize_pair_histogram(self) -> None:
        img_a = make_bgr(color=(50, 50, 50))
        img_b = make_bgr(color=(200, 200, 200))
        normalizer = PhotoNormalizer()
        a_out, b_out = normalizer.normalize_pair(img_a, img_b, method="histogram")
        assert a_out.shape == img_a.shape

    def test_normalize_pair_clahe(self) -> None:
        img_a = make_bgr()
        img_b = make_bgr()
        normalizer = PhotoNormalizer()
        a_out, b_out = normalizer.normalize_pair(img_a, img_b, method="clahe")
        assert a_out.shape == img_a.shape

    def test_match_histograms_output_shape(self) -> None:
        img_a = make_bgr(color=(60, 60, 60))
        img_b = make_bgr(color=(180, 180, 180))
        normalizer = PhotoNormalizer()
        matched = normalizer.match_histograms(img_a, img_b)
        assert matched.shape == img_b.shape
        assert matched.dtype == np.uint8

    def test_apply_clahe_output_shape(self) -> None:
        img = make_bgr()
        normalizer = PhotoNormalizer()
        out = normalizer.apply_clahe(img)
        assert out.shape == img.shape

    def test_config_init(self) -> None:
        cfg = NormalizationConfig(clahe_clip=3.0, clahe_grid=16)
        normalizer = PhotoNormalizer(config=cfg)
        assert normalizer._cfg.clahe_clip == 3.0
        assert normalizer._cfg.clahe_grid == 16

    def test_identical_images_stay_similar(self) -> None:
        img = make_bgr(color=(128, 128, 128))
        normalizer = PhotoNormalizer()
        a_out, b_out = normalizer.normalize_pair(img, img.copy())
        # Should be very similar (CLAHE applied equally to both)
        diff = np.abs(a_out.astype(int) - b_out.astype(int)).mean()
        assert diff < 5.0  # should be nearly identical
