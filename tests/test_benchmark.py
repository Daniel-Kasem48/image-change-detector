"""
tests/test_benchmark.py
-----------------------
Unit tests for dataset benchmark/evaluation utilities.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmark import (
    DatasetLayoutError,
    DatasetSampleError,
    discover_levir_samples,
    evaluate_binary_change_dataset,
)
from src.config import AppConfig, DetectorConfig


def _write_rgb(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    assert ok


def _write_gray(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    assert ok


def _make_constant_rgb(value: int, size: int = 64) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (value, value, value)
    return img


def _make_square_mask(size: int = 64, x0: int = 16, x1: int = 48) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[x0:x1, x0:x1] = 255
    return mask


class TestDiscoverLevirSamples:

    def test_discover_pairs(self, tmp_path: Path) -> None:
        _write_rgb(tmp_path / "A" / "001.png", _make_constant_rgb(0))
        _write_rgb(tmp_path / "B" / "001.png", _make_constant_rgb(255))
        _write_gray(tmp_path / "label" / "001.png", _make_square_mask())

        samples = discover_levir_samples(tmp_path)

        assert len(samples) == 1
        assert samples[0].sample_id == "001"

    def test_missing_layout_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetLayoutError):
            discover_levir_samples(tmp_path)

    def test_no_shared_ids_raises(self, tmp_path: Path) -> None:
        _write_rgb(tmp_path / "A" / "001.png", _make_constant_rgb(0))
        _write_rgb(tmp_path / "B" / "002.png", _make_constant_rgb(255))
        _write_gray(tmp_path / "label" / "003.png", _make_square_mask())

        with pytest.raises(DatasetSampleError):
            discover_levir_samples(tmp_path)


class TestEvaluateBinaryChangeDataset:

    def test_returns_metrics_and_sample_rows(self, tmp_path: Path) -> None:
        split_root = tmp_path / "test"

        # Sample 1: real change in square area
        img_a1 = _make_constant_rgb(0)
        img_b1 = _make_constant_rgb(0)
        img_b1[16:48, 16:48] = (255, 255, 255)
        mask1 = _make_square_mask()

        _write_rgb(split_root / "A" / "001.png", img_a1)
        _write_rgb(split_root / "B" / "001.png", img_b1)
        _write_gray(split_root / "label" / "001.png", mask1)

        # Sample 2: no change
        img_a2 = _make_constant_rgb(50)
        img_b2 = _make_constant_rgb(50)
        mask2 = np.zeros((64, 64), dtype=np.uint8)

        _write_rgb(split_root / "A" / "002.png", img_a2)
        _write_rgb(split_root / "B" / "002.png", img_b2)
        _write_gray(split_root / "label" / "002.png", mask2)

        cfg = AppConfig(
            detector=DetectorConfig(
                threshold=10,
                min_contour_area=20,
                blur_kernel=3,
            )
        )

        result = evaluate_binary_change_dataset(
            tmp_path,
            split="test",
            method="combined",
            app_config=cfg,
        )

        assert result["task"] == "binary_change_detection"
        assert result["num_samples"] == 2
        assert result["method"] == "combined"
        assert set(result["metrics"]).issuperset(
            {"precision", "recall", "f1", "iou", "accuracy", "tp", "tn", "fp", "fn"}
        )
        assert len(result["samples"]) == 2

    def test_limit_applies(self, tmp_path: Path) -> None:
        for sid in ("001", "002"):
            _write_rgb(tmp_path / "A" / f"{sid}.png", _make_constant_rgb(0))
            _write_rgb(tmp_path / "B" / f"{sid}.png", _make_constant_rgb(255))
            _write_gray(tmp_path / "label" / f"{sid}.png", _make_square_mask())

        result = evaluate_binary_change_dataset(
            tmp_path,
            method="pixel",
            app_config=AppConfig(detector=DetectorConfig(threshold=10, min_contour_area=20, blur_kernel=3)),
            limit=1,
        )

        assert result["num_samples"] == 1

    def test_batching_and_workers_preserve_sample_count(self, tmp_path: Path) -> None:
        for sid in ("001", "002", "003"):
            _write_rgb(tmp_path / "A" / f"{sid}.png", _make_constant_rgb(0))
            _write_rgb(tmp_path / "B" / f"{sid}.png", _make_constant_rgb(255))
            _write_gray(tmp_path / "label" / f"{sid}.png", _make_square_mask())

        result = evaluate_binary_change_dataset(
            tmp_path,
            method="pixel",
            app_config=AppConfig(detector=DetectorConfig(threshold=10, min_contour_area=20, blur_kernel=3)),
            batch_size=2,
            num_workers=2,
        )

        assert result["num_samples"] == 3
        assert len(result["samples"]) == 3
        assert result["batch_size"] == 2
        assert result["num_workers"] == 2

    def test_invalid_batch_controls_raise(self, tmp_path: Path) -> None:
        _write_rgb(tmp_path / "A" / "001.png", _make_constant_rgb(0))
        _write_rgb(tmp_path / "B" / "001.png", _make_constant_rgb(255))
        _write_gray(tmp_path / "label" / "001.png", _make_square_mask())

        cfg = AppConfig(detector=DetectorConfig(threshold=10, min_contour_area=20, blur_kernel=3))

        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            evaluate_binary_change_dataset(tmp_path, method="pixel", app_config=cfg, batch_size=0)

        with pytest.raises(ValueError, match="num_workers must be >= 0"):
            evaluate_binary_change_dataset(tmp_path, method="pixel", app_config=cfg, num_workers=-1)
