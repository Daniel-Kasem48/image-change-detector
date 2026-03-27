"""
tests/test_utils.py
-------------------
Unit tests for utility functions.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    downscale_if_large,
    ensure_bgr,
    load_image,
    normalize,
    preprocess,
    resize_to_match,
    result_to_json,
    result_to_json_string,
    save_result,
    validate_image,
)
from src.exceptions import ImageLoadError, ImageSaveError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bgr(h: int = 100, w: int = 100, color: tuple = (100, 100, 100)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def make_temp_image(h: int = 50, w: int = 50) -> Path:
    """Create a temporary JPEG image and return its path."""
    img = make_bgr(h, w)
    path = Path(tempfile.mktemp(suffix=".jpg"))
    cv2.imwrite(str(path), img)
    return path


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:

    def test_load_valid_image(self) -> None:
        path = make_temp_image()
        img = load_image(path)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        path.unlink()

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(ImageLoadError, match="file not found"):
            load_image("/tmp/nonexistent_image_xyz.jpg")

    def test_load_empty_file_raises(self) -> None:
        path = Path(tempfile.mktemp(suffix=".jpg"))
        path.write_bytes(b"")
        with pytest.raises(ImageLoadError, match="empty"):
            load_image(path)
        path.unlink()

    def test_load_directory_raises(self) -> None:
        with pytest.raises(ImageLoadError, match="not a regular file"):
            load_image("/tmp")

    def test_load_corrupt_file_raises(self) -> None:
        path = Path(tempfile.mktemp(suffix=".jpg"))
        path.write_bytes(b"this is not an image")
        with pytest.raises(ImageLoadError, match="could not decode"):
            load_image(path)
        path.unlink()


# ---------------------------------------------------------------------------
# save_result
# ---------------------------------------------------------------------------

class TestSaveResult:

    def test_save_creates_file(self) -> None:
        img = make_bgr()
        path = Path(tempfile.mktemp(suffix=".png"))
        result_path = save_result(img, path)
        assert result_path.exists()
        result_path.unlink()

    def test_save_creates_parent_dirs(self) -> None:
        img = make_bgr()
        path = Path(tempfile.mkdtemp()) / "sub" / "dir" / "out.png"
        result_path = save_result(img, path)
        assert result_path.exists()
        result_path.unlink()

    def test_save_invalid_image_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected numpy array"):
            save_result("not_an_image", "/tmp/out.png")  # type: ignore


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------

class TestValidateImage:

    def test_valid_bgr(self) -> None:
        validate_image(make_bgr())  # should not raise

    def test_valid_gray(self) -> None:
        validate_image(np.zeros((100, 100), dtype=np.uint8))

    def test_non_array_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected numpy array"):
            validate_image([1, 2, 3])  # type: ignore

    def test_4d_raises(self) -> None:
        with pytest.raises(ValueError, match="2D or 3D"):
            validate_image(np.zeros((1, 1, 1, 1), dtype=np.uint8))

    def test_empty_raises(self) -> None:
        # 1D empty array triggers ndim check
        with pytest.raises(ValueError):
            validate_image(np.array([], dtype=np.uint8))
        # 2D empty array triggers size check
        with pytest.raises(ValueError, match="empty"):
            validate_image(np.zeros((0, 0), dtype=np.uint8))


# ---------------------------------------------------------------------------
# ensure_bgr
# ---------------------------------------------------------------------------

class TestEnsureBGR:

    def test_gray_to_bgr(self) -> None:
        gray = np.zeros((10, 10), dtype=np.uint8)
        bgr = ensure_bgr(gray)
        assert bgr.ndim == 3
        assert bgr.shape[2] == 3

    def test_bgr_passthrough(self) -> None:
        img = make_bgr(10, 10)
        assert ensure_bgr(img) is img


# ---------------------------------------------------------------------------
# resize_to_match
# ---------------------------------------------------------------------------

class TestResizeToMatch:

    def test_same_size_no_change(self) -> None:
        a = make_bgr(100, 100)
        b = make_bgr(100, 100)
        ra, rb = resize_to_match(a, b)
        assert ra.shape == rb.shape

    def test_different_sizes(self) -> None:
        a = make_bgr(100, 200)
        b = make_bgr(50, 80)
        ra, rb = resize_to_match(a, b)
        assert ra.shape[:2] == rb.shape[:2]


# ---------------------------------------------------------------------------
# downscale_if_large
# ---------------------------------------------------------------------------

class TestDownscaleIfLarge:

    def test_small_image_no_change(self) -> None:
        img = make_bgr(100, 100)
        scaled, factor = downscale_if_large(img, max_dim=200)
        assert factor == 1.0
        assert scaled.shape == img.shape

    def test_large_image_downscaled(self) -> None:
        img = make_bgr(500, 1000)
        scaled, factor = downscale_if_large(img, max_dim=200)
        assert factor < 1.0
        assert max(scaled.shape[:2]) <= 200


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_output_range(self) -> None:
        gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        normed = normalize(gray)
        assert normed.min() >= 0
        assert normed.max() <= 255


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:

    def test_output_shape(self) -> None:
        img = make_bgr()
        out = preprocess(img)
        assert out.shape == img.shape

    def test_with_denoise(self) -> None:
        img = make_bgr()
        out = preprocess(img, denoise=True)
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# result_to_json
# ---------------------------------------------------------------------------

class TestResultToJson:

    def test_pixel_result(self) -> None:
        result = {
            "method": "pixel",
            "score": 0.95,
            "change_percent": 5.2,
            "bboxes": [(10, 20, 30, 40)],
        }
        j = result_to_json(result)
        assert j["method"] == "pixel"
        assert j["ssim_score"] == 0.95
        assert j["num_regions"] == 1
        # Verify JSON-serializable
        json.dumps(j)

    def test_json_string(self) -> None:
        result = {
            "method": "combined",
            "score": 0.99,
            "change_percent": 1.0,
            "bboxes": [],
        }
        s = result_to_json_string(result)
        parsed = json.loads(s)
        assert parsed["method"] == "combined"
