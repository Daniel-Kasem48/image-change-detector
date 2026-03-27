"""
tests/test_exceptions.py
------------------------
Unit tests for the exception hierarchy.
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.exceptions import (
    AlignmentError,
    ChangeDetectorError,
    ConfigError,
    DetectionError,
    ImageLoadError,
    ImageSaveError,
    ModelLoadError,
)


class TestExceptionHierarchy:

    def test_all_inherit_from_base(self) -> None:
        for cls in [ImageLoadError, ImageSaveError, ConfigError,
                    AlignmentError, DetectionError, ModelLoadError]:
            assert issubclass(cls, ChangeDetectorError)

    def test_image_load_error_attributes(self) -> None:
        exc = ImageLoadError("/path/to/img.jpg", "file not found")
        assert exc.path == "/path/to/img.jpg"
        assert exc.reason == "file not found"
        assert "file not found" in str(exc)

    def test_config_error_with_field(self) -> None:
        exc = ConfigError("must be positive", field="threshold")
        assert exc.field == "threshold"
        assert "threshold" in str(exc)

    def test_model_load_error(self) -> None:
        exc = ModelLoadError("yolov8n", "ultralytics not installed")
        assert exc.model_name == "yolov8n"
        assert "ultralytics" in str(exc)

    def test_catch_base_class(self) -> None:
        with pytest.raises(ChangeDetectorError):
            raise ImageLoadError("/x.jpg")
