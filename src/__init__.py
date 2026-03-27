"""
image-change-detector
~~~~~~~~~~~~~~~~~~~~~

Production-grade image change detection using computer vision and
deep learning.

Quick start::

    from src import ChangeDetector, load_image

    img_a = load_image("before.jpg")
    img_b = load_image("after.jpg")
    result = ChangeDetector().compare(img_a, img_b)
    print(result["change_percent"])

For the programmatic API, see :mod:`src.api`.
"""

from __future__ import annotations

__version__ = "1.0.0"

# Core classes — always available
from .detector import ChangeDetector
from .aligner import ImageAligner
from .normalizer import PhotoNormalizer
from .multiscale import MultiScaleDetector

# Utilities
from .utils import load_image, save_result, load_image_rgb

# Config
from .config import AppConfig, load_config

# Exceptions
from .exceptions import (
    ChangeDetectorError,
    ImageLoadError,
    ImageSaveError,
    ConfigError,
    AlignmentError,
    DetectionError,
    ModelLoadError,
)


def get_object_detector() -> type:
    """
    Lazy import for :class:`ObjectDetector` to avoid loading
    ``torch`` / ``ultralytics`` at package import time.
    """
    from .object_detector import ObjectDetector
    return ObjectDetector


__all__ = [
    # Version
    "__version__",
    # Core classes
    "ChangeDetector",
    "ImageAligner",
    "PhotoNormalizer",
    "MultiScaleDetector",
    # Lazy loader
    "get_object_detector",
    # Utilities
    "load_image",
    "load_image_rgb",
    "save_result",
    # Config
    "AppConfig",
    "load_config",
    # Exceptions
    "ChangeDetectorError",
    "ImageLoadError",
    "ImageSaveError",
    "ConfigError",
    "AlignmentError",
    "DetectionError",
    "ModelLoadError",
]
