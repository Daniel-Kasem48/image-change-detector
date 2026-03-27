"""
exceptions.py
-------------
Custom exception hierarchy for image-change-detector.

All exceptions inherit from ``ChangeDetectorError`` so callers can
catch a single base class or individual exception types as needed.
"""

from __future__ import annotations


class ChangeDetectorError(Exception):
    """Base exception for the image-change-detector package."""


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

class ImageLoadError(ChangeDetectorError):
    """Raised when an image file cannot be loaded or decoded."""

    def __init__(self, path: str, reason: str = "") -> None:
        self.path = path
        self.reason = reason
        msg = f"Failed to load image: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class ImageSaveError(ChangeDetectorError):
    """Raised when an output image cannot be saved to disk."""

    def __init__(self, path: str, reason: str = "") -> None:
        self.path = path
        self.reason = reason
        msg = f"Failed to save image: {path}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigError(ChangeDetectorError):
    """Raised when the YAML config is invalid or missing required fields."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        prefix = f"Config error [{field}]" if field else "Config error"
        super().__init__(f"{prefix}: {message}")


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

class AlignmentError(ChangeDetectorError):
    """Raised when image alignment fails irrecoverably."""

    def __init__(self, message: str = "Image alignment failed") -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class DetectionError(ChangeDetectorError):
    """Raised when change detection encounters an unrecoverable error."""

    def __init__(self, message: str = "Change detection failed") -> None:
        super().__init__(message)


class ModelLoadError(ChangeDetectorError):
    """Raised when a ML model (e.g. YOLO) cannot be loaded."""

    def __init__(self, model_name: str, reason: str = "") -> None:
        self.model_name = model_name
        msg = f"Failed to load model: {model_name}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)
