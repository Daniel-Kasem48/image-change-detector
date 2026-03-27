"""
config.py
---------
Typed configuration management with validation, defaults, and
environment-variable overrides.

Usage::

    from src.config import load_config, AppConfig

    cfg = load_config("config.yaml")
    print(cfg.detector.threshold)       # 60
    print(cfg.alignment.feature_type)   # "sift"

Environment variables override YAML values using the prefix
``ICD_`` (Image Change Detector)::

    ICD_DETECTOR_THRESHOLD=80
    ICD_ALIGNMENT_FEATURE_TYPE=orb
    ICD_OBJECT_DETECTOR_CONFIDENCE=0.5
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigError
from .logging_config import get_logger

logger = get_logger(__name__)

_ENV_PREFIX = "ICD_"


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectorConfig:
    """Pixel / SSIM detector settings."""
    threshold: int = 60
    min_contour_area: int = 8000
    blur_kernel: int = 11
    # Contours larger than this fraction of the total image area are treated as
    # alignment-noise floods and discarded.  Range (0, 1].  Default 0.6 = 60%.
    max_contour_area_ratio: float = 0.6

    def __post_init__(self) -> None:
        _validate_range("threshold", self.threshold, 0, 255)
        _validate_positive("min_contour_area", self.min_contour_area)
        _validate_odd("blur_kernel", self.blur_kernel)
        _validate_range("max_contour_area_ratio", self.max_contour_area_ratio, 0.0, 1.0)


@dataclass(frozen=True)
class ObjectDetectorConfig:
    """YOLO object-detection settings."""
    model: str = "yolov8n"
    confidence: float = 0.4
    iou_threshold: float = 0.3
    device: str = "cpu"
    max_center_distance: int = 150
    use_sahi: bool = False
    sahi_slice_height: int = 640
    sahi_slice_width: int = 640
    sahi_overlap_ratio: float = 0.2
    class_agnostic_output: bool = True

    def __post_init__(self) -> None:
        _validate_range("confidence", self.confidence, 0.0, 1.0)
        _validate_range("iou_threshold", self.iou_threshold, 0.0, 1.0)
        if self.device not in ("cpu", "cuda", "mps"):
            raise ConfigError(f"Invalid device: {self.device!r}", "object_detector.device")
        _validate_positive("sahi_slice_height", self.sahi_slice_height)
        _validate_positive("sahi_slice_width", self.sahi_slice_width)
        _validate_range("sahi_overlap_ratio", self.sahi_overlap_ratio, 0.0, 0.9)


@dataclass(frozen=True)
class AlignmentConfig:
    """Feature-based image alignment settings."""
    feature_type: str = "sift"
    max_features: int = 5000
    good_match_ratio: float = 0.75
    ransac_thresh: float = 5.0
    min_matches: int = 10
    # Drone mode: use similarity transform (scale+translation only).
    # More stable than full homography for nadir/vertical drone imagery.
    use_similarity_transform: bool = False

    def __post_init__(self) -> None:
        if self.feature_type not in ("sift", "orb"):
            raise ConfigError(
                f"Invalid feature_type: {self.feature_type!r}. Use 'sift' or 'orb'.",
                "alignment.feature_type",
            )
        _validate_positive("max_features", self.max_features)
        _validate_range("good_match_ratio", self.good_match_ratio, 0.0, 1.0)
        _validate_positive("ransac_thresh", self.ransac_thresh)
        _validate_positive("min_matches", self.min_matches)


@dataclass(frozen=True)
class NormalizationConfig:
    """Photometric normalization settings."""
    method: str = "combined"
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    def __post_init__(self) -> None:
        if self.method not in ("histogram", "clahe", "combined"):
            raise ConfigError(
                f"Invalid normalization method: {self.method!r}",
                "normalization.method",
            )
        _validate_positive("clahe_clip", self.clahe_clip)
        _validate_positive("clahe_grid", self.clahe_grid)


@dataclass(frozen=True)
class MultiscaleConfig:
    """Multi-scale Gaussian pyramid detection settings."""
    num_levels: int = 4
    threshold: int = 40
    consistency: int = 3
    min_contour_area: int = 2000
    blur_kernel: int = 7

    def __post_init__(self) -> None:
        _validate_range("num_levels", self.num_levels, 1, 10)
        _validate_range("threshold", self.threshold, 0, 255)
        _validate_range("consistency", self.consistency, 1, self.num_levels)
        _validate_positive("min_contour_area", self.min_contour_area)
        _validate_odd("blur_kernel", self.blur_kernel)


@dataclass(frozen=True)
class OutputConfig:
    """Output directory and rendering settings."""
    directory: str = "outputs/"
    dpi: int = 150

    def __post_init__(self) -> None:
        _validate_positive("dpi", self.dpi)





@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    multiscale: MultiscaleConfig = field(default_factory=MultiscaleConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> AppConfig:
    """
    Load configuration from a YAML file with env-var overrides.

    Args:
        path: Path to ``config.yaml``. If *None* or the file does not exist,
              returns defaults (with env-var overrides applied).

    Returns:
        A validated :class:`AppConfig` instance.

    Raises:
        ConfigError: If the YAML is malformed or values are out of range.
    """
    raw: dict[str, Any] = {}

    if path is not None:
        p = Path(path)
        if p.exists():
            logger.debug("Loading config from %s", p)
            try:
                with p.open() as f:
                    raw = yaml.safe_load(f) or {}
            except yaml.YAMLError as exc:
                raise ConfigError(f"Invalid YAML in {p}: {exc}") from exc
        else:
            logger.debug("Config file %s not found — using defaults", p)

    # Apply environment variable overrides
    raw = _apply_env_overrides(raw)

    try:
        return AppConfig(
            detector=_build_section(DetectorConfig, raw.get("detector")),
            alignment=_build_section(AlignmentConfig, raw.get("alignment")),
            normalization=_build_section(NormalizationConfig, raw.get("normalization")),
            multiscale=_build_section(MultiscaleConfig, raw.get("multiscale")),
            output=_build_section(OutputConfig, raw.get("output")),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_section(cls: type, data: dict[str, Any] | None) -> Any:
    """Instantiate a config dataclass from a dict, ignoring unknown keys."""
    if not data:
        return cls()
    # Filter to only fields the dataclass expects
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered)


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Scan environment for ``ICD_<SECTION>_<KEY>=value`` and overlay onto *raw*.

    Example: ``ICD_DETECTOR_THRESHOLD=80`` → ``raw["detector"]["threshold"] = 80``
    """
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        parts = key[len(_ENV_PREFIX):].lower().split("_", 1)
        if len(parts) != 2:
            continue
        section, field_name = parts

        # Map section names
        section_map = {
            "detector": "detector",
            "objectdetector": "object_detector",
            "alignment": "alignment",
            "normalization": "normalization",
            "multiscale": "multiscale",
            "output": "output"
        }
        section = section_map.get(section, section)

        if section not in raw:
            raw[section] = {}

        # Attempt type coercion
        raw[section][field_name] = _coerce(value)
        logger.debug("Env override: %s.%s = %r", section, field_name, raw[section][field_name])

    return raw


def _coerce(value: str) -> bool | int | float | str:
    """Best-effort coerce a string to int, float, or leave as str."""
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _validate_range(name: str, value: int | float, lo: int | float, hi: int | float) -> None:
    if not (lo <= value <= hi):
        raise ConfigError(f"{name} must be between {lo} and {hi}, got {value}", name)


def _validate_positive(name: str, value: int | float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive, got {value}", name)


def _validate_odd(name: str, value: int) -> None:
    _validate_positive(name, value)
    if value % 2 == 0:
        raise ConfigError(f"{name} must be odd, got {value}", name)
