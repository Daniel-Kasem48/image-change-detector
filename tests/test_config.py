"""
tests/test_config.py
--------------------
Unit tests for configuration loading and validation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    AppConfig,
    AlignmentConfig,
    DetectorConfig,
    MultiscaleConfig,
    NormalizationConfig,
    ObjectDetectorConfig,
    OutputConfig,
    load_config,
)
from src.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:

    def test_default_app_config(self) -> None:
        cfg = AppConfig()
        assert cfg.detector.threshold == 60
        assert cfg.alignment.feature_type == "sift"
        assert cfg.normalization.method == "combined"
        assert cfg.multiscale.num_levels == 4
        assert cfg.output.dpi == 150

    def test_load_config_no_file(self) -> None:
        cfg = load_config(None)
        assert isinstance(cfg, AppConfig)

    def test_load_config_missing_file(self) -> None:
        cfg = load_config("/tmp/nonexistent_config_xyz.yaml")
        assert isinstance(cfg, AppConfig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_threshold_out_of_range(self) -> None:
        with pytest.raises(ConfigError, match="threshold"):
            DetectorConfig(threshold=300)

    def test_negative_min_contour_area(self) -> None:
        with pytest.raises(ConfigError, match="min_contour_area"):
            DetectorConfig(min_contour_area=-1)

    def test_even_blur_kernel(self) -> None:
        with pytest.raises(ConfigError, match="blur_kernel"):
            DetectorConfig(blur_kernel=4)

    def test_invalid_feature_type(self) -> None:
        with pytest.raises(ConfigError, match="feature_type"):
            AlignmentConfig(feature_type="surf")

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ConfigError, match="confidence"):
            ObjectDetectorConfig(confidence=1.5)

    def test_invalid_device(self) -> None:
        with pytest.raises(ConfigError, match="device"):
            ObjectDetectorConfig(device="tpu")

    def test_invalid_normalization_method(self) -> None:
        with pytest.raises(ConfigError, match="normalization method"):
            NormalizationConfig(method="magic")

    def test_consistency_exceeds_levels(self) -> None:
        with pytest.raises(ConfigError, match="consistency"):
            MultiscaleConfig(num_levels=3, consistency=5)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

class TestYAMLLoading:

    def test_load_valid_yaml(self) -> None:
        yaml_content = """
detector:
  threshold: 80
  min_contour_area: 5000
  blur_kernel: 7
alignment:
  feature_type: orb
"""
        path = Path(tempfile.mktemp(suffix=".yaml"))
        path.write_text(yaml_content)

        cfg = load_config(path)
        assert cfg.detector.threshold == 80
        assert cfg.detector.min_contour_area == 5000
        assert cfg.alignment.feature_type == "orb"
        # Unset values should be defaults
        assert cfg.normalization.method == "combined"

        path.unlink()

    def test_load_empty_yaml(self) -> None:
        path = Path(tempfile.mktemp(suffix=".yaml"))
        path.write_text("")

        cfg = load_config(path)
        assert isinstance(cfg, AppConfig)

        path.unlink()

    def test_unknown_keys_ignored(self) -> None:
        yaml_content = """
detector:
  threshold: 50
  blur_kernel: 5
  unknown_key: "should be ignored"
"""
        path = Path(tempfile.mktemp(suffix=".yaml"))
        path.write_text(yaml_content)

        cfg = load_config(path)
        assert cfg.detector.threshold == 50

        path.unlink()

    def test_malformed_yaml_raises(self) -> None:
        path = Path(tempfile.mktemp(suffix=".yaml"))
        path.write_text("invalid: yaml: [: broken")

        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(path)

        path.unlink()


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ICD_DETECTOR_THRESHOLD", "42")
        cfg = load_config(None)
        assert cfg.detector.threshold == 42

    def test_env_override_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ICD_ALIGNMENT_GOOD_MATCH_RATIO", "0.6")
        cfg = load_config(None)
        assert cfg.alignment.good_match_ratio == pytest.approx(0.6)

    def test_env_override_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ICD_OBJECTDETECTOR_USE_SAHI", "true")
        cfg = load_config(None)
        assert cfg.object_detector.use_sahi is True

    def test_reasoning_env_override_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ICD_REASONING_PROVIDER", "openai")
        cfg = load_config(None)
        assert cfg.reasoning.provider == "openai"
