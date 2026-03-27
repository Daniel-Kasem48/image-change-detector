"""
normalizer.py
-------------
Photometric normalization to handle lighting, exposure, and white-balance
differences between images taken at different times or with different cameras.

Techniques:
    1. **Histogram matching** — match B's histogram to A's (LAB space)
    2. **CLAHE** — adaptive contrast enhancement
    3. **Combined** — histogram match + CLAHE (default, most robust)
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import NormalizationConfig
from .logging_config import get_logger
from .utils import validate_image, ensure_bgr

logger = get_logger(__name__)


class PhotoNormalizer:
    """
    Normalizes the photometric properties of image B to match image A.

    This reduces false change detections caused by:
        - Lighting changes (time of day, shadows)
        - Camera exposure / white balance differences
        - Sensor differences between cameras / drones

    Args:
        config: A :class:`NormalizationConfig` instance.  If ``None``,
                defaults are used.
    """

    def __init__(self, config: NormalizationConfig | None = None, **kwargs: object) -> None:
        if config is not None:
            self._cfg = config
        else:
            # Legacy dict-style init
            cfg_dict: dict = kwargs.pop("config_dict", {}) if not kwargs else {}  # type: ignore[assignment]
            if not cfg_dict and kwargs:
                cfg_dict = dict(kwargs)
            self._cfg = NormalizationConfig(
                clahe_clip=float(cfg_dict.get("clahe_clip", 2.0)),  # type: ignore[arg-type]
                clahe_grid=int(cfg_dict.get("clahe_grid", 8)),  # type: ignore[arg-type]
                method=str(cfg_dict.get("method", "combined")),
            )

        logger.debug(
            "PhotoNormalizer: method=%s, clahe_clip=%.1f, grid=%d",
            self._cfg.method, self._cfg.clahe_clip, self._cfg.clahe_grid,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_pair(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        method: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Normalize *img_b* to match *img_a*'s photometric properties.

        Args:
            img_a: Reference image (BGR).
            img_b: Target image (BGR).
            method: Override the configured method
                    (``'histogram'`` | ``'clahe'`` | ``'combined'``).

        Returns:
            ``(img_a_normalized, img_b_normalized)``
        """
        validate_image(img_a, "normalizer:img_a")
        validate_image(img_b, "normalizer:img_b")

        img_a = ensure_bgr(img_a)
        img_b = ensure_bgr(img_b)

        method = method or self._cfg.method

        logger.debug("Normalizing pair with method=%s", method)

        if method == "histogram":
            return img_a, self.match_histograms(img_a, img_b)
        elif method == "clahe":
            return self.apply_clahe(img_a), self.apply_clahe(img_b)
        else:  # combined
            img_b_matched = self.match_histograms(img_a, img_b)
            return self.apply_clahe(img_a), self.apply_clahe(img_b_matched)

    # ------------------------------------------------------------------
    # Histogram Matching
    # ------------------------------------------------------------------

    def match_histograms(
        self,
        reference: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Match *target*'s colour distribution to *reference* using
        histogram matching in LAB colour space.
        """
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
        tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

        result_channels = []
        for i in range(3):
            matched = self._match_channel(ref_lab[:, :, i], tgt_lab[:, :, i])
            result_channels.append(matched)

        result_lab = cv2.merge(result_channels)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # CLAHE
    # ------------------------------------------------------------------

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the L channel of LAB colour space.
        Enhances local contrast without over-amplifying noise.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self._cfg.clahe_clip,
            tileGridSize=(self._cfg.clahe_grid, self._cfg.clahe_grid),
        )
        l_channel = clahe.apply(l_channel)

        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _match_channel(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Match *target* channel histogram to *reference* using CDFs.
        """
        ref_hist, _ = np.histogram(reference.flatten(), bins=256, range=(0, 256))
        tgt_hist, _ = np.histogram(target.flatten(), bins=256, range=(0, 256))

        ref_cdf = ref_hist.cumsum().astype(np.float64)
        tgt_cdf = tgt_hist.cumsum().astype(np.float64)

        # Normalize CDFs to [0, 1]
        ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1
        tgt_cdf /= tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1

        # Build lookup table
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            j = np.argmin(np.abs(ref_cdf - tgt_cdf[i]))
            lut[i] = j

        return lut[target]
