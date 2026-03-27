"""
multiscale.py
-------------
Multi-scale (Gaussian pyramid) change detection.

Detects changes at multiple resolutions and keeps only those that appear
consistently across scales — eliminating noise while preserving real
structural changes.

Why multi-scale?
    - Single-scale detection is sensitive to image resolution.
    - Noise tends to appear at fine scales only.
    - Real changes (objects added/removed) persist across all scales.
    - More robust results for drone / satellite imagery.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any

from .config import MultiscaleConfig
from .logging_config import get_logger
from .utils import validate_image

logger = get_logger(__name__)


class MultiScaleDetector:
    """
    Performs change detection across a Gaussian image pyramid and
    fuses results to keep only consistent changes.

    Args:
        config: A :class:`MultiscaleConfig` instance.  If ``None``,
                defaults are used.  Legacy keyword arguments are
                also accepted for backwards compatibility.
    """

    def __init__(
        self,
        config: MultiscaleConfig | None = None,
        *,
        # Legacy kwargs
        num_levels: int | None = None,
        threshold: int | None = None,
        consistency: int | None = None,
        min_contour_area: int | None = None,
        blur_kernel: int | None = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        else:
            kwargs: dict = {}
            if num_levels is not None:
                kwargs["num_levels"] = num_levels
            if threshold is not None:
                kwargs["threshold"] = threshold
            if consistency is not None:
                kwargs["consistency"] = consistency
            if min_contour_area is not None:
                kwargs["min_contour_area"] = min_contour_area
            if blur_kernel is not None:
                kwargs["blur_kernel"] = blur_kernel
            self._cfg = MultiscaleConfig(**kwargs)

        logger.debug(
            "MultiScaleDetector: levels=%d, threshold=%d, consistency=%d",
            self._cfg.num_levels, self._cfg.threshold, self._cfg.consistency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        valid_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Run multi-scale change detection.

        Args:
            img_a: Reference image (BGR).
            img_b: Comparison image (BGR), should be aligned to *img_a*.
            valid_mask: Optional mask (from aligner) to ignore border regions.

        Returns:
            Dict with ``mask``, ``bboxes``, ``contours``, ``level_masks``,
            and ``change_percent``.
        """
        validate_image(img_a, "multiscale:img_a")
        validate_image(img_b, "multiscale:img_b")

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        # Build Gaussian pyramids
        pyr_a = self._build_pyramid(gray_a)
        pyr_b = self._build_pyramid(gray_b)

        # Detect changes at each level
        level_masks: list[np.ndarray] = []
        for level in range(self._cfg.num_levels):
            mask = self._detect_at_level(pyr_a[level], pyr_b[level])
            mask_full = self._upscale_mask(mask, gray_a.shape[:2], level)
            level_masks.append(mask_full)

        logger.debug("Built %d pyramid levels", len(level_masks))

        # Fuse: keep only changes in >= consistency levels
        fused = self._fuse_masks(level_masks)

        # Apply valid region mask
        if valid_mask is not None:
            fused = cv2.bitwise_and(fused, valid_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, kernel)
        fused = cv2.morphologyEx(fused, cv2.MORPH_OPEN, kernel)

        contours, bboxes = self._find_regions(fused)

        # Change percentage
        valid_pixels = int(np.sum(valid_mask > 0)) if valid_mask is not None else fused.size
        change_pixels = int(np.sum(fused > 0))
        change_percent = (change_pixels / valid_pixels * 100) if valid_pixels > 0 else 0.0

        logger.info(
            "Multi-scale detection: %d regions, %.1f%% change",
            len(bboxes), change_percent,
        )

        return {
            "mask": fused,
            "bboxes": bboxes,
            "contours": contours,
            "level_masks": level_masks,
            "change_percent": round(change_percent, 2),
        }

    # ------------------------------------------------------------------
    # Pyramid construction
    # ------------------------------------------------------------------

    def _build_pyramid(self, gray: np.ndarray) -> list[np.ndarray]:
        """Build a Gaussian pyramid with *num_levels* levels."""
        pyramid = [gray]
        current = gray
        for _ in range(1, self._cfg.num_levels):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    # ------------------------------------------------------------------
    # Per-level detection
    # ------------------------------------------------------------------

    def _detect_at_level(
        self, level_a: np.ndarray, level_b: np.ndarray,
    ) -> np.ndarray:
        """Compute change mask at a single pyramid level."""
        k = self._cfg.blur_kernel
        a_blur = cv2.GaussianBlur(level_a, (k, k), 0)
        b_blur = cv2.GaussianBlur(level_b, (k, k), 0)
        diff = cv2.absdiff(a_blur, b_blur)
        _, mask = cv2.threshold(diff, self._cfg.threshold, 255, cv2.THRESH_BINARY)
        return mask

    # ------------------------------------------------------------------
    # Mask manipulation
    # ------------------------------------------------------------------

    def _upscale_mask(
        self, mask: np.ndarray, target_shape: tuple[int, int], level: int,
    ) -> np.ndarray:
        """Upscale a pyramid-level mask back to the original resolution."""
        if level == 0:
            return mask
        h, w = target_shape
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    def _fuse_masks(self, level_masks: list[np.ndarray]) -> np.ndarray:
        """
        Fuse masks from multiple scales.
        A pixel is marked as changed only if it appears in ≥ *consistency* levels.
        """
        stack = np.stack([(m > 0).astype(np.uint8) for m in level_masks], axis=0)
        vote_map = stack.sum(axis=0)

        fused = np.zeros_like(level_masks[0])
        fused[vote_map >= self._cfg.consistency] = 255
        return fused

    # ------------------------------------------------------------------
    # Region extraction
    # ------------------------------------------------------------------

    def _find_regions(
        self, mask: np.ndarray,
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
        """Find contours and bounding boxes from the fused mask."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        bboxes: list[tuple[int, int, int, int]] = []
        valid_contours: list[np.ndarray] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self._cfg.min_contour_area:
                valid_contours.append(c)
                bboxes.append(cv2.boundingRect(c))

        return valid_contours, bboxes
