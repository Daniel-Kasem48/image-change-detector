"""
detector.py
-----------
Core image change detection module.

Provides pixel-level, structural (SSIM), contour-based, and **robust**
(align → normalize → multi-scale) change detection.

Modes:
    - ``pixel``    — Absolute pixel difference
    - ``ssim``     — Structural Similarity Index (perceptual)
    - ``combined`` — Pixel diff + SSIM combined mask
    - ``robust``   — Align → Normalize → Multi-scale detect
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Any

from .aligner import ImageAligner
from .config import AppConfig, DetectorConfig, load_config
from .exceptions import DetectionError
from .logging_config import get_logger
from .multiscale import MultiScaleDetector
from .normalizer import PhotoNormalizer
from .utils import resize_to_match, validate_image

logger = get_logger(__name__)

# Supported detection methods
METHODS = frozenset({"pixel", "ssim", "combined", "robust", "drone"})


class ChangeDetector:
    """
    Compares two images and detects regions of change.

    Supports four detection strategies:

    =========  =========================================================
    Method     Description
    =========  =========================================================
    pixel      Absolute pixel difference
    ssim       Structural Similarity Index (perceptual)
    combined   Pixel diff + SSIM combined mask (default)
    robust     Full pipeline: align → normalize → multi-scale detect
    =========  =========================================================

    Args:
        app_config: A full :class:`AppConfig` instance.
        config: Legacy dict for backwards compatibility (detector section
                keys only).
    """

    def __init__(
        self,
        app_config: AppConfig | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if app_config is not None:
            self._app_cfg = app_config
            self._det_cfg = app_config.detector
        elif config is not None:
            # Legacy dict-based init — build DetectorConfig
            self._det_cfg = DetectorConfig(
                threshold=config.get("threshold", 60),
                min_contour_area=config.get("min_contour_area", 8000),
                blur_kernel=config.get("blur_kernel", 11),
            )
            # Stash raw dict for robust-mode sub-configs
            self._legacy_cfg = config
            self._app_cfg = None
        else:
            self._app_cfg = AppConfig()
            self._det_cfg = self._app_cfg.detector

        logger.debug(
            "ChangeDetector: threshold=%d, min_area=%d, blur=%d",
            self._det_cfg.threshold,
            self._det_cfg.min_contour_area,
            self._det_cfg.blur_kernel,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        method: str = "combined",
    ) -> dict[str, Any]:
        """
        Compare two images and return a result dict.

        Args:
            image_a: Reference image (BGR).
            image_b: Comparison image (BGR).
            method: Detection mode (``pixel``, ``ssim``, ``combined``,
                    or ``robust``).

        Returns:
            Dict with keys: ``diff_image``, ``mask``, ``contours``,
            ``bboxes``, ``change_percent``, ``score``, ``method``.
            Robust mode adds ``alignment`` metadata.

        Raises:
            DetectionError: On unrecoverable processing error.
            ValueError: On invalid method or image input.
        """
        method = method.lower()
        if method not in METHODS:
            raise ValueError(
                f"Unknown method {method!r}. Choose from {sorted(METHODS)}."
            )

        validate_image(image_a, "compare:image_a")
        validate_image(image_b, "compare:image_b")

        img_a, img_b = resize_to_match(image_a, image_b)

        if method == "robust":
            return self._compare_robust(img_a, img_b)

        if method == "drone":
            return self._compare_drone(img_a, img_b)

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        try:
            if method == "pixel":
                mask, score = self._pixel_diff(gray_a, gray_b)
            elif method == "ssim":
                mask, score = self._ssim_diff(gray_a, gray_b)
            else:
                mask, score = self._combined_diff(gray_a, gray_b)
        except Exception as exc:
            raise DetectionError(f"{method} detection failed: {exc}") from exc

        contours, bboxes = self._find_changed_regions(mask)
        diff_image = self._build_diff_image(img_a, img_b, mask, bboxes)

        total_pixels = mask.shape[0] * mask.shape[1]
        change_percent = (np.sum(mask > 0) / total_pixels) * 100

        logger.info(
            "%s detection: score=%.4f, change=%.1f%%, regions=%d",
            method.upper(), score, change_percent, len(bboxes),
        )

        return {
            "diff_image": diff_image,
            "mask": mask,
            "contours": contours,
            "bboxes": bboxes,
            "change_percent": round(float(change_percent), 2),
            "score": round(float(score), 4),
            "method": method,
        }

    # ------------------------------------------------------------------
    # Robust mode
    # ------------------------------------------------------------------

    def _compare_robust(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
    ) -> dict[str, Any]:
        """
        Full robust pipeline: align → normalize → multi-scale detect.
        """
        # Resolve configs
        if self._app_cfg is not None:
            align_cfg = self._app_cfg.alignment
            norm_cfg = self._app_cfg.normalization
            ms_cfg = self._app_cfg.multiscale
        else:
            # Legacy dict mode
            from .config import AlignmentConfig, NormalizationConfig, MultiscaleConfig
            raw = getattr(self, "_legacy_cfg", {})
            align_cfg = AlignmentConfig(**{
                k: v for k, v in raw.get("alignment", {}).items()
                if k in AlignmentConfig.__dataclass_fields__
            })
            norm_cfg = NormalizationConfig(**{
                k: v for k, v in raw.get("normalization", {}).items()
                if k in NormalizationConfig.__dataclass_fields__
            })
            ms_cfg = MultiscaleConfig(**{
                k: v for k, v in raw.get("multiscale", {}).items()
                if k in MultiscaleConfig.__dataclass_fields__
            })

        # Step 1: Align
        logger.info("Robust pipeline — Step 1: Alignment")
        aligner = ImageAligner(config=align_cfg)
        alignment = aligner.align(img_a, img_b)
        img_b_aligned = alignment.warped
        valid_mask = alignment.valid_mask

        # Step 2: Normalize
        logger.info("Robust pipeline — Step 2: Normalization")
        normalizer = PhotoNormalizer(config=norm_cfg)
        img_a_norm, img_b_norm = normalizer.normalize_pair(
            img_a, img_b_aligned, method=norm_cfg.method,
        )

        # Step 3: Multi-scale detection
        logger.info("Robust pipeline — Step 3: Multi-scale detection")
        ms_detector = MultiScaleDetector(config=ms_cfg)
        ms_result = ms_detector.detect(img_a_norm, img_b_norm, valid_mask)

        # Step 4: SSIM on aligned, normalized images
        gray_a = cv2.cvtColor(img_a_norm, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b_norm, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray_a, gray_b, full=True)

        # Build visualization
        diff_image = self._build_diff_image(
            img_a, img_b_aligned, ms_result["mask"], ms_result["bboxes"],
        )

        logger.info(
            "Robust detection complete: score=%.4f, change=%.1f%%, regions=%d",
            score, ms_result["change_percent"], len(ms_result["bboxes"]),
        )

        return {
            "diff_image": diff_image,
            "mask": ms_result["mask"],
            "contours": ms_result["contours"],
            "bboxes": ms_result["bboxes"],
            "change_percent": ms_result["change_percent"],
            "score": round(float(score), 4),
            "method": "robust",
            "alignment": {
                "success": alignment.success,
                "num_matches": alignment.num_matches,
                "num_inliers": alignment.num_inliers,
                "confidence": round(alignment.confidence, 4),
            },
        }

    # ------------------------------------------------------------------
    # Detection strategies
    # ------------------------------------------------------------------

    def _compare_drone(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
    ) -> dict[str, Any]:
        """
        Optimised pipeline for vertical/nadir drone imagery with scale variation.

        Steps:
            1. Similarity transform alignment (scale + translation only).
            2. Photometric normalisation.
            3. Combined pixel+SSIM diff masked to valid region.

        The ``scale_factor`` in the result is the altitude ratio B/A.
        """
        from dataclasses import replace as dc_replace

        if self._app_cfg is not None:
            align_cfg = self._app_cfg.alignment
            norm_cfg = self._app_cfg.normalization
        else:
            from .config import AlignmentConfig, NormalizationConfig
            raw = getattr(self, "_legacy_cfg", {})
            align_cfg = AlignmentConfig(**{
                k: v for k, v in raw.get("alignment", {}).items()
                if k in AlignmentConfig.__dataclass_fields__
            })
            norm_cfg = NormalizationConfig(**{
                k: v for k, v in raw.get("normalization", {}).items()
                if k in NormalizationConfig.__dataclass_fields__
            })

        # Force similarity transform for drone mode
        align_cfg = dc_replace(align_cfg, use_similarity_transform=True)

        # Step 1: Similarity alignment
        logger.info("Drone pipeline — Step 1: Similarity alignment (scale+translation)")
        aligner = ImageAligner(config=align_cfg)
        alignment = aligner.align(img_a, img_b)
        img_b_aligned = alignment.warped
        valid_mask = alignment.valid_mask

        logger.info(
            "Scale factor B→A: %.4fx (altitude ratio ~%.2fx)",
            alignment.scale_factor,
            1.0 / alignment.scale_factor if alignment.scale_factor > 0 else 0,
        )

        # Step 2: Photometric normalisation
        logger.info("Drone pipeline — Step 2: Photometric normalisation")
        normalizer = PhotoNormalizer(config=norm_cfg)
        img_a_norm, img_b_norm = normalizer.normalize_pair(
            img_a, img_b_aligned, method=norm_cfg.method,
        )

        # Step 3: Combined diff masked to valid region
        logger.info("Drone pipeline — Step 3: Combined diff")
        gray_a = cv2.cvtColor(img_a_norm, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b_norm, cv2.COLOR_BGR2GRAY)

        pixel_mask, _ = self._pixel_diff(gray_a, gray_b)
        ssim_mask, score = self._ssim_diff(gray_a, gray_b)
        combined = cv2.bitwise_or(pixel_mask, ssim_mask)

        if valid_mask is not None:
            valid_binary = (valid_mask > 0).astype(np.uint8)
            combined = cv2.bitwise_and(combined, combined, mask=valid_binary)

        contours, bboxes = self._find_changed_regions(combined)
        diff_image = self._build_diff_image(img_a, img_b_aligned, combined, bboxes)

        total_valid = int(np.sum(valid_mask > 0)) if valid_mask is not None else (
            combined.shape[0] * combined.shape[1]
        )
        change_percent = (np.sum(combined > 0) / max(total_valid, 1)) * 100

        logger.info(
            "Drone detection complete: score=%.4f, change=%.1f%%, regions=%d, scale=%.4f",
            score, change_percent, len(bboxes), alignment.scale_factor,
        )

        return {
            "diff_image": diff_image,
            "mask": combined,
            "contours": contours,
            "bboxes": bboxes,
            "change_percent": round(float(change_percent), 2),
            "score": round(float(score), 4),
            "method": "drone",
            "alignment": {
                "success": alignment.success,
                "num_matches": alignment.num_matches,
                "num_inliers": alignment.num_inliers,
                "confidence": round(alignment.confidence, 4),
                "scale_factor": round(alignment.scale_factor, 4),
            },
        }


    def _pixel_diff(
        self, gray_a: np.ndarray, gray_b: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        diff = cv2.absdiff(gray_a, gray_b)
        k = self._det_cfg.blur_kernel
        diff = cv2.GaussianBlur(diff, (k, k), 0)
        _, mask = cv2.threshold(diff, self._det_cfg.threshold, 255, cv2.THRESH_BINARY)
        score, _ = ssim(gray_a, gray_b, full=True)
        return mask, float(score)

    def _ssim_diff(
        self, gray_a: np.ndarray, gray_b: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        score, diff_map = ssim(gray_a, gray_b, full=True)
        diff_map = (1 - diff_map) / 2
        diff_map = (diff_map * 255).astype(np.uint8)
        k = self._det_cfg.blur_kernel
        diff_map = cv2.GaussianBlur(diff_map, (k, k), 0)
        _, mask = cv2.threshold(diff_map, self._det_cfg.threshold, 255, cv2.THRESH_BINARY)
        return mask, float(score)

    def _combined_diff(
        self, gray_a: np.ndarray, gray_b: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        pixel_mask, _ = self._pixel_diff(gray_a, gray_b)
        ssim_mask, score = self._ssim_diff(gray_a, gray_b)
        combined = cv2.bitwise_or(pixel_mask, ssim_mask)
        return combined, score

    # ------------------------------------------------------------------
    # Contour / region extraction
    # ------------------------------------------------------------------

    def _find_changed_regions(
        self, mask: np.ndarray,
    ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
        """Find changed regions after morphological cleanup."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        total_area = mask.shape[0] * mask.shape[1]
        max_area = total_area * self._det_cfg.max_contour_area_ratio

        bboxes: list[tuple[int, int, int, int]] = []
        valid_contours: list[np.ndarray] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= self._det_cfg.min_contour_area and area <= max_area:
                valid_contours.append(c)
                bboxes.append(cv2.boundingRect(c))

        return valid_contours, bboxes

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _build_diff_image(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        mask: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Create a side-by-side diff visualization."""
        vis_a = img_a.copy()
        vis_b = img_b.copy()

        for x, y, w, h in bboxes:
            cv2.rectangle(vis_a, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(vis_b, (x, y), (x + w, y + h), (0, 255, 0), 2)

        overlay = cv2.merge([mask, np.zeros_like(mask), np.zeros_like(mask)])
        vis_a = cv2.addWeighted(vis_a, 1.0, overlay, 0.3, 0)
        vis_b = cv2.addWeighted(vis_b, 1.0, overlay, 0.3, 0)

        return np.hstack([vis_a, vis_b])
