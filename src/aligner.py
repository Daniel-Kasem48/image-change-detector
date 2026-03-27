"""
aligner.py
----------
Feature-based image registration / alignment.

Handles images taken from different cameras, angles, scales, and drones by:

1. Detecting keypoints (SIFT or ORB)
2. Matching features (FLANN or BruteForce + Lowe's ratio test)
3. Computing homography via RANSAC
4. Warping image B into image A's perspective
5. Creating a valid-region mask (to ignore warped borders)
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Sequence

from .config import AlignmentConfig
from .exceptions import AlignmentError
from .logging_config import get_logger
from .utils import validate_image

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AlignmentResult:
    """Result of image alignment."""

    warped: np.ndarray          # image B warped to A's perspective
    homography: np.ndarray      # 3×3 homography matrix (or embedded 2×3 affine)
    valid_mask: np.ndarray      # binary mask of valid (non-border) region
    num_matches: int            # total feature matches found
    num_inliers: int            # RANSAC inlier count
    confidence: float           # inlier_ratio = inliers / matches
    success: bool               # True if enough matches were found
    scale_factor: float = 1.0   # estimated scale B→A (>1 means B is zoomed out)


# ---------------------------------------------------------------------------
# ImageAligner
# ---------------------------------------------------------------------------

class ImageAligner:
    """
    Aligns image B to match image A's perspective using feature-based
    homography estimation.

    Handles:
        - Different camera angles
        - Different scales / zoom levels
        - Rotation & translation
        - Drone altitude differences

    Args:
        config: An :class:`AlignmentConfig` instance.  If ``None``, defaults
                are used.  Legacy keyword arguments are also accepted.
    """

    def __init__(
        self,
        config: AlignmentConfig | None = None,
        *,
        # Legacy kwargs for backwards compatibility
        feature_type: str | None = None,
        max_features: int | None = None,
        good_match_ratio: float | None = None,
        ransac_thresh: float | None = None,
        min_matches: int | None = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        else:
            kwargs: dict = {}
            if feature_type is not None:
                kwargs["feature_type"] = feature_type
            if max_features is not None:
                kwargs["max_features"] = max_features
            if good_match_ratio is not None:
                kwargs["good_match_ratio"] = good_match_ratio
            if ransac_thresh is not None:
                kwargs["ransac_thresh"] = ransac_thresh
            if min_matches is not None:
                kwargs["min_matches"] = min_matches
            self._cfg = AlignmentConfig(**kwargs)

        # Create detector + matcher
        if self._cfg.feature_type == "sift":
            self._detector = cv2.SIFT_create(nfeatures=self._cfg.max_features)
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            self._matcher = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            self._detector = cv2.ORB_create(nfeatures=self._cfg.max_features)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        logger.debug(
            "ImageAligner initialised: feature=%s, max_features=%d, ratio=%.2f",
            self._cfg.feature_type,
            self._cfg.max_features,
            self._cfg.good_match_ratio,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
    ) -> AlignmentResult:
        """
        Align *img_b* to *img_a*'s perspective.

        Returns:
            An :class:`AlignmentResult` with the warped image, homography
            matrix, valid-region mask, and quality metrics.
        """
        validate_image(img_a, "aligner:img_a")
        validate_image(img_b, "aligner:img_b")

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        # 1. Detect keypoints & compute descriptors
        kp_a, desc_a = self._detector.detectAndCompute(gray_a, None)
        kp_b, desc_b = self._detector.detectAndCompute(gray_b, None)

        if desc_a is None or desc_b is None or len(kp_a) < 4 or len(kp_b) < 4:
            logger.warning("Insufficient keypoints for alignment (A=%s, B=%s)",
                           len(kp_a) if kp_a else 0, len(kp_b) if kp_b else 0)
            return self._fail_result(img_b)

        # 2. Match features using Lowe's ratio test
        good_matches = self._match_features(desc_a, desc_b)

        logger.info(
            "Keypoints: A=%d, B=%d | Good matches: %d",
            len(kp_a), len(kp_b), len(good_matches),
        )

        if len(good_matches) < self._cfg.min_matches:
            logger.warning(
                "Not enough matches (%d < %d) — returning unwarped image",
                len(good_matches), self._cfg.min_matches,
            )
            return self._fail_result(img_b, num_matches=len(good_matches))

        # 3. Extract matched point coordinates
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 4. Compute transform with RANSAC
        if self._cfg.use_similarity_transform:
            # Similarity transform: scale + translation only.
            # Correct for nadir/vertical drone imagery where only altitude varies.
            M, mask = cv2.estimateAffinePartial2D(
                pts_b, pts_a,
                method=cv2.RANSAC,
                ransacReprojThreshold=self._cfg.ransac_thresh,
            )
            if M is None:
                logger.warning("Similarity transform estimation failed")
                return self._fail_result(img_b, num_matches=len(good_matches))

            scale_factor = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
            logger.info("Similarity transform: scale=%.4f", scale_factor)

            H = np.eye(3)
            H[:2, :] = M
            num_inliers = int(mask.sum()) if mask is not None else 0
            confidence = num_inliers / len(good_matches) if len(good_matches) > 0 else 0.0
            logger.info("Inliers: %d/%d (confidence: %.2f%%)",
                        num_inliers, len(good_matches), confidence * 100)
            h, w = img_a.shape[:2]
            warped = cv2.warpAffine(img_b, M, (w, h))
        else:
            # Full homography: handles perspective, rotation, shear.
            H, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, self._cfg.ransac_thresh)
            if H is None:
                logger.warning("Homography estimation failed")
                return self._fail_result(img_b, num_matches=len(good_matches))
            scale_factor = float(np.sqrt(abs(np.linalg.det(H[:2, :2]))))
            num_inliers = int(mask.sum()) if mask is not None else 0
            confidence = num_inliers / len(good_matches) if len(good_matches) > 0 else 0.0
            logger.info("Inliers: %d/%d (confidence: %.2f%%)",
                        num_inliers, len(good_matches), confidence * 100)
            h, w = img_a.shape[:2]
            warped = cv2.warpPerspective(img_b, H, (w, h))

        # 6. Create valid region mask
        valid_mask = self._compute_valid_mask(warped, h, w)

        return AlignmentResult(
            warped=warped,
            homography=H,
            valid_mask=valid_mask,
            num_matches=len(good_matches),
            num_inliers=num_inliers,
            confidence=confidence,
            success=True,
            scale_factor=scale_factor,
        )

    def draw_matches(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        max_display: int = 50,
    ) -> np.ndarray:
        """Visualize feature matches between two images (for debugging)."""
        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        kp_a, desc_a = self._detector.detectAndCompute(gray_a, None)
        kp_b, desc_b = self._detector.detectAndCompute(gray_b, None)

        if desc_a is None or desc_b is None:
            return np.hstack([img_a, img_b])

        good_matches = self._match_features(desc_a, desc_b)[:max_display]

        vis = cv2.drawMatches(
            img_a, kp_a, img_b, kp_b, good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return vis

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _match_features(
        self, desc_a: np.ndarray, desc_b: np.ndarray,
    ) -> list[cv2.DMatch]:
        """Match descriptors using knnMatch + Lowe's ratio test."""
        if self._cfg.feature_type == "sift":
            desc_a = np.float32(desc_a)
            desc_b = np.float32(desc_b)

        try:
            raw_matches = self._matcher.knnMatch(desc_a, desc_b, k=2)
        except cv2.error as exc:
            logger.warning("Feature matching failed: %s", exc)
            return []

        good: list[cv2.DMatch] = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self._cfg.good_match_ratio * n.distance:
                    good.append(m)

        good.sort(key=lambda x: x.distance)
        return good

    def _compute_valid_mask(
        self, warped: np.ndarray, h: int, w: int,
    ) -> np.ndarray:
        """
        Create a mask of valid (non-border) pixels after perspective warp.
        """
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.erode(mask, kernel, iterations=1)
        return mask

    def _fail_result(
        self, img_b: np.ndarray, num_matches: int = 0,
    ) -> AlignmentResult:
        """Return a failed-alignment result (no warping applied)."""
        h, w = img_b.shape[:2]
        return AlignmentResult(
            warped=img_b.copy(),
            homography=np.eye(3),
            valid_mask=np.ones((h, w), dtype=np.uint8) * 255,
            num_matches=num_matches,
            num_inliers=0,
            confidence=0.0,
            success=False,
            scale_factor=1.0,
        )
