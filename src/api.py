"""
api.py
------
Clean programmatic API for the image-change-detector library.

This module is the **recommended entry point** when using the package
as a library (not via CLI).  All functions return JSON-serializable
dicts and accept simple Python types.

Usage::

    from src.api import compare_images, compare_objects

    # Pixel / SSIM / combined / robust comparison
    result = compare_images("before.jpg", "after.jpg", method="combined")
    print(result["change_percent"])

    # Object-level comparison with YOLO
    result = compare_objects("before.jpg", "after.jpg")
    for change in result["changes"]:
        print(change)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .config import AppConfig, load_config
from .detector import ChangeDetector
from .logging_config import get_logger
from .object_detector import ObjectDetector, ObjectChange
from .utils import (
    load_image,
    resize_to_match,
    result_to_json,
    save_result,
    validate_image,
)

logger = get_logger(__name__)


def compare_images(
    image_a: str | Path,
    image_b: str | Path,
    *,
    method: str = "combined",
    config_path: str | Path | None = "config.yaml",
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compare two images for pixel/structural changes.

    Args:
        image_a: Path to the reference image.
        image_b: Path to the comparison image.
        method: Detection mode — ``pixel``, ``ssim``, ``combined``,
                or ``robust``.
        config_path: Path to ``config.yaml``.  ``None`` uses defaults.
        output_path: If provided, saves the diff visualization.

    Returns:
        A JSON-serializable dict with ``method``, ``ssim_score``,
        ``change_percent``, ``num_regions``, ``bboxes``, and optionally
        ``alignment`` metadata.
    """
    cfg = load_config(config_path)

    img_a = load_image(image_a)
    img_b = load_image(image_b)

    detector = ChangeDetector(app_config=cfg)
    result = detector.compare(img_a, img_b, method=method)

    if output_path:
        save_result(result["diff_image"], output_path)

    return result_to_json(result)


def compare_objects(
    image_a: str | Path,
    image_b: str | Path,
    *,
    config_path: str | Path | None = "config.yaml",
    output_path: str | Path | None = None,
    align_first: bool = True,
) -> dict[str, Any]:
    """
    Compare two images for object-level changes using YOLO.

    Args:
        image_a: Path to the reference image.
        image_b: Path to the comparison image.
        config_path: Path to ``config.yaml``.
        output_path: If provided, saves the annotated side-by-side image.
        align_first: Whether to align images before YOLO (recommended
                     for different camera angles).

    Returns:
        A JSON-serializable dict with ``summary`` and ``changes``.
    """
    cfg = load_config(config_path)

    img_a = load_image(image_a)
    img_b = load_image(image_b)

    img_a_r, img_b_r = resize_to_match(img_a, img_b)

    # Optionally align before object detection
    img_b_for_compare = img_b_r
    alignment_confidence: float | None = None
    if align_first:
        try:
            from .aligner import ImageAligner

            aligner = ImageAligner(config=cfg.alignment)
            alignment = aligner.align(img_a_r, img_b_r)
            alignment_confidence = alignment.confidence

            if alignment.success and alignment.confidence > 0.15:
                logger.info("Pre-aligned images (confidence: %.1f%%)",
                            alignment.confidence * 100)
                img_b_for_compare = alignment.warped
            else:
                logger.info("Alignment confidence too low — comparing without alignment")
        except Exception:
            logger.warning("Alignment failed — comparing without alignment")

    if cfg.object_detector.class_agnostic_output:
        import cv2

        detector = ChangeDetector(app_config=cfg)
        region_result = detector.compare(img_a_r, img_b_for_compare, method="combined")
        bboxes = region_result.get("bboxes", [])
        changes = [
            ObjectChange(
                change_type="changed",
                label="unknown_object",
                confidence=1.0,
                bbox_a=b,
                bbox_b=b,
            )
            for b in bboxes
        ]
        annotated_a = img_a_r.copy()
        annotated_b = img_b_for_compare.copy()
        for x, y, w, h in bboxes:
            cv2.rectangle(annotated_a, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cv2.rectangle(annotated_b, (x, y), (x + w, y + h), (0, 255, 100), 2)

        result = {
            "detections_a": [],
            "detections_b": [],
            "changes": changes,
            "annotated_a": annotated_a,
            "annotated_b": annotated_b,
            "summary": {
                "added": 0,
                "removed": 0,
                "moved": 0,
                "changed": len(changes),
                "total_changes": len(changes),
            },
            "class_agnostic": True,
        }
    else:
        obj_detector = ObjectDetector(config=cfg.object_detector)
        result = obj_detector.compare(
            img_a_r,
            img_b_for_compare,
            alignment_confidence=alignment_confidence,
        )

    if output_path:
        side_by_side = cv2.hconcat([result["annotated_a"], result["annotated_b"]])
        save_result(side_by_side, output_path)

    return result_to_json(result)
