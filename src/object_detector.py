"""
object_detector.py
------------------
Object-level change detection using YOLOv8.

Detects objects in each image independently, then compares the object
sets to identify:

    - **Added** objects   (appear in B but not A)
    - **Removed** objects (appear in A but not B)
    - **Moved** objects   (same class, different location)
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from .config import ObjectDetectorConfig
from .exceptions import ModelLoadError
from .logging_config import get_logger
from .utils import validate_image

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Detection:
    """A single YOLO detection."""

    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    center: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        x, y, w, h = self.bbox
        # frozen dataclass — use object.__setattr__
        object.__setattr__(self, "center", (x + w / 2, y + h / 2))


@dataclass(frozen=True)
class ObjectChange:
    """A detected change between two images at the object level."""

    change_type: str          # 'added' | 'removed' | 'moved'
    label: str
    confidence: float
    bbox_a: tuple[int, int, int, int] | None
    bbox_b: tuple[int, int, int, int] | None


# ---------------------------------------------------------------------------
# ObjectDetector
# ---------------------------------------------------------------------------

class ObjectDetector:
    """
    Uses YOLOv8 (via ``ultralytics``) to detect objects in two images
    and compute object-level differences.

    Args:
        config: An :class:`ObjectDetectorConfig` instance.  If ``None``,
                defaults are used.  Legacy keyword arguments are accepted
                for backwards compatibility.
    """

    def __init__(
        self,
        config: ObjectDetectorConfig | None = None,
        *,
        model_name: str | None = None,
        confidence: float | None = None,
        iou_threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        if config is not None:
            self._cfg = config
        else:
            kwargs: dict = {}
            if model_name is not None:
                kwargs["model"] = model_name
            if confidence is not None:
                kwargs["confidence"] = confidence
            if iou_threshold is not None:
                kwargs["iou_threshold"] = iou_threshold
            if device is not None:
                kwargs["device"] = device
            self._cfg = ObjectDetectorConfig(**kwargs)

        self._model: Any = None
        self._sahi_model: Any = None

        logger.debug(
            "ObjectDetector: model=%s, conf=%.2f, iou=%.2f, device=%s",
            self._cfg.model, self._cfg.confidence,
            self._cfg.iou_threshold, self._cfg.device,
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def model(self) -> Any:
        """Lazy-load the YOLO model on first use."""
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ModelLoadError(
                    self._cfg.model,
                    "ultralytics is required for object detection. "
                    "Install with: pip install ultralytics",
                ) from exc

            logger.info("Loading %s.pt ...", self._cfg.model)
            try:
                self._model = YOLO(f"{self._cfg.model}.pt")
            except Exception as exc:
                raise ModelLoadError(self._cfg.model, str(exc)) from exc

        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run YOLO inference on a single image."""
        validate_image(image, "ObjectDetector.detect")

        if self._cfg.use_sahi:
            detections = self._detect_with_sahi(image)
            if detections:
                logger.debug("Detected %d objects (SAHI)", len(detections))
                return detections

        results = self.model.predict(
            source=image,
            conf=self._cfg.confidence,
            device=self._cfg.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                label = result.names[int(box.cls[0])]
                conf = float(box.conf[0])
                detections.append(
                    Detection(label=label, confidence=conf, bbox=(x1, y1, w, h))
                )

        logger.debug("Detected %d objects", len(detections))
        return detections

    def _detect_with_sahi(self, image: np.ndarray) -> list[Detection]:
        """Run sliced inference for better small-object recall."""
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
        except ImportError:
            logger.warning(
                "SAHI requested but not installed. Falling back to regular YOLO inference. "
                "Install with: pip install sahi"
            )
            return []

        if self._sahi_model is None:
            # Use explicit weight path if present to avoid network fetch.
            import os

            weight_path = f"{self._cfg.model}.pt"
            model_path = weight_path if os.path.exists(weight_path) else self._cfg.model
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=model_path,
                confidence_threshold=self._cfg.confidence,
                device=self._cfg.device,
            )

        result = get_sliced_prediction(
            image,
            self._sahi_model,
            slice_height=self._cfg.sahi_slice_height,
            slice_width=self._cfg.sahi_slice_width,
            overlap_height_ratio=self._cfg.sahi_overlap_ratio,
            overlap_width_ratio=self._cfg.sahi_overlap_ratio,
            verbose=0,
        )

        detections: list[Detection] = []
        for pred in result.object_prediction_list:
            x1 = int(pred.bbox.minx)
            y1 = int(pred.bbox.miny)
            x2 = int(pred.bbox.maxx)
            y2 = int(pred.bbox.maxy)
            w, h = x2 - x1, y2 - y1
            detections.append(
                Detection(
                    label=str(pred.category.name),
                    confidence=float(pred.score.value),
                    bbox=(x1, y1, w, h),
                )
            )
        return detections

    def compare(
        self,
        image_a: np.ndarray,
        image_b: np.ndarray,
        *,
        alignment_confidence: float | None = None,
        min_alignment_confidence: float = 0.2,
        enforce_cross_view_guard: bool = True,
        reconcile_cross_label: bool = False,
    ) -> dict[str, Any]:
        """
        Detect objects in both images and return object-level changes.

        Returns:
            Dict with ``detections_a``, ``detections_b``, ``changes``,
            ``annotated_a``, ``annotated_b``, ``summary``.
        """
        validate_image(image_a, "ObjectDetector.compare:image_a")
        validate_image(image_b, "ObjectDetector.compare:image_b")

        dets_a = self.detect(image_a)
        dets_b = self.detect(image_b)

        changes = self._compute_changes(
            dets_a,
            dets_b,
            reconcile_cross_label=reconcile_cross_label,
        )
        guard_info = {
            "applied": False,
            "suppressed_count": 0,
            "alignment_confidence": alignment_confidence,
            "min_alignment_confidence": min_alignment_confidence,
            "reason": "",
        }

        # Cross-view safety guard:
        # if global alignment is weak, suppress hard add/remove claims because
        # they are often false positives from viewpoint changes.
        if (
            enforce_cross_view_guard
            and alignment_confidence is not None
            and alignment_confidence < min_alignment_confidence
        ):
            suppressed_count = sum(
                1 for c in changes if c.change_type in ("added", "removed")
            )
            if suppressed_count > 0:
                changes = [
                    c for c in changes if c.change_type not in ("added", "removed")
                ]
                guard_info["applied"] = True
                guard_info["suppressed_count"] = suppressed_count
                guard_info["reason"] = "low_alignment_confidence"
                logger.info(
                    "Cross-view guard applied: suppressed %d add/remove change(s) "
                    "(alignment %.2f < %.2f)",
                    suppressed_count,
                    alignment_confidence,
                    min_alignment_confidence,
                )

        annotated_a = self._annotate(image_a.copy(), dets_a, color=(0, 200, 255))
        annotated_b = self._annotate(image_b.copy(), dets_b, color=(0, 255, 100))

        summary = {
            "added": sum(1 for c in changes if c.change_type == "added"),
            "removed": sum(1 for c in changes if c.change_type == "removed"),
            "moved": sum(1 for c in changes if c.change_type == "moved"),
            "total_changes": len(changes),
        }

        logger.info(
            "Object comparison: +%d added, -%d removed, ~%d moved",
            summary["added"], summary["removed"], summary["moved"],
        )

        return {
            "detections_a": dets_a,
            "detections_b": dets_b,
            "changes": changes,
            "annotated_a": annotated_a,
            "annotated_b": annotated_b,
            "summary": summary,
            "guard": guard_info,
        }

    # ------------------------------------------------------------------
    # Change logic — two-pass matching
    # ------------------------------------------------------------------

    def _compute_changes(
        self,
        dets_a: list[Detection],
        dets_b: list[Detection],
        *,
        reconcile_cross_label: bool = False,
    ) -> list[ObjectChange]:
        """
        Two-pass object matching:

        1. IoU matching (strict geometric overlap)
        2. Center-distance fallback for different angles / scales
        """
        changes: list[ObjectChange] = []
        matched_b: set[int] = set()
        matched_a: set[int] = set()

        max_dist = self._cfg.max_center_distance

        # --- Pass 1: IoU matching ---
        for idx_a, det_a in enumerate(dets_a):
            best_iou = 0.0
            best_match: int | None = None
            for i, det_b in enumerate(dets_b):
                if det_b.label != det_a.label or i in matched_b:
                    continue
                iou = self._iou(det_a.bbox, det_b.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_match = i

            if best_match is not None and best_iou >= self._cfg.iou_threshold:
                matched_b.add(best_match)
                matched_a.add(idx_a)
                det_b = dets_b[best_match]
                dist = self._center_distance(det_a.center, det_b.center)
                if dist > 20:
                    changes.append(ObjectChange(
                        change_type="moved",
                        label=det_a.label,
                        confidence=max(det_a.confidence, det_b.confidence),
                        bbox_a=det_a.bbox,
                        bbox_b=det_b.bbox,
                    ))
                # else: same position — no change

        # --- Pass 2: Center-distance fallback ---
        for idx_a, det_a in enumerate(dets_a):
            if idx_a in matched_a:
                continue
            best_dist = float("inf")
            best_match = None
            for i, det_b in enumerate(dets_b):
                if det_b.label != det_a.label or i in matched_b:
                    continue
                dist = self._center_distance(det_a.center, det_b.center)
                if dist < best_dist:
                    best_dist = dist
                    best_match = i

            if best_match is not None and best_dist <= max_dist:
                matched_b.add(best_match)
                matched_a.add(idx_a)
                det_b = dets_b[best_match]
                if best_dist > 20:
                    changes.append(ObjectChange(
                        change_type="moved",
                        label=det_a.label,
                        confidence=max(det_a.confidence, det_b.confidence),
                        bbox_a=det_a.bbox,
                        bbox_b=det_b.bbox,
                    ))
            # unmatched handled later after cross-label reconciliation

        # --- Pass 3: optional cross-label spatial reconciliation ---
        # Useful for severe viewpoint shifts causing label flips.
        if reconcile_cross_label:
            unmatched_a = [i for i in range(len(dets_a)) if i not in matched_a]
            unmatched_b = [i for i in range(len(dets_b)) if i not in matched_b]

            relabel_a, relabel_b = self._reconcile_cross_label_matches(
                dets_a, dets_b, unmatched_a, unmatched_b,
            )
            matched_a.update(relabel_a)
            matched_b.update(relabel_b)

        # Remaining unmatched in A -> removed
        for idx_a in range(len(dets_a)):
            if idx_a not in matched_a:
                det_a = dets_a[idx_a]
                changes.append(ObjectChange(
                    change_type="removed",
                    label=det_a.label,
                    confidence=det_a.confidence,
                    bbox_a=det_a.bbox,
                    bbox_b=None,
                ))

        # Remaining unmatched in B -> added
        for idx_b in range(len(dets_b)):
            if idx_b not in matched_b:
                det_b = dets_b[idx_b]
                changes.append(ObjectChange(
                    change_type="added",
                    label=det_b.label,
                    confidence=det_b.confidence,
                    bbox_a=None,
                    bbox_b=det_b.bbox,
                ))

        return changes

    def _reconcile_cross_label_matches(
        self,
        dets_a: list[Detection],
        dets_b: list[Detection],
        unmatched_a: list[int],
        unmatched_b: list[int],
    ) -> tuple[set[int], set[int]]:
        """Pair unmatched detections across labels when geometry strongly agrees."""
        if not unmatched_a or not unmatched_b:
            return set(), set()

        candidates: list[tuple[float, float, int, int]] = []
        for idx_a in unmatched_a:
            da = dets_a[idx_a]
            area_a = da.bbox[2] * da.bbox[3]
            for idx_b in unmatched_b:
                db = dets_b[idx_b]
                area_b = db.bbox[2] * db.bbox[3]
                iou = self._iou(da.bbox, db.bbox)
                dist = self._center_distance(da.center, db.center)
                size_ratio = (area_a / area_b) if area_b > 0 else 0.0

                strong_overlap = iou >= max(0.15, self._cfg.iou_threshold * 0.5)
                strong_proximity = (
                    dist <= max(30.0, self._cfg.max_center_distance * 0.35)
                    and 0.25 <= size_ratio <= 4.0
                )

                if strong_overlap or strong_proximity:
                    # Sort candidates: higher IoU first, then lower distance.
                    candidates.append((iou, dist, idx_a, idx_b))

        candidates.sort(key=lambda x: (-x[0], x[1]))

        matched_a: set[int] = set()
        matched_b: set[int] = set()

        for _, _, idx_a, idx_b in candidates:
            if idx_a in matched_a or idx_b in matched_b:
                continue
            matched_a.add(idx_a)
            matched_b.add(idx_b)
            if dets_a[idx_a].label != dets_b[idx_b].label:
                logger.info(
                    "Reconciled cross-label match: %s -> %s",
                    dets_a[idx_a].label,
                    dets_b[idx_b].label,
                )

        return matched_a, matched_b

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(
        bbox_a: tuple[int, int, int, int],
        bbox_b: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b

        inter_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        inter_y = max(0, min(ay + ah, by + bh) - max(ay, by))
        intersection = inter_x * inter_y

        union = aw * ah + bw * bh - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _center_distance(
        c1: tuple[float, float], c2: tuple[float, float],
    ) -> float:
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

    @staticmethod
    def _annotate(
        image: np.ndarray,
        detections: list[Detection],
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                image, label_text, (x, max(y - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
        return image
