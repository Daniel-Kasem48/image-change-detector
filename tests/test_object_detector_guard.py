"""
tests/test_object_detector_guard.py
-----------------------------------
Unit tests for cross-view safety guard in ObjectDetector.
"""

from __future__ import annotations

import numpy as np

from src.object_detector import Detection, ObjectDetector


def make_bgr(h: int = 100, w: int = 100, color: tuple = (100, 100, 100)) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_guard_suppresses_added_removed_when_alignment_is_low(monkeypatch) -> None:
    img = make_bgr()
    dets_a = [Detection(label="cup", confidence=0.9, bbox=(10, 10, 40, 40))]
    dets_b = []
    calls = {"n": 0}

    def fake_detect(_self, _image):
        calls["n"] += 1
        return dets_a if calls["n"] == 1 else dets_b

    monkeypatch.setattr(ObjectDetector, "detect", fake_detect)

    detector = ObjectDetector()
    result = detector.compare(
        img,
        img,
        alignment_confidence=0.1,
        min_alignment_confidence=0.2,
    )

    assert result["summary"]["total_changes"] == 0
    assert result["guard"]["applied"] is True
    assert result["guard"]["suppressed_count"] == 1


def test_guard_keeps_added_removed_when_alignment_is_high(monkeypatch) -> None:
    img = make_bgr()
    dets_a = [Detection(label="cup", confidence=0.9, bbox=(10, 10, 40, 40))]
    dets_b = []
    calls = {"n": 0}

    def fake_detect(_self, _image):
        calls["n"] += 1
        return dets_a if calls["n"] == 1 else dets_b

    monkeypatch.setattr(ObjectDetector, "detect", fake_detect)

    detector = ObjectDetector()
    result = detector.compare(
        img,
        img,
        alignment_confidence=0.9,
        min_alignment_confidence=0.2,
    )

    assert result["summary"]["removed"] == 1
    assert result["summary"]["total_changes"] == 1
    assert result["guard"]["applied"] is False


def test_cross_label_spatial_reconciliation_suppresses_false_add_remove(monkeypatch) -> None:
    img = make_bgr()
    dets_a = [Detection(label="mouse", confidence=0.8, bbox=(100, 100, 80, 80))]
    dets_b = [Detection(label="toilet", confidence=0.8, bbox=(105, 105, 82, 82))]
    calls = {"n": 0}

    def fake_detect(_self, _image):
        calls["n"] += 1
        return dets_a if calls["n"] == 1 else dets_b

    monkeypatch.setattr(ObjectDetector, "detect", fake_detect)

    detector = ObjectDetector()
    result = detector.compare(
        img,
        img,
        alignment_confidence=0.9,
        min_alignment_confidence=0.2,
        reconcile_cross_label=True,
    )

    assert result["summary"]["added"] == 0
    assert result["summary"]["removed"] == 0
    assert result["summary"]["total_changes"] == 0


def test_cross_label_reconciliation_is_disabled_by_default(monkeypatch) -> None:
    img = make_bgr()
    dets_a = [Detection(label="mouse", confidence=0.8, bbox=(100, 100, 80, 80))]
    dets_b = [Detection(label="toilet", confidence=0.8, bbox=(105, 105, 82, 82))]
    calls = {"n": 0}

    def fake_detect(_self, _image):
        calls["n"] += 1
        return dets_a if calls["n"] == 1 else dets_b

    monkeypatch.setattr(ObjectDetector, "detect", fake_detect)

    detector = ObjectDetector()
    result = detector.compare(
        img,
        img,
        alignment_confidence=0.9,
        min_alignment_confidence=0.2,
    )

    assert result["summary"]["added"] == 1
    assert result["summary"]["removed"] == 1
    assert result["summary"]["total_changes"] == 2
