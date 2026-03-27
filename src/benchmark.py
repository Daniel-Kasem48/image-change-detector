"""
benchmark.py
------------
Dataset-level evaluation utilities for remote-sensing change detection.

The evaluator targets LEVIR-style folder layouts and computes binary
change metrics from predicted masks:

- precision / recall / F1
- IoU (Jaccard)
- accuracy

Expected dataset layout (split optional)::

    dataset_root/
      A/
      B/
      label/

or

    dataset_root/
      train/A train/B train/label
      val/A   val/B   val/label
      test/A  test/B  test/label
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import AppConfig
from .detector import ChangeDetector
from .logging_config import get_logger
from .utils import load_image

logger = get_logger(__name__)

_VALID_METHODS = frozenset({"pixel", "ssim", "combined", "robust"})


@dataclass(frozen=True)
class PairSample:
    """A matched (A, B, label) sample on disk."""

    sample_id: str
    image_a: Path
    image_b: Path
    label: Path


class DatasetLayoutError(ValueError):
    """Raised when the dataset directory does not match expected layout."""


class DatasetSampleError(ValueError):
    """Raised when there are no valid samples to evaluate."""


def discover_levir_samples(
    dataset_root: str | Path,
    *,
    split: str | None = None,
    dir_a: str = "A",
    dir_b: str = "B",
    dir_label: str = "label",
) -> list[PairSample]:
    """
    Discover paired samples from a LEVIR-style directory.

    Returns:
        A sorted list of :class:`PairSample`.
    """
    root = Path(dataset_root)
    base = root / split if split else root

    a_dir = base / dir_a
    b_dir = base / dir_b
    l_dir = base / dir_label

    missing = [str(p) for p in (a_dir, b_dir, l_dir) if not p.is_dir()]
    if missing:
        raise DatasetLayoutError(
            "Dataset layout not found. Expected directories: "
            f"{a_dir}, {b_dir}, {l_dir}. Missing: {', '.join(missing)}"
        )

    files_a = {p.stem: p for p in a_dir.iterdir() if p.is_file()}
    files_b = {p.stem: p for p in b_dir.iterdir() if p.is_file()}
    files_l = {p.stem: p for p in l_dir.iterdir() if p.is_file()}

    shared_ids = sorted(set(files_a) & set(files_b) & set(files_l))
    if not shared_ids:
        raise DatasetSampleError(
            "No shared sample IDs across A/B/label directories. "
            "Check file names and extensions."
        )

    samples = [
        PairSample(
            sample_id=sid,
            image_a=files_a[sid],
            image_b=files_b[sid],
            label=files_l[sid],
        )
        for sid in shared_ids
    ]

    logger.info("Discovered %d paired samples under %s", len(samples), base)
    return samples


def _load_binary_label(path: Path) -> np.ndarray:
    """Load a binary label mask where any non-zero pixel is change."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not decode label mask: {path}")
    return (mask > 0).astype(np.uint8)


def _mask_to_binary(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Convert a predicted uint8 mask to a binary mask with target shape."""
    if mask.ndim != 2:
        raise ValueError(f"Predicted mask must be 2D, got shape={mask.shape}")

    if mask.shape != target_shape:
        h, w = target_shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return (mask > 0).astype(np.uint8)


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _chunked(items: list[PairSample], batch_size: int) -> list[list[PairSample]]:
    """Split *items* into contiguous batches of size <= *batch_size*."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def _evaluate_single_sample(
    sample: PairSample,
    *,
    method: str,
    app_config: AppConfig | None,
) -> dict[str, Any]:
    """Evaluate one sample and return per-sample confusion counts + summary row."""
    img_a = load_image(sample.image_a)
    img_b = load_image(sample.image_b)
    gt = _load_binary_label(sample.label)

    detector = ChangeDetector(app_config=app_config)
    result = detector.compare(img_a, img_b, method=method)
    pred = _mask_to_binary(result["mask"], gt.shape)

    tp_i = int(np.sum((pred == 1) & (gt == 1)))
    tn_i = int(np.sum((pred == 0) & (gt == 0)))
    fp_i = int(np.sum((pred == 1) & (gt == 0)))
    fn_i = int(np.sum((pred == 0) & (gt == 1)))

    precision_i = _safe_div(tp_i, tp_i + fp_i)
    recall_i = _safe_div(tp_i, tp_i + fn_i)
    f1_i = _safe_div(2 * precision_i * recall_i, precision_i + recall_i)
    iou_i = _safe_div(tp_i, tp_i + fp_i + fn_i)

    return {
        "tp": tp_i,
        "tn": tn_i,
        "fp": fp_i,
        "fn": fn_i,
        "sample": {
            "sample_id": sample.sample_id,
            "precision": round(precision_i, 6),
            "recall": round(recall_i, 6),
            "f1": round(f1_i, 6),
            "iou": round(iou_i, 6),
            "change_percent_pred": round(float(np.mean(pred) * 100.0), 4),
            "change_percent_gt": round(float(np.mean(gt) * 100.0), 4),
        },
    }


def evaluate_binary_change_dataset(
    dataset_root: str | Path,
    *,
    split: str | None = None,
    method: str = "robust",
    app_config: AppConfig | None = None,
    limit: int | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    dir_a: str = "A",
    dir_b: str = "B",
    dir_label: str = "label",
) -> dict[str, Any]:
    """
    Evaluate detector outputs against binary change masks.

    Args:
        dataset_root: Root path of dataset.
        split: Optional split name (e.g. ``test``).
        method: Detector mode (pixel|ssim|combined|robust).
        app_config: Optional application config.
        limit: Optional max number of samples.
        batch_size: Number of pairs to process together as one batch.
        num_workers: Parallel workers per batch (0 = sequential).
        dir_a/dir_b/dir_label: Folder names for before/after/label.

    Returns:
        JSON-serializable metrics and per-sample summaries.
    """
    method = method.lower()
    if method not in _VALID_METHODS:
        raise ValueError(f"Invalid method {method!r}. Use one of {sorted(_VALID_METHODS)}")

    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    samples = discover_levir_samples(
        dataset_root,
        split=split,
        dir_a=dir_a,
        dir_b=dir_b,
        dir_label=dir_label,
    )
    if limit is not None:
        samples = samples[:limit]
    sample_batches = _chunked(samples, batch_size)

    tp = tn = fp = fn = 0
    sample_results: list[dict[str, Any]] = []

    processed = 0
    for batch in sample_batches:
        if num_workers > 0:
            def _eval(sample: PairSample) -> dict[str, Any]:
                return _evaluate_single_sample(sample, method=method, app_config=app_config)

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                batch_out = list(pool.map(_eval, batch))
        else:
            batch_out = [
                _evaluate_single_sample(s, method=method, app_config=app_config)
                for s in batch
            ]

        for out in batch_out:
            tp += out["tp"]
            tn += out["tn"]
            fp += out["fp"]
            fn += out["fn"]
            sample_results.append(out["sample"])

        processed += len(batch)
        if processed % 25 == 0 or processed == len(samples):
            logger.info("Evaluated %d/%d samples", processed, len(samples))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    metrics = {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "iou": round(iou, 6),
        "accuracy": round(accuracy, 6),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

    return {
        "task": "binary_change_detection",
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": split or "<root>",
        "method": method,
        "num_samples": len(samples),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "metrics": metrics,
        "samples": sample_results,
    }
