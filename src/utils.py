"""
utils.py
--------
Image I/O, preprocessing, visualization, and reporting utilities.

All functions include input validation, proper error handling, and
structured logging.  No bare ``print()`` calls except in
:func:`print_report` which is the user-facing console output.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .exceptions import ImageLoadError, ImageSaveError
from .logging_config import get_logger

logger = get_logger(__name__)

# Maximum image dimension (pixels) before automatic down-scaling
_MAX_DIMENSION: int = 8192


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    """
    Load an image from disk as a BGR ``numpy`` array (OpenCV convention).

    Supports JPEG, PNG, BMP, TIFF, WEBP.

    Args:
        path: Path to the image file.

    Returns:
        BGR ``uint8`` array with shape ``(H, W, 3)``.

    Raises:
        ImageLoadError: If the file does not exist, cannot be decoded,
            or has zero dimensions.
    """
    path = Path(path)

    if not path.exists():
        raise ImageLoadError(str(path), "file not found")

    if not path.is_file():
        raise ImageLoadError(str(path), "not a regular file")

    file_size = path.stat().st_size
    if file_size == 0:
        raise ImageLoadError(str(path), "file is empty (0 bytes)")

    image = cv2.imread(str(path))
    if image is None:
        raise ImageLoadError(str(path), "OpenCV could not decode the file")

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ImageLoadError(str(path), f"decoded image has zero dimensions ({w}×{h})")

    logger.debug("Loaded %s — %d×%d (%s)", path.name, w, h, _fmt_bytes(file_size))
    return image


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load an image as RGB (for Matplotlib / PIL usage)."""
    return cv2.cvtColor(load_image(path), cv2.COLOR_BGR2RGB)


def save_result(image: np.ndarray, output_path: str | Path) -> Path:
    """
    Save a BGR ``numpy`` array to disk.  Creates parent directories.

    Args:
        image: BGR image array.
        output_path: Destination path.

    Returns:
        Resolved :class:`~pathlib.Path` of the saved file.

    Raises:
        ImageSaveError: On write failure.
    """
    output_path = Path(output_path)

    validate_image(image, "save_result")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), image)
        if not success:
            raise ImageSaveError(str(output_path), "cv2.imwrite returned False")
    except OSError as exc:
        raise ImageSaveError(str(output_path), str(exc)) from exc

    logger.info("Saved → %s", output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_image(image: np.ndarray, context: str = "") -> None:
    """
    Validate that *image* is a usable image array.

    Raises:
        ValueError: On invalid input.
    """
    ctx = f" ({context})" if context else ""
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected numpy array{ctx}, got {type(image).__name__}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must be 2D or 3D{ctx}, got {image.ndim}D")
    if image.size == 0:
        raise ValueError(f"Image is empty (0 pixels){ctx}")


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale image to 3-channel BGR if needed."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def resize_to_match(
    img_a: np.ndarray,
    img_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize *img_b* to match *img_a*'s dimensions if they differ.

    Returns:
        ``(img_a, img_b_resized)`` — always the same spatial shape.
    """
    validate_image(img_a, "resize_to_match:img_a")
    validate_image(img_b, "resize_to_match:img_b")

    if img_a.shape[:2] == img_b.shape[:2]:
        return img_a, img_b

    h, w = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_AREA)
    logger.debug("Resized img_b from %s to %s", img_b.shape[:2], (h, w))
    return img_a, img_b_resized


def downscale_if_large(
    image: np.ndarray,
    max_dim: int = _MAX_DIMENSION,
) -> tuple[np.ndarray, float]:
    """
    Down-scale an image if any dimension exceeds *max_dim*.

    Returns:
        ``(scaled_image, scale_factor)`` where ``scale_factor <= 1.0``.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image, 1.0

    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info("Down-scaled %d×%d → %d×%d (factor %.2f)", w, h, new_w, new_h, scale)
    return scaled, scale


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize a grayscale image to the ``[0, 255]`` range."""
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def preprocess(image: np.ndarray, denoise: bool = False) -> np.ndarray:
    """
    Optional preprocessing pipeline: CLAHE contrast enhancement
    and optional Gaussian denoising.

    Returns a BGR image.
    """
    validate_image(image, "preprocess")
    image = ensure_bgr(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    if denoise:
        l_channel = cv2.GaussianBlur(l_channel, (3, 3), 0)

    lab = cv2.merge([l_channel, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparison(
    img_a: np.ndarray,
    img_b: np.ndarray,
    diff_image: np.ndarray,
    mask: np.ndarray,
    title: str = "Image Change Detection",
    save_path: str | None = None,
    dpi: int = 150,
    show: bool = True,
) -> None:
    """
    Display a 2×2 grid::

        [Image A]  [Image B]
        [Diff Map] [Change Mask]

    When *save_path* is given the figure is saved.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    axes[0, 0].imshow(cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Image A (Before)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Image B (After)")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Diff Visualization")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(mask, cmap="hot")
    axes[1, 1].set_title("Change Mask")
    axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Plot saved → %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_object_changes(
    annotated_a: np.ndarray,
    annotated_b: np.ndarray,
    changes: list[Any],
    save_path: str | None = None,
    dpi: int = 150,
    show: bool = True,
) -> None:
    """Display annotated images side-by-side with a change-summary legend."""
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Object-Level Change Detection", fontsize=14, fontweight="bold")

    axes[0].imshow(cv2.cvtColor(annotated_a, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image A — Detections")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(annotated_b, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image B — Detections")
    axes[1].axis("off")

    added = sum(1 for c in changes if c.change_type == "added")
    removed = sum(1 for c in changes if c.change_type == "removed")
    moved = sum(1 for c in changes if c.change_type == "moved")
    legend_patches = [
        mpatches.Patch(color="green", label=f"Added: {added}"),
        mpatches.Patch(color="red", label=f"Removed: {removed}"),
        mpatches.Patch(color="orange", label=f"Moved: {moved}"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=11)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Plot saved → %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_report(result: dict[str, Any]) -> str:
    """
    Build a human-readable report string from a detection result dict.

    Returns:
        Multi-line report string.
    """
    lines: list[str] = [
        "",
        "=" * 50,
        "  IMAGE CHANGE DETECTION REPORT",
        "=" * 50,
    ]

    if "method" in result:
        lines.append(f"  Method         : {result.get('method', 'N/A').upper()}")
        lines.append(f"  SSIM Score     : {result.get('score', 'N/A')} (1.0 = identical)")
        lines.append(f"  Change Area    : {result.get('change_percent', 'N/A')}% of image")
        lines.append(f"  Changed Regions: {len(result.get('bboxes', []))}")
    elif "summary" in result:
        s = result["summary"]
        lines.append(f"  Objects Added  : {s.get('added', 0)}")
        lines.append(f"  Objects Removed: {s.get('removed', 0)}")
        lines.append(f"  Objects Moved  : {s.get('moved', 0)}")
        lines.append(f"  Total Changes  : {s.get('total_changes', 0)}")
        lines.append("")
        for change in result.get("changes", []):
            lines.append(
                f"  [{change.change_type.upper():8s}] {change.label} "
                f"(conf: {change.confidence:.2f})"
            )

    lines.append("=" * 50)
    lines.append("")
    return "\n".join(lines)


def print_report(result: dict[str, Any]) -> None:
    """Pretty-print a change detection result to stdout."""
    print(format_report(result))


def result_to_json(result: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a detection result dict to a JSON-serializable dict.

    Strips numpy arrays and non-serializable objects, keeps only
    metrics and change descriptions.
    """
    output: dict[str, Any] = {}

    if "method" in result:
        output["method"] = result["method"]
        output["ssim_score"] = result.get("score")
        output["change_percent"] = result.get("change_percent")
        output["num_regions"] = len(result.get("bboxes", []))
        output["bboxes"] = [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            for x, y, w, h in result.get("bboxes", [])
        ]
        if "alignment" in result:
            output["alignment"] = result["alignment"]

    elif "summary" in result:
        output["summary"] = result["summary"]
        output["changes"] = [
            {
                "type": c.change_type,
                "label": c.label,
                "confidence": round(c.confidence, 4),
                "bbox_a": _bbox_dict(c.bbox_a),
                "bbox_b": _bbox_dict(c.bbox_b),
            }
            for c in result.get("changes", [])
        ]

    return output


def result_to_json_string(result: dict[str, Any], indent: int = 2) -> str:
    """Serialize a detection result to a JSON string."""
    return json.dumps(result_to_json(result), indent=indent)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _bbox_dict(bbox: tuple[int, ...] | None) -> dict[str, int] | None:
    if bbox is None:
        return None
    x, y, w, h = bbox
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"
