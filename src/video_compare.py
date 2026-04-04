"""
video_compare.py
----------------
Video-to-video comparison utilities.

Compares two videos (possibly different lengths), samples frames,
matches them by normalized timeline position, and saves only changed
frame results to an output folder.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import AppConfig
from .detector import ChangeDetector
from .logging_config import get_logger
from .utils import save_result

logger = get_logger(__name__)


def _format_timestamp_hms_ms(seconds: float) -> str:
    """Format seconds into HH-mm-ss-ms."""
    total_ms = max(0, int(round(seconds * 1000)))
    td = timedelta(milliseconds=total_ms)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = total_ms % 1000
    return f"{hours:02d}-{minutes:02d}-{secs:02d}-{millis:03d}"


@dataclass(frozen=True)
class FrameSample:
    """A sampled frame with index and timestamp."""

    frame_index: int
    timestamp_sec: float
    image: np.ndarray


def _sample_video_frames(video_path: str | Path, sample_fps: float) -> list[FrameSample]:
    """Sample frames from a video at approximately *sample_fps*."""
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if native_fps <= 0:
        native_fps = 30.0
    step = max(1, int(round(native_fps / max(sample_fps, 0.01))))

    samples: list[FrameSample] = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts = idx / native_fps
                samples.append(FrameSample(frame_index=idx, timestamp_sec=ts, image=frame))
            idx += 1
    finally:
        cap.release()

    logger.info(
        "Sampled %d frame(s) from %s at ~%.2f FPS (native %.2f, step=%d)",
        len(samples), path, sample_fps, native_fps, step,
    )
    return samples


def _estimate_affine(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray | None:
    """Estimate a partial affine transform from *curr* to *prev* frame."""
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

    pts_prev = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=300,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if pts_prev is None or len(pts_prev) < 10:
        return None

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)
    if pts_curr is None or status is None:
        return None

    good_prev = pts_prev[status.flatten() == 1]
    good_curr = pts_curr[status.flatten() == 1]
    if len(good_prev) < 10:
        return None

    mat, _inliers = cv2.estimateAffinePartial2D(
        good_curr,
        good_prev,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    return mat


def _stabilize_samples(samples: list[FrameSample]) -> list[FrameSample]:
    """Stabilize sampled frames against previous frame jitter."""
    if len(samples) <= 1:
        return samples

    stabilized: list[FrameSample] = [samples[0]]
    for i in range(1, len(samples)):
        prev = stabilized[-1].image
        curr = samples[i].image
        mat = _estimate_affine(prev, curr)
        if mat is None:
            stabilized.append(samples[i])
            continue

        h, w = curr.shape[:2]
        warped = cv2.warpAffine(
            curr,
            mat,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        stabilized.append(
            FrameSample(
                frame_index=samples[i].frame_index,
                timestamp_sec=samples[i].timestamp_sec,
                image=warped,
            )
        )

    logger.info("Applied stabilization to %d sampled frame(s)", len(stabilized))
    return stabilized


def _apply_clahe_bgr(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_chan)
    lab_eq = cv2.merge((l_eq, a_chan, b_chan))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _histogram_match_channel(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Match histogram of 1-channel *source* to *template*."""
    src = source.ravel()
    tmpl = template.ravel()

    src_vals, src_idx, src_counts = np.unique(src, return_inverse=True, return_counts=True)
    tmpl_vals, tmpl_counts = np.unique(tmpl, return_counts=True)

    src_cdf = np.cumsum(src_counts).astype(np.float64)
    src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1.0
    tmpl_cdf = np.cumsum(tmpl_counts).astype(np.float64)
    tmpl_cdf /= tmpl_cdf[-1] if tmpl_cdf[-1] > 0 else 1.0

    interp_vals = np.interp(src_cdf, tmpl_cdf, tmpl_vals)
    matched = interp_vals[src_idx].reshape(source.shape)
    return matched.astype(source.dtype)


def _normalize_illumination_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize illumination between a frame pair."""
    if mode == "none":
        return image_a, image_b
    if mode == "clahe":
        return _apply_clahe_bgr(image_a), _apply_clahe_bgr(image_b)
    if mode == "match":
        lab_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2LAB)
        lab_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2LAB)
        l_a, a_a, b_a = cv2.split(lab_a)
        l_b, a_b, b_b = cv2.split(lab_b)
        l_bm = _histogram_match_channel(l_b, l_a)
        lab_bm = cv2.merge((l_bm, a_b, b_b))
        return image_a, cv2.cvtColor(lab_bm, cv2.COLOR_LAB2BGR)
    raise ValueError("illumination_mode must be one of: none, clahe, match")


def _build_pairs(
    samples_a: list[FrameSample],
    samples_b: list[FrameSample],
    max_pairs: int | None = None,
) -> list[tuple[FrameSample, FrameSample]]:
    """Match sampled frames by normalized timeline position."""
    if not samples_a or not samples_b:
        return []

    if len(samples_a) <= len(samples_b):
        short, long_ = samples_a, samples_b
        short_is_a = True
    else:
        short, long_ = samples_b, samples_a
        short_is_a = False

    n_short = len(short)
    n_long = len(long_)
    n = n_short if max_pairs is None else min(n_short, max_pairs)

    out: list[tuple[FrameSample, FrameSample]] = []
    for i in range(n):
        if n_short == 1:
            j = 0
        else:
            j = int(round(i * (n_long - 1) / (n_short - 1)))
        s = short[i]
        l = long_[j]
        if short_is_a:
            out.append((s, l))
        else:
            out.append((l, s))
    return out


def _build_pairs_ratio(
    samples_a: list[FrameSample],
    samples_b: list[FrameSample],
    max_pairs: int | None = None,
) -> list[tuple[FrameSample, FrameSample, float]]:
    """Baseline timeline-ratio matching with synthetic confidence scores."""
    pairs = _build_pairs(samples_a, samples_b, max_pairs=max_pairs)
    return [(a, b, 1.0) for a, b in pairs]


def _orb_descriptors(frame: np.ndarray, max_features: int = 750) -> tuple[int, np.ndarray | None]:
    """Extract ORB descriptors from a resized grayscale frame."""
    h, w = frame.shape[:2]
    scale = 320.0 / max(h, w)
    if scale < 1.0:
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = frame
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    keypoints, desc = orb.detectAndCompute(gray, None)
    return len(keypoints), desc


def _orb_similarity(
    a_desc: np.ndarray | None,
    b_desc: np.ndarray | None,
) -> float:
    """Compute a normalized ORB similarity score in [0, 1]."""
    if a_desc is None or b_desc is None:
        return 0.0
    if len(a_desc) < 2 or len(b_desc) < 2:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        knn = bf.knnMatch(a_desc, b_desc, k=2)
    except cv2.error:
        return 0.0

    good = 0
    for m_n in knn:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good += 1

    denom = max(1, min(len(a_desc), len(b_desc)))
    return float(good / denom)


def _align_pairs_feature_dtw(
    samples_a: list[FrameSample],
    samples_b: list[FrameSample],
    *,
    gap_penalty: float = -0.05,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """
    Align sampled frame indices via monotonic dynamic programming.

    Returns:
        - list of (idx_a, idx_b, similarity_score) in temporal order
        - full similarity matrix used by the aligner
    """
    if not samples_a or not samples_b:
        return [], np.zeros((len(samples_a), len(samples_b)), dtype=np.float32)

    sig_a = [_orb_descriptors(s.image) for s in samples_a]
    sig_b = [_orb_descriptors(s.image) for s in samples_b]

    n = len(samples_a)
    m = len(samples_b)
    sim = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            sim[i, j] = _orb_similarity(sig_a[i][1], sig_b[j][1])

    dp = np.full((n + 1, m + 1), -1e9, dtype=np.float32)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)  # 1=diag, 2=up, 3=left
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_penalty
        back[i, 0] = 2
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + gap_penalty
        back[0, j] = 3

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1, j - 1] + sim[i - 1, j - 1]
            up = dp[i - 1, j] + gap_penalty
            left = dp[i, j - 1] + gap_penalty
            if diag >= up and diag >= left:
                dp[i, j] = diag
                back[i, j] = 1
            elif up >= left:
                dp[i, j] = up
                back[i, j] = 2
            else:
                dp[i, j] = left
                back[i, j] = 3

    pairs_rev: list[tuple[int, int, float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i, j]
        if move == 1:
            pairs_rev.append((i - 1, j - 1, float(sim[i - 1, j - 1])))
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        elif move == 3:
            j -= 1
        else:
            break

    pairs_rev.reverse()
    return pairs_rev, sim


def compare_videos(
    video_a: str | Path,
    video_b: str | Path,
    *,
    app_config: AppConfig,
    method: str = "robust",
    alignment_method: str = "feature-dtw",
    min_alignment_score: float = 0.08,
    gap_penalty: float = -0.05,
    min_detector_alignment_conf: float = 0.0,
    stabilize: bool = False,
    background_window: int = 0,
    static_std_thresh: float = 0.0,
    min_static_coverage: float = 0.0,
    illumination_mode: str = "none",
    sample_fps: float = 1.0,
    min_change_percent: float = 1.0,
    min_inner_change_percent: float | None = None,
    min_regions: int = 1,
    min_persistence: int = 1,
    edge_ignore_px: int = 0,
    max_pairs: int | None = None,
    output_dir: str | Path = "outputs/video_changes",
) -> dict[str, Any]:
    """
    Compare two videos and save only changed frame pair visualizations.

    Returns:
        Summary dict with changed frame metadata and output paths.
    """
    if sample_fps <= 0:
        raise ValueError("sample_fps must be > 0")
    if min_change_percent < 0:
        raise ValueError("min_change_percent must be >= 0")
    if min_regions < 0:
        raise ValueError("min_regions must be >= 0")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be a positive integer")
    if alignment_method not in {"ratio", "feature-dtw"}:
        raise ValueError("alignment_method must be 'ratio' or 'feature-dtw'")
    if min_alignment_score < 0:
        raise ValueError("min_alignment_score must be >= 0")
    if min_detector_alignment_conf < 0 or min_detector_alignment_conf > 1:
        raise ValueError("min_detector_alignment_conf must be in [0, 1]")
    if min_persistence <= 0:
        raise ValueError("min_persistence must be a positive integer")
    if edge_ignore_px < 0:
        raise ValueError("edge_ignore_px must be >= 0")
    if background_window < 0:
        raise ValueError("background_window must be >= 0")
    if static_std_thresh < 0:
        raise ValueError("static_std_thresh must be >= 0")
    if min_static_coverage < 0 or min_static_coverage > 1:
        raise ValueError("min_static_coverage must be in [0, 1]")
    if illumination_mode not in {"none", "clahe", "match"}:
        raise ValueError("illumination_mode must be 'none', 'clahe', or 'match'")

    out_root = Path(output_dir)
    processing_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    changed_dir = out_root / f"changed_frames_{processing_stamp}"
    changed_dir.mkdir(parents=True, exist_ok=True)

    samples_a = _sample_video_frames(video_a, sample_fps)
    samples_b = _sample_video_frames(video_b, sample_fps)
    if stabilize:
        samples_a = _stabilize_samples(samples_a)
        samples_b = _stabilize_samples(samples_b)
    alignment_pairs: list[tuple[FrameSample, FrameSample, float]]
    similarity_stats: dict[str, float | int] = {}

    if alignment_method == "ratio":
        alignment_pairs = _build_pairs_ratio(samples_a, samples_b, max_pairs=max_pairs)
        similarity_stats = {"avg_similarity": 1.0, "max_similarity": 1.0, "min_similarity": 1.0}
    else:
        idx_pairs, sim = _align_pairs_feature_dtw(
            samples_a,
            samples_b,
            gap_penalty=gap_penalty,
        )
        if max_pairs is not None:
            idx_pairs = idx_pairs[:max_pairs]

        alignment_pairs = [
            (samples_a[i], samples_b[j], score)
            for i, j, score in idx_pairs
            if score >= min_alignment_score
        ]
        if sim.size > 0:
            similarity_stats = {
                "avg_similarity": float(np.mean(sim)),
                "max_similarity": float(np.max(sim)),
                "min_similarity": float(np.min(sim)),
            }
        else:
            similarity_stats = {"avg_similarity": 0.0, "max_similarity": 0.0, "min_similarity": 0.0}
        similarity_stats["raw_aligned_pairs"] = len(idx_pairs)
        similarity_stats["kept_aligned_pairs"] = len(alignment_pairs)

    detector = ChangeDetector(app_config=app_config)

    index_a = {s.frame_index: i for i, s in enumerate(samples_a)}
    index_b = {s.frame_index: i for i, s in enumerate(samples_b)}

    candidate_changed: list[dict[str, Any]] = []
    skipped_low_detector_conf = 0
    for i, (fa, fb, align_score) in enumerate(alignment_pairs, start=1):
        if background_window > 0:
            # Build temporal median background from neighboring samples.
            def median_background(samples: list[FrameSample], center_idx: int) -> tuple[np.ndarray, np.ndarray | None]:
                lo = max(0, center_idx - background_window)
                hi = min(len(samples) - 1, center_idx + background_window)
                imgs = [samples[k].image for k in range(lo, hi + 1)]
                stack = np.stack(imgs, axis=0)
                median = np.median(stack, axis=0).astype(np.uint8)
                if static_std_thresh > 0:
                    # Stability mask from grayscale stddev across window
                    gray_stack = np.stack(
                        [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs],
                        axis=0,
                    ).astype(np.float32)
                    std = np.std(gray_stack, axis=0)
                    stable = (std <= static_std_thresh).astype(np.uint8) * 255
                else:
                    stable = None
                return median, stable

            idx_a = index_a.get(fa.frame_index, 0)
            idx_b = index_b.get(fb.frame_index, 0)
            bg_a, stable_a = median_background(samples_a, idx_a)
            bg_b, stable_b = median_background(samples_b, idx_b)
            norm_a, norm_b = _normalize_illumination_pair(bg_a, bg_b, illumination_mode)
            result = detector.compare(norm_a, norm_b, method=method)

            # Apply static mask if available
            if static_std_thresh > 0 and stable_a is not None and stable_b is not None:
                stable = cv2.bitwise_and(stable_a, stable_b)
                stable_area = int(np.sum(stable > 0))
                total_area = stable.shape[0] * stable.shape[1]
                coverage = stable_area / max(total_area, 1)
                if coverage < min_static_coverage:
                    continue
                mask = result.get("mask")
                if isinstance(mask, np.ndarray) and mask.ndim == 2:
                    masked = cv2.bitwise_and(mask, mask, mask=stable)
                    result["mask"] = masked
        else:
            norm_a, norm_b = _normalize_illumination_pair(fa.image, fb.image, illumination_mode)
            result = detector.compare(norm_a, norm_b, method=method)
        det_align_conf = float(result.get("alignment", {}).get("confidence", 1.0))
        if det_align_conf < min_detector_alignment_conf:
            skipped_low_detector_conf += 1
            continue

        change_percent = float(result.get("change_percent", 0.0))
        num_regions = len(result.get("bboxes", []))
        mask = result.get("mask")
        inner_change_percent = change_percent
        static_change_percent = change_percent
        if isinstance(mask, np.ndarray) and mask.ndim == 2 and edge_ignore_px > 0:
            h, w = mask.shape[:2]
            x1 = edge_ignore_px
            y1 = edge_ignore_px
            x2 = max(x1 + 1, w - edge_ignore_px)
            y2 = max(y1 + 1, h - edge_ignore_px)
            inner = mask[y1:y2, x1:x2]
            inner_total = inner.shape[0] * inner.shape[1]
            inner_change_percent = float(np.sum(inner > 0) / max(inner_total, 1) * 100.0)
            static_change_percent = inner_change_percent

        min_inner = min_change_percent if min_inner_change_percent is None else min_inner_change_percent
        is_changed = (
            change_percent >= min_change_percent
            and inner_change_percent >= min_inner
            and num_regions >= min_regions
        )
        if not is_changed:
            continue

        candidate_changed.append(
            {
                "pair_index": i,
                "frame_a_index": fa.frame_index,
                "frame_b_index": fb.frame_index,
                "timestamp_a_sec": round(fa.timestamp_sec, 3),
                "timestamp_b_sec": round(fb.timestamp_sec, 3),
                "alignment_score": round(float(align_score), 6),
                "detector_alignment_confidence": round(det_align_conf, 6),
                "change_percent": round(change_percent, 4),
                "inner_change_percent": round(inner_change_percent, 4),
                "static_change_percent": round(static_change_percent, 4),
                "num_regions": num_regions,
                "diff_image": result["diff_image"],
            }
        )

    kept: list[dict[str, Any]] = []
    if min_persistence <= 1:
        kept = candidate_changed
    else:
        run: list[dict[str, Any]] = []
        for row in candidate_changed:
            if not run:
                run = [row]
                continue
            if row["pair_index"] == run[-1]["pair_index"] + 1:
                run.append(row)
            else:
                if len(run) >= min_persistence:
                    kept.extend(run)
                run = [row]
        if run and len(run) >= min_persistence:
            kept.extend(run)

    changed_rows: list[dict[str, Any]] = []
    for row in kept:
        ts_a = _format_timestamp_hms_ms(float(row["timestamp_a_sec"]))
        ts_b = _format_timestamp_hms_ms(float(row["timestamp_b_sec"]))
        out_name = (
            f"video1_{ts_a}_____video2_{ts_b}.png"
        )
        out_path = changed_dir / out_name
        save_result(row.pop("diff_image"), out_path)
        row["output_image"] = str(out_path)
        changed_rows.append(row)
        if len(changed_rows) % 25 == 0:
            logger.info("Saved %d changed frame result(s)", len(changed_rows))

    summary = {
        "task": "video_change_detection",
        "video_a": str(Path(video_a).resolve()),
        "video_b": str(Path(video_b).resolve()),
        "method": method,
        "alignment_method": alignment_method,
        "min_alignment_score": min_alignment_score,
        "gap_penalty": gap_penalty,
        "min_detector_alignment_conf": min_detector_alignment_conf,
        "stabilize": stabilize,
        "background_window": background_window,
        "static_std_thresh": static_std_thresh,
        "min_static_coverage": min_static_coverage,
        "illumination_mode": illumination_mode,
        "sample_fps": sample_fps,
        "min_change_percent": min_change_percent,
        "min_inner_change_percent": min_inner_change_percent if min_inner_change_percent is not None else min_change_percent,
        "min_regions": min_regions,
        "min_persistence": min_persistence,
        "edge_ignore_px": edge_ignore_px,
        "num_sampled_a": len(samples_a),
        "num_sampled_b": len(samples_b),
        "num_pairs_compared": len(alignment_pairs),
        "num_pairs_skipped_low_detector_alignment_conf": skipped_low_detector_conf,
        "num_change_candidates_before_persistence": len(candidate_changed),
        "num_changed_pairs": len(changed_rows),
        "changed_dir": str(changed_dir.resolve()),
        "changed_frames": changed_rows,
    }
    summary.update(similarity_stats)

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved video comparison summary → %s", summary_path)
    return summary
