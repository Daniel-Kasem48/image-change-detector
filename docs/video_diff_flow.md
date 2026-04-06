# Video Diff Flow (Video-to-Video Comparison)

## Purpose

This document explains the end-to-end flow used by `compare_videos()` in `src/video_compare.py`.

The goal is to compare two videos and save only frame pairs that contain meaningful scene changes.

## Entry Point

- Function: `compare_videos(video_a, video_b, ...)`
- Inputs:
  - video file paths
  - alignment settings
  - detector method (`pixel`, `ssim`, `combined`, `robust`)
  - change filtering thresholds
  - output directory

## High-Level Flow

1. Validate input parameters.
2. Create output folder `changed_frames_<timestamp>`.
3. Sample frames from both videos at `sample_fps`.
4. Optionally stabilize sampled frames.
5. Align sampled frames between videos:
- `ratio`: timeline ratio matching
- `feature-dtw`: ORB similarity + monotonic dynamic-programming alignment
6. Drop weak alignments using `min_alignment_score` (feature-dtw mode).
7. For each aligned pair:
- Optionally build temporal median backgrounds
- Optionally normalize illumination (`none`, `clahe`, `match`)
- Run `ChangeDetector.compare(...)`
- Apply detector-alignment confidence filter
- Apply change filters (`min_change_percent`, `min_inner_change_percent`, `min_regions`)
8. Apply persistence filtering (`min_persistence`) to keep only stable runs of changed pairs.
9. Save diff images for kept pairs.
10. Write `summary.json` with metrics, thresholds, and changed-frame records.

## Frame Sampling

Sampling is done by `_sample_video_frames(...)`:

- Uses OpenCV `VideoCapture`.
- Computes frame step from native FPS and `sample_fps`.
- Stores for each sample:
  - `frame_index`
  - `timestamp_sec`
  - `image` (BGR)

## Optional Stabilization

If `stabilize=True`:

- `_stabilize_samples(...)` aligns each sampled frame to the previous stabilized frame.
- Uses optical flow (`calcOpticalFlowPyrLK`) + partial affine estimate (`estimateAffinePartial2D`).
- Reduces handheld jitter before alignment and differencing.

## Alignment Stage

### 1) Ratio Alignment

- `_build_pairs_ratio(...)` pairs by normalized timeline position.
- Confidence is synthetic (`1.0` per pair).
- Fast baseline option.

### 2) Feature-DTW Alignment

- `_align_pairs_feature_dtw(...)`:
  - Extract ORB descriptors per frame.
  - Build similarity matrix `sim[i, j]` via ratio-test match quality.
  - This computes similarity for every sampled combination (`n x m` sampled pairs).
  - Run monotonic DP with `gap_penalty` to allow unmatched segments.
  - Backtrack to produce aligned index pairs.
- Pairs with score `< min_alignment_score` are discarded.
- Practical meaning: exhaustive matching is only for sampled frames; actual change detection runs on the aligned subset, not all `n x m` pairs.

## Pair Comparison Stage

For each aligned pair `(fa, fb)`:

1. Optionally use temporal median backgrounds (`background_window > 0`).
2. Optionally compute static mask by window stddev (`static_std_thresh`).
3. Optionally enforce static coverage (`min_static_coverage`).
4. Normalize illumination (`illumination_mode`):
- `none`: no normalization
- `clahe`: CLAHE on L channel in LAB for both frames
- `match`: histogram-match frame B luminance to frame A
5. Run detector: `ChangeDetector.compare(norm_a, norm_b, method=method)`.
6. Reject pair if detector alignment confidence `< min_detector_alignment_conf`.
7. Compute metrics:
- `change_percent`
- `inner_change_percent` (if `edge_ignore_px > 0`)
- `num_regions`
8. Keep as candidate only if all conditions pass:
- `change_percent >= min_change_percent`
- `inner_change_percent >= min_inner_change_percent` (or fallback to `min_change_percent`)
- `num_regions >= min_regions`

## Persistence Filter

`min_persistence` controls temporal stability:

- `1`: keep every candidate changed pair.
- `>1`: keep only consecutive runs of changed pairs whose length is at least `min_persistence`.

This reduces one-off noise spikes.

## Output Artifacts

### Changed frame images

Saved in:

- `outputs/video_changes/changed_frames_<YYYY-MM-DD_HH-MM-SS-ms>/`

Filename format:

- `video1_<HH-mm-ss-ms>_____video2_<HH-mm-ss-ms>.png`

### Summary JSON

Saved at:

- `outputs/video_changes/summary.json`

Contains:

- input paths and runtime settings
- alignment stats (`avg_similarity`, `raw_aligned_pairs`, `kept_aligned_pairs`, ...)
- pair counts (sampled, compared, skipped, changed)
- per-changed-pair details and output image path

## Practical Tuning Notes

- Increase `sample_fps` for more temporal detail, decrease for speed.
- Use `alignment_method=feature-dtw` for videos with different pacing.
- Raise `min_alignment_score` to remove weak matches.
- Raise `min_change_percent` and `min_regions` to reduce false positives.
- Use `min_persistence > 1` for more stable, event-like changes.
- Enable `stabilize` for shaky handheld footage.
- Use `illumination_mode=clahe` or `match` when lighting differs strongly.
