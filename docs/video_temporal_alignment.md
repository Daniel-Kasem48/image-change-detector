# Video-to-Video Temporal Alignment for Change Detection

## Problem

When comparing two videos of the same place:

- Durations can be different (for example 1 minute vs 3 minutes).
- Camera speed can differ.
- Start/end times can be offset.
- Some segments may exist in only one video.

Because of this, matching frames by raw second (or even simple time ratio) is not reliable enough for production change detection.

## Why This Is a Known Real-World Problem

This is a standard problem in:

- CCTV and surveillance auditing
- Robotics and SLAM (place revisits)
- Autonomous driving map updates
- Drone/satellite monitoring
- Multi-camera event analysis

Common names:

- Temporal alignment
- Sequence matching
- Video synchronization
- Visual place recognition

## Goal

Find frame pairs `(A_i, B_j)` that are likely the **same place/moment context**, then run change detection only on those pairs.

## Recommended Pipeline

1. Frame sampling
- Sample both videos at fixed rate (e.g. 1 FPS).

2. Per-frame visual signature
- Option A (lightweight): ORB/SIFT keypoints + matching score.
- Option B (stronger): image embeddings (CLIP/DINO/NetVLAD) + cosine similarity.

3. Similarity matrix
- Build `S[i, j] = similarity(frame_A_i, frame_B_j)`.

4. Monotonic sequence alignment
- Use DTW (Dynamic Time Warping) or Needleman-Wunsch style alignment.
- Enforce timeline order to avoid unrealistic jumps.

5. Confidence filtering
- Keep pairs above thresholds (similarity, inliers, local consistency).
- Drop uncertain pairs.

6. Change detection stage
- Run local detector (`robust`/`combined`) only on accepted aligned pairs.
- Save only changed outputs.

## Confidence Signals for “Same Place”

Use multiple signals together:

- Global similarity (embedding cosine or normalized feature score)
- Geometric quality (homography/similarity transform inlier count + inlier ratio)
- Neighborhood consistency (pair agrees with nearby aligned pairs)

If confidence is low, mark pair as unmatched and skip change detection.

## Failure Cases to Handle

- Repeated textures/roads causing false matches
- Fast motion blur in one video
- Illumination/weather shifts
- Occlusions (vehicles, people)
- Long unmatched segments

Mitigation:

- Add minimum confidence thresholds
- Add temporal smoothing
- Add “unmatched segment” reporting

## Available Tools/Libraries

- OpenCV: ORB/SIFT, matching, homography/inliers
- `fastdtw` or `dtaidistance`: DTW alignment
- FAISS: fast nearest-neighbor search on embeddings
- PyTorch + CLIP/DINO: robust frame embeddings

## Practical Plan for This Repo

Phase 1 (fast to ship):

- Add feature-based temporal alignment (ORB + inlier score).
- Add DTW/monotonic matcher.
- Compare only aligned high-confidence frame pairs.

Phase 2 (accuracy upgrade):

- Add embedding-based alignment backend.
- Keep feature-based method as fallback.

## Acceptance Criteria

- Works with different video lengths.
- Returns aligned frame-pair index mapping.
- Produces changed-frame outputs only for confident aligned pairs.
- Writes summary JSON with:
  - aligned pairs
  - unmatched ranges
  - confidence metrics
  - changed frame outputs

