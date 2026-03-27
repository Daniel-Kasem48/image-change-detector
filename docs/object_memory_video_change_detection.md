# Object Memory for Video Change Detection

## Why Frame Diff Is Not Enough

For long videos, objects can:

- move slightly between captures
- be temporarily occluded
- reappear in nearby positions
- appear at different timestamps in each video

So raw frame differencing often over-reports false changes.

## Core Idea

Build a **spatio-temporal memory** of objects for each video, then compare memories.

Instead of asking:
"Are these two frames different?"

Ask:
"Is this the same object in the same place over time, with acceptable movement?"

## Data Model (Memory Table)

Store object observations per video:

- `video_id`
- `time_sec`
- `track_id`
- `object_signature` (appearance embedding / label)
- `bbox` / centroid
- `scene_coords` (normalized coordinates after alignment)
- `confidence`

## Pipeline

1. Scene alignment
- Estimate canonical scene coordinates (homography/similarity transform).
- Normalize object positions from both videos into that shared space.

2. Object detection + tracking
- Detect objects each frame.
- Track identities over time (Kalman + association / DeepSORT-like approach).

3. Build memory
- Save track trajectories and persistence windows.

4. Cross-video matching
- Match tracks by appearance + trajectory proximity in canonical space.
- Use temporal tolerance windows (because videos have different lengths/speeds).

5. Change classification
- `unchanged`: same object, stable location
- `moved_slightly`: same object, movement <= small threshold
- `moved_significantly`: same object, larger displacement
- `added`: appears only in video B memory
- `removed`: appears only in video A memory

## Key Thresholds

- `small_move_px` (or meters if calibrated)
- `large_move_px`
- `appearance_similarity_min`
- `persistence_min_sec` (ignore transient detections)
- `occlusion_grace_sec`

## Why This Solves Your Example

Example:
"Car at second 4 in video A moved a little in video B."

With memory-based matching:

- Same track signature + nearby canonical position
- Classified as `moved_slightly`
- Not treated as unrelated add/remove noise

## Output Format (Recommended)

Per object:

- object id / class / signature
- status (`unchanged|moved_slightly|moved_significantly|added|removed`)
- displacement magnitude
- first_seen / last_seen in each video
- representative evidence frames

Global summary:

- counts per status
- confidence distribution
- exported evidence images/video snippets

## Phased Implementation Plan

Phase 1:

- Centroid tracking + canonical coordinate normalization
- Distance-based track matching
- Basic status classification

Phase 2:

- Appearance embeddings for stronger identity matching
- Occlusion-aware re-identification
- Temporal smoothing and confidence calibration

Phase 3:

- Human-readable semantic reasoning layer (VLM) on only uncertain/important cases

