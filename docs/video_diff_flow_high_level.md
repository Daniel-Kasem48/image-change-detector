# Video Diff Flow (High-Level, Non-Technical)

## What This Does

This flow compares two videos of the same place and highlights meaningful changes.

Instead of showing every frame difference, it focuses on changes that are likely real and important.

## Simple End-to-End Flow

1. Load Video A and Video B.
2. Take snapshots from both videos at a chosen rate.
3. Match snapshots that represent roughly the same moment/scene.
4. Compare each matched snapshot pair.
5. Filter out weak or noisy differences.
6. Keep only consistent changes.
7. Save a visual result image for each kept change.
8. Save one summary file with counts and details.

## Why It Works Better Than Raw Frame-by-Frame Diff

- The two videos can have different lengths.
- Camera movement and speed can differ.
- Lighting may differ.
- Small random noise is filtered out.

So the pipeline first aligns similar moments, then detects changes, then removes unstable results.

## What You Get

- A folder with only changed frame-pair images.
- A `summary.json` file with:
  - how many frames were sampled
  - how many pairs were compared
  - how many changes were kept
  - where output images were saved

## Main Controls (Plain Language)

- `sample_fps`: How often to sample frames.
- `alignment_method`: How to match moments between videos.
- `min_alignment_score`: How strict matching should be.
- `min_change_percent`: Minimum visible change needed.
- `min_regions`: Minimum number of changed areas needed.
- `min_persistence`: How many consecutive changed pairs are required before accepting an event.
- `stabilize`: Helps when videos are shaky.

## Typical Outcome

You get a cleaner list of likely real scene changes, with much less noise than comparing every frame directly.
