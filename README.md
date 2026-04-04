# 🔍 image-change-detector

A Python project for detecting **meaningful changes** between two images, with a strong focus on **cross-view remote-sensing/drone imagery**.

It is designed to reduce false positives from:
- viewpoint/camera shifts
- illumination/exposure differences
- small registration errors

and produce usable outputs for both:
- region-level binary change detection
- object-level change reporting

---

## Features

| Mode | Method | Speed | Use Case |
|------|--------|-------|----------|
| `pixel` | Absolute pixel diff | ⚡ Fast | Simple brightness/color changes |
| `ssim` | Structural Similarity (SSIM) | ⚡ Fast | Perceptual / structural changes |
| `combined` | Pixel diff + SSIM | ⚡ Fast | General purpose (default) |
| `robust` | Align + Normalize + Multi-scale | ⚖️ Medium | Cross-view / different camera angle comparisons |
| `object` | YOLOv8 deep learning | 🐢 Slower | Object added / removed / moved |

---

## Project Structure

```
image-change-detector/
├── main.py                  # CLI entry point
├── config.yaml              # Configurable settings
├── requirements.txt         # Python dependencies
├── src/
│   ├── __init__.py
│   ├── detector.py          # Pixel/SSIM change detection
│   ├── object_detector.py   # YOLOv8 object-level detection
│   ├── benchmark.py         # Dataset-level evaluation (LEVIR-style)
│   ├── training.py          # Tiny UNet baseline training pipeline
│   └── utils.py             # Image I/O, preprocessing, visualisation
├── samples/                 # Put your test images here
├── outputs/                 # Results are saved here
└── tests/                   # Unit tests
```

---

## Installation

```bash
# 1. Clone / navigate to the project
cd image-change-detector

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU support:** Install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) and set `device: cuda` in `config.yaml`.

---

## Usage

### Basic pixel/SSIM comparison

```bash
python main.py compare samples/before.jpg samples/after.jpg
```

### Choose a detection method

```bash
python main.py compare before.jpg after.jpg --mode pixel
python main.py compare before.jpg after.jpg --mode ssim
python main.py compare before.jpg after.jpg --mode combined   # default
```

### Object-level detection (YOLOv8)

```bash
python main.py compare before.jpg after.jpg --mode object
```

Detects which objects were **added**, **removed**, or **moved** between the two images.

### Robust cross-view detection (recommended production mode)

```bash
python main.py compare before.jpg after.jpg --mode robust
```

Uses feature alignment + photometric normalization + multi-scale differencing to reduce false positives from viewpoint/lighting changes.

### Save output image

```bash
python main.py compare before.jpg after.jpg --output outputs/result.png
```

### Apply preprocessing (CLAHE contrast enhancement)

```bash
python main.py compare before.jpg after.jpg --preprocess
```

### Suppress the plot window

```bash
python main.py compare before.jpg after.jpg --no-show
```

### Evaluate on a LEVIR-style dataset

```bash
python main.py evaluate /path/to/dataset --split test --mode robust
```

Process image pairs in batches (and optionally parallel workers):

```bash
python main.py evaluate /path/to/dataset --split test --mode robust --batch-size 8 --num-workers 4
```

Outputs dataset-level metrics: `precision`, `recall`, `f1`, `iou`, `accuracy`, plus confusion counts.

Optional JSON export:

```bash
python main.py evaluate /path/to/dataset --split test --mode robust --format json --json-out outputs/eval.json
```

Expected folder layout:

```text
dataset_root/
  test/
    A/      # before images
    B/      # after images
    label/  # binary masks (non-zero = change)
```

or without split:

```text
dataset_root/
  A/
  B/
  label/
```

### Compare two videos (different lengths supported)

```bash
python main.py compare-videos /path/to/video_a.mp4 /path/to/video_b.mp4 \
  --mode robust \
  --alignment feature-dtw \
  --min-alignment-score 0.08 \
  --min-detector-align-conf 0.25 \
  --stabilize \
  --background-window 2 \
  --static-std-thresh 3.0 \
  --min-static-coverage 0.15 \
  --illumination match \
  --sample-fps 1 \
  --min-change-percent 1.0 \
  --min-inner-change-percent 1.0 \
  --edge-ignore-px 24 \
  --min-regions 1 \
  --min-persistence 2 \
  --output-dir outputs/video_changes
```

This saves only changed frame-pair visualizations under:
`outputs/video_changes/changed_frames/`
and writes a report:
`outputs/video_changes/summary.json`

If you want simple time-ratio pairing instead of visual alignment:

```bash
python main.py compare-videos /path/to/video_a.mp4 /path/to/video_b.mp4 --alignment ratio
```

### Desktop app for video comparison

If you prefer a GUI instead of CLI, you have 2 options.

#### Option A (non-technical, recommended)

Run one command:

```bash
./install_desktop_app.sh
```

This will:
- install everything automatically
- create an app-menu launcher: `Video Change Detector`
- create a desktop shortcut (if `~/Desktop` exists)

Then open the app by clicking the launcher/desktop icon.

#### Option B (manual)

```bash
./venv/bin/pip install -r requirements-desktop.txt
./venv/bin/python video_compare_desktop.py
```

The desktop app lets you:
- pick Video A / Video B
- tune comparison options (`mode`, `alignment`, thresholds, stabilization)
- run comparison and view JSON summary in-app
- open output folder quickly

This GUI uses `PySide6` (installed from pip), so you do not need system Tk/Tkinter setup.

#### Windows `.exe` (no terminal for end-users)

This repo includes GitHub Actions packaging for a native Windows executable:
- workflow file: `.github/workflows/build-windows-exe.yml`
- output artifact: `VideoChangeDetector.exe`

How to generate and download it:
1. Push your latest changes to GitHub.
2. Open the repo on GitHub.
3. Go to **Actions** -> **Build Windows EXE**.
4. Click **Run workflow**.
5. Open the finished run and download artifact **VideoChangeDetector-windows-exe**.
6. Share `VideoChangeDetector.exe` with non-technical users.

Optional local Windows build (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_exe.ps1
```

### Train a deep baseline (Tiny UNet)

```bash
python main.py train-baseline /path/to/dataset --train-split train --val-split val --epochs 20 --batch-size 4
```

Outputs are saved under `outputs/baseline_runs/...`:
- `best.pt` / `final.pt`
- `history.json`
- `summary.json`

### Semantic change reasoning with GPT (or other VLMs)

```bash
export OPENAI_API_KEY="your-key"
python main.py reason dataset/test/A/001.jpg dataset/test/B/001.jpg --provider openai --format json --json-out outputs/reason_001.json
```

You can also use `--provider gemini`, `--provider huggingface`, or `--provider openrouter`.

Region-based mode (recommended): detect changed regions with the local CV pipeline,
then send only those crops to the VLM:

```bash
python main.py reason dataset/test/A/001.jpg dataset/test/B/001.jpg \
  --provider openai \
  --use-regions \
  --region-method robust \
  --max-regions 5 \
  --region-pad 24 \
  --format json
```

This mode saves a colorized/bordered region preview image to `outputs/` before
the VLM call (or provide `--regions-image-out /path/to/file.png`).

---

## Python API

```python
from src.detector import ChangeDetector
from src.object_detector import ObjectDetector
from src.utils import load_image, print_report, plot_comparison

# Load images
img_a = load_image("samples/before.jpg")
img_b = load_image("samples/after.jpg")

# --- Pixel/SSIM detection ---
detector = ChangeDetector(config={"threshold": 25, "min_contour_area": 300})
result = detector.compare(img_a, img_b, method="combined")

print_report(result)
# result["score"]           → SSIM similarity score (1.0 = identical)
# result["change_percent"]  → % of image that changed
# result["bboxes"]          → list of changed region bounding boxes
# result["diff_image"]      → annotated comparison image (numpy array)

# --- Object detection ---
obj_detector = ObjectDetector(model_name="yolov8n", confidence=0.4)
obj_result = obj_detector.compare(img_a, img_b)

print_report(obj_result)
# obj_result["summary"]  → {"added": 2, "removed": 1, "moved": 0, ...}
# obj_result["changes"]  → list of ObjectChange objects
```

---

## Configuration

Edit `config.yaml` to tune detection behaviour:

```yaml
detector:
  threshold: 30          # pixel diff threshold (0-255)
  min_contour_area: 500  # ignore tiny changed regions
  blur_kernel: 5         # noise reduction kernel size

object_detector:
  model: yolov8n         # yolov8n / yolov8s / yolov8m / yolov8l / yolov8x
  confidence: 0.4        # minimum detection confidence
  iou_threshold: 0.3     # IoU for matching objects across images
  device: cpu            # cpu / cuda / mps
```

---

## Dependencies

- **OpenCV** — image processing, contour detection
- **scikit-image** — SSIM computation
- **ultralytics** — YOLOv8 object detection
- **PyTorch** — deep learning inference backend
- **matplotlib** — result visualisation
- **click** — CLI interface

---

## License

MIT
