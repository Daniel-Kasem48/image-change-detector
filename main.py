"""
main.py
-------
CLI entry point for image-change-detector.

Usage examples::

    # Pixel/SSIM-based comparison
    python main.py compare samples/before.jpg samples/after.jpg

    # Object-level comparison using YOLOv8
    python main.py compare samples/before.jpg samples/after.jpg --mode object

    # JSON output for automation pipelines
    python main.py compare before.jpg after.jpg --format json

    # Quiet mode (errors only)
    python main.py compare before.jpg after.jpg -q

    # Verbose / debug mode
    python main.py compare before.jpg after.jpg -v
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import cv2

# Allow running directly from the project root
sys.path.insert(0, str(Path(__file__).parent))

from src import __version__
from src.config import load_config
from src.detector import ChangeDetector
from src.exceptions import ChangeDetectorError
from src.logging_config import get_logger, setup_logging
from src.reasoning import ReasoningConfig, VLMReasoner
from src.utils import (
    load_image,
    preprocess,
    print_report,
    plot_comparison,
    plot_object_changes,
    resize_to_match,
    result_to_json,
    result_to_json_string,
    save_result,
)

logger = get_logger(__name__)

# Exit codes
EXIT_OK = 0
EXIT_INPUT_ERROR = 1
EXIT_DETECTION_ERROR = 2
EXIT_UNEXPECTED = 3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="image-change-detector")
def cli() -> None:
    """Image Change Detector — Compare two images for object/pixel changes."""


@cli.command()
@click.argument("image_a", type=click.Path(exists=True))
@click.argument("image_b", type=click.Path(exists=True))
@click.option(
    "--mode",
    type=click.Choice(
        ["pixel", "ssim", "combined", "robust", "drone"], case_sensitive=False,
    ),
    default="combined",
    show_default=True,
    help="Detection mode. 'drone' is optimised for vertical/nadir drone imagery.",
)
@click.option("--output", "-o", default=None, help="Path to save the output image.")
@click.option("--config", "-c", default="config.yaml", help="Path to config YAML.")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.  'json' for machine-readable output.",
)
@click.option("--show/--no-show", default=True, help="Display the result plot.")
@click.option(
    "--preprocess/--no-preprocess", "do_preprocess", default=False,
    help="Apply CLAHE preprocessing before comparison.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress info messages (errors only).")
def compare(
    image_a: str,
    image_b: str,
    mode: str,
    output: str | None,
    config: str,
    output_format: str,
    show: bool,
    do_preprocess: bool,
    verbose: bool,
    quiet: bool,
) -> None:
    """Compare IMAGE_A and IMAGE_B and report changes."""
    # Configure logging
    verbosity = 2 if verbose else (0 if quiet else 1)
    setup_logging(verbosity)

    try:
        cfg = load_config(config)

        logger.info("Loading images ...")
        img_a = load_image(image_a)
        img_b = load_image(image_b)

        if do_preprocess:
            logger.info("Preprocessing images ...")
            img_a = preprocess(img_a)
            img_b = preprocess(img_b)

        if mode == "robust":
            result = _run_robust_detection(img_a, img_b, cfg, output, show)
        elif mode == "drone":
            result = _run_drone_detection(img_a, img_b, cfg, output, show)
        else:
            result = _run_change_detection(img_a, img_b, mode, cfg, output, show)

        # Output
        if output_format == "json":
            click.echo(result_to_json_string(result))
        else:
            print_report(result)

            # Print alignment info for robust / drone mode
            align_info = result.get("alignment")
            if align_info:
                status = "✅ Success" if align_info["success"] else "⚠ Failed"
                click.echo(f"  🔗 Alignment: {status}")
                click.echo(
                    f"     Matches: {align_info['num_inliers']}/{align_info['num_matches']} "
                    f"inliers (confidence: {align_info['confidence']:.2%})"
                )
                if "scale_factor" in align_info:
                    sf = align_info["scale_factor"]
                    alt = 1 / sf if sf > 0 else 0
                    click.echo(f"     Scale factor B→A: {sf:.4f}x  (altitude ratio ~{alt:.2f}x)")
                click.echo()

    except ChangeDetectorError as exc:
        logger.error("%s", exc)
        raise SystemExit(EXIT_INPUT_ERROR)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise SystemExit(EXIT_OK)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        raise SystemExit(EXIT_UNEXPECTED)


@cli.command("evaluate")
@click.argument("dataset_root", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--mode",
    type=click.Choice(["pixel", "ssim", "combined", "robust"], case_sensitive=False),
    default="robust",
    show_default=True,
    help="Detection mode to evaluate.",
)
@click.option("--split", default=None, help="Optional dataset split (e.g. train/val/test).")
@click.option("--config", "-c", default="config.yaml", help="Path to config YAML.")
@click.option("--limit", type=int, default=None, help="Evaluate only the first N samples.")
@click.option("--batch-size", type=int, default=1, show_default=True, help="Number of pairs per evaluation batch.")
@click.option("--num-workers", type=int, default=0, show_default=True, help="Parallel workers per batch (0 = sequential).")
@click.option("--dir-a", default="A", help="Directory name for t1/before images.")
@click.option("--dir-b", default="B", help="Directory name for t2/after images.")
@click.option("--dir-label", default="label", help="Directory name for binary label masks.")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option("--json-out", default=None, help="Optional path to save JSON results.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress info messages (errors only).")
def evaluate(
    dataset_root: str,
    mode: str,
    split: str | None,
    config: str,
    limit: int | None,
    batch_size: int,
    num_workers: int,
    dir_a: str,
    dir_b: str,
    dir_label: str,
    output_format: str,
    json_out: str | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """Evaluate binary change detection on a LEVIR-style dataset."""
    verbosity = 2 if verbose else (0 if quiet else 1)
    setup_logging(verbosity)

    try:
        cfg = load_config(config)

        from src.benchmark import evaluate_binary_change_dataset

        result = evaluate_binary_change_dataset(
            dataset_root,
            split=split,
            method=mode,
            app_config=cfg,
            limit=limit,
            batch_size=batch_size,
            num_workers=num_workers,
            dir_a=dir_a,
            dir_b=dir_b,
            dir_label=dir_label,
        )

        if json_out:
            out_path = Path(json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2))
            logger.info("Saved evaluation JSON → %s", out_path)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            _print_eval_summary(result)

    except ChangeDetectorError as exc:
        logger.error("%s", exc)
        raise SystemExit(EXIT_INPUT_ERROR)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise SystemExit(EXIT_OK)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        raise SystemExit(EXIT_UNEXPECTED)


@cli.command("train-baseline")
@click.argument("dataset_root", type=click.Path(exists=True, file_okay=False))
@click.option("--train-split", default="train", show_default=True, help="Train split folder.")
@click.option("--val-split", default="val", show_default=True, help="Validation split folder.")
@click.option("--epochs", type=int, default=20, show_default=True, help="Training epochs.")
@click.option("--batch-size", type=int, default=4, show_default=True, help="Batch size.")
@click.option("--lr", type=float, default=1e-3, show_default=True, help="Learning rate.")
@click.option("--image-size", type=int, default=256, show_default=True, help="Input size.")
@click.option("--num-workers", type=int, default=0, show_default=True, help="DataLoader workers.")
@click.option("--threshold", type=float, default=0.5, show_default=True, help="Eval threshold.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed.")
@click.option("--limit-train", type=int, default=None, help="Optional cap on train samples.")
@click.option("--limit-val", type=int, default=None, help="Optional cap on val samples.")
@click.option("--device", default=None, help="Device override: cpu/cuda/mps.")
@click.option("--out-dir", default="outputs/baseline_runs", show_default=True, help="Run output dir.")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress info messages (errors only).")
def train_baseline_cmd(
    dataset_root: str,
    train_split: str,
    val_split: str,
    epochs: int,
    batch_size: int,
    lr: float,
    image_size: int,
    num_workers: int,
    threshold: float,
    seed: int,
    limit_train: int | None,
    limit_val: int | None,
    device: str | None,
    out_dir: str,
    output_format: str,
    verbose: bool,
    quiet: bool,
) -> None:
    """Train the tiny UNet baseline on LEVIR-style data."""
    verbosity = 2 if verbose else (0 if quiet else 1)
    setup_logging(verbosity)

    try:
        from src.training import TrainingConfig, train_baseline

        cfg = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            image_size=image_size,
            num_workers=num_workers,
            threshold=threshold,
            seed=seed,
            limit_train=limit_train,
            limit_val=limit_val,
        )
        summary = train_baseline(
            dataset_root,
            train_split=train_split,
            val_split=val_split,
            out_dir=out_dir,
            config=cfg,
            device=device,
        )

        if output_format == "json":
            click.echo(json.dumps(summary, indent=2))
        else:
            _print_train_summary(summary)

    except ImportError as exc:
        logger.error(
            "Training dependencies missing: %s. Install torch first "
            "(e.g. pip install torch torchvision).",
            exc,
        )
        raise SystemExit(EXIT_INPUT_ERROR)
    except ChangeDetectorError as exc:
        logger.error("%s", exc)
        raise SystemExit(EXIT_INPUT_ERROR)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise SystemExit(EXIT_OK)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        raise SystemExit(EXIT_UNEXPECTED)


@cli.command("reason")
@click.argument("image_a", type=click.Path(exists=True))
@click.argument("image_b", type=click.Path(exists=True))
@click.option(
    "--provider",
    type=click.Choice(["openai", "gemini", "huggingface", "openrouter"], case_sensitive=False),
    default="openai",
    show_default=True,
    help="Vision model provider.",
)
@click.option("--model", default="", help="Model override (provider default if omitted).")
@click.option("--api-key", default=None, help="Optional API key override.")
@click.option("--temperature", type=float, default=0.2, show_default=True, help="Sampling temperature.")
@click.option("--max-tokens", type=int, default=2048, show_default=True, help="Max response tokens.")
@click.option(
    "--use-regions/--no-use-regions",
    default=True,
    show_default=True,
    help="Use detector regions and send only changed crops to the VLM.",
)
@click.option(
    "--region-method",
    type=click.Choice(["pixel", "ssim", "combined", "robust", "drone"], case_sensitive=False),
    default="robust",
    show_default=True,
    help="Detector mode used to find changed regions.",
)
@click.option("--max-regions", type=int, default=5, show_default=True, help="Max changed regions to analyze.")
@click.option("--region-pad", type=int, default=24, show_default=True, help="Padding (px) around each region crop.")
@click.option(
    "--regions-image-out",
    default=None,
    help="Optional path to save detector region preview image before VLM call.",
)
@click.option("--config", "-c", default="config.yaml", help="Path to detector config YAML (for region extraction).")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option("--json-out", default=None, help="Optional path to save JSON output.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress info messages (errors only).")
def reason_cmd(
    image_a: str,
    image_b: str,
    provider: str,
    model: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    use_regions: bool,
    region_method: str,
    max_regions: int,
    region_pad: int,
    regions_image_out: str | None,
    config: str,
    output_format: str,
    json_out: str | None,
    verbose: bool,
    quiet: bool,
) -> None:
    """Reason about semantic changes between IMAGE_A and IMAGE_B with a VLM."""
    verbosity = 2 if verbose else (0 if quiet else 1)
    setup_logging(verbosity)

    try:
        cfg = ReasoningConfig(
            provider=provider.lower(),
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        reasoner = VLMReasoner(config=cfg, api_key=api_key)
        if not use_regions:
            result = reasoner.compare(image_a, image_b)
        else:
            if max_regions <= 0:
                raise ValueError("max_regions must be a positive integer")
            if region_pad < 0:
                raise ValueError("region_pad must be >= 0")

            det_cfg = load_config(config)
            img_a_arr = load_image(image_a)
            img_b_arr = load_image(image_b)
            detector = ChangeDetector(app_config=det_cfg)
            det_result = detector.compare(img_a_arr, img_b_arr, method=region_method.lower())
            regions_preview_out = _auto_regions_output(
                regions_image_out,
                image_a,
                region_method.lower(),
                det_cfg,
            )
            save_result(det_result["diff_image"], regions_preview_out)
            logger.info("Saved region preview → %s", regions_preview_out)

            bboxes = list(det_result.get("bboxes", []))
            bboxes = sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)
            selected = bboxes[:max_regions]
            h, w = img_a_arr.shape[:2]
            if not selected:
                result = {
                    "mode": "region_reasoning",
                    "provider": cfg.provider,
                    "model": cfg.resolved_model,
                    "api_called": False,
                    "detector": {
                        "method": region_method.lower(),
                        "score": det_result.get("score"),
                        "change_percent": det_result.get("change_percent"),
                        "num_regions_detected": 0,
                        "num_regions_analyzed": 0,
                        "region_preview_image": regions_preview_out,
                    },
                    "total_real_changes": 0,
                    "regions": [],
                    "summary": (
                        f"No changed regions detected by {region_method.lower()} detector. "
                        "Skipped VLM API call."
                    ),
                }
                if json_out:
                    out_path = Path(json_out)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(result, indent=2))
                    logger.info("Saved reasoning JSON → %s", out_path)

                if output_format == "json":
                    click.echo(json.dumps(result, indent=2))
                else:
                    _print_region_reasoning_summary(result)
                return

            region_results: list[dict] = []
            usage_total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            usage_seen = False

            with TemporaryDirectory(prefix="icd_regions_") as tmp_dir:
                tmp_root = Path(tmp_dir)
                for idx, (x, y, bw, bh) in enumerate(selected, start=1):
                    x1 = max(0, x - region_pad)
                    y1 = max(0, y - region_pad)
                    x2 = min(w, x + bw + region_pad)
                    y2 = min(h, y + bh + region_pad)

                    crop_a = img_a_arr[y1:y2, x1:x2]
                    crop_b = img_b_arr[y1:y2, x1:x2]
                    path_a = tmp_root / f"region_{idx:02d}_a.jpg"
                    path_b = tmp_root / f"region_{idx:02d}_b.jpg"
                    cv2.imwrite(str(path_a), crop_a)
                    cv2.imwrite(str(path_b), crop_b)

                    analysis = reasoner.compare(path_a, path_b)
                    usage = analysis.get("usage", {})
                    if usage:
                        usage_seen = True
                        for key in ("input_tokens", "output_tokens", "total_tokens"):
                            val = usage.get(key)
                            if isinstance(val, int):
                                usage_total[key] += val

                    region_results.append(
                        {
                            "region_id": idx,
                            "bbox": [int(x), int(y), int(bw), int(bh)],
                            "bbox_padded": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                            "analysis": analysis,
                        }
                    )

            total_real_changes = 0
            for item in region_results:
                val = item["analysis"].get("total_real_changes")
                if isinstance(val, int) and val > 0:
                    total_real_changes += val

            result = {
                "mode": "region_reasoning",
                "provider": cfg.provider,
                "model": cfg.resolved_model,
                "detector": {
                    "method": region_method.lower(),
                    "score": det_result.get("score"),
                    "change_percent": det_result.get("change_percent"),
                    "num_regions_detected": len(bboxes),
                    "num_regions_analyzed": len(region_results),
                    "region_preview_image": regions_preview_out,
                },
                "total_real_changes": total_real_changes,
                "regions": region_results,
            }
            if usage_seen:
                result["usage"] = {
                    "provider": cfg.provider,
                    "model": cfg.resolved_model,
                    "input_tokens": usage_total["input_tokens"],
                    "output_tokens": usage_total["output_tokens"],
                    "total_tokens": usage_total["total_tokens"],
                }
            result["summary"] = (
                f"Analyzed {len(region_results)} changed region(s) from detector method "
                f"{region_method.lower()}. Total real changes from region analyses: {total_real_changes}."
            )

        if json_out:
            out_path = Path(json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2))
            logger.info("Saved reasoning JSON → %s", out_path)

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            if result.get("mode") == "region_reasoning":
                _print_region_reasoning_summary(result)
            else:
                _print_reasoning_summary(result, cfg.provider, cfg.resolved_model)

    except ChangeDetectorError as exc:
        logger.error("%s", exc)
        raise SystemExit(EXIT_INPUT_ERROR)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise SystemExit(EXIT_OK)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        raise SystemExit(EXIT_UNEXPECTED)


@cli.command("compare-videos")
@click.argument("video_a", type=click.Path(exists=True))
@click.argument("video_b", type=click.Path(exists=True))
@click.option("--mode", type=click.Choice(["pixel", "ssim", "combined", "robust", "drone"], case_sensitive=False), default="robust", show_default=True, help="Detection mode used per frame pair.")
@click.option(
    "--alignment",
    "alignment_method",
    type=click.Choice(["feature-dtw", "ratio"], case_sensitive=False),
    default="feature-dtw",
    show_default=True,
    help="Frame alignment strategy before change detection.",
)
@click.option("--min-alignment-score", type=float, default=0.08, show_default=True, help="Minimum visual similarity score to keep aligned frame pairs.")
@click.option("--gap-penalty", type=float, default=-0.05, show_default=True, help="Gap penalty for DTW-style alignment (feature-dtw only).")
@click.option("--min-detector-align-conf", type=float, default=0.0, show_default=True, help="Minimum detector alignment confidence (0-1) to keep a frame pair.")
@click.option("--stabilize/--no-stabilize", default=False, show_default=True, help="Stabilize sampled frames before alignment (recommended for handheld/mobile video).")
@click.option("--background-window", type=int, default=0, show_default=True, help="Temporal median background window (+/- N frames). 0 disables.")
@click.option("--static-std-thresh", type=float, default=0.0, show_default=True, help="Stddev threshold for static-pixel mask (0 disables).")
@click.option("--min-static-coverage", type=float, default=0.0, show_default=True, help="Minimum static-mask coverage (0-1) to keep a pair.")
@click.option("--illumination", "illumination_mode", type=click.Choice(["none", "clahe", "match"], case_sensitive=False), default="none", show_default=True, help="Illumination normalization for frame pairs.")
@click.option("--sample-fps", type=float, default=1.0, show_default=True, help="Frame sampling rate from each video.")
@click.option("--min-change-percent", type=float, default=1.0, show_default=True, help="Minimum change percent to keep a frame pair.")
@click.option("--min-inner-change-percent", type=float, default=None, show_default=True, help="Minimum change percent inside cropped inner region (defaults to min-change-percent).")
@click.option("--min-regions", type=int, default=1, show_default=True, help="Minimum changed regions to keep a frame pair.")
@click.option("--min-persistence", type=int, default=1, show_default=True, help="Keep changes only if they persist across this many consecutive aligned pairs.")
@click.option("--edge-ignore-px", type=int, default=0, show_default=True, help="Ignore changes near borders by this many pixels.")
@click.option("--max-pairs", type=int, default=None, help="Optional cap on compared frame pairs.")
@click.option("--output-dir", default="outputs/video_changes", show_default=True, help="Output folder for changed frames and summary.")
@click.option("--config", "-c", default="config.yaml", help="Path to config YAML.")
@click.option(
    "--format", "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.option("-q", "--quiet", is_flag=True, help="Suppress info messages (errors only).")
def compare_videos_cmd(
    video_a: str,
    video_b: str,
    mode: str,
    alignment_method: str,
    min_alignment_score: float,
    gap_penalty: float,
    min_detector_align_conf: float,
    stabilize: bool,
    background_window: int,
    static_std_thresh: float,
    min_static_coverage: float,
    illumination_mode: str,
    sample_fps: float,
    min_change_percent: float,
    min_inner_change_percent: float | None,
    min_regions: int,
    min_persistence: int,
    edge_ignore_px: int,
    max_pairs: int | None,
    output_dir: str,
    config: str,
    output_format: str,
    verbose: bool,
    quiet: bool,
) -> None:
    """Compare two videos and save only changed frame-pair results."""
    verbosity = 2 if verbose else (0 if quiet else 1)
    setup_logging(verbosity)

    try:
        cfg = load_config(config)
        from src.video_compare import compare_videos

        result = compare_videos(
            video_a,
            video_b,
            app_config=cfg,
            method=mode.lower(),
            alignment_method=alignment_method.lower(),
            min_alignment_score=min_alignment_score,
            gap_penalty=gap_penalty,
            min_detector_alignment_conf=min_detector_align_conf,
            stabilize=stabilize,
            background_window=background_window,
            static_std_thresh=static_std_thresh,
            min_static_coverage=min_static_coverage,
            illumination_mode=illumination_mode.lower(),
            sample_fps=sample_fps,
            min_change_percent=min_change_percent,
            min_inner_change_percent=min_inner_change_percent,
            min_regions=min_regions,
            min_persistence=min_persistence,
            edge_ignore_px=edge_ignore_px,
            max_pairs=max_pairs,
            output_dir=output_dir,
        )

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            _print_video_summary(result)

    except ChangeDetectorError as exc:
        logger.error("%s", exc)
        raise SystemExit(EXIT_INPUT_ERROR)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        raise SystemExit(EXIT_OK)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        raise SystemExit(EXIT_UNEXPECTED)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _auto_output(output: str | None, mode: str, cfg) -> str:
    """Return *output* if set, otherwise generate a timestamped path in outputs/."""
    if output:
        return output
    out_dir = Path(cfg.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(out_dir / f"{mode}_{ts}.png")


def _auto_regions_output(
    output: str | None,
    image_a: str,
    method: str,
    cfg,
) -> str:
    """Return region preview path (custom or auto under configured outputs/)."""
    if output:
        return output
    out_dir = Path(cfg.output.directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_a).stem.replace(" ", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(out_dir / f"regions_{stem}_{method}_{ts}.png")


def _run_change_detection(
    img_a, img_b, method: str, cfg, output, show,
) -> dict:
    """Run pixel / SSIM / combined detection."""
    logger.info("Running %s change detection ...", method.upper())

    detector = ChangeDetector(app_config=cfg)
    result = detector.compare(img_a, img_b, method=method)

    out_path = _auto_output(output, method, cfg)
    save_result(result["diff_image"], out_path)
    logger.info("Saved result → %s", out_path)

    if show:
        plot_comparison(
            img_a, img_b, result["diff_image"], result["mask"],
            title=(
                f"Change Detection — {method.upper()} "
                f"(SSIM: {result['score']}, Δ: {result['change_percent']}%)"
            ),
        )

    return result


def _run_object_detection(img_a, img_b, cfg, output, show) -> dict:
    """Run YOLO object-level detection with optional pre-alignment."""
    logger.info("Running object-level change detection (YOLOv8) ...")

    img_a_r, img_b_r = resize_to_match(img_a, img_b)
    img_b_for_compare = img_b_r
    alignment_confidence: float | None = None

    # Try to align first
    try:
        from src.aligner import ImageAligner

        aligner = ImageAligner(config=cfg.alignment)
        alignment = aligner.align(img_a_r, img_b_r)
        alignment_confidence = alignment.confidence

        if alignment.success and alignment.confidence > 0.15:
            logger.info("Pre-aligned images (confidence: %.1f%%)",
                        alignment.confidence * 100)
            img_b_for_compare = alignment.warped
        else:
            logger.info("Alignment confidence too low — comparing without alignment")
    except Exception:
        logger.debug("Alignment skipped", exc_info=True)

    from src.object_detector import ObjectDetector, ObjectChange

    if cfg.object_detector.class_agnostic_output:
        # Class-agnostic change regions from image differencing.
        detector = ChangeDetector(app_config=cfg)
        region_result = detector.compare(img_a_r, img_b_for_compare, method="combined")
        bboxes = region_result.get("bboxes", [])
        changes = [
            ObjectChange(
                change_type="changed",
                label="unknown_object",
                confidence=1.0,
                bbox_a=b,
                bbox_b=b,
            )
            for b in bboxes
        ]

        annotated_a = img_a_r.copy()
        annotated_b = img_b_for_compare.copy()
        for x, y, w, h in bboxes:
            cv2.rectangle(annotated_a, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cv2.rectangle(annotated_b, (x, y), (x + w, y + h), (0, 255, 100), 2)
            cv2.putText(
                annotated_a, "changed", (x, max(y - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
            )
            cv2.putText(
                annotated_b, "changed", (x, max(y - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA,
            )

        summary = {
            "added": 0,
            "removed": 0,
            "moved": 0,
            "changed": len(changes),
            "total_changes": len(changes),
        }
        result = {
            "detections_a": [],
            "detections_b": [],
            "changes": changes,
            "annotated_a": annotated_a,
            "annotated_b": annotated_b,
            "summary": summary,
            "class_agnostic": True,
        }
        logger.info("Class-agnostic object mode: %d changed region(s)", len(changes))
    else:
        obj_detector = ObjectDetector(config=cfg.object_detector)
        result = obj_detector.compare(
            img_a_r,
            img_b_for_compare,
            alignment_confidence=alignment_confidence,
        )

    if output:
        side_by_side = cv2.hconcat([result["annotated_a"], result["annotated_b"]])
        save_result(side_by_side, output)

    if show:
        plot_object_changes(
            result["annotated_a"], result["annotated_b"], result["changes"],
        )

    return result



def _run_drone_detection(img_a, img_b, cfg, output, show) -> dict:
    """Run optimised drone pipeline: similarity align → normalize → combined diff."""
    logger.info("Running DRONE change detection ...")

    detector = ChangeDetector(app_config=cfg)
    result = detector.compare(img_a, img_b, method="drone")

    out_path = _auto_output(output, "drone", cfg)
    save_result(result["diff_image"], out_path)
    logger.info("Saved result → %s", out_path)

    if show:
        sf = result.get("alignment", {}).get("scale_factor", 1.0)
        plot_comparison(
            img_a, img_b, result["diff_image"], result["mask"],
            title=(
                f"Drone Change Detection "
                f"(SSIM: {result['score']}, Δ: {result['change_percent']}%, "
                f"scale: {sf:.3f}x)"
            ),
        )

    return result

def _run_robust_detection(img_a, img_b, cfg, output, show) -> dict:
    """Run align → normalize → multi-scale detection."""
    logger.info("Running ROBUST change detection ...")

    detector = ChangeDetector(app_config=cfg)
    result = detector.compare(img_a, img_b, method="robust")

    out_path = _auto_output(output, "robust", cfg)
    save_result(result["diff_image"], out_path)
    logger.info("Saved result → %s", out_path)

    if show:
        plot_comparison(
            img_a, img_b, result["diff_image"], result["mask"],
            title=(
                f"Robust Change Detection "
                f"(SSIM: {result['score']}, Δ: {result['change_percent']}%)"
            ),
        )

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_eval_summary(result: dict) -> None:
    """Render a compact dataset-level evaluation summary."""
    metrics = result["metrics"]

    click.echo()
    click.echo("📊 Dataset Evaluation")
    click.echo(f"  Dataset: {result['dataset_root']}")
    click.echo(f"  Split:   {result['split']}")
    click.echo(f"  Method:  {result['method']}")
    click.echo(f"  Samples: {result['num_samples']}")
    click.echo(f"  Batch:   {result.get('batch_size', 1)}")
    click.echo(f"  Workers: {result.get('num_workers', 0)}")
    click.echo()
    click.echo(f"  Precision: {metrics['precision']:.4f}")
    click.echo(f"  Recall:    {metrics['recall']:.4f}")
    click.echo(f"  F1:        {metrics['f1']:.4f}")
    click.echo(f"  IoU:       {metrics['iou']:.4f}")
    click.echo(f"  Accuracy:  {metrics['accuracy']:.4f}")
    click.echo()
    click.echo(
        f"  Confusion: TP={metrics['tp']} TN={metrics['tn']} "
        f"FP={metrics['fp']} FN={metrics['fn']}"
    )
    click.echo()


def _print_train_summary(summary: dict) -> None:
    """Render a compact training summary."""
    click.echo()
    click.echo("🏋️ Baseline Training Complete")
    click.echo(f"  Device:        {summary['device']}")
    click.echo(f"  Train samples: {summary['train_samples']}")
    click.echo(f"  Val samples:   {summary['val_samples']}")
    click.echo(f"  Epochs:        {summary['epochs']}")
    click.echo(f"  Batch size:    {summary['batch_size']}")
    click.echo(f"  LR:            {summary['learning_rate']}")
    click.echo(f"  Best val F1:   {summary['best_f1']:.4f}")
    click.echo()
    click.echo(f"  Run dir:       {summary['run_dir']}")
    click.echo(f"  Best ckpt:     {summary['best_checkpoint']}")
    click.echo(f"  Final ckpt:    {summary['final_checkpoint']}")
    click.echo(f"  History:       {summary['history_path']}")
    click.echo()


def _print_reasoning_summary(result: dict, provider: str, model: str) -> None:
    """Render a compact semantic change summary from VLM output."""
    changes = result.get("changes", [])
    usage = result.get("usage", {})
    click.echo()
    click.echo("🧠 Semantic Change Reasoning")
    click.echo(f"  Provider: {provider}")
    click.echo(f"  Model:    {model}")
    click.echo(f"  Changes:  {result.get('total_real_changes', len(changes))}")
    if usage:
        click.echo(
            "  Tokens:   "
            f"in={usage.get('input_tokens', 'N/A')} "
            f"out={usage.get('output_tokens', 'N/A')} "
            f"total={usage.get('total_tokens', 'N/A')}"
        )
    click.echo()
    click.echo("Summary:")
    click.echo(f"  {result.get('summary', 'N/A')}")
    click.echo()
    if changes:
        click.echo("Top change items:")
        for idx, item in enumerate(changes[:5], start=1):
            change_type = item.get("type", "unknown")
            obj = item.get("object", "unknown")
            conf = item.get("confidence", "N/A")
            desc = item.get("description", "")
            click.echo(f"  {idx}. [{change_type}] {obj} (conf: {conf})")
            if desc:
                click.echo(f"     {desc}")
        if len(changes) > 5:
            click.echo(f"  ... and {len(changes) - 5} more")
    click.echo()


def _print_region_reasoning_summary(result: dict) -> None:
    """Render a compact summary for detector-region + VLM reasoning."""
    det = result.get("detector", {})
    usage = result.get("usage", {})
    regions = result.get("regions", [])
    click.echo()
    click.echo("🧠 Region-Based Semantic Reasoning")
    click.echo(f"  Provider: {result.get('provider', 'N/A')}")
    click.echo(f"  Model:    {result.get('model', 'N/A')}")
    click.echo(f"  Method:   {det.get('method', 'N/A')}")
    click.echo(f"  Regions:  {det.get('num_regions_analyzed', 0)} analyzed / {det.get('num_regions_detected', 0)} detected")
    click.echo(f"  Changes:  {result.get('total_real_changes', 0)}")
    if det.get("region_preview_image"):
        click.echo(f"  Preview:  {det.get('region_preview_image')}")
    if usage:
        click.echo(
            "  Tokens:   "
            f"in={usage.get('input_tokens', 'N/A')} "
            f"out={usage.get('output_tokens', 'N/A')} "
            f"total={usage.get('total_tokens', 'N/A')}"
        )
    click.echo()
    click.echo("Summary:")
    click.echo(f"  {result.get('summary', 'N/A')}")
    click.echo()
    if regions:
        click.echo("Region details:")
        for item in regions:
            rid = item.get("region_id")
            bbox = item.get("bbox")
            analysis = item.get("analysis", {})
            n = analysis.get("total_real_changes", 0)
            s = analysis.get("summary", "")
            click.echo(f"  {rid}. bbox={bbox} changes={n}")
            if s:
                click.echo(f"     {s}")
    click.echo()


def _print_video_summary(result: dict) -> None:
    """Render a compact summary for video-to-video comparison."""
    click.echo()
    click.echo("🎬 Video Change Comparison")
    click.echo(f"  Video A:         {result.get('video_a')}")
    click.echo(f"  Video B:         {result.get('video_b')}")
    click.echo(f"  Method:          {result.get('method')}")
    click.echo(f"  Alignment:       {result.get('alignment_method')}")
    click.echo(f"  Stabilize:       {result.get('stabilize')}")
    click.echo(f"  BG window:       {result.get('background_window')}")
    click.echo(f"  Static std:      {result.get('static_std_thresh')}")
    click.echo(f"  Static cover:    {result.get('min_static_coverage')}")
    click.echo(f"  Illumination:    {result.get('illumination_mode')}")
    if result.get("alignment_method") == "feature-dtw":
        click.echo(f"  Min align score: {result.get('min_alignment_score')}")
        click.echo(f"  Min det align:   {result.get('min_detector_alignment_conf')}")
        click.echo(f"  Avg similarity:  {result.get('avg_similarity', 0.0):.4f}")
        click.echo(f"  Kept pairs:      {result.get('kept_aligned_pairs', 0)} / {result.get('raw_aligned_pairs', 0)}")
    click.echo(f"  Sample FPS:      {result.get('sample_fps')}")
    click.echo(f"  Sampled A/B:     {result.get('num_sampled_a')}/{result.get('num_sampled_b')}")
    click.echo(f"  Pairs compared:  {result.get('num_pairs_compared')}")
    click.echo(f"  Skipped low conf:{result.get('num_pairs_skipped_low_detector_alignment_conf', 0)}")
    click.echo(f"  Candidates pre-p:{result.get('num_change_candidates_before_persistence', 0)}")
    click.echo(f"  Edge ignore px:  {result.get('edge_ignore_px')}")
    click.echo(f"  Min inner %:     {result.get('min_inner_change_percent')}")
    click.echo(f"  Min persistence: {result.get('min_persistence')}")
    click.echo(f"  Changed pairs:   {result.get('num_changed_pairs')}")
    click.echo(f"  Output folder:   {result.get('changed_dir')}")
    changed_dir = Path(result.get("changed_dir", ""))
    summary_path = changed_dir.parent / "summary.json" if changed_dir else Path("summary.json")
    click.echo(f"  Summary JSON:    {summary_path}")
    click.echo()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Package entry point (``image-change-detector`` command)."""
    cli()


if __name__ == "__main__":
    main()
