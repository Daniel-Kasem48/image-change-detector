"""
training.py
-----------
Minimal deep-learning baseline for binary change detection.

This module provides:
- LEVIR-style dataset loader (A/B/label)
- Compact 6-channel UNet baseline
- Train/validation loop with F1/IoU metrics
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .benchmark import discover_levir_samples
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters for the baseline model."""

    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-3
    image_size: int = 256
    num_workers: int = 0
    threshold: float = 0.5
    seed: int = 42
    limit_train: int | None = None
    limit_val: int | None = None


class _ConvBlock:
    """Lazily-built nn.Module wrapper to keep imports local to torch usage."""

    def __init__(self, in_ch: int, out_ch: int):
        import torch.nn as nn

        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class TinyChangeUNet:  # runtime nn.Module
    """Tiny U-Net for 6-channel pair input (A||B)."""

    def __init__(self) -> None:
        import torch.nn as nn

        class _TinyChangeUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = _ConvBlock(6, 32).layer
                self.enc2 = _ConvBlock(32, 64).layer
                self.bottleneck = _ConvBlock(64, 128).layer
                self.pool = nn.MaxPool2d(2)
                self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.dec2 = _ConvBlock(128, 64).layer
                self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
                self.dec1 = _ConvBlock(64, 32).layer
                self.out_head = nn.Conv2d(32, 1, kernel_size=1)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                b = self.bottleneck(self.pool(e2))

                d2 = self.up2(b)
                d2 = self.dec2(_match_and_concat(d2, e2))

                d1 = self.up1(d2)
                d1 = self.dec1(_match_and_concat(d1, e1))

                return self.out_head(d1)

        self.model = _TinyChangeUNet()


def _match_and_concat(x_up, x_skip):
    """Pad/crop upsampled tensor when spatial dims differ by 1 due to pooling."""
    import torch

    _, _, h_u, w_u = x_up.shape
    _, _, h_s, w_s = x_skip.shape

    if h_u != h_s or w_u != w_s:
        x_up = torch.nn.functional.interpolate(
            x_up,
            size=(h_s, w_s),
            mode="bilinear",
            align_corners=False,
        )

    return torch.cat([x_up, x_skip], dim=1)


class PairChangeDataset:  # runtime torch Dataset
    """Torch dataset for LEVIR-style A/B/label folders."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str,
        image_size: int = 256,
        limit: int | None = None,
    ) -> None:
        self.image_size = image_size
        self.samples = discover_levir_samples(dataset_root, split=split)
        if limit is not None:
            self.samples = self.samples[:limit]

        if not self.samples:
            raise ValueError(f"No samples found for split={split!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch

        sample = self.samples[idx]
        img_a = cv2.imread(str(sample.image_a), cv2.IMREAD_COLOR)
        img_b = cv2.imread(str(sample.image_b), cv2.IMREAD_COLOR)
        label = cv2.imread(str(sample.label), cv2.IMREAD_GRAYSCALE)

        if img_a is None or img_b is None or label is None:
            raise ValueError(f"Failed loading sample {sample.sample_id}")

        size = (self.image_size, self.image_size)
        img_a = cv2.resize(img_a, size, interpolation=cv2.INTER_AREA)
        img_b = cv2.resize(img_b, size, interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)

        # BGR -> RGB and [0,1]
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        label = (label > 0).astype(np.float32)

        # [H,W,C] -> [C,H,W]
        t_a = torch.from_numpy(np.transpose(img_a, (2, 0, 1)))
        t_b = torch.from_numpy(np.transpose(img_b, (2, 0, 1)))
        t_y = torch.from_numpy(label[None, ...])

        # 6-channel pair baseline
        x = torch.cat([t_a, t_b], dim=0)
        return x, t_y


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def estimate_pos_weight(dataset: PairChangeDataset, max_samples: int = 256) -> float:
    """Estimate positive-class weight from dataset masks."""
    samples = min(len(dataset), max_samples)
    pos = 0.0
    total = 0.0

    for i in range(samples):
        _, y = dataset[i]
        y_np = y.numpy()
        pos += float(y_np.sum())
        total += float(y_np.size)

    neg = max(total - pos, 1.0)
    pos = max(pos, 1.0)
    return float(neg / pos)


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def evaluate_model(model, loader, device: str, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate binary metrics on a dataloader."""
    import torch

    model.eval()
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            prob = torch.sigmoid(logits)
            pred = (prob >= threshold).float()

            tp += int(((pred == 1) & (y == 1)).sum().item())
            tn += int(((pred == 0) & (y == 0)).sum().item())
            fp += int(((pred == 1) & (y == 0)).sum().item())
            fn += int(((pred == 0) & (y == 1)).sum().item())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": acc,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def train_baseline(
    dataset_root: str | Path,
    *,
    train_split: str = "train",
    val_split: str = "val",
    out_dir: str | Path = "outputs/baseline_runs",
    config: TrainingConfig | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Train the tiny UNet change-detection baseline.

    Returns:
        Summary dict with best metrics and saved checkpoints.
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    cfg = config or TrainingConfig()
    set_seed(cfg.seed)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PairChangeDataset(
        dataset_root,
        split=train_split,
        image_size=cfg.image_size,
        limit=cfg.limit_train,
    )
    val_ds = PairChangeDataset(
        dataset_root,
        split=val_split,
        image_size=cfg.image_size,
        limit=cfg.limit_val,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = TinyChangeUNet().model.to(resolved_device)

    pos_weight = estimate_pos_weight(train_ds)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=resolved_device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    run_dir = Path(out_dir) / f"tiny_unet_e{cfg.epochs}_bs{cfg.batch_size}_lr{cfg.learning_rate}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training on device=%s", resolved_device)
    logger.info("Train/Val samples: %d/%d", len(train_ds), len(val_ds))
    logger.info("Estimated pos_weight=%.4f", pos_weight)

    best_f1 = -1.0
    best_path = run_dir / "best.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for x, y in train_loader:
            x = x.to(resolved_device)
            y = y.to(resolved_device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=resolved_device,
            threshold=cfg.threshold,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
        }
        history.append(record)

        logger.info(
            "Epoch %d/%d - loss=%.5f val_f1=%.4f val_iou=%.4f",
            epoch,
            cfg.epochs,
            train_loss,
            val_metrics["f1"],
            val_metrics["iou"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)

    final_path = run_dir / "final.pt"
    torch.save(model.state_dict(), final_path)

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))

    summary = {
        "run_dir": str(run_dir.resolve()),
        "device": resolved_device,
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "image_size": cfg.image_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_f1": best_f1,
        "best_checkpoint": str(best_path.resolve()),
        "final_checkpoint": str(final_path.resolve()),
        "history_path": str(history_path.resolve()),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary
