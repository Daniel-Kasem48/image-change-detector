"""
tests/test_training.py
----------------------
Unit tests for deep baseline training utilities.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def _write_rgb(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)


def _write_gray(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img)


def _mk_img(v: int, size: int = 32) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = (v, v, v)
    return img


def _mk_mask(size: int = 32, changed: bool = True) -> np.ndarray:
    m = np.zeros((size, size), dtype=np.uint8)
    if changed:
        m[8:24, 8:24] = 255
    return m


@pytest.mark.gpu
def test_pair_change_dataset_shapes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    split_root = tmp_path / "train"
    _write_rgb(split_root / "A" / "001.png", _mk_img(10))
    _write_rgb(split_root / "B" / "001.png", _mk_img(250))
    _write_gray(split_root / "label" / "001.png", _mk_mask(changed=True))

    from src.training import PairChangeDataset

    ds = PairChangeDataset(tmp_path, split="train", image_size=64)
    x, y = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (6, 64, 64)
    assert y.shape == (1, 64, 64)


@pytest.mark.gpu
def test_estimate_pos_weight_positive(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    split_root = tmp_path / "train"
    _write_rgb(split_root / "A" / "001.png", _mk_img(10))
    _write_rgb(split_root / "B" / "001.png", _mk_img(20))
    _write_gray(split_root / "label" / "001.png", _mk_mask(changed=True))

    _write_rgb(split_root / "A" / "002.png", _mk_img(30))
    _write_rgb(split_root / "B" / "002.png", _mk_img(40))
    _write_gray(split_root / "label" / "002.png", _mk_mask(changed=False))

    from src.training import PairChangeDataset, estimate_pos_weight

    ds = PairChangeDataset(tmp_path, split="train", image_size=32)
    pw = estimate_pos_weight(ds)

    assert pw > 0
