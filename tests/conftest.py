"""
tests/conftest.py
-----------------
Shared pytest fixtures and configuration.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def bgr_image() -> np.ndarray:
    """A 200×200 BGR test image with some features."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (80, 80), (255, 0, 0), -1)
    cv2.circle(img, (150, 150), 30, (0, 255, 0), -1)
    return img


@pytest.fixture
def gray_image() -> np.ndarray:
    """A 200×200 grayscale test image."""
    return np.random.randint(0, 256, (200, 200), dtype=np.uint8)


@pytest.fixture
def image_pair() -> tuple[np.ndarray, np.ndarray]:
    """A pair of slightly different BGR images."""
    img_a = np.zeros((200, 200, 3), dtype=np.uint8)
    img_a[:] = (100, 100, 100)

    img_b = img_a.copy()
    img_b[60:140, 60:140] = (200, 200, 200)

    return img_a, img_b
