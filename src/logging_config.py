"""
logging_config.py
-----------------
Centralized logging configuration for image-change-detector.

Usage::

    from src.logging_config import get_logger, setup_logging

    # Module-level logger (use __name__ for automatic module path)
    logger = get_logger(__name__)

    # At application startup (CLI entry point)
    setup_logging(verbosity=1)  # 0=quiet, 1=normal, 2=verbose/debug
"""

from __future__ import annotations

import logging
import sys
from typing import Final

# Package root logger name — all child loggers inherit its config
PACKAGE_NAME: Final[str] = "image_change_detector"

# Format strings
_SIMPLE_FORMAT: Final[str] = "%(levelname)-8s %(message)s"
_VERBOSE_FORMAT: Final[str] = (
    "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
)
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger scoped under the package namespace.

    Args:
        name: Typically ``__name__`` from the calling module.

    Returns:
        A :class:`logging.Logger` instance.
    """
    # Ensure child loggers are namespaced under the package
    if not name.startswith(PACKAGE_NAME):
        name = f"{PACKAGE_NAME}.{name}"
    return logging.getLogger(name)


def setup_logging(verbosity: int = 1) -> None:
    """
    Configure the package-level logger.

    Args:
        verbosity:
            - ``0`` — QUIET: only warnings and errors.
            - ``1`` — NORMAL: info-level messages (default).
            - ``2`` — VERBOSE/DEBUG: debug-level with timestamps.
    """
    root_logger = logging.getLogger(PACKAGE_NAME)

    # Remove any existing handlers to avoid duplicates on re-init
    root_logger.handlers.clear()

    if verbosity <= 0:
        level = logging.WARNING
        fmt = _SIMPLE_FORMAT
    elif verbosity == 1:
        level = logging.INFO
        fmt = _SIMPLE_FORMAT
    else:
        level = logging.DEBUG
        fmt = _VERBOSE_FORMAT

    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))

    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("ultralytics", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
