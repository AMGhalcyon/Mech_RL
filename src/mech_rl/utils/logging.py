"""Project-wide logging configuration.

Uses stdlib `logging` with a single setup function. We avoid heavyweight
formatters; a concise format that includes time, level, and logger name
covers our needs.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

_DEFAULT_LEVEL: Final[int] = logging.INFO

_configured = False


def setup_logging(level: int = _DEFAULT_LEVEL) -> None:
    """Configure the root logger. Idempotent — safe to call multiple times."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module. Cheap; no setup needed."""
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
