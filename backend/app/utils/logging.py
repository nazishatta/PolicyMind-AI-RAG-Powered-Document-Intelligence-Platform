"""Structured logging configuration using structlog."""

from __future__ import annotations

import io
import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog with a clean console renderer."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Force UTF-8 on the output stream so Unicode characters in log messages
    # (arrows, Greek letters, box-drawing chars) don't crash on Windows cp1252.
    try:
        stream = sys.stdout.reconfigure(encoding="utf-8") or sys.stdout  # type: ignore[union-attr]
    except (AttributeError, io.UnsupportedOperation):
        stream = sys.stdout

    logging.basicConfig(
        format="%(message)s",
        stream=stream,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
