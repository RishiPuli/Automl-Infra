"""
utils/logger.py
---------------
Centralized structured logger for CloudAutoML.

Features:
  - Console output with colour-coded severity levels
  - Rotating file handler (logs/cloudautoml.log)
  - ISO-8601 timestamps
  - Thread-safe singleton
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "cloudautoml.log")

# ANSI colour codes for console output
_COLOURS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}


class ColourFormatter(logging.Formatter):
    """Apply ANSI colours to log level names in console output."""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATE_FMT = "%Y-%m-%dT%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour  = _COLOURS.get(record.levelname, _COLOURS["RESET"])
        reset   = _COLOURS["RESET"]
        record.levelname = f"{colour}{record.levelname}{reset}"
        formatter = logging.Formatter(self.FMT, datefmt=self.DATE_FMT)
        return formatter.format(record)


def get_logger(name: str = "CloudAutoML", level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a configured logger.  Calling this multiple times with the same
    name returns the same logger instance (Python's built-in dedup).
    """
    logger = logging.getLogger(name)

    if logger.handlers:          # already configured — return as-is
        return logger

    logger.setLevel(level)

    # ── Console handler ──────────────────────────────────────────────────────
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(ColourFormatter())
    logger.addHandler(sh)

    # ── Rotating file handler (optional — skipped on read-only filesystems) ──
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        fh = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        logger.addHandler(fh)
    except Exception:
        pass   # Cloud env may have read-only FS — console logging is enough

    return logger


# Module-level default logger
log = get_logger("CloudAutoML")
