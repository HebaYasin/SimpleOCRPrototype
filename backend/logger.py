"""
backend/logger.py
-----------------
Centralised logging configuration.
Import `get_logger(__name__)` in every module.
"""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger once at application startup."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ocr_engine.log", encoding="utf-8"),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Call as: logger = get_logger(__name__)"""
    return logging.getLogger(name)
