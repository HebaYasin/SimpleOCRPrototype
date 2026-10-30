"""
backend/utils.py
----------------
Miscellaneous utility functions shared across the backend:
  - File export helpers (return bytes/str for st.download_button)
  - Temporary file management
No Streamlit imports.
"""

import json
import os
import tempfile
from io import BytesIO

import pandas as pd
from PIL import Image

from backend.logger import get_logger

logger = get_logger(__name__)


# ── Export helpers ─────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame) -> bytes:
    """Return UTF-8 CSV bytes from the words DataFrame."""
    logger.info("Exporting CSV")
    return df.to_csv(index=False).encode("utf-8")


def export_json(structured_result: dict) -> bytes:
    """Return indented JSON bytes from a docTR export dict."""
    logger.info("Exporting JSON")
    return json.dumps(
        structured_result,
        indent=2,
        default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
    ).encode("utf-8")


def export_png(pil_image: Image.Image) -> bytes:
    """Return PNG bytes from a PIL Image."""
    logger.info("Exporting PNG")
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def export_plain_text(df: pd.DataFrame) -> str:
    """Return extracted text as a plain newline-separated string."""
    logger.info("Exporting plain text")
    return "\n".join(df["text"].tolist())


# ── Temp file helpers ──────────────────────────────────────────────────────

def save_temp_file(data: bytes, suffix: str) -> str:
    """
    Write *data* to a named temporary file and return its path.
    The caller is responsible for deleting it (use `cleanup_temp_file`).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    logger.info(f"Temp file created: {path}")
    return path


def cleanup_temp_file(path: str) -> None:
    """Delete a temporary file, silently ignoring errors."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.info(f"Temp file deleted: {path}")
    except Exception as exc:
        logger.warning(f"Could not delete temp file {path}: {exc}")
