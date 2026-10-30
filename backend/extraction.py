"""
backend/extraction.py
---------------------
Extract structured word data from a docTR Page and build analysis outputs.
No Streamlit imports.
"""

import numpy as np
import pandas as pd

from backend.logger import get_logger

logger = get_logger(__name__)


def extract_words(
    page,
    image_shape: tuple,
    conf_threshold: float,
    max_results: int,
) -> tuple[list[dict], int, int]:
    """
    Walk the docTR block → line → word hierarchy and collect per-word data.

    Parameters
    ----------
    page           : docTR Page object
    image_shape    : (H, W, C) tuple of the processed image
    conf_threshold : words below this confidence are skipped
    max_results    : hard cap on returned words

    Returns
    -------
    words_data     : list of word dicts (text, confidence, bounding box px coords)
    total_words    : total words seen before any filtering
    filtered_words : words that passed the confidence threshold
    """
    h, w = image_shape[:2]
    logger.info(f"Extracting words — image {w}×{h}, conf≥{conf_threshold}, cap={max_results}")

    words_data: list[dict] = []
    total_words = 0
    filtered_words = 0

    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                total_words += 1

                if word.confidence < conf_threshold:
                    continue
                filtered_words += 1

                coords = np.array(word.geometry)
                coords[:, 0] *= w
                coords[:, 1] *= h

                words_data.append(
                    {
                        "text":       word.value,
                        "confidence": round(word.confidence, 3),
                        "x_min":      int(coords[:, 0].min()),
                        "y_min":      int(coords[:, 1].min()),
                        "x_max":      int(coords[:, 0].max()),
                        "y_max":      int(coords[:, 1].max()),
                    }
                )

                if len(words_data) >= max_results:
                    logger.warning(f"Hit max_results cap ({max_results})")
                    return words_data, total_words, filtered_words

    logger.info(
        f"Extraction done — total={total_words}, passed={filtered_words}, returned={len(words_data)}"
    )
    return words_data, total_words, filtered_words


def build_dataframe(words_data: list[dict]) -> pd.DataFrame:
    """Convert the words list into a Pandas DataFrame."""
    return pd.DataFrame(words_data)


def confidence_band_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return word counts split into four confidence bands."""
    if df.empty:
        return {"High (≥0.9)": 0, "Good (0.7–0.9)": 0, "Medium (0.5–0.7)": 0, "Low (<0.5)": 0}
    c = df["confidence"]
    return {
        "High (≥0.9)":    int((c >= 0.9).sum()),
        "Good (0.7–0.9)": int(((c >= 0.7) & (c < 0.9)).sum()),
        "Medium (0.5–0.7)": int(((c >= 0.5) & (c < 0.7)).sum()),
        "Low (<0.5)":     int((c < 0.5).sum()),
    }
