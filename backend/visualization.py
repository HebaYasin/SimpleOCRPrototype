"""
backend/visualization.py
------------------------
All figure/image generation.  Returns PIL Images or Matplotlib Figures.
Callers decide how to render them.  No Streamlit imports.
"""

import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

from backend.config import DEJAVU_FONT_PATH
from backend.logger import get_logger

logger = get_logger(__name__)


# ── PIL overlay (fast) ─────────────────────────────────────────────────────

def create_pil_visualization(
    image: np.ndarray,
    words_data: list[dict],
    show_heatmap: bool,
    show_boxes: bool,
    show_text_labels: bool,
) -> Image.Image:
    """
    Draw confidence overlays using PIL.
    Recommended for large documents — significantly faster than Matplotlib.

    Returns a PIL RGBA Image.
    """
    logger.info(f"PIL viz — {len(words_data)} words")
    t0 = time.time()

    pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image.copy()
    draw = ImageDraw.Draw(pil_img, "RGBA")

    try:
        font = ImageFont.truetype(DEJAVU_FONT_PATH, 12)
        small_font = ImageFont.truetype(DEJAVU_FONT_PATH, 10)
    except Exception:
        logger.warning("DejaVu font not found — falling back to default")
        font = small_font = ImageFont.load_default()

    for i, w in enumerate(words_data):
        if i % 100 == 0:
            logger.debug(f"PIL drawing word {i}/{len(words_data)}")

        conf = w["confidence"]
        x0, y0, x1, y1 = w["x_min"], w["y_min"], w["x_max"], w["y_max"]

        # Red → yellow → green ramp
        r = 255 if conf < 0.5 else int(255 * (1 - (conf - 0.5) * 2))
        g = int(255 * conf * 2) if conf < 0.5 else 255
        b = 0

        if show_heatmap:
            alpha = int((0.15 + conf * 0.35) * 255)
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, alpha))

        if show_boxes:
            draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 200), width=2)

        if show_text_labels:
            label = f"{w['text']} ({conf:.2f})"
            ty = max(5, y0 - 15)
            bbox = draw.textbbox((x0, ty), label, font=small_font)
            draw.rectangle(bbox, fill=(255, 255, 255, 200))
            draw.text((x0, ty), label, fill=(0, 0, 255, 255), font=small_font)

    logger.info(f"PIL viz done in {time.time() - t0:.2f}s")
    return pil_img


# ── Matplotlib overlay (detailed) ─────────────────────────────────────────

def create_matplotlib_visualization(
    image: np.ndarray,
    words_data: list[dict],
    show_heatmap: bool,
    show_boxes: bool,
    show_text_labels: bool,
) -> plt.Figure:
    """
    Draw confidence overlays using Matplotlib patches.
    More configurable; slower for large word counts.
    Caller must call plt.close(fig) when done.
    """
    logger.info(f"Matplotlib viz — {len(words_data)} words")
    t0 = time.time()

    fig, ax = plt.subplots(1, figsize=(12, 16))
    ax.imshow(image)
    ax.axis("off")

    for i, w in enumerate(words_data):
        if i % 50 == 0:
            logger.debug(f"MPL drawing word {i}/{len(words_data)}")

        conf = w["confidence"]
        x0, y0, x1, y1 = w["x_min"], w["y_min"], w["x_max"], w["y_max"]
        color = (1 - conf, conf, 0)

        if show_heatmap:
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=0, facecolor=color, alpha=0.15 + conf * 0.4,
                )
            )
        if show_boxes:
            ax.add_patch(
                patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=1.2, edgecolor="red", facecolor="none",
                )
            )
        if show_text_labels:
            ax.text(
                x0, max(5, y0 - 3),
                f"{w['text']} ({conf:.2f})",
                color="blue", fontsize=6, backgroundcolor="white",
            )

    logger.info(f"Matplotlib viz done in {time.time() - t0:.2f}s")
    return fig


# ── Segmentation heatmap ───────────────────────────────────────────────────

def create_segmentation_figure(seg_map: np.ndarray) -> plt.Figure:
    """
    Render the raw detection probability map.
    Caller must call plt.close(fig) when done.
    """
    fig, ax = plt.subplots()
    ax.imshow(seg_map, cmap="hot")
    ax.axis("off")
    return fig


# ── Preprocessing diff map ─────────────────────────────────────────────────

def create_diff_map(original: np.ndarray, processed: np.ndarray) -> np.ndarray | None:
    """
    Return a contrast-enhanced absolute-difference image, or None if shapes differ.
    """
    if original.shape != processed.shape:
        return None
    diff = cv2.absdiff(original, processed)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
