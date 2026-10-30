"""
backend/inference.py
--------------------
Forward-pass helpers: full OCR run and raw segmentation map generation.
No Streamlit imports.
"""

import time

import cv2
import numpy as np
import torch
from doctr.io import DocumentFile

from backend.logger import get_logger

logger = get_logger(__name__)


def load_document(file_path: str, is_pdf: bool) -> list:
    """
    Load a file from disk into a docTR DocumentFile.

    Returns
    -------
    doc : list-like of page images (RGB NumPy arrays)
    """
    logger.info(f"Loading document — pdf={is_pdf}  path={file_path}")
    t0 = time.time()
    doc = DocumentFile.from_pdf(file_path) if is_pdf else DocumentFile.from_images(file_path)
    logger.info(f"Loaded {len(doc)} page(s) in {time.time() - t0:.2f}s")
    return doc


def run_ocr(predictor, image: np.ndarray) -> tuple:
    """
    Run full OCR on a single image.

    Returns
    -------
    result : docTR Document
    page   : first Page object from result.pages
    """
    logger.info("Running OCR inference")
    t0 = time.time()
    result = predictor([image])
    page = result.pages[0]
    logger.info(f"OCR done in {time.time() - t0:.2f}s")
    return result, page


def get_segmentation_map(
    predictor,
    image: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Run only the detection stage to obtain the raw segmentation probability map.
    The map is resized to match the input image dimensions.

    Returns
    -------
    seg_map : 2-D float32 NumPy array, H × W
    """
    logger.info("Generating segmentation map")
    t0 = time.time()

    with torch.no_grad():
        batches = predictor.det_predictor.pre_processor([image])
        out = predictor.det_predictor.model(batches[0].to(device), return_model_output=True)

    seg_map = np.squeeze(out["out_map"].to("cpu").numpy())
    seg_map = cv2.resize(
        seg_map,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    logger.info(f"Segmentation map ready in {time.time() - t0:.2f}s")
    return seg_map


def resize_image(image: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Downscale *image* so its longest edge is at most *max_dim*.
    Returns the original array unchanged if it already fits.
    """
    if max(image.shape[:2]) <= max_dim:
        return image

    scale = max_dim / max(image.shape[:2])
    new_h, new_w = int(image.shape[0] * scale), int(image.shape[1] * scale)
    logger.info(f"Resizing image {image.shape[:2]} → ({new_h}, {new_w})")
    return cv2.resize(image, (new_w, new_h))
