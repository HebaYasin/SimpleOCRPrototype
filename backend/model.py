"""
backend/model.py
----------------
OCR model construction and caching.

`load_model` is decorated with @st.cache_resource so the heavy model object
is only instantiated once per process (or when its parameters change).
This is the only Streamlit import in the entire backend — it's a decorator,
not a UI concern, so it lives here.
"""

import time

import torch
import streamlit as st
from doctr.models import ocr_predictor

from backend.logger import get_logger

logger = get_logger(__name__)


@st.cache_resource
def load_model(
    det_arch: str,
    reco_arch: str,
    assume_straight_pages: bool,
    straighten_pages: bool,
    export_as_straight_boxes: bool,
    disable_page_orientation: bool,
    disable_crop_orientation: bool,
    bin_thresh: float,
    box_thresh: float,
) -> tuple:
    """
    Build, configure, and return the docTR predictor + active device.
    Results are cached by Streamlit; the model re-loads only when parameters change.

    Returns
    -------
    predictor : docTR OCR predictor
    device    : torch.device in use (cuda or cpu)
    """
    logger.info(f"Loading model — det={det_arch}  reco={reco_arch}")
    t0 = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    predictor = ocr_predictor(
        det_arch,
        reco_arch,
        pretrained=True,
        assume_straight_pages=assume_straight_pages,
        straighten_pages=straighten_pages,
        export_as_straight_boxes=export_as_straight_boxes,
        detect_orientation=not assume_straight_pages,
        disable_page_orientation=disable_page_orientation,
        disable_crop_orientation=disable_crop_orientation,
        preserve_aspect_ratio=True,
    ).to(device)

    predictor.det_predictor.model.postprocessor.bin_thresh = bin_thresh
    predictor.det_predictor.model.postprocessor.box_thresh = box_thresh

    logger.info(f"Model ready in {time.time() - t0:.2f}s")
    return predictor, device
