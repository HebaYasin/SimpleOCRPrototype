"""
backend/preprocessing.py
------------------------
Pure image preprocessing functions and the pipeline orchestrator.
All functions operate on RGB NumPy arrays and return RGB NumPy arrays.
Zero Streamlit dependency.
"""

import time

import cv2
import numpy as np

from backend.logger import get_logger

logger = get_logger(__name__)


# ── Individual operations ──────────────────────────────────────────────────

def apply_contrast_enhancement(
    image: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
) -> np.ndarray:
    """Boost contrast via CLAHE, histogram equalisation, or adaptive CLAHE."""
    logger.info(f"Contrast enhancement — method={method}, clip={clip_limit}")

    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
    else:
        l = image.copy()
        a = b = None

    if method == "clahe":
        l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l)
    elif method == "histogram":
        l = cv2.equalizeHist(l)
    elif method == "adaptive":
        l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16)).apply(l)

    if len(image.shape) == 3:
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return l


def apply_denoising(
    image: np.ndarray,
    method: str = "bilateral",
    strength: int = 5,
) -> np.ndarray:
    """Reduce image noise while trying to preserve edges."""
    logger.info(f"Denoising — method={method}, strength={strength}")

    if method == "bilateral":
        return cv2.bilateralFilter(image, 9, strength * 10, strength * 10)
    if method == "gaussian":
        k = strength * 2 + 1
        return cv2.GaussianBlur(image, (k, k), 0)
    if method == "median":
        k = strength * 2 + 1
        return cv2.medianBlur(image, k)
    if method == "nlmeans":
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    return image


def apply_sharpening(
    image: np.ndarray,
    method: str = "unsharp",
    strength: float = 1.0,
) -> np.ndarray:
    """Sharpen text edges."""
    logger.info(f"Sharpening — method={method}, strength={strength}")

    if method == "unsharp":
        blur = cv2.GaussianBlur(image, (0, 0), 2.0)
        out = cv2.addWeighted(image, 1.0 + strength, blur, -strength, 0)
    elif method == "laplacian":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        lap = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
        if len(image.shape) == 3:
            lap = cv2.cvtColor(lap, cv2.COLOR_GRAY2RGB)
        out = cv2.addWeighted(image, 1.0, lap, strength * 0.5, 0)
    elif method == "kernel":
        k = np.array([[-1, -1, -1], [-1, 9 + strength * 2, -1], [-1, -1, -1]])
        out = cv2.filter2D(image, -1, k)
    else:
        out = image

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_binarization(
    image: np.ndarray,
    method: str = "adaptive_gaussian",
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """Convert to high-contrast binary image."""
    logger.info(f"Binarization — method={method}")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_mean":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
        )
    elif method == "adaptive_gaussian":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
        )
    elif method == "sauvola":
        from skimage.filters import threshold_sauvola
        thresh = threshold_sauvola(gray, window_size=block_size)
        binary = (gray > thresh).astype(np.uint8) * 255
    else:
        binary = gray

    return cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB) if len(image.shape) == 3 else binary


def apply_morphological_operations(
    image: np.ndarray,
    operation: str = "close",
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply morphological transforms to clean up text regions."""
    logger.info(f"Morphological — operation={operation}, kernel={kernel_size}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    morph_map = {
        "dilate":   lambda: cv2.dilate(image, kernel, iterations=1),
        "erode":    lambda: cv2.erode(image, kernel, iterations=1),
        "open":     lambda: cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel),
        "close":    lambda: cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel),
        "gradient": lambda: cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel),
    }
    return morph_map[operation]() if operation in morph_map else image


def apply_deskew(image: np.ndarray) -> np.ndarray:
    """Detect and correct document skew via Hough line analysis."""
    logger.info("Deskew — detecting skew angle")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    if lines is not None and len(lines) > 0:
        angles = [np.degrees(theta) - 90 for rho, theta in lines[:, 0]]
        angle = np.median(angles)
        if abs(angle) > 0.5:
            logger.info(f"Correcting skew: {angle:.2f}°")
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )

    logger.info("No significant skew detected")
    return image


def remove_borders(image: np.ndarray, border_percentage: int = 2) -> np.ndarray:
    """Crop border artifacts by a percentage of each dimension."""
    logger.info(f"Removing borders — {border_percentage}%")
    h, w = image.shape[:2]
    bh, bw = int(h * border_percentage / 100), int(w * border_percentage / 100)
    return image[bh : h - bh, bw : w - bw]


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply gamma correction (< 1 darkens, > 1 lightens)."""
    logger.info(f"Gamma correction — gamma={gamma}")
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# ── Pipeline orchestrator ──────────────────────────────────────────────────

def preprocess_image(
    image: np.ndarray,
    config: dict,
) -> tuple[np.ndarray, list[str]]:
    """
    Run every enabled preprocessing step in the correct order.

    Parameters
    ----------
    image  : RGB NumPy array
    config : dict from the sidebar preprocessing section

    Returns
    -------
    processed     : transformed RGB NumPy array
    steps_applied : human-readable list of steps that ran
    """
    logger.info("Preprocessing pipeline starting")
    t0 = time.time()
    out = image.copy()
    steps: list[str] = []

    if config.get("remove_borders"):
        out = remove_borders(out, config.get("border_percentage", 2))
        steps.append("Border removal")

    if config.get("deskew"):
        out = apply_deskew(out)
        steps.append("Deskewing")

    if config.get("gamma_correction"):
        g = config.get("gamma", 1.0)
        if g != 1.0:
            out = apply_gamma_correction(out, g)
            steps.append(f"Gamma ({g})")

    if config.get("contrast_enhancement"):
        m = config.get("contrast_method", "clahe")
        out = apply_contrast_enhancement(out, m, config.get("contrast_clip_limit", 2.0))
        steps.append(f"Contrast/{m}")

    if config.get("denoising"):
        m = config.get("denoise_method", "bilateral")
        out = apply_denoising(out, m, config.get("denoise_strength", 5))
        steps.append(f"Denoise/{m}")

    if config.get("sharpening"):
        m = config.get("sharpen_method", "unsharp")
        out = apply_sharpening(out, m, config.get("sharpen_strength", 1.0))
        steps.append(f"Sharpen/{m}")

    if config.get("morphological"):
        op = config.get("morph_operation", "close")
        out = apply_morphological_operations(out, op, config.get("morph_kernel_size", 3))
        steps.append(f"Morph/{op}")

    if config.get("binarization"):
        m = config.get("binarization_method", "adaptive_gaussian")
        out = apply_binarization(out, m, config.get("binarization_block_size", 11), config.get("binarization_c", 2))
        steps.append(f"Binarize/{m}")

    logger.info(f"Preprocessing done in {time.time() - t0:.2f}s — {', '.join(steps) or 'no steps'}")
    return out, steps
