"""
backend/config.py
-----------------
All application-wide constants and default values.
No logic, no imports beyond stdlib — safe to import from anywhere.
"""

# ── Detection architectures ────────────────────────────────────────────────
DET_ARCHS: list[str] = [
    "fast_base",
    "fast_small",
    "fast_tiny",
    "db_resnet50",
    "db_resnet34",
    "db_mobilenet_v3_large",
    "linknet_resnet18",
    "linknet_resnet34",
    "linknet_resnet50",
]

# ── Recognition architectures ──────────────────────────────────────────────
RECO_ARCHS: list[str] = [
    "crnn_vgg16_bn",
    "crnn_mobilenet_v3_small",
    "crnn_mobilenet_v3_large",
    "master",
    "sar_resnet31",
    "vitstr_small",
    "vitstr_base",
    "parseq",
    "viptr_tiny",
]

# ── Sidebar defaults ───────────────────────────────────────────────────────
DEFAULT_DET_INDEX: int = 0       # fast_base
DEFAULT_RECO_INDEX: int = 7      # parseq

DEFAULT_BIN_THRESH: float = 0.3
DEFAULT_BOX_THRESH: float = 0.1
DEFAULT_CONF_THRESHOLD: float = 0.5
DEFAULT_MAX_RESULTS: int = 1_000
DEFAULT_MAX_IMAGE_DIM: int = 1_600

# ── Quick-preset configurations ────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "clean": {
        "contrast_enhancement": True,
        "contrast_method": "clahe",
        "contrast_clip_limit": 2.0,
        "sharpening": True,
        "sharpen_method": "unsharp",
        "sharpen_strength": 1.0,
    },
    "faded": {
        "gamma_correction": True,
        "gamma": 0.8,
        "contrast_enhancement": True,
        "contrast_method": "clahe",
        "contrast_clip_limit": 3.0,
        "binarization": True,
        "binarization_method": "adaptive_gaussian",
        "binarization_block_size": 15,
        "binarization_c": 2,
    },
    "photo": {
        "deskew": True,
        "denoising": True,
        "denoise_method": "bilateral",
        "denoise_strength": 5,
        "contrast_enhancement": True,
        "contrast_method": "clahe",
        "contrast_clip_limit": 2.5,
    },
    "low_quality": {
        "denoising": True,
        "denoise_method": "nlmeans",
        "denoise_strength": 7,
        "contrast_enhancement": True,
        "contrast_method": "adaptive",
        "contrast_clip_limit": 3.0,
        "sharpening": True,
        "sharpen_method": "unsharp",
        "sharpen_strength": 1.5,
        "morphological": True,
        "morph_operation": "close",
        "morph_kernel_size": 3,
    },
}

# ── Font path for PIL text labels ──────────────────────────────────────────
DEJAVU_FONT_PATH: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
