# OCR & HCR Engine

A Streamlit application for Optical and Handwritten Character Recognition, built on [docTR](https://github.com/mindee/doctr). Supports configurable image preprocessing, multiple visualisation modes, and structured data export.

---

## Project Structure

```
simple-ocr-engine/
│
├── frontend/
│   └── app.py            ← Streamlit UI — all st.* calls live here
│
├── backend/
│   ├── config.py         ← Architecture lists, defaults, presets (no imports)
│   ├── logger.py         ← Logging setup; use get_logger(__name__) everywhere
│   ├── preprocessing.py  ← Image enhancement pipeline (pure NumPy/OpenCV)
│   ├── model.py          ← docTR model loading with st.cache_resource
│   ├── inference.py      ← Document loading, OCR run, segmentation map
│   ├── extraction.py     ← Word extraction from docTR page, DataFrame build
│   ├── visualization.py  ← PIL & Matplotlib figure builders
│   └── utils.py          ← Export helpers (CSV/JSON/PNG/TXT), temp file I/O
│
└── requirements.txt
```

**Design rules:**
- `backend/` is Streamlit-free (except `model.py`'s `@st.cache_resource` decorator, which is not a UI concern).
- `frontend/app.py` is the only file that calls `st.*` widget functions.
- Data flows in one direction: sidebar → pipeline → results. No shared mutable state.

---

## Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.10 |
| CUDA *(optional)* | 11.x or 12.x for GPU |

Install the DejaVu system font for text label rendering (Linux):

```bash
sudo apt-get install fonts-dejavu
```

---

## Installation

```bash
# 1. Clone / copy the project
git clone <your-repo-url>
cd ocr_engine

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**GPU users:** install the CUDA-enabled PyTorch build *before* the rest:

```bash
# Example for CUDA 12.1 — adjust to your driver version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Running the App

```bash
# From the ocr_engine/ directory
streamlit run frontend/app.py
```

Streamlit opens `http://localhost:8501` in your default browser.

Custom port:

```bash
streamlit run frontend/app.py --server.port 8080
```

---

## Interface Guide

### Sidebar

#### Model Selection

| Control | Description |
|---|---|
| **Detection Model** | Locates text regions. `fast_base` is a good starting point; `db_resnet50` gives higher recall at the cost of speed. |
| **Recognition Model** | Reads text inside each detected region. `parseq` gives the best accuracy; `crnn_mobilenet_v3_small` is fastest. |

#### OCR Parameters

| Control | Description |
|---|---|
| **Assume straight pages** | Enable for upright documents — skips orientation detection and speeds up inference. |
| **Disable page/crop orientation** | Skip individual orientation steps for additional speed. |
| **Straighten pages** | Rotate detected pages to upright before recognition. |
| **Export as straight boxes** | Force all bounding boxes to be axis-aligned (useful for downstream parsing). |
| **Binarization threshold** | Controls how the segmentation map is binarized. Lower values produce more detections; higher values reduce false positives. |
| **Box threshold** | Minimum detection confidence to keep a box. Raise to eliminate weak detections. |
| **Confidence filter** | Words below this score are excluded from results and visualisations. |

#### Visualisation

| Control | Description |
|---|---|
| **Render method** | *PIL (Fast)* handles thousands of words per second. *Matplotlib (Detailed)* is slower but produces vector-quality output. |
| **Confidence heatmap** | Colour-fill each word region: red = low confidence, green = high. |
| **Bounding boxes** | Draw outlines around detected words. |
| **Text labels** | Overlay the recognised text and its confidence score. |
| **Segmentation heatmap** | Show the raw detection probability map from the detector. |
| **docTR visualisation** | Render the built-in docTR page summary figure. |
| **Page reconstitution** | Synthesise a clean text page from the recognition output. |

#### Performance

| Control | Description |
|---|---|
| **Max words to display** | Hard cap on rendered words. Reduce if the browser tab becomes slow. |
| **Max image dimension** | Images whose longest edge exceeds this are downsampled before processing. |

#### Image Preprocessing

Enable **preprocessing** to reveal per-step controls:

| Step | Best for |
|---|---|
| Remove borders | Scanned docs with dark border artifacts |
| Auto-deskew | Slightly rotated photographs or scans |
| Gamma correction | Dark (`gamma < 1`) or washed-out (`gamma > 1`) images |
| Enhance contrast | Low-contrast or faded documents |
| Reduce noise | Noisy photographs, faxes |
| Sharpen | Blurry scans |
| Morphological ops | Broken or merged character strokes |
| Binarize | Documents with mixed or uneven backgrounds |

**Quick presets** apply recommended parameter combinations in one click:

| Preset | Use case |
|---|---|
| 📄 Clean doc | Standard office documents |
| 📃 Faded doc | Old, photocopied, or low-ink documents |
| 📷 Photo/scan | Camera captures of documents |
| 🖨️ Low quality | Poor-quality faxes or printouts |

---

### Upload

Drag and drop or click to upload. Accepted formats: **PDF, PNG, JPG, JPEG**.

Multi-page PDFs show a **page selector** — choose the page to process.

---

### Results

#### 🔬 Preprocessing *(only shown when preprocessing is enabled)*
Side-by-side original vs. processed view, per-step list, and a pixel-difference map.

#### 🖼️ Visualisations
Up to five side-by-side panels: original image, segmentation heatmap, confidence overlay, docTR output, and reconstituted page (controlled by sidebar toggles).

#### 📊 Data Table
Full word-level results with confidence progress bars and an inline text-search box.

#### 📈 Analytics
Confidence score histogram, descriptive statistics, and a breakdown of words by confidence band (High / Good / Medium / Low).

#### ⬇️ Downloads

| Format | Contents |
|---|---|
| **CSV** | Text, confidence, and bounding-box pixel coordinates for every word |
| **JSON** | Full docTR document hierarchy: blocks → lines → words + normalised geometry |
| **PNG** | Confidence overlay image *(PIL mode only)* |
| **TXT** | Plain text — one word per line |

#### 🔍 JSON
Interactive collapsible viewer for the full structured document output.


