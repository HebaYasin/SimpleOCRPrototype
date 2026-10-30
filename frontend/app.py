"""
frontend/app.py
---------------
The complete Streamlit UI.  All st.* calls live here.
This file orchestrates the backend pipeline and renders every widget.

Sections (in order):
  1. Bootstrap         — page config, logging, imports
  2. Sidebar           — all controls, returns config dicts
  3. Upload            — file uploader + page selector
  4. Pipeline          — calls backend modules with progress feedback
  5. Results           — statistics strip + all tabs
"""

import os
import sys
import time
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from doctr.utils.visualization import visualize_page

# ── Make sure the project root is on sys.path ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.config import (
    DET_ARCHS, RECO_ARCHS,
    DEFAULT_DET_INDEX, DEFAULT_RECO_INDEX,
    DEFAULT_BIN_THRESH, DEFAULT_BOX_THRESH, DEFAULT_CONF_THRESHOLD,
    DEFAULT_MAX_RESULTS, DEFAULT_MAX_IMAGE_DIM,
    PRESETS,
)
from backend.extraction  import extract_words, build_dataframe, confidence_band_counts
from backend.inference   import load_document, run_ocr, get_segmentation_map, resize_image
from backend.logger      import setup_logging, get_logger
from backend.model       import load_model
from backend.preprocessing import preprocess_image
from backend.utils       import export_csv, export_json, export_png, export_plain_text, save_temp_file, cleanup_temp_file
from backend.visualization import (
    create_pil_visualization,
    create_matplotlib_visualization,
    create_segmentation_figure,
    create_diff_map,
)

# ── Bootstrap ──────────────────────────────────────────────────────────────

setup_logging()
logger = get_logger(__name__)

st.set_page_config(layout="wide", page_title="OCR & HCR Engine", page_icon="📋")

logger.info("=" * 72)
logger.info("APPLICATION STARTED")
logger.info("=" * 72)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.title("OCR & HCR Engine Prototype")
st.markdown(
    "**Optical & Human Character Recognition — full model control, "
    "confidence analysis, and multiple visualisation modes.**"
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.title("⚙️ Configuration")

# ── Model selection ────────────────────────────────────────────────────────
st.sidebar.header("Model Selection")

det_arch = st.sidebar.selectbox(
    "Detection Model", DET_ARCHS, index=DEFAULT_DET_INDEX,
    help="Architecture used to locate text regions.",
)
reco_arch = st.sidebar.selectbox(
    "Recognition Model", RECO_ARCHS, index=DEFAULT_RECO_INDEX,
    help="Architecture used to read text inside each detected region.",
)

# ── OCR parameters ─────────────────────────────────────────────────────────
st.sidebar.header("OCR Parameters")

assume_straight_pages   = st.sidebar.checkbox("Assume straight pages",                 value=True,  help="Enable when documents are not rotated — faster and more accurate.")
disable_page_orientation = st.sidebar.checkbox("Disable page orientation detection",    value=False, help="Skip page-level orientation detection.")
disable_crop_orientation = st.sidebar.checkbox("Disable crop orientation detection",    value=False, help="Skip word-level orientation detection.")
straighten_pages         = st.sidebar.checkbox("Straighten pages",                      value=False, help="Automatically rotate detected pages to upright.")
export_straight_boxes    = st.sidebar.checkbox("Export as straight boxes",              value=False, help="Force all bounding boxes to be axis-aligned rectangles.")

st.sidebar.markdown("**Thresholds**")
bin_thresh      = st.sidebar.slider("Binarization threshold", 0.1, 0.9, DEFAULT_BIN_THRESH,      0.1,  help="Segmentation map binarization threshold. Lower = more detections.")
box_thresh      = st.sidebar.slider("Box threshold",          0.1, 0.9, DEFAULT_BOX_THRESH,      0.1,  help="Minimum detection score to keep a bounding box.")
conf_threshold  = st.sidebar.slider("Confidence filter",      0.0, 1.0, DEFAULT_CONF_THRESHOLD,  0.01, help="Words below this score are hidden from results.")

# ── Visualisation ──────────────────────────────────────────────────────────
st.sidebar.header("Visualisation")

viz_method       = st.sidebar.radio("Render method", ["PIL (Fast)", "Matplotlib (Detailed)"], help="PIL is significantly faster for large documents.")
show_heatmap     = st.sidebar.checkbox("Confidence heatmap",      value=True)
show_boxes       = st.sidebar.checkbox("Bounding boxes",          value=True)
show_text_labels = st.sidebar.checkbox("Text labels",             value=True)
show_segmentation = st.sidebar.checkbox("Segmentation heatmap",   value=True)
show_doctr_viz   = st.sidebar.checkbox("docTR visualisation",     value=False)
show_reconstitution = st.sidebar.checkbox("Page reconstitution",  value=True)

# ── Performance ────────────────────────────────────────────────────────────
st.sidebar.header("Performance")

max_results    = int(st.sidebar.number_input("Max words to display", 10, 10_000, DEFAULT_MAX_RESULTS,   100, help="Cap word count rendered — reduce if the UI slows down."))
max_image_dim  = int(st.sidebar.number_input("Max image dimension",  800,  3_200, DEFAULT_MAX_IMAGE_DIM, 200, help="Downsample images larger than this before processing."))

# ── Preprocessing ──────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.header("🔬 Image Preprocessing")
st.sidebar.caption("Enhance image quality before OCR")

enable_preprocessing = st.sidebar.checkbox("Enable preprocessing", value=False)
preprocessing_cfg: dict = {"enabled": enable_preprocessing}

if enable_preprocessing:
    with st.sidebar.expander("Preprocessing options", expanded=True):

        preprocessing_cfg["remove_borders"] = st.checkbox("Remove borders", value=False)
        if preprocessing_cfg["remove_borders"]:
            preprocessing_cfg["border_percentage"] = st.slider("Border %", 1, 10, 2)

        preprocessing_cfg["deskew"] = st.checkbox("Auto-deskew", value=False, help="Correct document rotation automatically.")

        preprocessing_cfg["gamma_correction"] = st.checkbox("Gamma correction", value=False)
        if preprocessing_cfg["gamma_correction"]:
            preprocessing_cfg["gamma"] = st.slider("Gamma", 0.5, 2.0, 1.0, 0.1, help="< 1 darkens, > 1 lightens.")

        preprocessing_cfg["contrast_enhancement"] = st.checkbox("Enhance contrast", value=False)
        if preprocessing_cfg["contrast_enhancement"]:
            preprocessing_cfg["contrast_method"] = st.selectbox("Contrast method", ["clahe", "histogram", "adaptive"])
            if preprocessing_cfg["contrast_method"] == "clahe":
                preprocessing_cfg["contrast_clip_limit"] = st.slider("Clip limit", 1.0, 5.0, 2.0, 0.5)

        preprocessing_cfg["denoising"] = st.checkbox("Reduce noise", value=False)
        if preprocessing_cfg["denoising"]:
            preprocessing_cfg["denoise_method"]   = st.selectbox("Denoise method",   ["bilateral", "gaussian", "median", "nlmeans"])
            preprocessing_cfg["denoise_strength"]  = st.slider("Noise strength", 1, 10, 5)

        preprocessing_cfg["sharpening"] = st.checkbox("Sharpen", value=False)
        if preprocessing_cfg["sharpening"]:
            preprocessing_cfg["sharpen_method"]   = st.selectbox("Sharpen method",   ["unsharp", "laplacian", "kernel"])
            preprocessing_cfg["sharpen_strength"]  = st.slider("Sharpen strength", 0.5, 2.0, 1.0, 0.1)

        preprocessing_cfg["morphological"] = st.checkbox("Morphological ops", value=False, help="Clean up broken or merged characters.")
        if preprocessing_cfg["morphological"]:
            preprocessing_cfg["morph_operation"]   = st.selectbox("Operation", ["close", "open", "dilate", "erode", "gradient"])
            preprocessing_cfg["morph_kernel_size"]  = st.slider("Kernel size", 1, 9, 3, 2)

        preprocessing_cfg["binarization"] = st.checkbox("Binarize", value=False, help="Convert to black & white.")
        if preprocessing_cfg["binarization"]:
            preprocessing_cfg["binarization_method"] = st.selectbox(
                "Binarization method", ["adaptive_gaussian", "adaptive_mean", "otsu", "sauvola"]
            )
            if "adaptive" in preprocessing_cfg["binarization_method"]:
                preprocessing_cfg["binarization_block_size"] = st.slider("Block size", 3, 51, 11, 2)
                preprocessing_cfg["binarization_c"]           = st.slider("C value", -10, 10, 2)

    # Quick presets
    with st.sidebar.expander("⚡ Quick presets"):
        c1, c2 = st.columns(2)
        if c1.button("📄 Clean doc",    use_container_width=True): st.session_state["_preset"] = "clean";       st.rerun()
        if c2.button("📃 Faded doc",    use_container_width=True): st.session_state["_preset"] = "faded";       st.rerun()
        c3, c4 = st.columns(2)
        if c3.button("📷 Photo/scan",   use_container_width=True): st.session_state["_preset"] = "photo";       st.rerun()
        if c4.button("🖨️ Low quality",  use_container_width=True): st.session_state["_preset"] = "low_quality"; st.rerun()

    preset_key = st.session_state.pop("_preset", None)
    if preset_key and preset_key in PRESETS:
        preprocessing_cfg.update(PRESETS[preset_key])

logger.info(
    f"Config — det={det_arch}, reco={reco_arch}, "
    f"straight={assume_straight_pages}, preprocessing={enable_preprocessing}"
)


# ══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

st.header("📂 Document Upload")
uploaded_file = st.file_uploader(
    "Drop your PDF or image here",
    type=["pdf", "png", "jpg", "jpeg"],
    help="Supports PDF, PNG, JPG, JPEG.",
)


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if uploaded_file:
    logger.info(f"Upload: {uploaded_file.name}  {uploaded_file.type}  {uploaded_file.size} B")

    progress = st.progress(0)
    status   = st.empty()
    tmp_path: str | None = None
    pipeline_start = time.time()

    try:
        # 1 ── Save temp file ───────────────────────────────────────────────
        status.text("📁 Saving temporary file…")
        progress.progress(5)
        suffix   = os.path.splitext(uploaded_file.name)[1]
        tmp_path = save_temp_file(uploaded_file.read(), suffix)

        # 2 ── Load document ────────────────────────────────────────────────
        status.text("📄 Loading document…")
        progress.progress(10)
        is_pdf = uploaded_file.type == "application/pdf"
        doc    = load_document(tmp_path, is_pdf)

        # Page selector (multi-page PDFs only)
        page_idx = 0
        if len(doc) > 1:
            page_idx = st.selectbox(
                "Select page",
                range(1, len(doc) + 1),
                format_func=lambda x: f"Page {x}",
            ) - 1

        image: np.ndarray = doc[page_idx]

        # 3 ── Resize ───────────────────────────────────────────────────────
        image = resize_image(image, max_image_dim)
        original_image = image.copy()

        # 4 ── Preprocessing ────────────────────────────────────────────────
        preprocessing_steps: list[str] = []
        if enable_preprocessing:
            status.text("🔬 Preprocessing image…")
            progress.progress(25)
            image, preprocessing_steps = preprocess_image(image, preprocessing_cfg)

        # 5 ── Load model ───────────────────────────────────────────────────
        status.text("🔧 Loading model…")
        progress.progress(30)
        predictor, device = load_model(
            det_arch, reco_arch,
            assume_straight_pages, straighten_pages, export_straight_boxes,
            disable_page_orientation, disable_crop_orientation,
            bin_thresh, box_thresh,
        )

        # 6 ── OCR ──────────────────────────────────────────────────────────
        status.text("🔍 Running OCR…")
        progress.progress(45)
        result, page = run_ocr(predictor, image)

        # 7 ── Segmentation map ─────────────────────────────────────────────
        status.text("🗺️ Generating segmentation map…")
        progress.progress(55)
        seg_map = get_segmentation_map(predictor, image, device)

        # 8 ── Extract words ────────────────────────────────────────────────
        status.text("📊 Extracting word data…")
        progress.progress(65)
        words_data, total_words, filtered_words = extract_words(
            page, image.shape, conf_threshold, max_results
        )
        df = build_dataframe(words_data)

        # 9 ── Build visualisations ─────────────────────────────────────────
        status.text("🎨 Building visualisations…")
        progress.progress(80)

        viz_output = None
        if viz_method == "PIL (Fast)":
            viz_output = create_pil_visualization(image, words_data, show_heatmap, show_boxes, show_text_labels)
        else:
            viz_output = create_matplotlib_visualization(image, words_data, show_heatmap, show_boxes, show_text_labels)

        doctr_fig = None
        if show_doctr_viz:
            doctr_fig = visualize_page(page.export(), page.page, interactive=False, add_labels=False)

        reconstituted_img = None
        if show_reconstitution and (
            assume_straight_pages or (not assume_straight_pages and straighten_pages)
        ):
            reconstituted_img = page.synthesize()

        # 10 ── Structured export ───────────────────────────────────────────
        status.text("📦 Preparing exports…")
        progress.progress(90)
        structured_json = result.export()

        # Done ──────────────────────────────────────────────────────────────
        progress.progress(100)
        status.text("✅ Done!")
        time.sleep(0.3)
        status.empty()
        progress.empty()

        total_time = time.time() - pipeline_start
        logger.info(f"Pipeline complete in {total_time:.2f}s")


        # ══════════════════════════════════════════════════════════════════
        # RESULTS
        # ══════════════════════════════════════════════════════════════════

        st.markdown("---")
        st.header("Analysis Results")

        # ── Statistics strip ───────────────────────────────────────────────
        avg_conf = df["confidence"].mean() if not df.empty else 0.0
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total words",      total_words)
        s2.metric("After filter",     filtered_words)
        s3.metric("Avg confidence",   f"{avg_conf:.2%}")
        s4.metric("Processing time",  f"{total_time:.1f}s")
        s5.metric("Pages",            len(doc))

        # ── Tab layout ─────────────────────────────────────────────────────
        has_preprocessing = enable_preprocessing and bool(preprocessing_steps)
        tab_labels = (["🔬 Preprocessing"] if has_preprocessing else []) + [
            "🖼️ Visualisations", "📊 Data Table", "📈 Analytics",
            "⬇️ Downloads",      "🔍 JSON",
        ]
        all_tabs = st.tabs(tab_labels)
        offset   = 1 if has_preprocessing else 0

        # ── TAB: Preprocessing ─────────────────────────────────────────────
        if has_preprocessing:
            with all_tabs[0]:
                st.subheader("Preprocessing Comparison")

                st.success(f"✅ {len(preprocessing_steps)} step(s) applied:")
                for i, step in enumerate(preprocessing_steps, 1):
                    st.markdown(f"**{i}.** {step}")

                st.markdown("---")
                col_o, col_p = st.columns(2)
                with col_o:
                    st.markdown("**Original**")
                    st.image(original_image, use_container_width=True)
                    st.caption(f"Mean {original_image.mean():.1f} · Std {original_image.std():.1f}")
                with col_p:
                    st.markdown("**Preprocessed**")
                    st.image(image, use_container_width=True)
                    st.caption(f"Mean {image.mean():.1f} · Std {image.std():.1f}")

                diff = create_diff_map(original_image, image)
                if diff is not None:
                    st.markdown("---")
                    st.markdown("**Difference map** — brighter pixels changed more")
                    _, mid, _ = st.columns([1, 2, 1])
                    with mid:
                        st.image(diff, use_container_width=True)
                    raw_diff = cv2.absdiff(original_image, image)
                    dc1, dc2 = st.columns(2)
                    dc1.metric("Avg pixel change", f"{raw_diff.mean():.2f}")
                    dc2.metric("Max pixel change", int(raw_diff.max()))

        # ── TAB: Visualisations ────────────────────────────────────────────
        with all_tabs[offset]:
            st.subheader("Visual Analysis")

            panels = ["Original"]
            if show_segmentation:          panels.append("Segmentation")
            if show_heatmap or show_boxes or show_text_labels: panels.append("Confidence")
            if show_doctr_viz and doctr_fig:   panels.append("docTR")
            if show_reconstitution and reconstituted_img is not None: panels.append("Reconstituted")

            cols = st.columns(len(panels))
            ci   = 0

            with cols[ci]:
                st.markdown("**Original image**")
                st.image(image, use_container_width=True)
            ci += 1

            if show_segmentation:
                with cols[ci]:
                    st.markdown("**Segmentation heatmap**")
                    seg_fig = create_segmentation_figure(seg_map)
                    st.pyplot(seg_fig)
                    plt.close(seg_fig)
                ci += 1

            if show_heatmap or show_boxes or show_text_labels:
                with cols[ci]:
                    st.markdown("**Confidence overlay**")
                    if viz_method == "PIL (Fast)":
                        st.image(viz_output, use_container_width=True)
                    else:
                        st.pyplot(viz_output)
                        plt.close(viz_output)
                ci += 1

            if show_doctr_viz and doctr_fig is not None:
                with cols[ci]:
                    st.markdown("**docTR output**")
                    st.pyplot(doctr_fig)
                    plt.close(doctr_fig)
                ci += 1

            if show_reconstitution and reconstituted_img is not None:
                with cols[ci]:
                    st.markdown("**Reconstituted page**")
                    st.image(reconstituted_img, use_container_width=True, clamp=True)

        # ── TAB: Data Table ────────────────────────────────────────────────
        with all_tabs[offset + 1]:
            st.subheader("Extracted Text Data")
            st.dataframe(
                df,
                use_container_width=True,
                height=500,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "Confidence", format="%.3f", min_value=0, max_value=1
                    )
                },
            )
            search = st.text_input("🔍 Search extracted text")
            if search:
                hits = df[df["text"].str.contains(search, case=False, na=False)]
                st.write(f"{len(hits)} match(es):")
                st.dataframe(hits, use_container_width=True)

        # ── TAB: Analytics ─────────────────────────────────────────────────
        with all_tabs[offset + 2]:
            st.subheader("Confidence Distribution")
            a1, a2 = st.columns(2)

            with a1:
                if not df.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(df["confidence"], bins=20, color="steelblue", edgecolor="black", alpha=0.7)
                    ax.set_xlabel("Confidence score")
                    ax.set_ylabel("Word count")
                    ax.set_title("Confidence distribution")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close(fig)

            with a2:
                if not df.empty:
                    st.markdown("**Statistical summary**")
                    st.write(df["confidence"].describe())
                    st.markdown("**Confidence bands**")
                    for label, count in confidence_band_counts(df).items():
                        pct = count / len(df) * 100
                        st.metric(label, f"{count} words ({pct:.1f}%)")

        # ── TAB: Downloads ─────────────────────────────────────────────────
        with all_tabs[offset + 3]:
            st.subheader("Export Options")
            ts   = time.strftime("%Y%m%d_%H%M%S")
            d1, d2, d3, d4 = st.columns(4)

            d1.download_button(
                "📥 CSV",  export_csv(df),
                file_name=f"ocr_{ts}.csv", mime="text/csv",
                use_container_width=True,
            )
            d2.download_button(
                "📥 JSON", export_json(structured_json),
                file_name=f"ocr_{ts}.json", mime="application/json",
                use_container_width=True,
            )
            if viz_method == "PIL (Fast)" and viz_output is not None:
                d3.download_button(
                    "📥 PNG", export_png(viz_output),
                    file_name=f"ocr_{ts}.png", mime="image/png",
                    use_container_width=True,
                )
            d4.download_button(
                "📥 TXT", export_plain_text(df),
                file_name=f"ocr_{ts}.txt", mime="text/plain",
                use_container_width=True,
            )
            st.info("💡 The JSON export contains the full document hierarchy: blocks → lines → words + geometry.")

        # ── TAB: JSON viewer ───────────────────────────────────────────────
        with all_tabs[offset + 4]:
            st.subheader("Structured JSON Output")
            st.markdown("Complete docTR document structure:")
            st.json(structured_json, expanded=False)

    except Exception as exc:
        logger.error(f"Pipeline failed: {exc}", exc_info=True)
        st.error(f"❌ {exc}")
        with st.expander("Show full traceback"):
            st.code(traceback.format_exc())

    finally:
        cleanup_temp_file(tmp_path)

else:
    st.info("Upload a PDF or image file to begin.")

logger.info("Render cycle complete")
