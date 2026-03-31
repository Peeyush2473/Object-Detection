"""
streamlit_app.py – Smart Surveillance Dashboard

Launch with:
    streamlit run dashboard/streamlit_app.py

Provides a Streamlit-based UI for:
- Real-time object detection from webcam or uploaded video
- Live bounding-box visualization with confidence scores
- Alert panel for suspicious objects
- Detection log viewer & CSV export
- Model configuration sidebar
"""

import os
import sys
import time
import tempfile
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Project path setup ──────────────────────────────────────────── #
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_DASHBOARD_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import config
from app.detection import SurveillanceDetector
from app.video_processor import draw_detections, draw_fps
from app.utils import (
    log_detections,
    load_detection_logs,
    clear_detection_logs,
    should_alert,
    play_alert_sound,
    send_email_alert,
    format_detection_summary,
    ensure_log_file,
)


# ================================================================== #
#  Page configuration
# ================================================================== #
st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon=config.DASHBOARD_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a premium dark dashboard look ──────────────────── #
st.markdown("""
<style>
    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117, #161b22);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(56,189,248,0.12);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(6px);
    }

    /* Alert box */
    .alert-box {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.4);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        color: #fca5a5;
        font-weight: 600;
    }

    .safe-box {
        background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(34,197,94,0.04));
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        color: #86efac;
        font-weight: 600;
    }

    /* Detection list items */
    .det-item {
        background: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 8px 14px;
        margin-bottom: 6px;
        font-family: 'Segoe UI', sans-serif;
        font-size: 0.9rem;
    }

    /* Header */
    .main-header {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }

    .sub-header {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }

    /* Status indicator */
    .status-live {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 8px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
</style>
""", unsafe_allow_html=True)


# ================================================================== #
#  Session state defaults
# ================================================================== #
_DEFAULTS = {
    "detector": None,
    "running": False,
    "frame_count": 0,
    "total_detections": 0,
    "alert_count": 0,
    "fps": 0.0,
    "last_detections": [],
    "enabled_classes": list(config.MONITORED_CLASSES),
}

for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ================================================================== #
#  Sidebar – Model & class configuration
# ================================================================== #
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    model_choice = st.selectbox(
        "YOLO model variant",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0,
        help="Nano (n) is fastest; Extra-large (x) is most accurate.",
    )

    confidence = st.slider(
        "Confidence threshold", 0.1, 1.0, config.CONFIDENCE_THRESHOLD, 0.05
    )

    st.markdown("---")
    st.markdown("### 🎯 Detection Classes")
    enabled_classes = []
    for cls in config.MONITORED_CLASSES:
        if st.checkbox(cls, value=True, key=f"cls_{cls}"):
            enabled_classes.append(cls)
    st.session_state["enabled_classes"] = enabled_classes

    st.markdown("---")
    st.markdown("### 🔔 Alerts")
    enable_sound = st.checkbox("Enable sound alert", value=False)
    config.ENABLE_SOUND_ALERT = enable_sound

    st.markdown("---")
    if st.button("🗑️ Clear Detection Logs"):
        clear_detection_logs()
        st.success("Logs cleared!")


# ================================================================== #
#  Initialize / reinitialize detector when settings change
# ================================================================== #
@st.cache_resource
def load_detector(model_name: str, conf: float, classes: tuple):
    """Cache the detector so it isn't reloaded on every rerun."""
    return SurveillanceDetector(
        confidence=conf,
        monitored_classes=list(classes),
    )


detector = load_detector(model_choice, confidence, tuple(enabled_classes))


# ================================================================== #
#  Header
# ================================================================== #
st.markdown('<div class="main-header">🔒 Smart Surveillance Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Real-time object detection powered by YOLOv8 · '
    'Monitor, detect, and alert</div>',
    unsafe_allow_html=True,
)


# ================================================================== #
#  Source selection
# ================================================================== #
source_tab, logs_tab, about_tab = st.tabs(["📹 Live Feed", "📊 Detection Logs", "ℹ️ About"])

with source_tab:
    col_source, col_info = st.columns([3, 1])

    with col_source:
        mode = st.radio(
            "Input source",
            ["📷 Webcam", "📁 Upload Video"],
            horizontal=True,
        )

    with col_info:
        st.markdown(
            f"**Model:** `{model_choice}` &nbsp;|&nbsp; "
            f"**Confidence:** `{confidence:.0%}` &nbsp;|&nbsp; "
            f"**Classes:** `{len(enabled_classes)}`"
        )

    st.markdown("---")

    # ── Metrics row ────────────────────────────────────────────── #
    m1, m2, m3, m4 = st.columns(4)
    metric_fps = m1.empty()
    metric_frames = m2.empty()
    metric_detections = m3.empty()
    metric_alerts = m4.empty()

    # ── Alert banner ───────────────────────────────────────────── #
    alert_banner = st.empty()

    # ── Video + detection side-by-side ─────────────────────────── #
    col_video, col_detlist = st.columns([3, 1])
    video_placeholder = col_video.empty()
    detection_list_placeholder = col_detlist.empty()

    # ── Controls ──────────────────────────────────────────────── #
    uploaded_file = None
    if mode == "📁 Upload Video":
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
        )

    start_col, stop_col, _ = st.columns([1, 1, 4])
    start_btn = start_col.button("▶️ Start Detection", type="primary", use_container_width=True)
    stop_btn = stop_col.button("⏹ Stop", use_container_width=True)

    # ── Main processing loop ──────────────────────────────────── #
    if start_btn:
        # Determine video source
        if mode == "📷 Webcam":
            source = 0
        else:
            if uploaded_file is None:
                st.warning("Please upload a video file first.")
                st.stop()
            # Save uploaded file to a temp location
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded_file.read())
            tmp.close()
            source = tmp.name

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.error(f"Cannot open video source: {source}")
            st.stop()

        st.session_state["running"] = True
        st.session_state["frame_count"] = 0
        st.session_state["total_detections"] = 0
        st.session_state["alert_count"] = 0

        prev_time = time.time()
        log_batch = []  # buffer for CSV writes

        while cap.isOpened() and st.session_state["running"]:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize
            h, w = frame.shape[:2]
            if w > config.MAX_DISPLAY_WIDTH:
                scale = config.MAX_DISPLAY_WIDTH / w
                frame = cv2.resize(frame, (config.MAX_DISPLAY_WIDTH, int(h * scale)))

            # Detect
            detections = detector.detect(frame)
            annotated = draw_detections(frame.copy(), detections)

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            draw_fps(annotated, fps)

            # Counters
            st.session_state["frame_count"] += 1
            st.session_state["total_detections"] += len(detections)
            alerts_this_frame = [d for d in detections if d["is_alert"]]
            st.session_state["alert_count"] += len(alerts_this_frame)

            # Update metrics
            metric_fps.metric("⚡ FPS", f"{fps:.1f}")
            metric_frames.metric("🖼️ Frames", st.session_state["frame_count"])
            metric_detections.metric("🔍 Detections", st.session_state["total_detections"])
            metric_alerts.metric("🚨 Alerts", st.session_state["alert_count"])

            # Alert banner
            if alerts_this_frame:
                alert_names = ", ".join(set(d["class_name"] for d in alerts_this_frame))
                alert_banner.markdown(
                    f'<div class="alert-box">🚨 <span class="status-live"></span>'
                    f'ALERT — Suspicious object detected: <b>{alert_names}</b> '
                    f'({datetime.now().strftime("%H:%M:%S")})</div>',
                    unsafe_allow_html=True,
                )
                if should_alert():
                    play_alert_sound()
                    send_email_alert(
                        body=f"Suspicious object detected: {alert_names}"
                    )
            else:
                alert_banner.markdown(
                    '<div class="safe-box">✅ Scene is clear — No suspicious objects</div>',
                    unsafe_allow_html=True,
                )

            # Show frame (convert BGR → RGB)
            video_placeholder.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            # Detection list panel
            if detections:
                items_html = ""
                for d in detections:
                    emoji = "🔴" if d["is_alert"] else "🟢"
                    items_html += (
                        f'<div class="det-item">{emoji} '
                        f'<b>{d["class_name"]}</b> — '
                        f'{d["confidence"]:.0%}</div>'
                    )
                detection_list_placeholder.markdown(items_html, unsafe_allow_html=True)
            else:
                detection_list_placeholder.markdown(
                    '<div class="det-item" style="color:#94a3b8;">No detections</div>',
                    unsafe_allow_html=True,
                )

            # Buffer logs (write every 30 frames to reduce I/O)
            log_batch.extend(detections)
            if st.session_state["frame_count"] % 30 == 0 and log_batch:
                log_detections(log_batch)
                log_batch = []

        # Flush remaining logs
        if log_batch:
            log_detections(log_batch)

        cap.release()
        st.session_state["running"] = False
        st.success("✅ Processing complete.")

    if stop_btn:
        st.session_state["running"] = False
        st.info("Detection stopped by user.")


# ================================================================== #
#  Logs tab
# ================================================================== #
with logs_tab:
    st.markdown("### 📋 Detection Log Viewer")

    df = load_detection_logs()
    if df.empty:
        st.info("No detection logs recorded yet. Start a detection session first!")
    else:
        st.markdown(f"**Total records:** {len(df)}")

        # Filters
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            filter_class = st.multiselect(
                "Filter by class",
                options=df["class_name"].unique().tolist() if "class_name" in df.columns else [],
            )
        with fcol2:
            filter_alert = st.selectbox("Filter alerts", ["All", "Alerts Only", "Non-alerts"])

        filtered = df.copy()
        if filter_class:
            filtered = filtered[filtered["class_name"].isin(filter_class)]
        if filter_alert == "Alerts Only":
            filtered = filtered[filtered["is_alert"] == True]
        elif filter_alert == "Non-alerts":
            filtered = filtered[filtered["is_alert"] == False]

        st.dataframe(
            filtered.tail(config.LOG_MAX_ROWS_DISPLAY),
            use_container_width=True,
            height=400,
        )

        # Download button
        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Logs as CSV",
            data=csv_data,
            file_name=f"detection_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        # Summary stats
        st.markdown("---")
        st.markdown("### 📈 Summary Statistics")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            st.metric("Total Detections", len(filtered))
        with scol2:
            alert_ct = filtered["is_alert"].sum() if "is_alert" in filtered.columns else 0
            st.metric("Total Alerts", int(alert_ct))
        with scol3:
            unique_cls = filtered["class_name"].nunique() if "class_name" in filtered.columns else 0
            st.metric("Unique Classes", unique_cls)

        if "class_name" in filtered.columns:
            st.bar_chart(filtered["class_name"].value_counts())


# ================================================================== #
#  About tab
# ================================================================== #
with about_tab:
    st.markdown("""
    ### ℹ️ About Smart Surveillance

    **Smart Surveillance** is a real-time object detection system built for
    monitoring environments and flagging suspicious objects or activities.

    #### 🔧 How It Works
    1. **Video input** is captured from a webcam or uploaded file.
    2. Each frame is processed by a **YOLOv8** model from Ultralytics.
    3. Detections are filtered to **monitored classes** (e.g., person, knife).
    4. Bounding boxes and confidence scores are drawn on the frame.
    5. **Alerts** are raised when suspicious objects (like knives) appear.
    6. All detections are logged to a **CSV file** for later review.

    #### 🧰 Tech Stack
    | Component | Technology |
    |-----------|-----------|
    | Detection Engine | YOLOv8 (Ultralytics) |
    | Computer Vision | OpenCV |
    | Deep Learning | PyTorch |
    | Dashboard | Streamlit |
    | Data | NumPy, Pandas |

    #### 📂 Project Structure
    ```
    smart-surveillance/
    ├── app/                    # Core application modules
    │   ├── detection.py        # YOLO wrapper
    │   ├── video_processor.py  # Frame processing pipeline
    │   └── utils.py            # Logging & alerts
    ├── dashboard/
    │   └── streamlit_app.py    # This dashboard
    ├── models/                 # YOLO weights (auto-downloaded)
    ├── data/                   # Sample videos & logs
    ├── notebooks/              # Training notebooks
    ├── config.py               # Central configuration
    └── requirements.txt
    ```

    ---
    *Built with ❤️ using Python, YOLOv8, and Streamlit*
    """)
