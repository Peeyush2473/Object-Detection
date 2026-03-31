"""
Configuration file for Smart Surveillance System.
All configurable parameters are centralized here to avoid hardcoding.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_FILE = os.path.join(DATA_DIR, "detection_logs.csv")

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50

# ──────────────────────────────────────────────
# Surveillance Classes of Interest
# ──────────────────────────────────────────────
# COCO class IDs mapped to human-readable names
# Full COCO: https://docs.ultralytics.com/datasets/detect/coco/
SUSPICIOUS_CLASSES = {
    0: "person",
    43: "knife",
    # Guns are not in COCO – handled via custom model or label override
}

# Classes that trigger HIGH-priority alerts
ALERT_CLASSES = ["knife"]

# All classes to track (COCO names)
MONITORED_CLASSES = [
    "person", "knife", "scissors", "backpack",
    "handbag", "suitcase", "cell phone", "laptop",
]

# ──────────────────────────────────────────────
# Alert Configuration
# ──────────────────────────────────────────────
ENABLE_SOUND_ALERT = False
ALERT_SOUND_PATH = os.path.join(BASE_DIR, "assets", "alert.wav")
ALERT_COOLDOWN_SECONDS = 10  # Minimum seconds between repeated alerts

# Email alert (optional – fill in to enable)
ENABLE_EMAIL_ALERT = False
EMAIL_SENDER = ""
EMAIL_PASSWORD = ""
EMAIL_RECEIVER = ""

# ──────────────────────────────────────────────
# Video / Camera Settings
# ──────────────────────────────────────────────
DEFAULT_VIDEO_SOURCE = 0  # 0 = default webcam
MAX_DISPLAY_WIDTH = 720
FPS_DISPLAY = True

# ──────────────────────────────────────────────
# Dashboard Settings
# ──────────────────────────────────────────────
DASHBOARD_TITLE = "🔒 Smart Surveillance Dashboard"
DASHBOARD_ICON = "🔒"
LOG_MAX_ROWS_DISPLAY = 200    # Max rows shown in log viewer
