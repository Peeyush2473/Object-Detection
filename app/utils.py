"""
utils.py – Shared utilities for the Smart Surveillance system.

Includes:
- CSV logging of detections
- Sound / email alert helpers
- Miscellaneous helpers
"""

import csv
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# Resolve project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, _PROJECT_ROOT)
import config


# ──────────────────────────────────────────────────────────────────── #
#  CSV Logging
# ──────────────────────────────────────────────────────────────────── #

_CSV_HEADER = ["timestamp", "class_name", "confidence", "bbox", "is_alert"]


def ensure_log_file(path: str = config.LOG_FILE) -> str:
    """
    Create the CSV log file with headers if it doesn't exist.

    Returns the absolute path to the log file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_HEADER)
    return path


def log_detections(
    detections: List[Dict],
    path: str = config.LOG_FILE,
) -> None:
    """
    Append a batch of detections to the CSV log.

    Parameters
    ----------
    detections : list[dict]
        Output from ``SurveillanceDetector.detect()``.
    path : str
        Path to the CSV log file.
    """
    if not detections:
        return

    ensure_log_file(path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        for det in detections:
            writer.writerow([
                timestamp,
                det["class_name"],
                round(det["confidence"], 4),
                str(det["bbox"]),
                det["is_alert"],
            ])


def load_detection_logs(path: str = config.LOG_FILE) -> pd.DataFrame:
    """
    Load the detection log CSV into a DataFrame.

    Returns an empty DataFrame with proper columns if the file doesn't exist.
    """
    ensure_log_file(path)
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=_CSV_HEADER)
    return df


def clear_detection_logs(path: str = config.LOG_FILE) -> None:
    """Reset the log file (keep header only)."""
    ensure_log_file(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)


# ──────────────────────────────────────────────────────────────────── #
#  Alert System
# ──────────────────────────────────────────────────────────────────── #

_last_alert_time: float = 0.0


def should_alert(cooldown: float = config.ALERT_COOLDOWN_SECONDS) -> bool:
    """
    Rate-limiter for alerts.  Returns True only if enough time has
    elapsed since the last alert was triggered.
    """
    global _last_alert_time
    now = time.time()
    if now - _last_alert_time >= cooldown:
        _last_alert_time = now
        return True
    return False


def play_alert_sound(sound_path: str = config.ALERT_SOUND_PATH) -> None:
    """
    Play an alert sound in a background thread (non-blocking).
    Falls back silently if the file or library is missing.
    """
    if not config.ENABLE_SOUND_ALERT:
        return
    if not os.path.exists(sound_path):
        return

    def _play():
        try:
            from playsound import playsound
            playsound(sound_path)
        except Exception:
            pass  # Fail silently – sound is a nice-to-have

    threading.Thread(target=_play, daemon=True).start()


def send_email_alert(
    subject: str = "⚠️ Surveillance Alert",
    body: str = "A suspicious object was detected.",
) -> None:
    """
    Send an email alert using yagmail (if configured).
    Runs in a background thread to avoid blocking the main loop.
    """
    if not config.ENABLE_EMAIL_ALERT:
        return
    if not config.EMAIL_SENDER or not config.EMAIL_RECEIVER:
        return

    def _send():
        try:
            import yagmail
            yag = yagmail.SMTP(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            yag.send(
                to=config.EMAIL_RECEIVER,
                subject=subject,
                contents=body,
            )
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


# ──────────────────────────────────────────────────────────────────── #
#  Miscellaneous
# ──────────────────────────────────────────────────────────────────── #

def format_detection_summary(detections: List[Dict]) -> str:
    """
    Return a human-readable one-line summary of detections.

    Example: "2 person, 1 knife [ALERT]"
    """
    if not detections:
        return "No objects detected"

    counts: Dict[str, int] = {}
    has_alert = False
    for d in detections:
        counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
        if d["is_alert"]:
            has_alert = True

    parts = [f"{v} {k}" for k, v in sorted(counts.items())]
    summary = ", ".join(parts)
    if has_alert:
        summary += " [⚠️ ALERT]"
    return summary
