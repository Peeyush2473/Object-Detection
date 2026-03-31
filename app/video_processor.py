"""
video_processor.py – Video capture and frame-level processing.

Provides a generator-based pipeline that reads frames from a webcam or
video file, runs detection, draws annotations, and yields results.
"""

import time
from typing import Generator, Dict, Optional, Tuple

import cv2
import numpy as np

from app.detection import SurveillanceDetector


# ──────────────────────────────────────────────────────────────────── #
#  Drawing helpers
# ──────────────────────────────────────────────────────────────────── #

# Colour palette: normal = green, alert = red
_COLOR_NORMAL = (0, 200, 80)       # BGR green
_COLOR_ALERT = (0, 0, 230)         # BGR red
_COLOR_TEXT_BG = (30, 30, 30)      # dark background
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes, labels, and confidence scores on the frame.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to annotate (modified **in-place** and also returned).
    detections : list[dict]
        Output from ``SurveillanceDetector.detect()``.

    Returns
    -------
    np.ndarray
        Annotated frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = _COLOR_ALERT if det["is_alert"] else _COLOR_NORMAL
        label = f'{det["class_name"]} {det["confidence"]:.0%}'

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), _FONT, 0.55, (255, 255, 255), 1)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS counter on the top-left corner."""
    label = f"FPS: {fps:.1f}"
    cv2.putText(frame, label, (10, 28), _FONT, 0.7, (0, 255, 255), 2)
    return frame


# ──────────────────────────────────────────────────────────────────── #
#  Video processing generator
# ──────────────────────────────────────────────────────────────────── #

def process_video(
    source,
    detector: SurveillanceDetector,
    max_width: int = 720,
    show_fps: bool = True,
) -> Generator[Tuple[np.ndarray, list, float], None, None]:
    """
    Generator that yields ``(annotated_frame, detections, fps)`` tuples.

    Parameters
    ----------
    source : int | str
        ``0`` for webcam, or a file path / URL for video.
    detector : SurveillanceDetector
        Initialised detector instance.
    max_width : int
        Frames wider than this are resized (aspect-ratio preserved).
    show_fps : bool
        Whether to overlay an FPS counter on the frame.

    Yields
    ------
    tuple[np.ndarray, list, float]
        (annotated_frame, raw_detections, current_fps)
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    prev_time = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize for display / performance
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (max_width, int(h * scale)))

            # Detect
            detections = detector.detect(frame)

            # Annotate
            annotated = draw_detections(frame.copy(), detections)

            # FPS
            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 1e-6)
            prev_time = current_time

            if show_fps:
                draw_fps(annotated, fps)

            yield annotated, detections, fps
    finally:
        cap.release()
