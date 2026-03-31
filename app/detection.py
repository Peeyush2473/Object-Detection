"""
detection.py – Core object detection module.

Wraps the Ultralytics YOLOv8 model and provides a clean inference
interface tailored for the Smart Surveillance system.
"""

import os
import time
from typing import List, Dict, Optional, Set

import numpy as np
from ultralytics import YOLO

# Resolve project root (one level up from app/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import config from project root
import sys
sys.path.insert(0, _PROJECT_ROOT)
import config


class SurveillanceDetector:
    """
    High-level wrapper around YOLOv8 for surveillance-specific
    object detection.

    Attributes
    ----------
    model : YOLO
        Loaded YOLOv8 model instance.
    confidence : float
        Minimum confidence threshold for detections.
    monitored_classes : set
        Set of class names to report.
    alert_classes : set
        Subset that triggers high-priority alerts.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = config.CONFIDENCE_THRESHOLD,
        iou: float = config.IOU_THRESHOLD,
        monitored_classes: Optional[List[str]] = None,
        alert_classes: Optional[List[str]] = None,
    ):
        """
        Initialize the detector.

        Parameters
        ----------
        model_path : str, optional
            Path to a YOLO .pt weights file.  Falls back to
            ``config.MODEL_NAME`` which downloads automatically the
            first time.
        confidence : float
            Minimum confidence score for a detection to be kept.
        iou : float
            IoU threshold for NMS.
        monitored_classes : list[str], optional
            Class names to track. Defaults to ``config.MONITORED_CLASSES``.
        alert_classes : list[str], optional
            Class names that trigger alerts. Defaults to ``config.ALERT_CLASSES``.
        """
        if model_path is None:
            model_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)

        # If weights file doesn't exist locally, Ultralytics will auto-download
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou = iou

        self.monitored_classes: Set[str] = set(
            monitored_classes or config.MONITORED_CLASSES
        )
        self.alert_classes: Set[str] = set(
            alert_classes or config.ALERT_CLASSES
        )

        # Build a reverse map: class_id → name from the model
        self._class_names: Dict[int, str] = self.model.names  # {0: 'person', …}

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (H×W×3).

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``class_name`` (str)
            - ``class_id`` (int)
            - ``confidence`` (float, 0-1)
            - ``bbox`` (list[int], [x1, y1, x2, y2])
            - ``is_alert`` (bool) – True when the class is suspicious
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self._class_names.get(class_id, "unknown")

                # Only keep monitored classes
                if class_name not in self.monitored_classes:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "class_name": class_name,
                    "class_id": class_id,
                    "confidence": round(conf, 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "is_alert": class_name in self.alert_classes,
                })

        return detections

    def get_model_info(self) -> Dict:
        """Return basic information about the loaded model."""
        return {
            "model_type": str(self.model.model_name) if hasattr(self.model, "model_name") else config.MODEL_NAME,
            "confidence_threshold": self.confidence,
            "iou_threshold": self.iou,
            "monitored_classes": sorted(self.monitored_classes),
            "alert_classes": sorted(self.alert_classes),
        }
