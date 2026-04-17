"""HSV-based colored-marker detector.

Finds colored markers in a BGR image and returns their subpixel centroids.
Expected to be called on already-rectified images (pinhole camera model,
zero distortion).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MarkerConfig:
    name: str
    tf_frame: str
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    min_area_px: int = 30
    max_area_px: int = 5000


@dataclass
class MarkerDetection:
    name: str
    uv: Tuple[float, float]
    area_px: float
    confidence: float


class MarkerDetector:
    def __init__(self, configs: List[MarkerConfig]):
        self._configs = {c.name: c for c in configs}

    @staticmethod
    def _mask_hue_range(hsv: np.ndarray, low: Tuple[int, int, int],
                        high: Tuple[int, int, int]) -> np.ndarray:
        """HSV threshold with optional hue wrap-around (e.g. red)."""
        if low[0] <= high[0]:
            return cv2.inRange(hsv, np.array(low), np.array(high))
        lower_a = (0, low[1], low[2])
        upper_a = (high[0], high[1], high[2])
        lower_b = (low[0], low[1], low[2])
        upper_b = (179, high[1], high[2])
        return cv2.bitwise_or(
            cv2.inRange(hsv, np.array(lower_a), np.array(upper_a)),
            cv2.inRange(hsv, np.array(lower_b), np.array(upper_b)),
        )

    def detect(self, bgr: np.ndarray) -> Dict[str, MarkerDetection]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        out: Dict[str, MarkerDetection] = {}
        for name, cfg in self._configs.items():
            det = self._detect_one(hsv, cfg)
            if det is not None:
                out[name] = det
        return out

    def _detect_one(self, hsv: np.ndarray,
                    cfg: MarkerConfig) -> Optional[MarkerDetection]:
        mask = self._mask_hue_range(hsv, cfg.hsv_lower, cfg.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(biggest))
        if area < cfg.min_area_px or area > cfg.max_area_px:
            return None
        m = cv2.moments(biggest)
        if m['m00'] <= 0:
            return None
        u = m['m10'] / m['m00']
        v = m['m01'] / m['m00']
        # Confidence: contour compactness (1.0 = perfect circle).
        perim = cv2.arcLength(biggest, True)
        compactness = 4.0 * np.pi * area / (perim * perim) if perim > 0 else 0.0
        return MarkerDetection(
            name=cfg.name,
            uv=(float(u), float(v)),
            area_px=area,
            confidence=float(np.clip(compactness, 0.0, 1.0)),
        )

    def draw_overlay(self, bgr: np.ndarray,
                     detections: Dict[str, MarkerDetection],
                     reprojections: Optional[Dict[str, Tuple[float, float]]]
                     = None) -> np.ndarray:
        out = bgr.copy()
        for name, det in detections.items():
            u, v = int(det.uv[0]), int(det.uv[1])
            cv2.circle(out, (u, v), 8, (0, 255, 255), 2)
            cv2.putText(out, name, (u + 10, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if reprojections:
            for name, (u, v) in reprojections.items():
                cv2.drawMarker(out, (int(u), int(v)), (0, 0, 255),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=14, thickness=2)
        return out
