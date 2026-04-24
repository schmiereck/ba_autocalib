"""HSV-based and CNN-based marker detectors.

Finds colored markers in a BGR image and returns their subpixel centroids.
Expected to be called on already-rectified images.
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
    hsv_lower_strict: Optional[Tuple[int, int, int]] = None
    hsv_upper_strict: Optional[Tuple[int, int, int]] = None
    min_area_px: int = 30
    max_area_px: int = 5000
    min_compactness: float = 0.55
    edge_margin_px: int = 1
    close_kernel_px: int = 9


@dataclass
class HSVStats:
    n_pixels: int
    h_low: Tuple[int, int]
    h_high: Tuple[int, int]
    s_p5_p95: Tuple[int, int]
    v_p5_p95: Tuple[int, int]
    wrap: bool


@dataclass
class MarkerDetection:
    name: str
    uv: Tuple[float, float]
    area_px: float
    confidence: float
    hsv_stats: Optional['HSVStats'] = None


class MarkerDetector:
    def __init__(self, configs: List[MarkerConfig],
                 logger: Optional[object] = None):
        self._configs = {c.name: c for c in configs}
        self._logger = logger

    @staticmethod
    def _mask_hue_range(hsv: np.ndarray, low: Tuple[int, int, int],
                        high: Tuple[int, int, int]) -> np.ndarray:
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

    def detect(self, bgr: np.ndarray,
               verbose: bool = False) -> Dict[str, MarkerDetection]:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        out: Dict[str, MarkerDetection] = {}
        for name, cfg in self._configs.items():
            det = self._detect_one(hsv, cfg, verbose=verbose)
            if det is not None:
                out[name] = det
        return out

    def _detect_one(self, hsv: np.ndarray,
                    cfg: MarkerConfig,
                    verbose: bool = False) -> Optional[MarkerDetection]:
        mask = self._mask_hue_range(hsv, cfg.hsv_lower, cfg.hsv_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        k = max(1, int(cfg.close_kernel_px))
        if k % 2 == 0: k += 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))

        if (cfg.hsv_lower_strict is not None and cfg.hsv_upper_strict is not None):
            strict = self._mask_hue_range(hsv, cfg.hsv_lower_strict, cfg.hsv_upper_strict)
            num_cc, cc_labels = cv2.connectedComponents(mask)
            if num_cc > 1:
                flat_labels = cc_labels.ravel()
                flat_strict = strict.ravel()
                has_core = np.bincount(flat_labels,
                                       weights=(flat_strict > 0).astype(np.int32),
                                       minlength=num_cc) > 0
                has_core[0] = False
                keep_mask = has_core[cc_labels]
                mask = np.where(keep_mask, mask, 0).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        def _compactness(c: np.ndarray) -> float:
            hull = cv2.convexHull(c)
            ha = cv2.contourArea(hull)
            hp = cv2.arcLength(hull, True)
            return 4.0 * np.pi * ha / (hp * hp) if hp > 0 else 0.0

        H, W = hsv.shape[:2]
        candidates = []
        for c in contours:
            a = cv2.contourArea(c)
            comp = _compactness(c)
            if cfg.min_area_px <= a <= cfg.max_area_px and comp >= cfg.min_compactness:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bx >= cfg.edge_margin_px and by >= cfg.edge_margin_px and \
                   bx+bw <= W-cfg.edge_margin_px and by+bh <= H-cfg.edge_margin_px:
                    candidates.append(c)

        if not candidates: return None
        best = max(candidates, key=lambda c: _compactness(c) ** 2 * cv2.contourArea(c))
        area = float(cv2.contourArea(best))
        m = cv2.moments(best)
        if m['m00'] <= 0: return None
        u, v = m['m10'] / m['m00'], m['m01'] / m['m00']
        return MarkerDetection(
            name=cfg.name, uv=(float(u), float(v)), area_px=area,
            confidence=float(np.clip(_compactness(best), 0.0, 1.0)),
            hsv_stats=self._pixel_stats(hsv, mask, best, cfg)
        )

    @staticmethod
    def _pixel_stats(hsv: np.ndarray, hsv_match_mask: np.ndarray,
                     contour: np.ndarray, cfg: MarkerConfig) -> Optional[HSVStats]:
        contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], 0, 255, cv2.FILLED)
        effective = cv2.bitwise_and(contour_mask, hsv_match_mask)
        pixels = hsv[effective > 0]
        if pixels.shape[0] < 20: return None
        h_vals, s_vals, v_vals = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        s_p = np.percentile(s_vals, [5, 95]).astype(int)
        v_p = np.percentile(v_vals, [5, 95]).astype(int)
        wrap = cfg.hsv_lower[0] > cfg.hsv_upper[0]
        if not wrap:
            h_p = np.percentile(h_vals, [5, 95]).astype(int)
            h_low, h_high = (int(h_p[0]), int(h_p[1])), (0, 0)
        else:
            ls = h_vals[h_vals >= cfg.hsv_lower[0]]
            hs = h_vals[h_vals <= cfg.hsv_upper[0]]
            h_low = tuple(np.percentile(ls, [5, 95]).astype(int)) if ls.size >= 10 else (cfg.hsv_lower[0],)*2
            h_high = tuple(np.percentile(hs, [5, 95]).astype(int)) if hs.size >= 10 else (cfg.hsv_upper[0],)*2
        return HSVStats(len(pixels), h_low, h_high, (s_p[0], s_p[1]), (v_p[0], v_p[1]), wrap)

    # --- Shared helper methods for visualization ---
    _MASK_TINTS: Dict[str, Tuple[int, int, int]] = {
        'grasp': (0, 128, 255), 'forearm': (0, 255, 0), 'base': (255, 0, 0),
    }

    def draw_mask_overlay(self, bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        out = bgr.copy()
        for name, cfg in self._configs.items():
            mask = self._mask_hue_range(hsv, cfg.hsv_lower, cfg.hsv_upper)
            if not np.any(mask): continue
            tint = np.array(self._MASK_TINTS.get(name, (255, 255, 255)), dtype=np.uint8)
            blended = cv2.addWeighted(out, 1.0 - alpha, np.full_like(out, tint), alpha, 0.0)
            out = np.where(mask[..., None].astype(bool), blended, out)
        return out

    def draw_overlay(self, bgr: np.ndarray, detections: Dict[str, MarkerDetection],
                     reprojections: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        out = bgr.copy()
        for name, det in detections.items():
            u, v = int(det.uv[0]), int(det.uv[1])
            # Draw KI result in bright cyan
            cv2.circle(out, (u, v), 8, (255, 255, 0), 2)
            cv2.putText(out, f'{name} c={det.confidence:.2f}', (u + 12, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        
        missing = [n for n in self._configs if n not in detections]
        if missing:
            cv2.putText(out, 'missing: ' + ', '.join(missing), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        if reprojections:
            for name, (u, v) in reprojections.items():
                # Draw reprojected URDF model in red cross
                cv2.drawMarker(out, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        return out


class CNNMarkerDetector:
    def __init__(self, model_path: str, configs: List[MarkerConfig], logger: Optional[object] = None):
        self._logger = logger
        self._configs = {c.name: c for c in configs}
        from ultralytics import YOLO
        if logger: logger.info(f'CNNMarkerDetector: Loading model from {model_path}')
        self._model = YOLO(model_path)
        self._class_names = {0: 'grasp', 1: 'forearm', 2: 'base'}

    def detect(self, bgr: np.ndarray, verbose: bool = False, conf: float = 0.5) -> Dict[str, MarkerDetection]:
        results = self._model.predict(source=bgr, conf=conf, verbose=False)
        out: Dict[str, MarkerDetection] = {}
        if not results: return out
        
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
        
        res = results[0]
        for i in range(len(res.boxes)):
            cls_idx = int(res.boxes.cls[i].item())
            name = self._class_names.get(cls_idx)
            if not name: continue
            
            kpts = res.keypoints.xy[i].cpu().numpy()
            if kpts.shape[0] > 0:
                u_ki, v_ki = kpts[0]
                if u_ki == 0 and v_ki == 0: continue
                
                # --- Subpixel Refinement via Centroid ---
                # Crop a small region around the KI point
                u_int, v_vint = int(u_ki), int(v_ki)
                win = 15
                h, w = bgr.shape[:2]
                y1, y2 = max(0, v_vint-win), min(h, v_vint+win)
                x1, x2 = max(0, u_int-win), min(w, u_int+win)
                
                roi_hsv = hsv[y1:y2, x1:x2]
                cfg = self._configs.get(name)
                
                # Use a very simple moment-based refinement if we have config
                if cfg:
                    mask = MarkerDetector._mask_hue_range(roi_hsv, cfg.hsv_lower, cfg.hsv_upper)
                    M = cv2.moments(mask)
                    if M['m00'] > 5: # If enough pixels match the expected color
                        u_ki = x1 + M['m10'] / M['m00']
                        v_ki = y1 + M['m01'] / M['m00']

                out[name] = MarkerDetection(name=name, uv=(float(u_ki), float(v_ki)),
                                            area_px=1000.0, confidence=float(res.boxes.conf[i].item()))
        
        if verbose and self._logger:
            self._logger.info(f'CNN Detection: found {list(out.keys())}')
        return out

    def draw_mask_overlay(self, bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        return bgr

    def draw_overlay(self, bgr: np.ndarray, detections: Dict[str, MarkerDetection],
                     reprojections: Optional[Dict[str, Tuple[float, float]]] = None) -> np.ndarray:
        # Use the same drawing logic as the HSV detector
        temp_detector = MarkerDetector(list(self._configs.values()))
        return temp_detector.draw_overlay(bgr, detections, reprojections)
