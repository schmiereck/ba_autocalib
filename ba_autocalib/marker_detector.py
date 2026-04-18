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
    # Optional tight "core" HSV bounds. If set, the detector requires
    # each candidate blob (from the loose hsv_lower/upper range) to
    # contain at least one pixel that also passes the strict range.
    # This lets the loose range be generous enough to keep the ball
    # mask solid (no graininess from shadow/gloss/edge pixels) while
    # strict confirmation kills background distractors that happen to
    # fall in the loose range but never reach "saturated ball red".
    # Both None (default) = single-pass behaviour.
    hsv_lower_strict: Optional[Tuple[int, int, int]] = None
    hsv_upper_strict: Optional[Tuple[int, int, int]] = None
    min_area_px: int = 30
    max_area_px: int = 5000
    min_compactness: float = 0.55
    # Reject detections whose bounding box actually touches the image
    # boundary — the marker is likely clipped, its centroid is biased
    # toward the remaining half, and feeding that into hand-eye solves
    # introduces a pose-dependent bias. A non-zero margin is too strict:
    # a ball 15 px from the edge is fully visible but the contour is a
    # clean circle, and the detection is perfectly usable.
    edge_margin_px: int = 1
    # MORPH_CLOSE kernel size (odd). Larger bridges wider specular
    # highlights on glossy markers; smaller keeps the ball mask from
    # tendrilling out to nearby same-color distractors. Tune per
    # marker: forearm needed 13 (big specular stripe), but grasp with
    # a loose HSV sees reddish distractors nearby that CLOSE 13 merged
    # into the ball contour. 7-9 is typical.
    close_kernel_px: int = 9


@dataclass
class HSVStats:
    """HSV percentiles of the pixels inside a detected marker contour.

    For non-wrap hues (S, V always, and H when lower[0] <= upper[0])
    the percentile fields hold raw p5/p50/p95 directly. For a
    wrap-around hue (red: lower[0]=170, upper[0]=10), H is reported as
    TWO disjoint ranges so both sides of H=0 are visible; the
    percentiles are computed on a shifted axis and mapped back.
    """
    n_pixels: int
    h_low: Tuple[int, int]    # (p5, p95) on the [>=lower[0]] side of the wheel
    h_high: Tuple[int, int]   # (p5, p95) on the [<=upper[0]] side; None if no wrap
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

    def detect(self, bgr: np.ndarray,
               verbose: bool = False) -> Dict[str, MarkerDetection]:
        """Run detection over all configured markers.

        verbose=True enables per-marker rejection diagnostics (only pass
        True on capture events; the debug tick runs at ~1 Hz and the
        logs would drown everything else).
        """
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
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        # CLOSE kernel size is per-marker (see MarkerConfig.close_kernel_px).
        # Enforce odd kernel size — cv2 requires it; silently fix even input.
        k = max(1, int(cfg.close_kernel_px))
        if k % 2 == 0:
            k += 1
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((k, k), np.uint8))

        # Strict-core confirmation: if a tighter HSV range is configured,
        # keep only connected components of the loose mask that contain
        # at least one pixel passing the strict range. This lets the
        # loose range be generous (solid ball mask, no graininess) while
        # strict kills background distractors that never reach the
        # "confident ball core" color profile.
        if (cfg.hsv_lower_strict is not None
                and cfg.hsv_upper_strict is not None):
            strict = self._mask_hue_range(hsv, cfg.hsv_lower_strict,
                                          cfg.hsv_upper_strict)
            num_cc, cc_labels = cv2.connectedComponents(mask)
            if num_cc > 1:
                # Per-label max over the strict mask: any value > 0 means
                # the component contains at least one strict pixel.
                # Vectorised via np.bincount on labels weighted by strict.
                flat_labels = cc_labels.ravel()
                flat_strict = strict.ravel()
                has_core = np.bincount(flat_labels,
                                       weights=(flat_strict > 0).astype(np.int32),
                                       minlength=num_cc) > 0
                has_core[0] = False  # background label always dropped
                keep_mask = has_core[cc_labels]
                mask = np.where(keep_mask, mask, 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # Convex-hull compactness: 4πA_hull / p_hull².
        # Using the hull rather than the raw contour makes the metric
        # robust against small physical protrusions (mounting clips,
        # gloss reflections that extend a few pixels beyond the ball) —
        # the hull of (ball + tiny bump) is still a near-perfect circle
        # while the hull of a tendril-blob or cable is large and sparse.
        # Centroid is still computed from the raw contour moments, so
        # the reported uv is the actual mask centre, not the hull centre.
        def _compactness(c: np.ndarray) -> float:
            hull = cv2.convexHull(c)
            ha = cv2.contourArea(hull)
            hp = cv2.arcLength(hull, True)
            return 4.0 * np.pi * ha / (hp * hp) if hp > 0 else 0.0

        H, W = hsv.shape[:2]
        margin = cfg.edge_margin_px
        # Classify every contour, for diagnostic logging when nothing passes.
        classified: List[Tuple[float, float, str]] = []  # (area, compactness, verdict)
        candidates = []
        for c in contours:
            a = cv2.contourArea(c)
            comp = _compactness(c)
            if a < cfg.min_area_px:
                classified.append((a, comp, 'too_small'))
            elif a > cfg.max_area_px:
                classified.append((a, comp, 'too_big'))
            elif comp < cfg.min_compactness:
                classified.append((a, comp, 'not_compact'))
            else:
                if margin > 0:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    if (bx < margin or by < margin
                            or bx + bw > W - margin
                            or by + bh > H - margin):
                        classified.append((a, comp, 'near_edge'))
                        continue
                classified.append((a, comp, 'pass'))
                candidates.append(c)
        if not candidates:
            if verbose and self._logger is not None and classified:
                top = sorted(classified, key=lambda x: x[0], reverse=True)[:5]
                summary = ', '.join(f'a={a:.0f} c={comp:.2f} {v}'
                                    for a, comp, v in top)
                self._logger.info(
                    f'[{cfg.name}] no candidate passed. top-5 by area: {summary} '
                    f'(window [{cfg.min_area_px},{cfg.max_area_px}] px, '
                    f'min_compactness {cfg.min_compactness})')
            return None
        best = max(candidates,
                   key=lambda c: _compactness(c) ** 2 * cv2.contourArea(c))
        biggest = best
        area = float(cv2.contourArea(biggest))
        m = cv2.moments(biggest)
        if m['m00'] <= 0:
            return None
        u = m['m10'] / m['m00']
        v = m['m01'] / m['m00']
        # Confidence: hull compactness (1.0 = perfect circle), consistent
        # with the candidate filter above.
        compactness = _compactness(biggest)
        stats = self._pixel_stats(hsv, mask, biggest, cfg)
        return MarkerDetection(
            name=cfg.name,
            uv=(float(u), float(v)),
            area_px=area,
            confidence=float(np.clip(compactness, 0.0, 1.0)),
            hsv_stats=stats,
        )

    @staticmethod
    def _pixel_stats(hsv: np.ndarray, hsv_match_mask: np.ndarray,
                     contour: np.ndarray,
                     cfg: 'MarkerConfig') -> Optional['HSVStats']:
        """HSV percentiles of pixels inside `contour` that also matched
        the HSV range. Morphologically-closed filler pixels are excluded
        (their HSV isn't in-band, so they'd bias S/V floors downward).
        """
        contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], 0, 255, cv2.FILLED)
        effective = cv2.bitwise_and(contour_mask, hsv_match_mask)
        pixels = hsv[effective > 0]
        if pixels.shape[0] < 20:
            return None
        h_vals = pixels[:, 0].astype(np.int32)
        s_vals = pixels[:, 1].astype(np.int32)
        v_vals = pixels[:, 2].astype(np.int32)
        s_p = np.percentile(s_vals, [5, 95]).astype(int)
        v_p = np.percentile(v_vals, [5, 95]).astype(int)
        wrap = cfg.hsv_lower[0] > cfg.hsv_upper[0]
        if not wrap:
            h_p = np.percentile(h_vals, [5, 95]).astype(int)
            h_low = (int(h_p[0]), int(h_p[1]))
            h_high = (0, 0)
        else:
            # Split pixels: "low" side = H >= lower[0], "high" side = H <= upper[0]
            low_side = h_vals[h_vals >= cfg.hsv_lower[0]]
            high_side = h_vals[h_vals <= cfg.hsv_upper[0]]
            if low_side.size >= 10:
                lp = np.percentile(low_side, [5, 95]).astype(int)
                h_low = (int(lp[0]), int(lp[1]))
            else:
                h_low = (cfg.hsv_lower[0], cfg.hsv_lower[0])
            if high_side.size >= 10:
                hp = np.percentile(high_side, [5, 95]).astype(int)
                h_high = (int(hp[0]), int(hp[1]))
            else:
                h_high = (cfg.hsv_upper[0], cfg.hsv_upper[0])
        return HSVStats(
            n_pixels=int(pixels.shape[0]),
            h_low=h_low,
            h_high=h_high,
            s_p5_p95=(int(s_p[0]), int(s_p[1])),
            v_p5_p95=(int(v_p[0]), int(v_p[1])),
            wrap=bool(wrap),
        )

    # BGR tints for mask overlay, keyed by marker name.
    _MASK_TINTS: Dict[str, Tuple[int, int, int]] = {
        'grasp': (0, 128, 255),
        'forearm': (0, 255, 0),
        'base': (255, 0, 0),
    }

    def draw_mask_overlay(self, bgr: np.ndarray,
                          alpha: float = 0.45) -> np.ndarray:
        """Tint pixels matched by each marker's HSV range.

        Use this to diagnose HSV thresholds: if a marker is never detected,
        its tint will also be absent, pointing at the threshold as culprit.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        out = bgr.copy()
        for name, cfg in self._configs.items():
            mask = self._mask_hue_range(hsv, cfg.hsv_lower, cfg.hsv_upper)
            if not np.any(mask):
                continue
            tint = np.array(self._MASK_TINTS.get(name, (255, 255, 255)),
                            dtype=np.uint8)
            tinted = np.full_like(out, tint)
            blended = cv2.addWeighted(out, 1.0 - alpha, tinted, alpha, 0.0)
            out = np.where(mask[..., None].astype(bool), blended, out)
        return out

    def draw_overlay(self, bgr: np.ndarray,
                     detections: Dict[str, MarkerDetection],
                     reprojections: Optional[Dict[str, Tuple[float, float]]]
                     = None) -> np.ndarray:
        out = bgr.copy()
        for name, det in detections.items():
            u, v = int(det.uv[0]), int(det.uv[1])
            cv2.circle(out, (u, v), 8, (0, 255, 255), 2)
            label = f'{name} a={int(det.area_px)} c={det.confidence:.2f}'
            cv2.putText(out, label, (u + 10, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # Annotate which configured markers are missing (top-left corner).
        missing = [name for name in self._configs
                   if name not in detections]
        if missing:
            cv2.putText(out, 'missing: ' + ', '.join(missing), (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if reprojections:
            for name, (u, v) in reprojections.items():
                cv2.drawMarker(out, (int(u), int(v)), (0, 0, 255),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=14, thickness=2)
        return out
