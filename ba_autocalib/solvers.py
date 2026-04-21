"""Hand-Eye and Depth solvers.

Works on a homogeneous list of DataPoint objects:

    P_base  : (3,) marker position in base_link (from TF)
    uv      : (2,) marker pixel in the rectified image
    d_rel   : float relative depth at (u, v) from DA V2
    marker  : str (for bookkeeping)

Hand-Eye solution: cv2.solvePnP (SQPNP) -> T_cam_base (4x4)
Depth solution:    curve fit Z_cam = a / d_rel + b using the PnP result.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DataPoint:
    marker: str
    P_base: np.ndarray   # (3,) float
    uv: np.ndarray       # (2,) float
    d_rel: float
    timestamp: float


@dataclass
class HandEyeResult:
    T_cam_base: np.ndarray        # (4, 4) camera = T @ base
    reprojection_px: np.ndarray   # (N,) per-point pixel error (all points)
    median_px: float              # median over inliers only
    n_points: int                 # total correspondences fed in
    n_inliers: int                # RANSAC inliers (== n_points if RANSAC skipped)
    inlier_mask: np.ndarray       # (N,) bool — True for RANSAC inliers


@dataclass
class DepthResult:
    a: float
    b: float
    rmse_m: float
    n_samples: int
    d_range: Tuple[float, float]
    depth_range_m: Tuple[float, float]


_RANSAC_REPROJ_PX = 20.0   # inlier threshold for RANSAC
_RANSAC_MIN_INLIERS = 6    # fall back to full-set SQPNP below this count


def solve_hand_eye(points: List[DataPoint], K: np.ndarray,
                   dist: Optional[np.ndarray] = None) -> HandEyeResult:
    """Solve T_cam_base from N >= 4 (P_base, uv) correspondences.

    Uses RANSAC to discard outlier correspondences (wrong detections, TF
    timing jitter), then refines with SQPNP on the inlier subset.
    Falls back to plain SQPNP on all points if RANSAC finds too few inliers.
    """
    if len(points) < 4:
        raise ValueError(f'Need at least 4 points, got {len(points)}')

    obj = np.array([p.P_base for p in points], dtype=np.float32)
    img = np.array([p.uv for p in points], dtype=np.float32)
    if dist is None:
        dist = np.zeros(5, dtype=np.float32)

    K32 = K.astype(np.float32)
    dist32 = dist.astype(np.float32)
    n = len(points)
    inlier_mask = np.ones(n, dtype=bool)

    # --- RANSAC pass ---
    ok_r, rvec_r, tvec_r, ransac_inliers = cv2.solvePnPRansac(
        obj, img, K32, dist32,
        reprojectionError=_RANSAC_REPROJ_PX,
        iterationsCount=5000,
        confidence=0.999,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    use_ransac = (ok_r and ransac_inliers is not None
                  and len(ransac_inliers) >= _RANSAC_MIN_INLIERS)

    if use_ransac:
        idx = ransac_inliers.flatten()
        inlier_mask = np.zeros(n, dtype=bool)
        inlier_mask[idx] = True
        obj_in = obj[idx]
        img_in = img[idx]
        # Refine on inliers with SQPNP
        ok, rvec, tvec = cv2.solvePnP(
            obj_in, img_in, K32, dist32,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            use_ransac = False   # fall through to full-set solve below

    if not use_ransac:
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, K32, dist32,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            raise RuntimeError('cv2.solvePnP failed')
        inlier_mask = np.ones(n, dtype=bool)

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    projected, _ = cv2.projectPoints(obj, rvec, tvec, K32, dist32)
    projected = projected.reshape(-1, 2)
    per_point = np.linalg.norm(projected - img, axis=1)

    return HandEyeResult(
        T_cam_base=T,
        reprojection_px=per_point,
        median_px=float(np.median(per_point[inlier_mask])),
        n_points=n,
        n_inliers=int(inlier_mask.sum()),
        inlier_mask=inlier_mask,
    )


def solve_depth(points: List[DataPoint],
                T_cam_base: np.ndarray) -> DepthResult:
    """Fit Z_cam = a / d_rel + b using PnP-derived ground truth Z_cam."""
    if len(points) < 5:
        raise ValueError(f'Need at least 5 points for depth fit, got {len(points)}')

    P_base = np.array([p.P_base for p in points], dtype=np.float64)
    d_rel = np.array([p.d_rel for p in points], dtype=np.float64)

    P_base_h = np.concatenate([P_base, np.ones((len(points), 1))], axis=1)
    P_cam = (T_cam_base @ P_base_h.T).T[:, :3]
    Z_cam = P_cam[:, 2]

    # Mask non-positive depths defensively (behind-camera points from a bad
    # PnP should have been filtered upstream, but be safe).
    mask = (Z_cam > 0) & (d_rel > 1e-4)
    if mask.sum() < 5:
        raise RuntimeError('Too few valid (Z_cam, d_rel) pairs after filtering')
    Z_cam = Z_cam[mask]
    d_rel = d_rel[mask]

    # Linear LSQ on 1/d: Z = a * (1/d) + b
    x = 1.0 / d_rel
    A = np.column_stack([x, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(A, Z_cam, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])

    pred = a / d_rel + b
    rmse = float(np.sqrt(np.mean((pred - Z_cam) ** 2)))

    return DepthResult(
        a=a, b=b, rmse_m=rmse, n_samples=int(mask.sum()),
        d_range=(float(d_rel.min()), float(d_rel.max())),
        depth_range_m=(float(Z_cam.min()), float(Z_cam.max())),
    )


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform (faster and stabler than np.linalg.inv)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti
