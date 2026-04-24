"""Hand-Eye and Depth solvers.

Works on a homogeneous list of DataPoint objects:

    P_base  : (3,) marker position in base_link (from TF)
    uv      : (2,) marker pixel in the rectified image
    d_rel   : float relative depth at (u, v) from DA V2
    marker  : str (for bookkeeping)

Hand-Eye solution: cv2.solvePnP (SQPNP) -> T_cam_base (4x4)
Depth solution:    fits both Z_cam = a / d_rel + b and Z_cam = a * d_rel + b.
                   Uses marker-balanced weighting to ensure base markers (anchors)
                   are as important as moving arm markers.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

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
    model_type: str  # "linear" or "inverse"
    a: float
    b: float
    rmse_m: float
    n_samples: int
    d_range: Tuple[float, float]
    depth_range_m: Tuple[float, float]


_RANSAC_REPROJ_PX = 20.0
_RANSAC_MIN_INLIERS = 6


def solve_hand_eye(points: List[DataPoint], K: np.ndarray,
                   dist: Optional[np.ndarray] = None) -> HandEyeResult:
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
        ok, rvec, tvec = cv2.solvePnP(obj_in, img_in, K32, dist32, flags=cv2.SOLVEPNP_SQPNP)
        if not ok: use_ransac = False

    if not use_ransac:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K32, dist32, flags=cv2.SOLVEPNP_SQPNP)
        if not ok: raise RuntimeError('cv2.solvePnP failed')
        inlier_mask = np.ones(n, dtype=bool)

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()

    projected, _ = cv2.projectPoints(obj, rvec, tvec, K32, dist32)
    projected = projected.reshape(-1, 2)
    per_point = np.linalg.norm(projected - img, axis=1)

    return HandEyeResult(
        T_cam_base=T, reprojection_px=per_point,
        median_px=float(np.median(per_point[inlier_mask])),
        n_points=n, n_inliers=int(inlier_mask.sum()), inlier_mask=inlier_mask,
    )


def solve_depth(points: List[DataPoint],
                T_cam_base: np.ndarray) -> DepthResult:
    """Fit Z_cam = f(d_rel) using marker-balanced weighting."""
    if len(points) < 5:
        raise ValueError(f'Need at least 5 points, got {len(points)}')

    # 1. Map all points to camera frame to get ground truth Z
    P_base = np.array([p.P_base for p in points], dtype=np.float64)
    P_base_h = np.concatenate([P_base, np.ones((len(points), 1))], axis=1)
    Z_cam_all = (T_cam_base @ P_base_h.T).T[:, 2]
    d_rel_all = np.array([p.d_rel for p in points], dtype=np.float64)
    
    # 2. Assign weights so each marker type contributes equally to the fit
    marker_names = [p.marker for p in points]
    unique_markers = list(set(marker_names))
    marker_counts = {m: marker_names.count(m) for m in unique_markers}
    
    # Weight = 1.0 / (number of samples of this type)
    weights = np.array([1.0 / marker_counts[m] for m in marker_names])
    # Normalize weights so they sum to N
    weights = weights * len(points) / np.sum(weights)

    mask = (Z_cam_all > 0) & (d_rel_all > 1e-4)
    Z_cam = Z_cam_all[mask]
    d_rel = d_rel_all[mask]
    W = weights[mask]
    W_mat = np.diag(np.sqrt(W)) # For weighted LSQ

    # --- Model 1: Inverse (Z = a/d + b) ---
    x_inv = 1.0 / d_rel
    A_inv = np.column_stack([x_inv, np.ones_like(x_inv)])
    # Weighted LSQ: W*A*x = W*Z
    coeffs_inv, *_ = np.linalg.lstsq(W_mat @ A_inv, W_mat @ Z_cam, rcond=None)
    a_inv, b_inv = float(coeffs_inv[0]), float(coeffs_inv[1])
    rmse_inv = float(np.sqrt(np.mean(((a_inv / d_rel + b_inv) - Z_cam) ** 2)))

    # --- Model 2: Linear (Z = a*d + b) ---
    A_lin = np.column_stack([d_rel, np.ones_like(d_rel)])
    coeffs_lin, *_ = np.linalg.lstsq(W_mat @ A_lin, W_mat @ Z_cam, rcond=None)
    a_lin, b_lin = float(coeffs_lin[0]), float(coeffs_lin[1])
    rmse_lin = float(np.sqrt(np.mean(((a_lin * d_rel + b_lin) - Z_cam) ** 2)))

    # Choose winner
    if rmse_lin < rmse_inv:
        return DepthResult(
            model_type="linear", a=a_lin, b=b_lin, rmse_m=rmse_lin,
            n_samples=len(Z_cam),
            d_range=(float(d_rel.min()), float(d_rel.max())),
            depth_range_m=(float(Z_cam.min()), float(Z_cam.max())),
        )
    else:
        return DepthResult(
            model_type="inverse", a=a_inv, b=b_inv, rmse_m=rmse_inv,
            n_samples=len(Z_cam),
            d_range=(float(d_rel.min()), float(d_rel.max())),
            depth_range_m=(float(Z_cam.min()), float(Z_cam.max())),
        )

def invert_transform(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3], Ti[:3, 3] = R.T, -R.T @ t
    return Ti
