#!/usr/bin/env python3
"""Offline solvePnP analysis with hardcoded datapoints from the latest run.

Usage:
  python3 analyze_calib.py                  # uses hardcoded K guess
  python3 analyze_calib.py fx fy cx cy      # override K (e.g. 800 800 320 240)

  K from autocalib startup log:
    [ba_autocalib_node]: Got camera_info: WxH, fx=..., fy=...
"""

import sys
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# DATA from the latest run (315px result, 2026-04-20)
# Format: (pose_name, marker, P_base, uv)
# P_base = (x, y, z) in base_link, uv = (u, v) pixel
# ---------------------------------------------------------------------------
DATA = [
    # pose 1 – extend_5_forward (grasp + forearm detected, logged before "Pose ... +2 points")
    # We only have pose 1 totals (+2 points) without individual log lines shown.
    # Skipping pose 1 — use poses 3-21 which are fully logged.

    # pose 3 – extend_5_right
    ("extend_5_right",        "grasp",   (0.034, 0.093, 0.148), (394.2,  90.7)),

    # pose 4 – extend_10_forward
    ("extend_10_forward",     "grasp",   (0.000, 0.171, 0.118), (264.1, 156.4)),
    ("extend_10_forward",     "forearm", (0.000, 0.066, 0.179), (391.4, 117.6)),

    # pose 5 – extend_10_left
    ("extend_10_left",        "grasp",   (-0.034, 0.191, 0.096), (470.4, 284.3)),

    # pose 6 – extend_10_right
    ("extend_10_right",       "grasp",   (0.058, 0.159, 0.115), (268.6,  86.0)),

    # pose 7 – flat_15_forward
    ("flat_15_forward",       "grasp",   (-0.004, 0.238, 0.068), (467.8, 312.3)),
    ("flat_15_forward",       "forearm", (-0.002, 0.103, 0.096), (342.4, 214.8)),

    # pose 8 – flat_15_left
    ("flat_15_left",          "grasp",   (-0.041, 0.234, 0.063), (474.1, 335.5)),

    # pose 9 – flat_15_right
    ("flat_15_right",         "grasp",   (0.075, 0.195, 0.064), (247.7, 114.6)),
    ("flat_15_right",         "forearm", (0.038, 0.098, 0.095), (349.9, 151.1)),

    # pose 10 – flat_height_10_forward
    ("flat_height_10_fwd",    "grasp",   (0.000, 0.170, 0.197), (230.5,  54.1)),

    # pose 11 – upward_10_forward
    ("upward_10_forward",     "grasp",   (0.000, 0.231, 0.141), (222.6, 139.5)),

    # pose 12 – downward_10_forward (compactness fix worked!)
    ("downward_10_forward",   "grasp",   (0.000, 0.138, 0.114), (301.1, 164.1)),
    ("downward_10_forward",   "forearm", (0.000, 0.048, 0.204), (419.8,  95.8)),

    # pose 14 – flat_middle_5_forward
    ("flat_middle_5_fwd",     "grasp",   (0.000, 0.272, 0.134), (124.0, 150.2)),

    # pose 15 – upward_10_left
    ("upward_10_left",        "grasp",   (-0.034, 0.194, 0.160), (110.4, 150.4)),
    ("upward_10_left",        "forearm", (-0.023, 0.128, 0.089), (272.0, 176.0)),

    # pose 16 – upward_10_right
    ("upward_10_right",       "grasp",   (0.034, 0.195, 0.156), (271.6, 163.0)),

    # pose 18 – flat_height_10_right
    ("flat_height_10_right",  "grasp",   (0.027, 0.172, 0.179), (238.1,  38.1)),

    # pose 19 – downward_10_left
    ("downward_10_left",      "grasp",   (-0.021, 0.119, 0.144), (262.1, 139.4)),

    # pose 21 – mid_height_left
    ("mid_height_left",       "grasp",   (-0.028, 0.160, 0.170), (269.6, 174.4)),
]

# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------
if len(sys.argv) == 5:
    fx, fy, cx, cy = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    print(f"K from args: fx={fx} fy={fy} cx={cx} cy={cy}")
else:
    # Logitech C930e @ 640x480 — from ba_camera_bridge/config/overview_camera_info.yaml
    # Node rectifies with initUndistortRectifyMap(K, D, None, K, ...) so same K applies;
    # dist is zeroed out after rectification.
    fx, fy, cx, cy = 518.94459, 518.92238, 340.04769, 238.39717
    print(f"K from overview_camera_info.yaml: fx={fx} fy={fy} cx={cx} cy={cy}")

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1.0]], dtype=np.float64)
dist = np.zeros(5, dtype=np.float64)

obj = np.array([d[2] for d in DATA], dtype=np.float32)
img = np.array([d[3] for d in DATA], dtype=np.float32)

# ---------------------------------------------------------------------------
# Solve (SQPNP, all points)
# ---------------------------------------------------------------------------
ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
if not ok:
    print("solvePnP FAILED")
    sys.exit(1)

projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
projected = projected.reshape(-1, 2)
errors = np.linalg.norm(projected - img, axis=1)

R, _ = cv2.Rodrigues(rvec)
T = np.eye(4)
T[:3, :3] = R
T[:3, 3] = tvec.flatten()

print(f"\nAll points:  median={np.median(errors):.1f}px  mean={np.mean(errors):.1f}px  max={np.max(errors):.1f}px  (n={len(errors)})")
print(f"a (depth, positive=valid): {T[2,3]:.4f}")

print("\nPer-point residuals (sorted):")
order = np.argsort(errors)[::-1]
for i in order:
    pose, marker, pb, uv = DATA[i]
    print(f"  {errors[i]:6.1f}px  [{marker:7s}]  pose={pose:30s}  "
          f"P_base=({pb[0]:+.3f},{pb[1]:+.3f},{pb[2]:+.3f})  "
          f"detected=({uv[0]:.0f},{uv[1]:.0f})  "
          f"projected=({projected[i,0]:.0f},{projected[i,1]:.0f})")

# ---------------------------------------------------------------------------
# RANSAC-PnP: find largest consistent inlier subset
# ---------------------------------------------------------------------------
print("\n--- RANSAC-PnP (reprojectionError=20px) ---")
ok_r, rvec_r, tvec_r, inliers = cv2.solvePnPRansac(
    obj, img, K, dist,
    reprojectionError=20.0,
    iterationsCount=5000,
    confidence=0.999,
    flags=cv2.SOLVEPNP_SQPNP)
if ok_r and inliers is not None:
    idx = inliers.flatten()
    proj_r, _ = cv2.projectPoints(obj, rvec_r, tvec_r, K, dist)
    proj_r = proj_r.reshape(-1, 2)
    errs_r = np.linalg.norm(proj_r - img, axis=1)
    R_r, _ = cv2.Rodrigues(rvec_r)
    T_r = np.eye(4); T_r[:3,:3] = R_r; T_r[:3,3] = tvec_r.flatten()
    print(f"Inliers: {len(idx)}/{len(DATA)}  median={np.median(errs_r[idx]):.1f}px  "
          f"a={T_r[2,3]:.4f}")
    print("Camera transform (base→cam T matrix):")
    for row in T_r:
        print(f"  [{row[0]:+.5f}, {row[1]:+.5f}, {row[2]:+.5f}, {row[3]:+.5f}]")
    print("Inlier points:")
    for i in idx:
        pose, marker, pb, uv = DATA[i]
        print(f"  {errs_r[i]:6.1f}px  [{marker:7s}]  pose={pose}")
    print("Outlier points:")
    all_idx = set(range(len(DATA)))
    for i in sorted(all_idx - set(idx.tolist())):
        pose, marker, pb, uv = DATA[i]
        print(f"  {errs_r[i]:6.1f}px  [{marker:7s}]  pose={pose}  "
              f"P_base=({pb[0]:+.3f},{pb[1]:+.3f},{pb[2]:+.3f})  uv=({uv[0]:.0f},{uv[1]:.0f})")
else:
    print("RANSAC failed or no inliers found")

# Iterative inlier refinement: drop highest residual, re-solve, repeat
print("\n--- Iterative outlier removal (drop until median < 15px or n < 8) ---")
cur_data = list(DATA)
while len(cur_data) >= 8:
    obj_c = np.array([d[2] for d in cur_data], dtype=np.float32)
    img_c = np.array([d[3] for d in cur_data], dtype=np.float32)
    ok_c, rv_c, tv_c = cv2.solvePnP(obj_c, img_c, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok_c:
        break
    pr_c, _ = cv2.projectPoints(obj_c, rv_c, tv_c, K, dist)
    pr_c = pr_c.reshape(-1, 2)
    er_c = np.linalg.norm(pr_c - img_c, axis=1)
    med = float(np.median(er_c))
    R_c, _ = cv2.Rodrigues(rv_c)
    T_c = np.eye(4); T_c[:3,:3] = R_c; T_c[:3,3] = tv_c.flatten()
    worst_i = int(np.argmax(er_c))
    print(f"  n={len(cur_data):2d}  median={med:7.1f}px  max={er_c[worst_i]:.1f}px  "
          f"a={T_c[2,3]:.4f}  drop={cur_data[worst_i][0]}[{cur_data[worst_i][1]}]")
    if med < 15.0:
        print(f"  => CONVERGED at n={len(cur_data)}, median={med:.1f}px")
        break
    cur_data.pop(worst_i)

# ---------------------------------------------------------------------------
# K sweep (find best fx if default K used)
# ---------------------------------------------------------------------------
if len(sys.argv) < 5:
    print(f"\nK sweep (cx={cx:.1f}, cy={cy:.1f}, fx=fy, range 350-750):")
    best_fx, best_err = 0, 1e9
    for fx_try in range(350, 750, 10):
        K_try = np.array([[fx_try, 0, cx], [0, fx_try, cy], [0, 0, 1.0]], dtype=np.float64)
        ok2, rv2, tv2 = cv2.solvePnP(obj, img, K_try, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok2:
            continue
        proj2, _ = cv2.projectPoints(obj, rv2, tv2, K_try, dist)
        err2 = float(np.median(np.linalg.norm(proj2.reshape(-1,2) - img, axis=1)))
        print(f"  fx={fx_try:5d}  median={err2:7.1f}px")
        if err2 < best_err:
            best_err, best_fx = err2, fx_try
    print(f"\n  Best: fx={best_fx}  median={best_err:.1f}px")
