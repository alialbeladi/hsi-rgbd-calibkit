#!/usr/bin/env python3
"""
Standalone HSI-RGB Calibration Debug Script
============================================
Zero dependencies on the hsi_rgbd_calib package.
Everything is explicit and self-contained so you can audit every step.

COORDINATE CONVENTIONS (explicitly documented):
  RGB Frame  (OpenCV, right-handed):
    X_c = Right
    Y_c = Down
    Z_c = Forward (optical axis)

  HSI Frame  (physical rig, right-handed):
    X_h = Up  (parallel to -Y_c, since HSI is rotated 90 deg around Z with respect to RGB)
    Y_h = Right (parallel to X_c, this is the slit direction)
    Z_h = Forward (optical axis, parallel to Z_c)

  Therefore the nominal R_rgb2hsi (such that P_h = R*P_c + T) is:
    R_rgb2hsi = [[  0, -1,  0 ],    (X_h = -Y_c)
                 [  1,  0,  0 ],    (Y_h = +X_c)
                 [  0,  0,  1 ]]    (Z_h = +Z_c)

  And if the HSI camera is ~6 cm ABOVE the RGB camera in the real world,
  its origin is at P_c = [0, -0.06, 0] in the RGB frame.
  From P_h = R_rgb2hsi * P_c + T_rgb2hsi, we get T = -R * [0, -0.06, 0]:
    T_rgb2hsi = [-0.06, 0, 0]^T  (HSI origin is at -X_h = down in HSI frame = 6cm above RGB)

PATTERN (Li-Wen-Qiu target on Z=0 plane in BOARD frame, coordinates in METERS):
  Feature lines L1..L6: each described as a*X + b*Y + c = 0 (with c scaled to meters)
  L1: Horizontal (constant Y), L2: Diagonal, L3: Horizontal, L4: Diagonal, L5: Horizontal, L6: Diagonal

OBSERVATION MODEL for the HSI line-scan camera:
  Given a 3D point P_h = [X_h, Y_h, Z_h] in the HSI frame:
    v_predicted = f * (Y_h / Z_h) * (1 + k * (Y_h/Z_h)^2) + v0

OPTIMIZER solves for: [f, v0, k, rvec(3), T(3)]  (9 parameters total)
  The rotation from RGB to HSI is parameterized as a Rodrigues rotation vector.

For each view j:
  1. P_h = R_rgb2hsi * (R_board_j * P_board + T_board_j) + T_rgb2hsi
     = R0 * P_board + T0   where R0 = R_rgb2hsi @ R_board_j
                                  T0 = R_rgb2hsi @ T_board_j + T_rgb2hsi
  2. Scan line in board plane: the HSI slit is X_h=0 in HSI frame
     -> 0 = R0[0,:] dot P_board + T0[0]
     -> scan line: a*X + b*Y + c = 0 in board coords
  3. Intersect scan line with each feature line L_i -> intersection point P_i
  4. Compute P_h = R0 @ P_i_3d + T0
  5. v_pred = f * (P_h[1]/P_h[2]) * (1 + k*(P_h[1]/P_h[2])**2) + v0
  6. Cost: sum of (v_pred - v_obs)^2

NOTE ON WHAT R IS vs. WHAT WE PRINT:
  The optimizer solves for R_rgb2hsi (i.e. P_{hsi} = R * P_{rgb} + T).
  At the end we INVERT it to get R_h2c (P_{rgb} = R_h2c * P_{hsi} + t_h2c)
  since that is the more common "HSI pose in RGB frame" convention.
    R_h2c = R_rgb2hsi.T
    t_h2c = -R_rgb2hsi.T @ T_rgb2hsi
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# ════════════════════════════════════════════════════════════════════════════
# Paths - edit these if needed
# ════════════════════════════════════════════════════════════════════════════
DETECTIONS_JSON = Path("output/phase1/detections.json")
CALIB_JSON      = Path("C:/Users/albeladi/OneDrive - KFUPM/Documents/GitHub/oak-record-replay/dataset_20260302_165711/calibration.json")
TARGET_YAML     = Path("assets/calibration_targets/combined_target.yaml")
OUTPUT_JSON     = Path("output/phase2/hsi_calib_debug.json")

# ════════════════════════════════════════════════════════════════════════════
# Physical initialization
# ════════════════════════════════════════════════════════════════════════════
HSI_ABOVE_RGB_M = 0.06      # 6 cm

# R_rgb2hsi: transforms points FROM rgb frame TO hsi frame
# Physical logic: RGB-X goes to HSI-Y, RGB-Y goes to HSI-(-X)
R_INIT_rgb2hsi = np.array([
    [ 0., -1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0.,  1.],
], dtype=np.float64)

# T_rgb2hsi: origin of RGB camera expressed in HSI frame
# HSI is 6cm above RGB -> RGB is 6cm below HSI -> T = -R * [0, -0.06, 0]
T_INIT_rgb2hsi = -(R_INIT_rgb2hsi @ np.array([0., -HSI_ABOVE_RGB_M, 0.]))

F_INIT  =  800.0   # HSI focal length initial guess (pixels)
V0_INIT =  640.0   # HSI principal point initial guess (pixels)
K_INIT  =    0.0   # radial distortion initial guess


# ════════════════════════════════════════════════════════════════════════════
# Load data
# ════════════════════════════════════════════════════════════════════════════

def load_intrinsics(calib_json: Path, img_w: int, img_h: int):
    with open(calib_json) as f:
        data = json.load(f)
    cam = data["rgb"]
    K_raw = np.array(cam["intrinsic_matrix_3x3"], dtype=np.float64)
    dist  = None  # images are already undistorted, pass None to solvePnP
    cal_w = cam["resolution_width"]
    cal_h = cam["resolution_height"]
    
    sx = img_w / cal_w
    sy = img_h / cal_h
    K_scaled = K_raw.copy()
    K_scaled[0, 0] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[0, 2] *= sx
    K_scaled[1, 2] *= sy
    return K_scaled, dist


def load_feature_lines(target_yaml: Path):
    """Load L1..L6 line equations scaled to meters (from mm in YAML)."""
    with open(target_yaml) as f:
        cfg = yaml.safe_load(f)
    lines = []
    for i in range(1, 7):
        eq_mm = cfg["li_wen_qiu"]["feature_lines"][f"L{i}"]["eq"]
        a, b, c_mm = eq_mm
        # Scale a,b, and c (keep everything in meters): (a/1000)*x_m + (b/1000)*y_m + (c/1000000) = 0
        lines.append((float(a)/1000.0, float(b)/1000.0, float(c_mm)/1000000.0))
    return lines


def load_views(detections_json: Path, calib_json: Path, target_yaml: Path):
    """Return list of dicts: {R_board, T_board, v_obs} for each view."""
    with open(detections_json) as f:
        detections = json.load(f)
    
    # Use first detected image to infer resolution
    first_rgb = next(d["rgb_file"] for d in detections if d.get("pose_found"))
    img = cv2.imread(first_rgb)
    h, w = img.shape[:2]
    
    K, dist = load_intrinsics(calib_json, w, h)
    
    with open(target_yaml) as f:
        cfg = yaml.safe_load(f)
    
    # Build ArUco board
    aruco_cfg = cfg["aruco"]
    dict_id   = getattr(cv2.aruco, aruco_cfg["dictionary"])
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    obj_pts_list = []
    ids_list     = []
    for mid in aruco_cfg["marker_ids"]:
        corners_mm = aruco_cfg["markers"][mid]["corners_mm"]
        corners_3d = np.array([[c[0]/1000., c[1]/1000., 0.] for c in corners_mm], dtype=np.float32)
        obj_pts_list.append(corners_3d)
        ids_list.append(mid)
    board = cv2.aruco.Board(obj_pts_list, dictionary, np.array(ids_list, dtype=np.int32))
    
    views = []
    for det in detections:
        if not det.get("pose_found"):
            continue
        v_obs = np.array(det["v_observations"], dtype=np.float64)
        R_board = np.array(det["R"], dtype=np.float64)
        T_board = np.array(det["T"], dtype=np.float64).ravel()
        views.append({"R_board": R_board, "T_board": T_board, "v_obs": v_obs,
                      "view_id": det["view_id"]})
    return views, K


# ════════════════════════════════════════════════════════════════════════════
# Core math
# ════════════════════════════════════════════════════════════════════════════

def intersect_lines_2d(l1, l2):
    """Intersect two homogeneous 2D lines. Returns (x, y) or None."""
    x = np.cross(l1, l2)
    if abs(x[2]) < 1e-12:
        return None
    return (x[0] / x[2], x[1] / x[2])


def predict_v(R_rgb2hsi, T_rgb2hsi, R_board, T_board, feature_lines, f, v0, k):
    """
    Predict HSI pixel coordinates [v1..v6] for a single view.
    
    The HSI slit is the Y_h = 0 plane in HSI frame.
    We intersect that plane with the board plane Z_board = 0.
    The intersection line in board coords becomes our 'scan line'.
    Then we intersect that scan line with each feature line L_i.
    
    Note: The scan line here is in BOARD coordinates (2D: X_b, Y_b).
    Feature lines are also in board coordinates.
    """
    T_board = T_board.reshape(3, 1)
    
    # Combined transform: P_h = R0 @ P_board_3d + T0
    R0 = R_rgb2hsi @ R_board           # 3x3
    T0 = (R_rgb2hsi @ T_board).ravel() + T_rgb2hsi  # (3,)
    
    # HSI slit plane in board coords: X_h = 0  (all HSI rays lie in the Y_h-Z_h plane)
    # X_h = R0[0,0]*X_b + R0[0,1]*Y_b + T0[0] = 0
    # => scan line: R0[0,0]*X_b + R0[0,1]*Y_b + T0[0] = 0
    scan_line = np.array([R0[0, 0], R0[0, 1], T0[0]])
    
    v_preds = []
    for (a, b, c) in feature_lines:
        feat_line = np.array([a, b, c])
        pt = intersect_lines_2d(scan_line, feat_line)
        if pt is None:
            v_preds.append(np.nan)
            continue
        
        # 3D point on board (Z_b = 0)
        P_board = np.array([pt[0], pt[1], 0.0])
        
        # Transform to HSI frame
        P_h = R0 @ P_board + T0
        
        if abs(P_h[2]) < 1e-10:
            v_preds.append(np.nan)
            continue
        
        # Project onto HSI 1D sensor
        # Projection: v = f * (Y_h / Z_h) * (1 + k*(Y_h/Z_h)^2) + v0
        angle = P_h[1] / P_h[2]
        v = f * angle * (1.0 + k * angle**2) + v0
        v_preds.append(v)
    
    return np.array(v_preds)


def cost_fn(params, feature_lines, views):
    """Total reprojection cost."""
    f    = params[0]
    v0   = params[1]
    k    = params[2]
    rvec = params[3:6]
    T    = params[6:9]
    
    R = Rotation.from_rotvec(rvec).as_matrix()
    
    total = 0.0
    for view in views:
        v_pred = predict_v(R, T, view["R_board"], view["T_board"],
                           feature_lines, f, v0, k)
        for vp, vo in zip(v_pred, view["v_obs"]):
            if np.isnan(vp):
                total += 1e6
            else:
                total += (vp - vo)**2
    return total


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("HSI-RGB Calibration — Standalone Debug Script")
    print("=" * 60)
    
    feature_lines = load_feature_lines(TARGET_YAML)
    print(f"\nLoaded {len(feature_lines)} feature lines from {TARGET_YAML}")
    for i, (a, b, c) in enumerate(feature_lines, 1):
        print(f"  L{i}: {a:.4f}*X + {b:.4f}*Y + {c:.6f} = 0")
    
    views, K = load_views(DETECTIONS_JSON, CALIB_JSON, TARGET_YAML)
    print(f"\nLoaded {len(views)} views from {DETECTIONS_JSON}")
    for v in views:
        print(f"  {v['view_id']}: v_obs = {np.round(v['v_obs'], 1)}")
    
    # Initial guess
    t_init = T_INIT_rgb2hsi.copy()
    rvec_init = Rotation.from_matrix(R_INIT_rgb2hsi).as_rotvec()
    params0 = np.array([F_INIT, V0_INIT, K_INIT,
                        rvec_init[0], rvec_init[1], rvec_init[2],
                        t_init[0], t_init[1], t_init[2]])
    
    print(f"\nInitial parameters:")
    print(f"  f    = {params0[0]:.2f} px")
    print(f"  v0   = {params0[1]:.2f} px")
    print(f"  k    = {params0[2]:.6f}")
    print(f"  rvec = {params0[3:6]}")
    print(f"  T    = {params0[6:9]}")
    print(f"\nInitial R_rgb2hsi (should look like [[ 0,-1,0],[1,0,0],[0,0,1]]):")
    print(f"  {R_INIT_rgb2hsi}")
    print(f"\nT_rgb2hsi (origin of RGB in HSI frame, expect [-0.06, 0, 0]):")
    print(f"  {t_init}")
    
    cost0 = cost_fn(params0, feature_lines, views)
    print(f"\nInitial cost: {cost0:.4f}")
    
    print("\nRunning optimizer...")
    result = minimize(
        cost_fn,
        params0,
        args=(feature_lines, views),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-8, "fatol": 1e-8, "disp": False},
    )
    
    print(f"Optimizer status: {result.message}")
    print(f"Final cost: {result.fun:.6f}   (initial: {cost0:.4f})")
    
    # Unpack optimized params
    f_est    = result.x[0]
    v0_est   = result.x[1]
    k_est    = result.x[2]
    rvec_est = result.x[3:6]
    T_est    = result.x[6:9]
    R_rgb2hsi_est = Rotation.from_rotvec(rvec_est).as_matrix()
    
    # Invert to get the "HSI in RGB frame" convention
    R_h2c = R_rgb2hsi_est.T
    t_h2c = -R_rgb2hsi_est.T @ T_est
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nHSI Intrinsics:")
    print(f"  f  = {f_est:.4f} px")
    print(f"  v0 = {v0_est:.4f} px")
    print(f"  k  = {k_est:.6f}")
    
    print(f"\nR_rgb2hsi (optimizer output — P_hsi = R * P_rgb + T):")
    print(f"  {np.round(R_rgb2hsi_est, 5)}")
    
    print(f"\nT_rgb2hsi (origin of RGB camera in HSI frame):")
    print(f"  {np.round(T_est, 5)}")
    
    print(f"\nR_h2c = R_rgb2hsi.T (HSI -> RGB, i.e. P_rgb = R_h2c * P_hsi + t_h2c):")
    print(f"  {np.round(R_h2c, 5)}")
    
    print(f"\nt_h2c = -R_rgb2hsi.T @ T_rgb2hsi (origin of HSI in RGB frame):")
    print(f"  {np.round(t_h2c, 5)}")
    print(f"  [X: {t_h2c[0]*100:.1f} cm, Y: {t_h2c[1]*100:.1f} cm, Z: {t_h2c[2]*100:.1f} cm]")
    
    # Per-view residuals
    print("\nPer-view residuals [v_pred - v_obs] (px):")
    for view in views:
        v_pred = predict_v(R_rgb2hsi_est, T_est, view["R_board"], view["T_board"],
                           feature_lines, f_est, v0_est, k_est)
        residuals = v_pred - view["v_obs"]
        print(f"  {view['view_id']}: {np.round(residuals, 3)}")
    
    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as fp:
        json.dump({
            "R_rgb2hsi": R_rgb2hsi_est.tolist(),
            "T_rgb2hsi": T_est.tolist(),
            "R_h2c": R_h2c.tolist(),
            "t_h2c": t_h2c.tolist(),
            "hsi_intrinsics": {"f_px": f_est, "v0_px": v0_est, "k": k_est},
        }, fp, indent=2)
    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
