#!/usr/bin/env python3
"""
Live Slit Projection

Connects to the OAK-D camera, detects the Li-Wen-Qiu calibration target,
and uniquely projects the HSI slit line onto the live RGB feed using the
optimized HSI->RGB Extrinsics result.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import cv2
import depthai as dai
import numpy as np
import yaml

from hsi_rgbd_calib.boards.li_wen_qiu_pattern import LiWenQiuPattern
from hsi_rgbd_calib.cal_method.interface import Intrinsics

# Import the core detection functions from our existing script
import detect_and_visualize as dv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_oakd_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()
    
    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutVideo = pipeline.create(dai.node.XLinkOut)
    xoutVideo.setStreamName("rgb")

    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_C)  # Assuming center RGB camera
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    camRgb.setFps(30)
    camRgb.setInterleaved(False)

    # Linking
    camRgb.video.link(xoutVideo.input)
    
    return pipeline

def load_intrinsics(calib_json: Path, rgb_size: tuple[int, int]) -> Intrinsics:
    with open(calib_json, 'r') as f:
        cal_data = json.load(f)
    
    # Usually "cam_c" is the RGB camera in standard OAK-D
    rgb_cal = None
    if "cameraData" in cal_data and "cam_c" in cal_data["cameraData"]:
        rgb_cal = cal_data["cameraData"]["cam_c"]
    else:
        raise ValueError("Could not find 'cam_c' in calibration JSON")
        
    K_raw = np.array(rgb_cal["intrinsicMatrix"])
    dist_raw = np.array(rgb_cal["distortionCoeff"])
    cal_size = tuple(rgb_cal["resolution"])
    
    K_scaled = Intrinsics.scale_intrinsics(K_raw, cal_size, rgb_size)
    
    return Intrinsics(
        camera_matrix=K_scaled,
        dist_coeffs=dist_raw,
        width=rgb_size[0],
        height=rgb_size[1]
    )

def main():
    parser = argparse.ArgumentParser(description="Live OAK-D HSI Slit Projection")
    parser.add_argument("--calib", type=Path, default="C:/Users/albeladi/OneDrive - KFUPM/Documents/GitHub/oak-record-replay/dataset_20260302_165711/calibration.json", help="Path to OAK-D calibration JSON")
    parser.add_argument("--target", type=Path, default="assets/calibration_targets/combined_target.yaml", help="Path to target YAML")
    parser.add_argument("--extrinsics", type=Path, default="output/phase2/hsi_rgb_extrinsics.json", help="Path to optimized Extrinsics")
    args = parser.parse_args()

    # 1. Load Extrinsics (HSI -> RGB)
    if not args.extrinsics.exists():
        logging.error(f"Extrinsics file not found: {args.extrinsics}")
        sys.exit(1)
    with open(args.extrinsics, 'r') as f:
        ext_data = json.load(f)
    R_h2c = np.array(ext_data["R_h2c"])
    t_h2c = np.array(ext_data["t_h2c"]).reshape((3, 1))
    logging.info("Loaded HSI->RGB Extrinsics.")

    # 2. Load ArUco Target config
    with open(args.target, 'r') as f:
        target_cfg = yaml.safe_load(f)
    board, dictionary = dv.setup_aruco(target_cfg)

    # 3. Setup OAK-D Pipeline
    pipeline = get_oakd_pipeline()

    # 4. Connect to device and start
    with dai.Device(pipeline) as device:
        video_q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        logging.info("Connected to OAK-D. Press 'q' to quit.")
        
        # Load Intrinsics on first frame using actual size
        intrinsics = None
        
        while True:
            # Get RGB frame
            in_video = video_q.get()
            frame = in_video.getCvFrame()
            
            h, w = frame.shape[:2]
            if intrinsics is None:
                intrinsics = load_intrinsics(args.calib, (w, h))
                
            K = intrinsics.camera_matrix
            dist = intrinsics.dist_coeffs
            
            # Detect ArUco and estimate pose
            rvec, tvec, _, _ = dv.detect_aruco_and_pose(frame, board, dictionary, K, dist)
            
            # Draw standard overlay (axes and triangle wireframe)
            vis = dv.draw_rgb_overlay(frame, rvec, tvec, None, None, K, dist, target_cfg)
            
            # If board detected, intersect HSI slit and project
            if rvec is not None and tvec is not None:
                R_board, _ = cv2.Rodrigues(rvec)
                T_board = tvec.reshape((3, 1))
                
                # Slit plane defined in HSI frame: X_h = 0
                n_h = np.array([[1.0], [0.0], [0.0]])
                
                # Transform slit plane to RGB frame: n_c^T * P_rgb + d_c = 0
                n_c = R_h2c @ n_h
                d_c = -(n_c.T @ t_h2c)[0, 0]
                
                # Intersect with Board Plane (Z_b = 0 in Board frame)
                # n_c^T * (R_board * [X_b, Y_b, 0]^T + T_board) + d_c = 0
                A = (n_c.T @ R_board[:, 0:1])[0, 0]
                B = (n_c.T @ R_board[:, 1:2])[0, 0]
                C = (n_c.T @ T_board)[0, 0] + d_c
                
                pts_3d_rgb = []
                if abs(B) > 1e-4:
                    # Sample at left (X_b = 0) and right (X_b = 0.24m)
                    for x_b in [0.0, 0.24]:
                        y_b = -(A * x_b + C) / B
                        P_b = np.array([[x_b], [y_b], [0.0]])
                        P_c = R_board @ P_b + T_board
                        pts_3d_rgb.append(P_c)
                else:
                    # Vertical line
                    x_b = -C / A
                    for y_b in [0.0, 0.18]:
                        P_b = np.array([[x_b], [y_b], [0.0]])
                        P_c = R_board @ P_b + T_board
                        pts_3d_rgb.append(P_c)
                        
                # Project 3D points back to 2D
                pts_2d = []
                for P_c in pts_3d_rgb:
                    p_img = K @ P_c
                    if p_img[2, 0] > 0: # Ensure point is in front of camera
                        u_px = int(round(p_img[0, 0] / p_img[2, 0]))
                        v_px = int(round(p_img[1, 0] / p_img[2, 0]))
                        pts_2d.append((u_px, v_px))
                
                # Draw thick yellow line representing the HSI Slit intersect
                if len(pts_2d) == 2:
                    cv2.line(vis, pts_2d[0], pts_2d[1], (0, 255, 255), 3)
                    
                    # Add descriptive text near the middle of the line
                    mid_x = (pts_2d[0][0] + pts_2d[1][0]) // 2
                    mid_y = (pts_2d[0][1] + pts_2d[1][1]) // 2
                    cv2.putText(vis, "HSI Slit Match", (mid_x, mid_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display
            cv2.imshow("Live HSI Slit Projection", vis)
            
            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
