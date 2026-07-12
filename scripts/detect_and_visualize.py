#!/usr/bin/env python3
"""Phase 1: Detect calibration target in RGB + HSI images and visualize.

For each matched pair of RGB + FLIR images:
  1. Detect ArUco markers (IDs 0-3) in RGB, estimate camera pose via solvePnP
  2. Detect 6 edge transitions in FLIR (HSI) spatial profile
  3. Save overlay visualization (markers + edges)

Usage:
    python scripts/detect_and_visualize.py \
        --rgb-dir <path_to_calibration_images> \
        --target-config assets/calibration_targets/combined_target.yaml \
        --calib-json <path_to_calibration.json> \
        --out output/phase1
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

try:
    import matplotlib
    # Try multiple interactive backends until one works
    for backend in ['Qt5Agg', 'TkAgg', 'macosx', 'GTK3Agg', 'wxAgg']:
        try:
            matplotlib.use(backend)
            import matplotlib.pyplot as plt
            # test if it works
            fig = plt.figure()
            plt.close(fig)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ═══════════════════════════════════════════════════════════════════════
# ArUco detection + pose estimation
# ═══════════════════════════════════════════════════════════════════════
def load_target_config(path: Path) -> dict:
    """Load combined target YAML config."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_rgb_intrinsics(calib_json: Path) -> tuple[NDArray, NDArray]:
    """Load RGB camera matrix and distortion from calibration.json."""
    with open(calib_json, "r") as f:
        data = json.load(f)
    rgb = data["rgb"]
    K = np.array(rgb["intrinsic_matrix_3x3"], dtype=np.float64)
    dist = np.array(rgb["distortion_coeffs"], dtype=np.float64)
    return K, dist


def build_aruco_board(target_cfg: dict):
    """Build a cv2.aruco.Board from target config marker corners.

    The corners in the YAML are in mm, pattern-frame (origin at L1 left end).
    We convert to meters for solvePnP (standard OpenCV convention).
    """
    aruco_cfg = target_cfg["aruco"]
    marker_ids = aruco_cfg["marker_ids"]
    marker_size_mm = aruco_cfg["marker_size_mm"]

    dict_name = aruco_cfg["dictionary"]
    dict_id = getattr(cv2.aruco, dict_name)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    # Build objPoints: list of (4,3) arrays, one per marker
    # Each array has the 4 corners in 3D (Z=0, in meters)
    all_obj_points = []
    all_ids = []
    for mid in marker_ids:
        corners_mm = target_cfg["aruco"]["markers"][mid]["corners_mm"]
        # Convert mm -> meters and add Z=0
        corners_3d = np.array(
            [[c[0] / 1000.0, c[1] / 1000.0, 0.0] for c in corners_mm],
            dtype=np.float32,
        )
        all_obj_points.append(corners_3d)
        all_ids.append(mid)

    board = cv2.aruco.Board(
        all_obj_points,
        dictionary,
        np.array(all_ids, dtype=np.int32),
    )
    return board, dictionary


def detect_aruco_and_pose(
    image: NDArray,
    board,
    dictionary,
    camera_matrix: NDArray,
    dist_coeffs: NDArray,
) -> tuple:
    """Detect ArUco markers and estimate board pose.

    Returns (rvec, tvec, detected_corners, detected_ids) or Nones if failed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 2:
        return None, None, corners, ids

    # Estimate board pose
    obj_pts, img_pts = board.matchImagePoints(corners, ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None, None, corners, ids

    success, rvec, tvec = cv2.solvePnP(
        obj_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None, None, corners, ids

    return rvec, tvec, corners, ids


def draw_rgb_overlay(
    image: NDArray,
    rvec: NDArray | None,
    tvec: NDArray | None,
    corners,
    ids,
    camera_matrix: NDArray,
    dist_coeffs: NDArray,
    target_cfg: dict,
) -> NDArray:
    """Draw detected markers and pattern outline on RGB image."""
    vis = image.copy()

    # Draw detected markers
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    if rvec is not None and tvec is not None:
        # Draw coordinate axes (50mm = 0.05m)
        cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

        # Draw triangle outlines
        hp = target_cfg["li_wen_qiu"]["hp_mm"]
        wp = target_cfg["li_wen_qiu"]["wp_mm"]

        # Triangle vertices in pattern frame (meters, Z=0)
        for i in range(3):
            y_top = i * hp / 1000.0
            y_bot = (i + 1) * hp / 1000.0
            pts_3d = np.array([
                [0, y_top, 0],
                [wp / 1000.0, y_top, 0],
                [0, y_bot, 0],
            ], dtype=np.float64)

            pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, dist_coeffs)
            pts_2d = pts_2d.reshape(-1, 2).astype(np.int32)

            # Draw triangle edges
            cv2.line(vis, tuple(pts_2d[0]), tuple(pts_2d[1]), (0, 255, 0), 2)  # L_odd (horizontal)
            cv2.line(vis, tuple(pts_2d[1]), tuple(pts_2d[2]), (0, 0, 255), 2)  # L_even (diagonal)
            cv2.line(vis, tuple(pts_2d[2]), tuple(pts_2d[0]), (255, 0, 0), 1)  # left edge

        # Label the feature lines
        for i, name in enumerate(["L1", "L3", "L5"]):
            y = i * hp / 1000.0
            pt_3d = np.array([[wp / 2000.0, y, 0]], dtype=np.float64)
            pt_2d, _ = cv2.projectPoints(pt_3d, rvec, tvec, camera_matrix, dist_coeffs)
            pt = tuple(pt_2d.reshape(-1, 2)[0].astype(int))
            cv2.putText(vis, name, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return vis


# ═══════════════════════════════════════════════════════════════════════
# HSI/FLIR edge detection
# ═══════════════════════════════════════════════════════════════════════
def detect_hsi_edges_interactive(
    flir_img: NDArray,
    num_edges: int = 6,
    band_range: tuple[int, int] | None = None,
    sigma: float = 3.0,
) -> tuple[NDArray, NDArray, tuple[int, int]]:
    """Semi-automated edge detection with interactive user selection.

    Args:
        flir_img: Mono8 image (H x W).
        num_edges: Expected number of edges to select.
        band_range: Optional spectral rows to average.
        sigma: Gaussian smoothing sigma.

    Returns:
        (edge_positions, spatial_profile, band_range)
    """
    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required for interactive edge selection.")
        print("Please install it: pip install matplotlib PyQt5")
        sys.exit(1)
        
    h, w = flir_img.shape[:2]
    if len(flir_img.shape) == 3:
        flir_img = cv2.cvtColor(flir_img, cv2.COLOR_BGR2GRAY)

    img_f = flir_img.astype(np.float64)

    if band_range is None:
        # Auto-select brightest band
        row_means = np.mean(img_f, axis=1)
        from scipy.ndimage import gaussian_filter1d
        row_means_smooth = gaussian_filter1d(row_means, sigma=5)
        peak_row = int(np.argmax(row_means_smooth))
        band_half = 30
        band_range = (max(0, peak_row - band_half), min(h, peak_row + band_half))

    # Average across spectral rows for SNR
    profile = np.mean(img_f[band_range[0]:band_range[1], :], axis=0)
    
    # Smooth
    from scipy.ndimage import gaussian_filter1d
    profile_smooth = gaussian_filter1d(profile, sigma=sigma)

    # Enhance image for display
    img_norm = cv2.normalize(flir_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create interactive plot
    print(f"\n[INTERACTIVE] Drag the 6 dashed lines to the transition points. Close window when done.")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.imshow(img_norm, cmap='gray', aspect='auto')
    ax1.axhline(band_range[0], color='r', linestyle='--', alpha=0.5)
    ax1.axhline(band_range[1], color='r', linestyle='--', alpha=0.5)
    ax1.set_title("HSI Image (Drag vertical lines to edges)")
    ax1.set_ylabel("Spectral Y")

    ax2.plot(profile_smooth, 'k-', label='Averaged Profile')
    ax2.set_title("Spatial Profile (Averaged over red bands)")
    ax2.set_xlabel("Spatial X")
    ax2.set_ylabel("Intensity")
    
    # Initialize 6 vertical lines spaced evenly
    x_init = np.linspace(w * 0.2, w * 0.8, num_edges)
    lines_ax1 = []
    lines_ax2 = []
    labels_ax1 = []
    
    # We explicitly assign L1..L6 to these 6 starting lines
    # From combined_target.yaml:
    # L1: Horizontal
    # L2: Diagonal
    # L3: Horizontal
    # L4: Diagonal
    # L5: Horizontal
    # L6: Diagonal
    line_names = ["L1 (H)", "L2 (D)", "L3 (H)", "L4 (D)", "L5 (H)", "L6 (D)"]
    
    for i, x in enumerate(x_init):
        l1 = ax1.axvline(x, color='r', linestyle='--')
        l2 = ax2.axvline(x, color='r', linestyle='--')
        
        # Add a text label near the top of ax1
        txt = ax1.text(x, band_range[0] - 10, line_names[i], color='yellow', 
                       fontweight='bold', ha='center', va='bottom',
                       bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
        
        lines_ax1.append(l1)
        lines_ax2.append(l2)
        labels_ax1.append(txt)
        
    print(f"\n[INTERACTIVE] Drag the 6 dashed lines to the transition points.")
    print(f"CRITICAL: Make sure the label (e.g., L1, L2) perfectly matches the physical line!")
    print(f"H = Horizontal line, D = Diagonal line.")
    print(f"Close window when done.")
    
    # Simple drag logic using matplotlib event handling
    class DraggableLines:
        def __init__(self, lines1, lines2, labels1, profile_smooth):
            self.lines1 = lines1
            self.lines2 = lines2
            self.labels1 = labels1
            self.profile_smooth = profile_smooth
            self.active_line_idx = None
            
            # Precompute gradient for snapping
            self.gradient = np.abs(np.gradient(self.profile_smooth))
            
            fig.canvas.mpl_connect('button_press_event', self.on_press)
            fig.canvas.mpl_connect('button_release_event', self.on_release)
            fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
            
        def on_press(self, event):
            if event.inaxes not in [ax1, ax2] or event.button != 1:
                return
            # Find closest line
            x_coords = [l.get_xdata()[0] for l in self.lines1]
            distances = [abs(x - event.xdata) for x in x_coords]
            min_idx = np.argmin(distances)
            # Only grab if within 20 pixels
            if distances[min_idx] < max(20, w * 0.05):
                self.active_line_idx = min_idx
                
        def on_release(self, event):
            if self.active_line_idx is None:
                return
            
            # Snap to nearest local maximum in the gradient
            current_x = int(round(self.lines1[self.active_line_idx].get_xdata()[0]))
            
            # Search window: ±15 pixels
            search_radius = 15
            start_x = max(0, current_x - search_radius)
            end_x = min(len(self.gradient), current_x + search_radius + 1)
            
            if end_x > start_x:
                window_grad = self.gradient[start_x:end_x]
                peak_offset = np.argmax(window_grad)
                snapped_x = start_x + peak_offset
                
                # Update line position to the snapped peak
                self.lines1[self.active_line_idx].set_xdata([snapped_x, snapped_x])
                self.lines2[self.active_line_idx].set_xdata([snapped_x, snapped_x])
                self.labels1[self.active_line_idx].set_position((snapped_x, band_range[0] - 10))
                fig.canvas.draw_idle()
                
            self.active_line_idx = None
            
        def on_motion(self, event):
            if self.active_line_idx is None or event.inaxes not in [ax1, ax2]:
                return
            x = event.xdata
            self.lines1[self.active_line_idx].set_xdata([x, x])
            self.lines2[self.active_line_idx].set_xdata([x, x])
            self.labels1[self.active_line_idx].set_position((x, band_range[0] - 10))
            fig.canvas.draw_idle()

    drag_handler = DraggableLines(lines_ax1, lines_ax2, labels_ax1, profile_smooth)
    plt.tight_layout()
    plt.show(block=True)
    
    # Extract final X coordinates in their strictly preserved L1..L6 order. 
    # Do NOT sort them left-to-right, because we need the array index to correspond directly to [L1, L2, L3, L4, L5, L6]
    top_edges = np.array([l.get_xdata()[0] for l in drag_handler.lines1])
    
    return top_edges, profile_smooth, band_range


def draw_hsi_overlay(
    flir_img: NDArray,
    edge_positions: NDArray,
    band_range: tuple[int, int] | None = None,
) -> NDArray:
    """Draw detected edges on FLIR image (contrast-enhanced)."""
    # Enhance contrast for visualization
    if len(flir_img.shape) == 2:
        img_norm = cv2.normalize(flir_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vis = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    else:
        vis = flir_img.copy()

    h, w = vis.shape[:2]

    # Draw detected edge positions as vertical lines
    colors = [(0, 255, 0), (0, 255, 255), (0, 255, 0),
              (0, 255, 255), (0, 255, 0), (0, 255, 255)]
    labels = ["v1", "v2", "v3", "v4", "v5", "v6"]
    for i, x in enumerate(edge_positions):
        x = int(x)
        col = colors[i % len(colors)]
        cv2.line(vis, (x, 0), (x, h), col, 1)
        label = labels[i] if i < len(labels) else f"v{i+1}"
        cv2.putText(vis, f"{label}={x}", (x + 3, 20 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    # Draw band range
    if band_range is not None:
        cv2.line(vis, (0, band_range[0]), (w, band_range[0]), (255, 0, 0), 1)
        cv2.line(vis, (0, band_range[1]), (w, band_range[1]), (255, 0, 0), 1)

    return vis


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description="Phase 1: Detect calibration target and visualize.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--rgb-dir", type=Path, required=True,
                    help="Directory containing rgb_*.png and flir_*.png files")
    ap.add_argument("--target-config", type=Path,
                    default=Path("assets/calibration_targets/combined_target.yaml"))
    ap.add_argument("--calib-json", type=Path, required=True,
                    help="Path to OAK-D calibration.json")
    ap.add_argument("--out", type=Path, default=Path("output/phase1"))
    ap.add_argument("--hsi-scale", type=float, default=1.0,
                    help="Scale factor to apply to FLIR/HSI images before processing")
    ap.add_argument("--show", action="store_true", help="Display images interactively")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load config and intrinsics
    target_cfg = load_target_config(args.target_config)
    K, dist = load_rgb_intrinsics(args.calib_json)
    board, dictionary = build_aruco_board(target_cfg)

    print(f"Camera matrix:\n{K}")
    print(f"Distortion: {dist[:5]}...")

    # Find matched image pairs
    rgb_files = sorted(glob.glob(str(args.rgb_dir / "rgb_*.png")))
    flir_files = sorted(glob.glob(str(args.rgb_dir / "flir_*.png")))

    if len(rgb_files) != len(flir_files):
        print(f"WARNING: {len(rgb_files)} RGB files vs {len(flir_files)} FLIR files")

    # Match by timestamp suffix
    pairs = []
    for rgb_path in rgb_files:
        ts = Path(rgb_path).stem.replace("rgb_", "")
        flir_path = args.rgb_dir / f"flir_{ts}.png"
        if flir_path.exists():
            pairs.append((Path(rgb_path), flir_path))
        else:
            print(f"  No FLIR match for {Path(rgb_path).name}")

    print(f"\nFound {len(pairs)} matched RGB+FLIR pairs\n")

    # Also save detections for Phase 2
    all_detections = []

    for idx, (rgb_path, flir_path) in enumerate(pairs):
        ts = rgb_path.stem.replace("rgb_", "")
        print(f"--- View {idx+1}/{len(pairs)}: {ts} ---")

        # Load images
        rgb_img = cv2.imread(str(rgb_path))
        flir_img = cv2.imread(str(flir_path), cv2.IMREAD_GRAYSCALE)

        if rgb_img is None:
            print(f"  ERROR: cannot read {rgb_path}")
            continue
        if flir_img is None:
            print(f"  ERROR: cannot read {flir_path}")
            continue
            
        # Scale K and zero distortion for this image
        h_img, w_img = rgb_img.shape[:2]
        scale_x = w_img / 1920.0
        scale_y = h_img / 1080.0
        
        K_scaled = K.copy()
        K_scaled[0, 0] *= scale_x  # fx
        K_scaled[1, 1] *= scale_y  # fy
        K_scaled[0, 2] *= scale_x  # cx
        K_scaled[1, 2] *= scale_y  # cy
        
        # OAK-D RGB images are already rectified by ISP, so dist is 0
        dist_zero = np.zeros(5)

        if args.hsi_scale != 1.0:
            new_w = int(flir_img.shape[1] * args.hsi_scale)
            new_h = int(flir_img.shape[0] * args.hsi_scale)
            flir_img = cv2.resize(flir_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # --- RGB: ArUco detection + pose ---
        rvec, tvec, corners, ids = detect_aruco_and_pose(
            rgb_img, board, dictionary, K_scaled, dist_zero
        )

        if rvec is not None:
            R, _ = cv2.Rodrigues(rvec)
            print(f"  RGB: detected {len(ids)} markers, pose OK")
            print(f"    tvec = [{tvec.ravel()[0]:.4f}, {tvec.ravel()[1]:.4f}, {tvec.ravel()[2]:.4f}] m")
        else:
            n_det = len(ids) if ids is not None else 0
            print(f"  RGB: detected {n_det} markers, pose FAILED")
            R = None

        rgb_vis = draw_rgb_overlay(rgb_img, rvec, tvec, corners, ids, K_scaled, dist_zero, target_cfg)

        # --- FLIR: interactive edge detection ---
        edges, profile, band_range = detect_hsi_edges_interactive(flir_img, num_edges=6)
        print(f"  HSI: selected 6 edges at x = {np.round(edges, 1).tolist()}")

        flir_vis = draw_hsi_overlay(flir_img, edges, band_range)

        # --- Combine side-by-side ---
        # Resize to same height
        h_target = 720
        rgb_resized = cv2.resize(rgb_vis, (int(rgb_vis.shape[1] * h_target / rgb_vis.shape[0]), h_target))
        flir_resized = cv2.resize(flir_vis, (int(flir_vis.shape[1] * h_target / flir_vis.shape[0]), h_target))
        combined = np.hstack([rgb_resized, flir_resized])

        # Add title
        cv2.putText(combined, f"View {idx+1}: {ts}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save
        out_path = args.out / f"view_{idx+1}_{ts}.png"
        cv2.imwrite(str(out_path), combined)
        print(f"  Saved -> {out_path}")

        # Also save a profile plot
        _save_profile_plot(profile, edges, args.out / f"profile_{idx+1}_{ts}.png")

        # Store detection for Phase 2
        detection = {
            "view_id": f"view_{idx+1}",
            "timestamp": ts,
            "rgb_file": str(rgb_path),
            "flir_file": str(flir_path),
            "v_observations": edges.tolist(),
            "pose_found": rvec is not None,
        }
        if rvec is not None:
            R_mat, _ = cv2.Rodrigues(rvec)
            detection["R"] = R_mat.tolist()
            detection["T"] = tvec.ravel().tolist()
        all_detections.append(detection)

        if args.show:
            cv2.imshow("Detection", combined)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    # Save detections JSON for Phase 2
    det_path = args.out / "detections.json"
    with open(det_path, "w") as f:
        json.dump(all_detections, f, indent=2)
    print(f"\nDetections saved -> {det_path}")

    if args.show:
        cv2.destroyAllWindows()



def _save_profile_plot(profile: NDArray, edges: NDArray, path: Path):
    """Save a simple profile + edges plot using matplotlib if available."""
    try:
        if not HAS_MATPLOTLIB:
            return
            
        import matplotlib.pyplot as plt

        # Create a new figure without displaying it
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

        ax1.plot(profile, 'k-', linewidth=0.8)
        for x in edges:
            ax1.axvline(x, color='r', linestyle='--', alpha=0.7, linewidth=0.8)
        ax1.set_ylabel("Intensity (averaged)")
        ax1.set_title("Spatial profile (spectral-averaged)")

        gradient = np.abs(np.gradient(profile))
        ax2.plot(gradient, 'b-', linewidth=0.8)
        for x in edges:
            ax2.axvline(x, color='r', linestyle='--', alpha=0.7, linewidth=0.8)
        ax2.set_ylabel("|Gradient|")
        ax2.set_xlabel("Pixel position (spatial)")

        plt.tight_layout()
        plt.savefig(str(path), dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"  Warning: Could not save profile plot: {e}")
    except Exception as e:
        print(f"  Warning: Could not save profile plot: {e}")


if __name__ == "__main__":
    main()
