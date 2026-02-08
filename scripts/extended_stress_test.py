"""Extended stress test for Li-Wen-Qiu calibration.

Tests: 100 views x 5 noise levels

Reports:
1. Parameter estimation errors (init vs refined): rotation, translation, f, v0
2. Conditioning analysis: min_sep correlation with errors
3. Worst 5 views at sigma=0.2 and sigma=0.5
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation

from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern
from hsi_rgbd_calib.boards.geometry import intersect_lines_2d
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import (
    simulate_views, NoiseConfig, get_default_ground_truth
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    compute_scan_line_in_pattern,
    compute_transform_pattern_to_linescan,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.cross_ratio import (
    recover_pattern_points_from_observations,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.closed_form import closed_form_init
from hsi_rgbd_calib.cal_method.li_wen_qiu import LiWenQiuBackend
from hsi_rgbd_calib.cal_method.interface import CalibrationConfig


@dataclass
class ViewAnalysis:
    """Full analysis for a single view."""
    view_id: int
    noise_sigma: float
    
    # v observations
    v_obs: np.ndarray
    min_sep: float  # Conditioning proxy
    
    # Point recovery
    point_errors: List[float]
    max_point_error: float
    point_recovery_success: bool
    
    # Init parameters (closed-form)
    init_rot_error_deg: float = 0.0
    init_trans_error_mm: float = 0.0
    init_f_error_pct: float = 0.0
    init_v0_error_px: float = 0.0
    init_rmse: float = 0.0
    
    # Final parameters (refined)
    final_rot_error_deg: float = 0.0
    final_trans_error_mm: float = 0.0
    final_f_error_pct: float = 0.0
    final_v0_error_px: float = 0.0
    final_rmse: float = 0.0


def compute_rotation_error(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Compute rotation error in degrees."""
    R_err = R_est @ R_true.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    return np.degrees(angle)


def compute_translation_error(T_est: np.ndarray, T_true: np.ndarray) -> float:
    """Compute translation error in mm."""
    return np.linalg.norm(T_est - T_true) * 1000


def compute_min_sep(v: np.ndarray) -> float:
    """Compute minimum separation between any two v values."""
    min_sep = float('inf')
    for i in range(len(v)):
        for j in range(i+1, len(v)):
            sep = abs(v[i] - v[j])
            if sep < min_sep:
                min_sep = sep
    return min_sep


def compute_point_error(p_rec: Tuple, p_true: Tuple) -> float:
    """Euclidean error between recovered and true point."""
    return np.sqrt((p_rec[0] - p_true[0])**2 + (p_rec[1] - p_true[1])**2)


def run_extended_stress_test(
    n_views: int = 100,
    noise_levels: List[float] = None,
) -> Dict[float, List[ViewAnalysis]]:
    """Run comprehensive stress test."""
    
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    pattern = get_default_li_wen_qiu_pattern()
    gt = get_default_ground_truth()
    
    results = {}
    
    for noise_sigma in noise_levels:
        print(f"\n{'='*60}")
        print(f"Testing sigma = {noise_sigma} px")
        print(f"{'='*60}")
        
        # Simulate views
        sim = simulate_views(
            n_views=n_views,
            ground_truth=gt,
            noise_config=NoiseConfig(sigma_v=noise_sigma),
            seed=42,
        )
        
        # Also get noiseless for GT points
        sim_noiseless = simulate_views(
            n_views=n_views,
            ground_truth=gt,
            noise_config=NoiseConfig(sigma_v=0.0),
            seed=42,
        )
        
        view_analyses = []
        
        for i, view in enumerate(sim.views):
            v = view.v_observations
            min_sep = compute_min_sep(v)
            
            # Get true pattern points
            R0, T0 = compute_transform_pattern_to_linescan(
                view.R_frame_pattern, view.T_frame_pattern, gt.R, gt.T
            )
            scan_line = compute_scan_line_in_pattern(R0, T0)
            true_pts = [intersect_lines_2d(scan_line, fl) for fl in pattern.feature_lines]
            
            # Point recovery
            try:
                recovered = recover_pattern_points_from_observations(
                    v_obs=list(v),
                    wp1=pattern.wp1,
                    wp2=pattern.wp2,
                    pattern_lines=pattern.feature_lines,
                )
                point_errors = [compute_point_error(recovered[j][:2], true_pts[j]) for j in range(6)]
                max_point_error = max(point_errors)
                point_recovery_success = True
            except Exception:
                point_errors = [float('inf')] * 6
                max_point_error = float('inf')
                point_recovery_success = False
            
            analysis = ViewAnalysis(
                view_id=i,
                noise_sigma=noise_sigma,
                v_obs=v,
                min_sep=min_sep,
                point_errors=point_errors,
                max_point_error=max_point_error,
                point_recovery_success=point_recovery_success,
            )
            view_analyses.append(analysis)
        
        # Now run full calibration to get init and final parameter errors
        print("  Running calibration...")
        backend = LiWenQiuBackend()
        config = CalibrationConfig.from_dict({
            "max_iterations": 2000,
            "convergence_threshold": 1e-10,
        })
        
        result = backend.estimate_from_observations(sim.views, config)
        
        # Get final estimated parameters
        R_final = result.T_oakrgb_hsi[:3, :3]
        T_final = result.T_oakrgb_hsi[:3, 3]
        f_final = result.hsi_intrinsics.focal_length_slit
        v0_final = result.hsi_intrinsics.principal_point_u0
        
        # Compute final errors
        final_rot_err = compute_rotation_error(R_final, gt.R)
        final_trans_err = compute_translation_error(T_final, gt.T)
        final_f_err = abs(f_final - gt.f) / gt.f * 100
        final_v0_err = abs(v0_final - gt.v0)
        
        # Get init estimates from closed-form
        # (We need to run closed-form separately to get init errors)
        pattern_points = []
        frame_poses = []
        v_observations = []
        
        for view in sim.views:
            try:
                rec = recover_pattern_points_from_observations(
                    v_obs=list(view.v_observations),
                    wp1=pattern.wp1,
                    wp2=pattern.wp2,
                    pattern_lines=pattern.feature_lines,
                )
                pattern_points.append(np.array([[p[0], p[1], 0.0] for p in rec]))
                frame_poses.append((view.R_frame_pattern, view.T_frame_pattern))
                v_observations.append(view.v_observations)
            except:
                pass
        
        if pattern_points:
            init_result = closed_form_init(pattern_points, frame_poses, v_observations)
            init_rot_err = compute_rotation_error(init_result.R, gt.R)
            init_trans_err = compute_translation_error(init_result.T, gt.T)
            init_f_err = abs(init_result.f - gt.f) / gt.f * 100
            init_v0_err = abs(init_result.v0 - gt.v0)
        else:
            init_rot_err = init_trans_err = init_f_err = init_v0_err = float('inf')
        
        # Update each view analysis with per-view RMSE from result
        for i, vr in enumerate(result.per_view):
            view_analyses[i].final_rmse = vr.residual_rmse
            # Assign global init/final errors (same for all views in this batch)
            view_analyses[i].init_rot_error_deg = init_rot_err
            view_analyses[i].init_trans_error_mm = init_trans_err
            view_analyses[i].init_f_error_pct = init_f_err
            view_analyses[i].init_v0_error_px = init_v0_err
            view_analyses[i].final_rot_error_deg = final_rot_err
            view_analyses[i].final_trans_error_mm = final_trans_err
            view_analyses[i].final_f_error_pct = final_f_err
            view_analyses[i].final_v0_error_px = final_v0_err
        
        results[noise_sigma] = view_analyses
        
        print(f"  Final RMSE: {result.reprojection_error_rmse:.4f} px")
        print(f"  Init rot err: {init_rot_err:.4f} deg, Final: {final_rot_err:.4f} deg")
        print(f"  Init trans err: {init_trans_err:.4f} mm, Final: {final_trans_err:.4f} mm")
    
    return results


def print_parameter_error_summary(results: Dict[float, List[ViewAnalysis]]):
    """Print parameter estimation error summary."""
    
    print("\n" + "="*80)
    print("1. PARAMETER ESTIMATION ERRORS")
    print("="*80)
    
    print("\n--- CLOSED-FORM INIT ---")
    print("-"*70)
    print(f"{'sigma':<8} {'Rot(deg)':<12} {'Trans(mm)':<12} {'f(%)':<12} {'v0(px)':<12}")
    print("-"*70)
    for sigma, views in sorted(results.items()):
        if views:
            print(f"{sigma:<8.1f} {views[0].init_rot_error_deg:<12.4f} {views[0].init_trans_error_mm:<12.4f} {views[0].init_f_error_pct:<12.4f} {views[0].init_v0_error_px:<12.4f}")
    
    print("\n--- FINAL REFINED ---")
    print("-"*70)
    print(f"{'sigma':<8} {'Rot(deg)':<12} {'Trans(mm)':<12} {'f(%)':<12} {'v0(px)':<12}")
    print("-"*70)
    for sigma, views in sorted(results.items()):
        if views:
            print(f"{sigma:<8.1f} {views[0].final_rot_error_deg:<12.4f} {views[0].final_trans_error_mm:<12.4f} {views[0].final_f_error_pct:<12.4f} {views[0].final_v0_error_px:<12.4f}")


def print_conditioning_analysis(results: Dict[float, List[ViewAnalysis]]):
    """Print conditioning analysis: min_sep vs errors."""
    
    print("\n" + "="*80)
    print("2. CONDITIONING ANALYSIS (min_sep)")
    print("="*80)
    
    for sigma in [0.2, 0.5, 1.0]:
        if sigma not in results:
            continue
        views = results[sigma]
        
        # Compute correlation
        min_seps = [v.min_sep for v in views if v.point_recovery_success]
        max_errors = [v.max_point_error * 1000 for v in views if v.point_recovery_success]  # mm
        
        if len(min_seps) > 1:
            corr = np.corrcoef(min_seps, max_errors)[0, 1]
        else:
            corr = float('nan')
        
        print(f"\n--- sigma = {sigma} px ---")
        print(f"Correlation(min_sep, max_P_error): {corr:.4f}")
        
        # Summary stats
        print(f"min_sep: min={min(min_seps):.2f}, max={max(min_seps):.2f}, mean={np.mean(min_seps):.2f}")
        print(f"max_P_error (mm): min={min(max_errors):.4f}, max={max(max_errors):.4f}, mean={np.mean(max_errors):.4f}")


def print_worst_views(results: Dict[float, List[ViewAnalysis]], n_worst: int = 5):
    """Print worst views analysis."""
    
    print("\n" + "="*80)
    print("3. WORST VIEWS ANALYSIS")
    print("="*80)
    
    for sigma in [0.2, 0.5]:
        if sigma not in results:
            continue
        views = results[sigma]
        
        # Sort by max point error
        sorted_views = sorted(
            [v for v in views if v.point_recovery_success],
            key=lambda v: v.max_point_error,
            reverse=True
        )
        
        print(f"\n--- sigma = {sigma} px: Top {n_worst} worst views by max P_i error ---")
        print("-"*90)
        print(f"{'View':<8} {'min_sep':<10} {'maxP_err(mm)':<14} {'finalRMSE':<12} {'Rot(deg)':<10} {'Trans(mm)':<12}")
        print("-"*90)
        
        for v in sorted_views[:n_worst]:
            print(f"{v.view_id:<8} {v.min_sep:<10.2f} {v.max_point_error*1000:<14.4f} {v.final_rmse:<12.4f} {v.final_rot_error_deg:<10.4f} {v.final_trans_error_mm:<12.4f}")


def print_point_error_distribution(results: Dict[float, List[ViewAnalysis]]):
    """Print point error distribution."""
    
    print("\n" + "="*80)
    print("4. POINT RECOVERY ERROR DISTRIBUTION")
    print("="*80)
    
    print("\n" + "-"*70)
    print(f"{'sigma':<8} {'Mean(mm)':<12} {'Median(mm)':<12} {'Std(mm)':<12} {'Max(mm)':<12}")
    print("-"*70)
    
    for sigma, views in sorted(results.items()):
        all_errors = []
        for v in views:
            if v.point_recovery_success:
                all_errors.extend([e * 1000 for e in v.point_errors])  # Convert to mm
        
        if all_errors:
            print(f"{sigma:<8.1f} {np.mean(all_errors):<12.4f} {np.median(all_errors):<12.4f} {np.std(all_errors):<12.4f} {max(all_errors):<12.4f}")


if __name__ == "__main__":
    results = run_extended_stress_test(n_views=100, noise_levels=[0.0, 0.1, 0.2, 0.5, 1.0])
    
    print_parameter_error_summary(results)
    print_conditioning_analysis(results)
    print_worst_views(results)
    print_point_error_distribution(results)
