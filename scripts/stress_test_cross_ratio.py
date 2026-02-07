"""Stress test for cross-ratio recovery.

Tests: 100 views × noise levels σ={0, 0.1, 0.2, 0.5, 1.0} px

Reports:
  (a) RMSE distribution
  (b) % of failed/unstable cross-ratio recoveries
  (c) Worst-case error in recovered P_i vs GT intersection points
  (d) Point ordering changes and formula validity
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import traceback

from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern
from hsi_rgbd_calib.boards.geometry import intersect_lines_2d
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import simulate_views, NoiseConfig, get_default_ground_truth
from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    compute_scan_line_in_pattern,
    compute_transform_pattern_to_linescan,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.cross_ratio import (
    recover_pattern_points_from_observations,
    compute_cross_ratio,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu import LiWenQiuBackend
from hsi_rgbd_calib.cal_method.interface import CalibrationConfig


@dataclass
class ViewResult:
    """Result for a single view."""
    view_id: int
    noise_sigma: float
    success: bool
    error_message: str
    point_errors: List[float]  # Error for each of 6 points
    max_point_error: float
    v_order_original: List[int]  # Argsort of v values
    v_order_noisy: List[int]  # Argsort after noise
    order_changed: bool


@dataclass
class NoiseResults:
    """Aggregated results for one noise level."""
    noise_sigma: float
    n_views: int
    n_success: int
    n_failed: int
    failure_rate: float
    point_errors_all: List[float]  # All point errors
    max_error: float
    mean_error: float
    median_error: float
    std_error: float
    n_order_changes: int
    order_change_rate: float
    rmse_values: List[float]  # Per-view cross-ratio RMSE


def get_v_order(v: np.ndarray) -> List[int]:
    """Get order of v values (sorted indices)."""
    return list(np.argsort(v))


def compute_point_error(p_rec: Tuple, p_true: Tuple) -> float:
    """Compute Euclidean error between recovered and true point."""
    return np.sqrt((p_rec[0] - p_true[0])**2 + (p_rec[1] - p_true[1])**2)


def run_single_view(
    pattern,
    gt,
    view,
    noise_sigma: float,
    view_id: int,
    gt_v: np.ndarray,  # Noiseless v values
) -> ViewResult:
    """Test cross-ratio recovery on a single view."""
    
    v = view.v_observations
    
    # Get original v order (noiseless) and noisy order
    v_order_original = get_v_order(gt_v)
    v_order_noisy = get_v_order(v)
    order_changed = v_order_original != v_order_noisy
    
    # Get true pattern points from ground truth
    R0, T0 = compute_transform_pattern_to_linescan(
        view.R_frame_pattern, view.T_frame_pattern, gt.R, gt.T
    )
    scan_line = compute_scan_line_in_pattern(R0, T0)
    true_points = [intersect_lines_2d(scan_line, fl) for fl in pattern.feature_lines]
    
    # Try cross-ratio recovery
    try:
        recovered = recover_pattern_points_from_observations(
            v_obs=list(v),
            wp1=pattern.wp1,
            wp2=pattern.wp2,
            pattern_lines=pattern.feature_lines,
        )
        
        # Compute errors
        point_errors = []
        for i in range(6):
            err = compute_point_error(recovered[i][:2], true_points[i])
            point_errors.append(err)
        
        max_error = max(point_errors)
        
        return ViewResult(
            view_id=view_id,
            noise_sigma=noise_sigma,
            success=True,
            error_message="",
            point_errors=point_errors,
            max_point_error=max_error,
            v_order_original=v_order_original,
            v_order_noisy=v_order_noisy,
            order_changed=order_changed,
        )
        
    except Exception as e:
        return ViewResult(
            view_id=view_id,
            noise_sigma=noise_sigma,
            success=False,
            error_message=str(e),
            point_errors=[float('inf')] * 6,
            max_point_error=float('inf'),
            v_order_original=v_order_original,
            v_order_noisy=v_order_noisy,
            order_changed=order_changed,
        )


def run_stress_test(n_views: int = 100, noise_levels: List[float] = None) -> Dict[float, NoiseResults]:
    """Run comprehensive stress test."""
    
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
    
    pattern = get_default_li_wen_qiu_pattern()
    gt = get_default_ground_truth()
    
    results_by_noise = {}
    
    for noise_sigma in noise_levels:
        print(f"\n{'='*60}")
        print(f"Testing noise level sigma = {noise_sigma} px")
        print(f"{'='*60}")
        
        view_results = []
        
        # First simulate noiseless to get GT v values
        sim_noiseless = simulate_views(
            n_views=n_views,
            ground_truth=gt,
            noise_config=NoiseConfig(sigma_v=0.0),
            seed=42,
        )
        
        # Then simulate with noise
        if noise_sigma > 0:
            sim_noisy = simulate_views(
                n_views=n_views,
                ground_truth=gt,
                noise_config=NoiseConfig(sigma_v=noise_sigma),
                seed=42,
            )
        else:
            sim_noisy = sim_noiseless
        
        for i in range(n_views):
            result = run_single_view(
                pattern=pattern,
                gt=gt,
                view=sim_noisy.views[i],
                noise_sigma=noise_sigma,
                view_id=i,
                gt_v=sim_noiseless.views[i].v_observations,
            )
            view_results.append(result)
            
            if not result.success:
                print(f"  View {i}: FAILED - {result.error_message}")
            elif result.order_changed:
                print(f"  View {i}: Order changed! Max error = {result.max_point_error*1000:.3f} mm")
        
        # Aggregate results
        n_success = sum(1 for r in view_results if r.success)
        n_failed = n_views - n_success
        n_order_changes = sum(1 for r in view_results if r.order_changed)
        
        all_point_errors = []
        for r in view_results:
            if r.success:
                all_point_errors.extend(r.point_errors)
        
        valid_errors = [e for e in all_point_errors if np.isfinite(e)]
        
        if valid_errors:
            max_error = max(valid_errors)
            mean_error = np.mean(valid_errors)
            median_error = np.median(valid_errors)
            std_error = np.std(valid_errors)
        else:
            max_error = mean_error = median_error = std_error = float('inf')
        
        # Also compute RMSE per view (for recovered points)
        rmse_values = []
        for r in view_results:
            if r.success:
                rmse = np.sqrt(np.mean([e**2 for e in r.point_errors]))
                rmse_values.append(rmse)
        
        results_by_noise[noise_sigma] = NoiseResults(
            noise_sigma=noise_sigma,
            n_views=n_views,
            n_success=n_success,
            n_failed=n_failed,
            failure_rate=n_failed / n_views * 100,
            point_errors_all=valid_errors,
            max_error=max_error,
            mean_error=mean_error,
            median_error=median_error,
            std_error=std_error,
            n_order_changes=n_order_changes,
            order_change_rate=n_order_changes / n_views * 100,
            rmse_values=rmse_values,
        )
    
    return results_by_noise


def print_summary(results: Dict[float, NoiseResults]):
    """Print summary table."""
    
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    
    print("\n(a) RMSE Distribution (recovered P_i vs GT intersection)")
    print("-" * 70)
    print(f"{'s (px)':<10} {'Mean (mm)':<12} {'Median (mm)':<12} {'Std (mm)':<12} {'Max (mm)':<12}")
    print("-" * 70)
    for sigma, r in sorted(results.items()):
        print(f"{sigma:<10.1f} {r.mean_error*1000:<12.4f} {r.median_error*1000:<12.4f} {r.std_error*1000:<12.4f} {r.max_error*1000:<12.4f}")
    
    print("\n(b) Failure Rate")
    print("-" * 50)
    print(f"{'s (px)':<10} {'Failed':<10} {'Total':<10} {'Rate (%)':<12}")
    print("-" * 50)
    for sigma, r in sorted(results.items()):
        print(f"{sigma:<10.1f} {r.n_failed:<10} {r.n_views:<10} {r.failure_rate:<12.2f}")
    
    print("\n(c) Worst-Case Error (max point error)")
    print("-" * 40)
    print(f"{'s (px)':<10} {'Max Error (mm)':<15}")
    print("-" * 40)
    for sigma, r in sorted(results.items()):
        print(f"{sigma:<10.1f} {r.max_error*1000:<15.4f}")
    
    print("\n(d) Point Ordering Changes (v values change order with noise)")
    print("-" * 50)
    print(f"{'s (px)':<10} {'# Changes':<12} {'Rate (%)':<12}")
    print("-" * 50)
    for sigma, r in sorted(results.items()):
        print(f"{sigma:<10.1f} {r.n_order_changes:<12} {r.order_change_rate:<12.2f}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Check if formulas remain valid with order changes
    has_failures_with_order_changes = False
    for sigma, r in sorted(results.items()):
        if r.n_order_changes > 0 and r.n_failed > 0:
            has_failures_with_order_changes = True
    
    if has_failures_with_order_changes:
        print("\n[!] Some failures correlate with ordering changes.")
        print("   The formulas may not be robust to reordering.")
    else:
        print("\n[OK] No correlation between ordering changes and failures.")
    
    # Check high-noise behavior
    if 1.0 in results:
        r = results[1.0]
        if r.failure_rate > 5:
            print(f"\n[!] At s=1.0px, {r.failure_rate:.1f}% failures.")
        else:
            print(f"\n[OK] At s=1.0px, only {r.failure_rate:.1f}% failures.")


if __name__ == "__main__":
    results = run_stress_test(n_views=100, noise_levels=[0.0, 0.1, 0.2, 0.5, 1.0])
    print_summary(results)
