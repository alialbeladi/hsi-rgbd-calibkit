"""Test the new closed-form implementation's parameter recovery."""

import numpy as np
from scipy.spatial.transform import Rotation

from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import (
    simulate_views, NoiseConfig, get_default_ground_truth
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.cross_ratio import (
    recover_pattern_points_from_observations,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.closed_form import closed_form_init


def compute_rotation_error(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Compute rotation error in degrees."""
    R_err = R_est @ R_true.T
    angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
    return np.degrees(angle)


def test_closed_form():
    print("="*60)
    print("TESTING CLOSED-FORM PARAMETER RECOVERY")
    print("="*60)
    
    pattern = get_default_li_wen_qiu_pattern()
    gt = get_default_ground_truth()
    
    print(f"\nGround Truth:")
    print(f"  R =\n{gt.R}")
    print(f"  T = {gt.T}")
    print(f"  f = {gt.f}")
    print(f"  v0 = {gt.v0}")
    
    # Simulate noiseless views
    sim = simulate_views(
        n_views=15,
        ground_truth=gt,
        noise_config=NoiseConfig(sigma_v=0.0),
        seed=42,
    )
    
    # Prepare input data for closed-form
    pattern_points = []
    frame_poses = []
    v_observations = []
    
    for view in sim.views:
        # Recover pattern points using cross-ratio
        recovered = recover_pattern_points_from_observations(
            v_obs=list(view.v_observations),
            wp1=pattern.wp1,
            wp2=pattern.wp2,
            pattern_lines=pattern.feature_lines,
        )
        
        # Convert to numpy array
        pts = np.array([[p[0], p[1], 0.0] for p in recovered])
        pattern_points.append(pts)
        frame_poses.append((view.R_frame_pattern, view.T_frame_pattern))
        v_observations.append(view.v_observations)
    
    print(f"\nPrepared {len(pattern_points)} views for closed-form")
    
    # Run closed-form
    result = closed_form_init(pattern_points, frame_poses, v_observations)
    
    print(f"\nClosed-Form Result:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  R =\n{result.R}")
    print(f"  T = {result.T}")
    print(f"  f = {result.f}")
    print(f"  v0 = {result.v0}")
    
    # Compute errors
    rot_err = compute_rotation_error(result.R, gt.R)
    trans_err = np.linalg.norm(result.T - gt.T) * 1000
    f_err = abs(result.f - gt.f) / gt.f * 100
    v0_err = abs(result.v0 - gt.v0)
    
    print(f"\nErrors vs Ground Truth:")
    print(f"  Rotation:    {rot_err:.4f} deg")
    print(f"  Translation: {trans_err:.4f} mm")
    print(f"  f:           {f_err:.4f} %")
    print(f"  v0:          {v0_err:.4f} px")
    
    print("\n" + "="*60)
    if rot_err < 0.1 and trans_err < 1.0 and f_err < 0.1 and v0_err < 1.0:
        print("SUCCESS! Closed-form recovers GT accurately.")
    else:
        print("FAILURE: Closed-form has significant errors.")
    print("="*60)


if __name__ == "__main__":
    test_closed_form()
