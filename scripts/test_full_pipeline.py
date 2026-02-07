"""Test full pipeline RMSE."""

from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import (
    simulate_views, NoiseConfig, get_default_ground_truth, compute_estimation_errors
)
from hsi_rgbd_calib.cal_method.li_wen_qiu import LiWenQiuBackend
from hsi_rgbd_calib.cal_method.interface import CalibrationConfig

gt = get_default_ground_truth()
sim = simulate_views(
    n_views=15,
    ground_truth=gt,
    noise_config=NoiseConfig(sigma_v=0.0),
    seed=42
)

backend = LiWenQiuBackend()
config = CalibrationConfig.from_dict({
    'max_iterations': 1000,
    'convergence_threshold': 1e-8
})

result = backend.estimate_from_observations(sim.views, config)

print(f"RMSE: {result.reprojection_error_rmse:.4f}")
print(f"Success: {result.success}")

R_est = result.T_oakrgb_hsi[:3, :3]
T_est = result.T_oakrgb_hsi[:3, 3]
f_est = result.hsi_intrinsics.focal_length_slit
v0_est = result.hsi_intrinsics.principal_point_u0

print(f"f={f_est:.2f} (GT: {gt.f:.2f})")
print(f"v0={v0_est:.2f} (GT: {gt.v0:.2f})")

errors = compute_estimation_errors(
    {'R': R_est, 'T': T_est, 'f': f_est, 'v0': v0_est},
    gt
)
print(f"Rotation error: {errors['rotation_error_rad']:.4f} rad")
print(f"Translation error: {errors['translation_error_m']:.4f} m")
print(f"Focal length error: {errors['focal_length_error_pct']:.2f}%")
print(f"v0 error: {errors['v0_error_px']:.2f} px")
