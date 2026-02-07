#!/usr/bin/env python
"""Simulation script for Li-Wen-Qiu calibration.

This script generates synthetic calibration data, runs the calibration
pipeline, and reports errors compared to ground truth.

Usage:
    python scripts/sim_run_li_wen_qiu.py --n-views 20 --noise 0.2 --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from hsi_rgbd_calib.cal_method.li_wen_qiu import LiWenQiuBackend
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import (
    simulate_views,
    get_default_ground_truth,
    compute_estimation_errors,
    NoiseConfig,
)
from hsi_rgbd_calib.cal_method.interface import CalibrationConfig
from hsi_rgbd_calib.common.logging import setup_logging, print_info, print_success, print_error


def main():
    parser = argparse.ArgumentParser(
        description="Run Li-Wen-Qiu calibration on simulated data"
    )
    parser.add_argument(
        "--n-views", type=int, default=15,
        help="Number of views to simulate (default: 15)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.2,
        help="Pixel noise standard deviation (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output JSON report path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    import logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    
    print_info(f"Simulating {args.n_views} views with noise sigma={args.noise} px")
    
    # Generate ground truth and simulation
    gt = get_default_ground_truth()
    noise_config = NoiseConfig(sigma_v=args.noise)
    
    sim_result = simulate_views(
        n_views=args.n_views,
        ground_truth=gt,
        noise_config=noise_config,
        seed=args.seed,
    )
    
    print_info(f"Generated {len(sim_result.views)} valid views")
    
    # Run calibration
    print_info("Running Li-Wen-Qiu calibration...")
    
    backend = LiWenQiuBackend()
    config = CalibrationConfig.from_dict({
        "max_iterations": 2000,
        "convergence_threshold": 1e-10,
    })
    
    result = backend.estimate_from_observations(sim_result.views, config)
    
    # Compute errors
    R_est = result.T_oakrgb_hsi[:3, :3]
    T_est = result.T_oakrgb_hsi[:3, 3]
    f_est = result.hsi_intrinsics.focal_length_slit
    v0_est = result.hsi_intrinsics.principal_point_u0
    k_est = result.hsi_intrinsics.distortion_coeffs[0] if result.hsi_intrinsics.distortion_coeffs else 0.0
    
    errors = compute_estimation_errors(
        estimated={"R": R_est, "T": T_est, "f": f_est, "v0": v0_est, "k": k_est},
        ground_truth=gt,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    
    print(f"\nReprojection Error:")
    print(f"  RMSE:   {result.reprojection_error_rmse:.4f} px")
    print(f"  Median: {result.reprojection_error_median:.4f} px")
    print(f"  Max:    {result.reprojection_error_max:.4f} px")
    
    print(f"\nEstimated Parameters:")
    print(f"  f:  {f_est:.2f} (GT: {gt.f:.2f})")
    print(f"  v0: {v0_est:.2f} (GT: {gt.v0:.2f})")
    print(f"  k:  {k_est:.2e} (GT: {gt.k:.2e})")
    
    print(f"\nParameter Errors:")
    print(f"  Rotation:    {errors['rotation_error_deg']:.4f} deg")
    print(f"  Translation: {errors['translation_error_m']*1000:.4f} mm")
    print(f"  f (rel):     {errors['f_error_relative']*100:.4f} %")
    print(f"  v0:          {errors['v0_error_px']:.4f} px")
    
    # Check acceptance criteria
    print("\n" + "-" * 60)
    print("ACCEPTANCE CRITERIA CHECK")
    print("-" * 60)
    
    if args.noise == 0:
        # Noiseless criteria
        criteria = {
            "rotation_error_rad": (errors["rotation_error_rad"], 1e-3),
            "translation_error_m": (errors["translation_error_m"], 1e-6),
            "f_error_relative": (errors["f_error_relative"], 1e-6),
            "v0_error_px": (errors["v0_error_px"], 1e-6),
        }
    else:
        # Noisy criteria
        criteria = {
            "rotation_error_deg": (errors["rotation_error_deg"], 0.5),
            "translation_error_mm": (errors["translation_error_m"] * 1000, 5.0),
            "f_error_percent": (errors["f_error_relative"] * 100, 1.0),
            "v0_error_px": (errors["v0_error_px"], 2.0),
        }
    
    all_pass = True
    for name, (value, threshold) in criteria.items():
        passed = value <= threshold
        all_pass = all_pass and passed
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {value:.6f} <= {threshold:.6f} [{status}]")
    
    if all_pass:
        print_success("\nAll acceptance criteria passed!")
    else:
        print_error("\nSome acceptance criteria failed.")
    
    # Save report if requested
    if args.out:
        report = {
            "n_views": args.n_views,
            "noise_sigma": args.noise,
            "seed": args.seed,
            "ground_truth": gt.to_dict(),
            "estimated": {
                "f": f_est,
                "v0": v0_est,
                "k": k_est,
                "R": R_est.tolist(),
                "T": T_est.tolist(),
            },
            "reprojection": {
                "rmse": result.reprojection_error_rmse,
                "median": result.reprojection_error_median,
                "max": result.reprojection_error_max,
            },
            "errors": {k: float(v) for k, v in errors.items()},
            "success": result.success,
            "all_criteria_passed": all_pass,
        }
        
        out_path = Path(args.out)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print_info(f"\nReport saved to {out_path}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
