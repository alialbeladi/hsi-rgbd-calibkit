"""Generate visualization plots for Li-Wen-Qiu calibration.

This script:
1. Simulates calibration views
2. Runs the calibration pipeline
3. Generates all visualization plots
"""

import sys
from pathlib import Path

from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern
from hsi_rgbd_calib.cal_method.li_wen_qiu import LiWenQiuBackend
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import (
    simulate_views, 
    get_default_ground_truth, 
    NoiseConfig
)
from hsi_rgbd_calib.cal_method.interface import CalibrationConfig
from hsi_rgbd_calib.viz import VisualizationData, generate_all_plots, GroundTruthViz


def main():
    # Output directory for plots
    output_dir = Path("outputs/viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Li-Wen-Qiu Calibration Visualization")
    print("=" * 60)
    
    # 1. Simulate views
    print("\n[1] Simulating calibration views...")
    gt = get_default_ground_truth()
    pattern = get_default_li_wen_qiu_pattern()
    
    sim_result = simulate_views(
        n_views=15,
        ground_truth=gt,
        noise_config=NoiseConfig(sigma_v=0.2),  # Small noise
        seed=42,
    )
    print(f"    Generated {len(sim_result.views)} views")
    
    # 2. Run calibration
    print("\n[2] Running calibration...")
    backend = LiWenQiuBackend()
    config = CalibrationConfig.from_dict({
        "max_iterations": 2000,
        "convergence_threshold": 1e-10,
        "debug_output_dir": str(output_dir),
    })
    
    result = backend.estimate_from_observations(sim_result.views, config)
    print(f"    RMSE: {result.reprojection_error_rmse:.4f} px")
    print(f"    Success: {result.success}")
    
    # 3. Prepare visualization data
    print("\n[3] Preparing visualization data...")
    
    # Create ground truth viz data
    gt_viz = GroundTruthViz(
        R_true=gt.R,
        T_true=gt.T,
        f_true=gt.f,
        v0_true=gt.v0,
        k_true=gt.k,
    )
    
    # Create visualization data from result
    viz_data = VisualizationData.from_calibration_result(
        result=result,
        pattern=pattern,
        gt=gt_viz,
    )
    
    # 4. Generate plots
    print("\n[4] Generating plots...")
    plot_paths = generate_all_plots(viz_data, output_dir, include_optional=True)
    
    print("\n" + "=" * 60)
    print("Generated Plots")
    print("=" * 60)
    for name, path in plot_paths.items():
        if path:
            print(f"  {name}: {path}")
        else:
            print(f"  {name}: (skipped)")
    
    print(f"\nPlots saved to: {output_dir.absolute()}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
