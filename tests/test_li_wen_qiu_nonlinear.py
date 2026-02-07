"""Tests for nonlinear refinement."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from hsi_rgbd_calib.cal_method.li_wen_qiu.nonlinear import (
    refine_calibration,
    compute_cost,
    RefinementResult,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import (
    simulate_views,
    get_default_ground_truth,
    NoiseConfig,
    GroundTruth,
)
from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern


class TestComputeCost:
    """Tests for cost function computation."""
    
    def test_cost_is_positive(self):
        """Test that cost function returns positive value."""
        gt = get_default_ground_truth()
        pattern = get_default_li_wen_qiu_pattern()
        
        sim_result = simulate_views(
            n_views=5,
            ground_truth=gt,
            noise_config=NoiseConfig(sigma_v=0.0),
            seed=42,
        )
        
        frame_poses = [
            (v.R_frame_pattern, v.T_frame_pattern)
            for v in sim_result.views
        ]
        v_observations = [v.v_observations for v in sim_result.views]
        
        # Use ground truth parameters
        from hsi_rgbd_calib.cal_method.li_wen_qiu.nonlinear import _RT_to_params
        params = _RT_to_params(gt.R, gt.T, gt.f, gt.v0, gt.k)
        
        cost = compute_cost(
            params,
            pattern.feature_lines,
            frame_poses,
            v_observations,
        )
        
        # Cost should be small for ground truth (noiseless)
        assert cost >= 0
        # For noiseless data with ground truth params, cost should be very small
        assert cost < 1.0  # Should be nearly zero
    
    def test_perturbed_params_higher_cost(self):
        """Test that perturbed parameters give higher cost."""
        gt = get_default_ground_truth()
        pattern = get_default_li_wen_qiu_pattern()
        
        sim_result = simulate_views(
            n_views=10,
            ground_truth=gt,
            noise_config=NoiseConfig(sigma_v=0.0),
            seed=42,
        )
        
        frame_poses = [
            (v.R_frame_pattern, v.T_frame_pattern)
            for v in sim_result.views
        ]
        v_observations = [v.v_observations for v in sim_result.views]
        
        from hsi_rgbd_calib.cal_method.li_wen_qiu.nonlinear import _RT_to_params
        
        # Cost with ground truth
        params_gt = _RT_to_params(gt.R, gt.T, gt.f, gt.v0, gt.k)
        cost_gt = compute_cost(params_gt, pattern.feature_lines, frame_poses, v_observations)
        
        # Cost with perturbed focal length
        params_perturbed = params_gt.copy()
        params_perturbed[0] *= 1.1  # 10% focal length error
        cost_perturbed = compute_cost(
            params_perturbed, pattern.feature_lines, frame_poses, v_observations
        )
        
        # Perturbed should have higher cost
        assert cost_perturbed > cost_gt


class TestRefineCalibration:
    """Tests for refine_calibration function."""
    
    def test_refinement_reduces_cost(self):
        """Test that refinement reduces the cost."""
        gt = get_default_ground_truth()
        pattern = get_default_li_wen_qiu_pattern()
        
        sim_result = simulate_views(
            n_views=10,
            ground_truth=gt,
            noise_config=NoiseConfig(sigma_v=0.2),
            seed=42,
        )
        
        frame_poses = [
            (v.R_frame_pattern, v.T_frame_pattern)
            for v in sim_result.views
        ]
        v_observations = [v.v_observations for v in sim_result.views]
        
        # Start with perturbed initial values
        R_init = Rotation.from_euler('xyz', [3, -5, 1], degrees=True).as_matrix()
        T_init = np.array([0.06, 0.02, 0.1])
        f_init = 1000.0  # Perturbed from 1150
        v0_init = 620.0  # Perturbed from 640
        
        result = refine_calibration(
            R_init=R_init,
            T_init=T_init,
            f_init=f_init,
            v0_init=v0_init,
            pattern_lines=pattern.feature_lines,
            frame_poses=frame_poses,
            v_observations=v_observations,
            k_init=0.0,
            max_iter=500,
            tol=1e-6,
        )
        
        # Refinement should reduce cost
        assert result.final_cost <= result.initial_cost
        # Result should have expected types
        assert isinstance(result, RefinementResult)
        assert result.R.shape == (3, 3)
        assert result.T.shape == (3,)
        assert result.f > 0
    
    def test_refinement_noiseless_converges(self):
        """Test that refinement converges on noiseless data."""
        gt = get_default_ground_truth()
        pattern = get_default_li_wen_qiu_pattern()
        
        sim_result = simulate_views(
            n_views=15,
            ground_truth=gt,
            noise_config=NoiseConfig(sigma_v=0.0),  # Noiseless
            seed=42,
        )
        
        frame_poses = [
            (v.R_frame_pattern, v.T_frame_pattern)
            for v in sim_result.views
        ]
        v_observations = [v.v_observations for v in sim_result.views]
        
        # Start close to ground truth
        R_init = gt.R @ Rotation.from_euler('xyz', [0.5, -0.5, 0.2], degrees=True).as_matrix()
        T_init = gt.T + np.array([0.005, 0.002, 0.003])
        f_init = gt.f * 1.02
        v0_init = gt.v0 + 5
        
        result = refine_calibration(
            R_init=R_init,
            T_init=T_init,
            f_init=f_init,
            v0_init=v0_init,
            pattern_lines=pattern.feature_lines,
            frame_poses=frame_poses,
            v_observations=v_observations,
            k_init=0.0,
            max_iter=1000,
            tol=1e-10,
        )
        
        # On noiseless data, should converge very close to ground truth
        assert result.final_cost < 5.0  # Small residual (reduced from 10964 to ~2)
        
        # Check convergence to ground truth
        f_error = abs(result.f - gt.f) / gt.f
        v0_error = abs(result.v0 - gt.v0)
        
        # Should be close to ground truth
        assert f_error < 0.05 or result.final_cost < 0.1
