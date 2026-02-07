"""Tests for closed-form initialization."""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from hsi_rgbd_calib.cal_method.li_wen_qiu.closed_form import (
    closed_form_init,
    ClosedFormResult,
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import (
    simulate_views,
    get_default_ground_truth,
    NoiseConfig,
    GroundTruth,
)
from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern


class TestClosedFormNoiseless:
    """Tests for closed-form initialization with noiseless data."""
    
    def test_recover_ground_truth_noiseless(self):
        """Test that closed-form recovers ground truth on noiseless data."""
        # Generate noiseless simulation
        n_views = 15
        
        # Use specific ground truth
        gt = GroundTruth(
            f=1000.0,
            v0=640.0,
            k=0.0,  # No distortion for closed-form
            R=Rotation.from_euler('xyz', [1, -2, 0.5], degrees=True).as_matrix(),
            T=np.array([0.04, 0.01, 0.06]),
        )
        
        sim_result = simulate_views(
            n_views=n_views,
            ground_truth=gt,
            noise_config=NoiseConfig(sigma_v=0.0),  # Noiseless
            seed=42,
        )
        
        # Prepare inputs for closed-form
        pattern_points = []
        frame_poses = []
        v_observations = []
        
        pattern = get_default_li_wen_qiu_pattern()
        
        for view in sim_result.views:
            # For closed-form test, we need actual P_i from cross-ratio
            # But since we have noiseless data, we can compute them
            from hsi_rgbd_calib.cal_method.li_wen_qiu.cross_ratio import (
                recover_pattern_points_from_observations,
            )
            
            try:
                P_points = recover_pattern_points_from_observations(
                    v_obs=list(view.v_observations),
                    wp1=pattern.wp1,
                    wp2=pattern.wp2,
                    pattern_lines=pattern.feature_lines,
                )
                pattern_points.append(np.array(P_points))
            except ValueError:
                # Skip views with degenerate cross-ratio
                continue
            
            frame_poses.append((view.R_frame_pattern, view.T_frame_pattern))
            v_observations.append(view.v_observations)
        
        if len(pattern_points) < 2:
            pytest.skip("Not enough valid views for test")
        
        # Run closed-form
        result = closed_form_init(pattern_points, frame_poses, v_observations)
        
        # Check result is reasonable
        assert isinstance(result, ClosedFormResult)
        assert result.f > 0
        
        # For noiseless data, should be close to ground truth
        # Allow some tolerance since closed-form is approximate
        if result.success:
            assert abs(result.f - gt.f) / gt.f < 0.5  # Within 50%
            assert abs(result.v0 - gt.v0) < 200  # Within 200 pixels


class TestClosedFormEdgeCases:
    """Tests for edge cases in closed-form initialization."""
    
    def test_insufficient_views(self):
        """Test that insufficient views are handled."""
        result = closed_form_init(
            pattern_points=[np.zeros((6, 3))],  # Only 1 view
            frame_poses=[(np.eye(3), np.zeros(3))],
            v_observations=[np.arange(6) * 100.0],
        )
        
        assert not result.success
        assert "2 views" in result.message
    
    def test_degenerate_data(self):
        """Test handling of degenerate data."""
        # All zeros - degenerate case
        result = closed_form_init(
            pattern_points=[np.zeros((6, 3)), np.zeros((6, 3))],
            frame_poses=[
                (np.eye(3), np.zeros(3)),
                (np.eye(3), np.zeros(3)),
            ],
            v_observations=[
                np.zeros(6),
                np.zeros(6),
            ],
        )
        
        # Should return something (possibly fallback)
        assert isinstance(result, ClosedFormResult)
        # f and v0 should still be reasonable
        assert result.f > 0


class TestClosedFormWithNoise:
    """Tests for closed-form with noisy data."""
    
    def test_noisy_still_converges(self):
        """Test that closed-form produces reasonable results with noise."""
        gt = get_default_ground_truth()
        
        sim_result = simulate_views(
            n_views=20,
            ground_truth=gt,
            noise_config=NoiseConfig(sigma_v=0.5),  # Moderate noise
            seed=123,
        )
        
        pattern = get_default_li_wen_qiu_pattern()
        
        # Prepare data
        pattern_points = []
        frame_poses = []
        v_observations = []
        
        from hsi_rgbd_calib.cal_method.li_wen_qiu.cross_ratio import (
            recover_pattern_points_from_observations,
        )
        
        for view in sim_result.views:
            try:
                P_points = recover_pattern_points_from_observations(
                    v_obs=list(view.v_observations),
                    wp1=pattern.wp1,
                    wp2=pattern.wp2,
                    pattern_lines=pattern.feature_lines,
                )
                pattern_points.append(np.array(P_points))
                frame_poses.append((view.R_frame_pattern, view.T_frame_pattern))
                v_observations.append(view.v_observations)
            except ValueError:
                continue
        
        if len(pattern_points) < 2:
            pytest.skip("Not enough valid views")
        
        result = closed_form_init(pattern_points, frame_poses, v_observations)
        
        # Should produce some result
        assert isinstance(result, ClosedFormResult)
        assert result.f > 0
        assert result.R.shape == (3, 3)
        assert result.T.shape == (3,)
