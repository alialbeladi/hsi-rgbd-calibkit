"""Tests for cross-ratio computation."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from hsi_rgbd_calib.cal_method.li_wen_qiu.cross_ratio import (
    compute_cross_ratio,
    compute_cross_ratios,
    recover_y_from_cross_ratio,
    recover_pattern_points_from_observations,
)
from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import simulate_views, NoiseConfig
from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    compute_scan_line_in_pattern,
    compute_transform_pattern_to_linescan,
)
from hsi_rgbd_calib.boards.geometry import intersect_lines_2d


class TestComputeCrossRatio:
    """Tests for cross-ratio computation."""
    
    def test_cross_ratio_basic(self):
        """Test basic cross-ratio formula."""
        # CR(a, b, c, d) = ((a-c)*(b-d)) / ((b-c)*(a-d))
        # With a=0, b=1, c=2, d=3:
        # CR = ((0-2)*(1-3)) / ((1-2)*(0-3)) = (-2)*(-2) / ((-1)*(-3)) = 4/3
        cr = compute_cross_ratio(0, 1, 2, 3)
        expected = 4.0 / 3.0
        assert_allclose(cr, expected)
    
    def test_cross_ratio_invariant(self):
        """Test that cross-ratio is projective invariant."""
        # If we scale all values by a constant, CR should remain the same
        a, b, c, d = 10, 20, 30, 50
        cr1 = compute_cross_ratio(a, b, c, d)
        cr2 = compute_cross_ratio(2*a, 2*b, 2*c, 2*d)  # Scaled
        assert_allclose(cr1, cr2)
        
        # If we translate all values, CR should remain the same
        cr3 = compute_cross_ratio(a+100, b+100, c+100, d+100)
        assert_allclose(cr1, cr3)
    
    def test_degenerate_case_raises(self):
        """Test that degenerate cases raise ValueError."""
        with pytest.raises(ValueError, match="Degenerate"):
            compute_cross_ratio(0, 1, 1, 2)  # b = c causes denominator = 0


class TestRecoverYFromCrossRatio:
    """Tests for Y-coordinate recovery from cross-ratio."""
    
    def test_basic_recovery(self):
        """Test Y recovery formula."""
        wp1, wp2 = 0.1, 0.05
        
        # For a known Y value, compute what CR should be, then verify recovery
        y_true = 0.04  # Target Y
        
        # CR(y1, y2, y, y3) = ((y1-y)*(y2-y3)) / ((y2-y)*(y1-y3))
        # = ((0 - y)*(wp2 - wp1)) / ((wp2 - y)*(0 - wp1))
        # = ((-y)*(wp2 - wp1)) / ((wp2 - y)*(-wp1))
        # = (y*(wp2 - wp1)) / ((wp2 - y)*wp1)  [negatives cancel]
        cr = (y_true * (wp2 - wp1)) / ((wp2 - y_true) * wp1)
        
        y_recovered = recover_y_from_cross_ratio(cr, wp1, wp2)
        assert_allclose(y_recovered, y_true, atol=1e-10)
    
    def test_degenerate_case_raises(self):
        """Test that degenerate case raises ValueError."""
        wp1, wp2 = 0.1, 0.05
        # CR = 1 - wp2/wp1 would make denominator zero
        cr_degenerate = 1 - wp2/wp1
        with pytest.raises(ValueError, match="Degenerate"):
            recover_y_from_cross_ratio(cr_degenerate, wp1, wp2)


class TestRecoverPatternPoints:
    """Tests for full pattern point recovery."""
    
    def test_roundtrip_noiseless(self):
        """Test that pattern points can be recovered from observations."""
        pattern = get_default_li_wen_qiu_pattern()
        
        # Simulate with no noise
        sim_result = simulate_views(
            n_views=1,
            noise_config=NoiseConfig(sigma_v=0.0),
            seed=42,
        )
        gt = sim_result.ground_truth
        view = sim_result.views[0]
        
        # Get true pattern points
        R0, T0 = compute_transform_pattern_to_linescan(
            view.R_frame_pattern, view.T_frame_pattern, gt.R, gt.T
        )
        scan_line = compute_scan_line_in_pattern(R0, T0)
        true_points = [intersect_lines_2d(scan_line, fl) for fl in pattern.feature_lines]
        
        # Recover using cross-ratio
        recovered = recover_pattern_points_from_observations(
            v_obs=list(view.v_observations),
            wp1=pattern.wp1,
            wp2=pattern.wp2,
            pattern_lines=pattern.feature_lines,
        )
        
        # Compare - should be exact for noiseless case
        for i in range(6):
            assert_allclose(recovered[i][:2], true_points[i], atol=1e-6)
    
    def test_multiple_views(self):
        """Test recovery across multiple views."""
        pattern = get_default_li_wen_qiu_pattern()
        
        sim_result = simulate_views(
            n_views=5,
            noise_config=NoiseConfig(sigma_v=0.0),
            seed=123,
        )
        gt = sim_result.ground_truth
        
        for view in sim_result.views:
            # Get true pattern points
            R0, T0 = compute_transform_pattern_to_linescan(
                view.R_frame_pattern, view.T_frame_pattern, gt.R, gt.T
            )
            scan_line = compute_scan_line_in_pattern(R0, T0)
            true_points = [intersect_lines_2d(scan_line, fl) for fl in pattern.feature_lines]
            
            # Recover
            recovered = recover_pattern_points_from_observations(
                v_obs=list(view.v_observations),
                wp1=pattern.wp1,
                wp2=pattern.wp2,
                pattern_lines=pattern.feature_lines,
            )
            
            for i in range(6):
                assert_allclose(recovered[i][:2], true_points[i], atol=1e-6)
    
    def test_with_noise(self):
        """Test that recovery is approximate with noise."""
        pattern = get_default_li_wen_qiu_pattern()
        
        sim_result = simulate_views(
            n_views=1,
            noise_config=NoiseConfig(sigma_v=0.5),  # Small noise
            seed=42,
        )
        gt = sim_result.ground_truth
        view = sim_result.views[0]
        
        # Get true pattern points
        R0, T0 = compute_transform_pattern_to_linescan(
            view.R_frame_pattern, view.T_frame_pattern, gt.R, gt.T
        )
        scan_line = compute_scan_line_in_pattern(R0, T0)
        true_points = [intersect_lines_2d(scan_line, fl) for fl in pattern.feature_lines]
        
        # Recover - should still work but not exact
        recovered = recover_pattern_points_from_observations(
            v_obs=list(view.v_observations),
            wp1=pattern.wp1,
            wp2=pattern.wp2,
            pattern_lines=pattern.feature_lines,
        )
        
        # With small noise, should be within reasonable tolerance
        for i in range(6):
            assert_allclose(recovered[i][:2], true_points[i], atol=0.01)  # 1cm tolerance
