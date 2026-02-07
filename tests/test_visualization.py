"""Tests for the visualization module."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from hsi_rgbd_calib.viz import (
    VisualizationData,
    ViewVizData,
    GroundTruthViz,
    CalibratedRig,
    check_view_validity,
    generate_all_plots,
    get_pattern_bounds,
)


def create_mock_viz_data(n_views: int = 5, with_gt: bool = False) -> VisualizationData:
    """Create mock visualization data for testing."""
    views = []
    np.random.seed(42)
    
    for j in range(n_views):
        # Random pose
        angle = np.random.uniform(-0.3, 0.3)
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        T = np.array([0.0, 0.0, 0.5 + j * 0.1])
        
        # Mock observations
        v_obs = np.array([100, 200, 300, 400, 500, 600]) + np.random.randn(6) * 2
        v_init = v_obs + np.random.randn(6) * 5
        v_final = v_obs + np.random.randn(6) * 1
        
        # Pattern points
        P = np.array([
            [0.05, 0.0, 0.0],
            [0.05, 0.05, 0.0],
            [0.05, 0.1, 0.0],
            [0.1, 0.1, 0.0],
            [0.1, 0.15, 0.0],
            [0.15, 0.15, 0.0],
        ])
        
        views.append(ViewVizData(
            view_id=f"view_{j}",
            R_frame_pattern=R,
            T_frame_pattern=T,
            R0=R,
            T0=T,
            scan_line=(0.1, 0.9, -0.05),
            v_observed=v_obs,
            v_init=v_init,
            v_final=v_final,
            P_pattern_init=P,
            P_pattern_final=P,
            residual_rmse=float(np.sqrt(np.mean((v_final - v_obs) ** 2))),
        ))
    
    # Pattern lines
    pattern_lines = [
        (0, 1, 0),          # Y = 0
        (1, -1, 0),         # X = Y
        (0, 1, -0.1),       # Y = 0.1
        (1, -1, 0.1),       # X - Y = -0.1
        (0, 1, -0.15),      # Y = 0.15
        (1, -1, 0.15),      # X - Y = -0.15
    ]
    
    # Transform
    T_oakrgb_hsi = np.eye(4)
    T_oakrgb_hsi[:3, 3] = [0.05, 0.0, 0.0]
    
    # Ground truth (optional)
    gt = None
    if with_gt:
        gt = GroundTruthViz(
            R_true=np.eye(3),
            T_true=np.array([0.05, 0.0, 0.0]),
            f_true=1000.0,
            v0_true=640.0,
            k_true=0.0,
        )
    
    return VisualizationData(
        wp1=0.15,
        wp2=0.1,
        pattern_lines=pattern_lines,
        T_oakrgb_hsi=T_oakrgb_hsi,
        f=1000.0,
        v0=640.0,
        k=0.0,
        views=views,
        cost_history=[100, 50, 20, 10, 5, 3, 2, 1.5],
        gt=gt,
    )


class TestDataContracts:
    """Test data contracts."""
    
    def test_view_viz_data_creation(self):
        """Test ViewVizData can be created."""
        view = ViewVizData(
            view_id="test",
            R_frame_pattern=np.eye(3),
            T_frame_pattern=np.zeros(3),
            R0=np.eye(3),
            T0=np.zeros(3),
            scan_line=(0, 1, 0),
            v_observed=np.zeros(6),
            v_init=np.zeros(6),
            v_final=np.zeros(6),
            P_pattern_init=np.zeros((6, 3)),
            P_pattern_final=np.zeros((6, 3)),
            residual_rmse=0.0,
        )
        assert view.view_id == "test"
    
    def test_visualization_data_creation(self):
        """Test VisualizationData can be created."""
        data = create_mock_viz_data()
        assert len(data.views) == 5
        assert data.wp1 == 0.15


class TestTransforms:
    """Test transform utilities."""
    
    def test_calibrated_rig_creation(self):
        """Test CalibratedRig can be created."""
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.0, 0.0]
        rig = CalibratedRig(T)
        
        assert rig.T_oakrgb_hsi.shape == (4, 4)
        assert np.allclose(rig.hsi_origin_in_oakrgb(), [0.1, 0.0, 0.0])
    
    def test_calibrated_rig_inverse(self):
        """Test inverse transform consistency."""
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.02, -0.01]
        rig = CalibratedRig(T)
        
        # Inverse should invert back
        T_inv = rig.T_hsi_oakrgb
        product = T @ T_inv
        assert np.allclose(product, np.eye(4), atol=1e-10)
    
    def test_check_view_validity(self):
        """Test validity checks."""
        R = np.eye(3)
        T = np.array([0, 0, 1.0])  # Points will be in front
        P = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [0.1, 0.1, 0],
            [0, 0.1, 0],
            [0.05, 0.05, 0],
            [0.15, 0.15, 0],
        ])
        
        result = check_view_validity(R, T, P)
        assert result["centroid_in_front"] == True
        assert result["det_R0_positive"] == True
        assert result["points_visible"] == True


class TestPatternBounds:
    """Test pattern bounds calculation."""
    
    def test_pattern_bounds_reasonable(self):
        """Test bounds are reasonable for given dimensions."""
        wp1, wp2 = 0.15, 0.1
        x_range, y_range = get_pattern_bounds(wp1, wp2)
        
        assert x_range[0] < 0
        assert x_range[1] > 2 * wp2
        assert y_range[0] < 0
        assert y_range[1] > wp1


class TestGenerateAllPlots:
    """Test plot generation."""
    
    def test_generate_all_plots_creates_files(self):
        """Test that all required plots are created."""
        viz_data = create_mock_viz_data()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            results = generate_all_plots(viz_data, output_dir)
            
            # Required plots should exist
            assert (output_dir / "pattern_scanlines.png").exists()
            assert (output_dir / "residuals_bar.png").exists()
            assert (output_dir / "residuals_hist.png").exists()
            assert (output_dir / "rig_3d.png").exists()
            assert (output_dir / "init_vs_final.png").exists()
            assert (output_dir / "chirality_check.png").exists()
            
            # Optional cost trace
            assert (output_dir / "cost_trace.png").exists()
    
    def test_generate_all_plots_with_gt(self):
        """Test plot generation with ground truth."""
        viz_data = create_mock_viz_data(with_gt=True)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            results = generate_all_plots(viz_data, output_dir)
            
            assert (output_dir / "gt_comparison.png").exists()
    
    def test_generate_all_plots_no_cost_history(self):
        """Test graceful handling when cost_history is missing."""
        viz_data = create_mock_viz_data()
        viz_data.cost_history = None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            results = generate_all_plots(viz_data, output_dir)
            
            # Should still create other plots
            assert (output_dir / "pattern_scanlines.png").exists()
            # cost_trace should be None
            assert results["cost_trace"] is None
