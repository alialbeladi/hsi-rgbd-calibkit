"""Tests for calibration method functionality."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np

from hsi_rgbd_calib.cal_method.interface import (
    CalibrationConfig,
    CalibrationResult,
    estimate_calibration,
)
from hsi_rgbd_calib.cal_method.stub_backend import (
    StubBackend,
    load_precomputed_results,
    save_precomputed_results,
)
from hsi_rgbd_calib.io.session import load_session


class TestCalibrationConfig:
    """Tests for CalibrationConfig."""
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "backend": "stub",
            "method_name": "li_wen_qiu",
            "max_iterations": 50,
        }
        
        config = CalibrationConfig.from_dict(data)
        
        assert config.backend == "stub"
        assert config.method_name == "li_wen_qiu"
        assert config.max_iterations == 50
    
    def test_from_yaml(self, config_path: Path):
        """Test loading config from YAML file."""
        config = CalibrationConfig.from_yaml(config_path / "li_wen_qiu.yaml")
        
        assert config.backend == "stub"
        assert config.method_name == "li_wen_qiu"
    
    def test_defaults(self):
        """Test default values."""
        config = CalibrationConfig()
        
        assert config.backend == "stub"
        assert config.use_ransac is True
        assert config.min_correspondences == 20


class TestStubBackend:
    """Tests for StubBackend."""
    
    def test_load_precomputed_results(self, sample_session_path: Path):
        """Test loading precomputed calibration results."""
        import yaml
        
        output_path = sample_session_path / "cal_method" / "cal_method_output.yaml"
        with open(output_path) as f:
            data = yaml.safe_load(f)
        
        result = load_precomputed_results(data)
        
        assert isinstance(result, CalibrationResult)
        assert result.success is True
        assert result.T_oakrgb_hsi.shape == (4, 4)
        assert result.hsi_intrinsics.slit_width == 1280
    
    def test_stub_backend_estimate(self, sample_session_path: Path):
        """Test StubBackend.estimate with real session."""
        session = load_session(sample_session_path)
        config = CalibrationConfig(backend="stub")
        backend = StubBackend()
        
        result = backend.estimate(session, config)
        
        assert result.success is True
        assert result.reprojection_error_rmse > 0
        assert result.T_oakrgb_hsi.shape == (4, 4)
    
    def test_stub_backend_no_precomputed(self, sample_session_path: Path):
        """Test StubBackend raises error when no precomputed results."""
        from hsi_rgbd_calib.io.session import SessionData
        
        # Create session without precomputed results
        session = SessionData(
            session_path=sample_session_path,
            session_yaml={},
            oak_intrinsics={},
            oak_depth_config={},
            hsi_metadata={},
            cal_method_output=None,  # No precomputed
        )
        
        config = CalibrationConfig(backend="stub")
        backend = StubBackend()
        
        with pytest.raises(ValueError, match="precomputed"):
            backend.estimate(session, config)
    
    def test_save_load_roundtrip(self, temp_output_dir: Path):
        """Test saving and loading precomputed results."""
        from hsi_rgbd_calib.cal_method.interface import HsiSlitIntrinsicsResult
        
        result = CalibrationResult(
            T_oakrgb_hsi=np.eye(4),
            hsi_intrinsics=HsiSlitIntrinsicsResult(
                focal_length_slit=1000.0,
                principal_point_u0=640.0,
                slit_width=1280,
            ),
            reprojection_error_rmse=0.5,
            reprojection_error_median=0.4,
            reprojection_error_max=1.0,
            num_correspondences=100,
            num_inliers=95,
        )
        
        output_path = temp_output_dir / "test_output.yaml"
        save_precomputed_results(result, output_path)
        
        import yaml
        with open(output_path) as f:
            data = yaml.safe_load(f)
        
        loaded = load_precomputed_results(data)
        
        assert loaded.reprojection_error_rmse == result.reprojection_error_rmse
        assert loaded.hsi_intrinsics.slit_width == result.hsi_intrinsics.slit_width


class TestEstimateCalibration:
    """Tests for estimate_calibration function."""
    
    def test_estimate_with_stub(self, sample_session_path: Path):
        """Test estimate_calibration with stub backend."""
        session = load_session(sample_session_path)
        config = CalibrationConfig(backend="stub")
        
        result = estimate_calibration(session, config)
        
        assert result.success is True
        assert result.method == "li_wen_qiu"
    
    def test_estimate_with_python_backend(self, sample_session_path: Path):
        """Test estimate_calibration with Python backend."""
        session = load_session(sample_session_path)
        config = CalibrationConfig(backend="python")
        
        # Python backend should run (with placeholder implementation)
        result = estimate_calibration(session, config)
        
        assert result.success is True
    
    def test_invalid_backend(self, sample_session_path: Path):
        """Test that invalid backend raises error."""
        session = load_session(sample_session_path)
        config = CalibrationConfig(backend="invalid")
        
        with pytest.raises(ValueError, match="backend"):
            estimate_calibration(session, config)
