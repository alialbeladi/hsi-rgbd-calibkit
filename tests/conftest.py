"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any

import pytest
import numpy as np
from scipy.spatial.transform import Rotation

# Add src to path for development testing
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_session_path() -> Path:
    """Path to the sample session directory."""
    return Path(__file__).parent.parent / "datasets" / "sample_session"


@pytest.fixture
def config_path() -> Path:
    """Path to the config directory."""
    return Path(__file__).parent.parent / "configs"


@pytest.fixture
def sample_transform() -> np.ndarray:
    """Sample 4x4 transformation matrix with a proper orthogonal rotation."""
    # Create a valid rotation from Euler angles (small rotations like a camera rig)
    rot = Rotation.from_euler('xyz', [0.01, 0.02, -0.01])  # ~1 degree rotations
    R = rot.as_matrix()
    t = np.array([0.052, -0.018, 0.095])
    
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@pytest.fixture
def identity_transform() -> np.ndarray:
    """Identity 4x4 transformation matrix."""
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def sample_artifact_data() -> Dict[str, Any]:
    """Sample calibration artifact data."""
    return {
        "artifact_version": "1.0",
        "created_at": "2024-01-15T10:30:00.123456",
        "frames": {
            "rig": {
                "name": "rig",
                "description": "Rig reference frame",
                "convention": "X: Right, Y: Down, Z: Forward",
            },
            "oak_rgb": {
                "name": "oak_rgb",
                "description": "OAK-D RGB camera optical frame",
                "convention": "X: Right, Y: Down, Z: Forward",
            },
            "oak_depth": {
                "name": "oak_depth",
                "description": "OAK-D depth camera optical frame",
                "convention": "X: Right, Y: Down, Z: Forward",
            },
            "hsi": {
                "name": "hsi",
                "description": "HSI line-scan camera slit frame",
                "convention": "X: Along slit, Y: Cross-track, Z: Forward",
            },
        },
        "oak": {
            "rgb_intrinsics": {
                "fx": 1000.0,
                "fy": 1000.0,
                "cx": 640.0,
                "cy": 360.0,
                "width": 1280,
                "height": 720,
                "distortion_model": "radtan",
                "distortion_coeffs": [0.05, -0.08, 0.0, 0.0, 0.02],
            },
            "depth_alignment": {
                "aligned_to_rgb": True,
                "output_width": 1280,
                "output_height": 720,
                "stereo_mode": "standard",
                "subpixel_enabled": True,
                "lr_check_enabled": True,
                "notes": "",
            },
            "device_serial": "14110250AB",
            "device_model": "OAK-D S2",
        },
        "hsi": {
            "slit_intrinsics": {
                "focal_length_slit": 1150.0,
                "principal_point_u0": 640.0,
                "slit_width": 1280,
                "distortion_model": "none",
                "distortion_coeffs": [],
            },
            "wavelengths_nm": None,
            "wavelengths_file": None,
            "num_bands": 224,
            "integration_time_us": 5000.0,
        },
        "extrinsics": {
            "T_oakrgb_hsi": [
                [0.9998, 0.0175, -0.0087, 0.052],
                [-0.0174, 0.9998, 0.0052, -0.018],
                [0.0088, -0.0050, 0.9999, 0.095],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "uncertainty": {
                "translation_std_m": 0.002,
                "rotation_std_deg": 0.15,
            },
            "method": "li_wen_qiu",
            "notes": "",
        },
        "validation_summary": {
            "reprojection_rmse_px": 0.423,
            "median_abs_error_px": 0.356,
            "max_error_px": 1.205,
            "repeatability_translation_mm": None,
            "repeatability_rotation_deg": None,
            "num_validation_points": 156,
            "slow_motion_check_passed": True,
            "slow_motion_notes": "",
        },
        "session_path": "/path/to/session",
        "notes": "",
    }


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
