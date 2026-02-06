"""Tests for artifact schema validation."""

from __future__ import annotations

import pytest
from typing import Dict, Any

from hsi_rgbd_calib.io.artifact_schema import (
    validate_artifact_dict,
    CalibrationArtifact,
    validate_artifact,
)


class TestValidateArtifactDict:
    """Tests for validate_artifact_dict function."""
    
    def test_valid_artifact(self, sample_artifact_data: Dict[str, Any]):
        """Test that a valid artifact passes validation."""
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_missing_artifact_version(self, sample_artifact_data: Dict[str, Any]):
        """Test that missing artifact_version is detected."""
        del sample_artifact_data["artifact_version"]
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("artifact_version" in e for e in errors)
    
    def test_missing_created_at(self, sample_artifact_data: Dict[str, Any]):
        """Test that missing created_at is detected."""
        del sample_artifact_data["created_at"]
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("created_at" in e for e in errors)
    
    def test_invalid_timestamp(self, sample_artifact_data: Dict[str, Any]):
        """Test that invalid timestamp is detected."""
        sample_artifact_data["created_at"] = "not-a-timestamp"
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("timestamp" in e.lower() for e in errors)
    
    def test_missing_frames(self, sample_artifact_data: Dict[str, Any]):
        """Test that missing frames is detected."""
        del sample_artifact_data["frames"]
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("frames" in e for e in errors)
    
    def test_missing_frame_definitions(self, sample_artifact_data: Dict[str, Any]):
        """Test that missing frame definitions are detected."""
        sample_artifact_data["frames"] = {"rig": sample_artifact_data["frames"]["rig"]}
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("frame" in e.lower() for e in errors)
    
    def test_missing_oak_intrinsics(self, sample_artifact_data: Dict[str, Any]):
        """Test that missing RGB intrinsics is detected."""
        del sample_artifact_data["oak"]["rgb_intrinsics"]
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("rgb_intrinsics" in e for e in errors)
    
    def test_missing_intrinsic_field(self, sample_artifact_data: Dict[str, Any]):
        """Test that missing intrinsic fields are detected."""
        del sample_artifact_data["oak"]["rgb_intrinsics"]["fx"]
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("fx" in e for e in errors)
    
    def test_missing_extrinsics(self, sample_artifact_data: Dict[str, Any]):
        """Test that missing extrinsics is detected."""
        del sample_artifact_data["extrinsics"]
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("extrinsics" in e for e in errors)
    
    def test_invalid_transform_matrix(self, sample_artifact_data: Dict[str, Any]):
        """Test that invalid transform matrix is detected."""
        sample_artifact_data["extrinsics"]["T_oakrgb_hsi"] = [[1, 0], [0, 1]]
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("4x4" in e for e in errors)
    
    def test_missing_slit_intrinsics(self, sample_artifact_data: Dict[str, Any]):
        """Test that missing HSI slit intrinsics is detected."""
        del sample_artifact_data["hsi"]["slit_intrinsics"]
        
        is_valid, errors = validate_artifact_dict(sample_artifact_data)
        
        assert is_valid is False
        assert any("slit_intrinsics" in e for e in errors)


class TestCalibrationArtifact:
    """Tests for CalibrationArtifact class."""
    
    def test_to_dict_roundtrip(self, sample_artifact_data: Dict[str, Any]):
        """Test that to_dict produces validatable output."""
        from hsi_rgbd_calib.io.export import _dict_to_artifact
        
        artifact = _dict_to_artifact(sample_artifact_data)
        data = artifact.to_dict()
        
        is_valid, errors = validate_artifact_dict(data)
        
        assert is_valid is True, f"Validation failed: {errors}"
