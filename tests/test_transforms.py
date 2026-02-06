"""Tests for SE(3) transformation utilities."""

from __future__ import annotations

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from hsi_rgbd_calib.common.transforms import (
    compose,
    invert,
    apply_transform,
    make_transform,
    decompose_transform,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    is_valid_rotation_matrix,
    is_valid_transform,
)


class TestCompose:
    """Tests for compose function."""
    
    def test_compose_identity(self, identity_transform):
        """Composing with identity should not change the transform."""
        T = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        
        result = compose(identity_transform, T)
        assert_array_almost_equal(result, T)
        
        result = compose(T, identity_transform)
        assert_array_almost_equal(result, T)
    
    def test_compose_translations(self):
        """Test composing pure translations."""
        T1 = make_transform(np.eye(3), [1, 0, 0])
        T2 = make_transform(np.eye(3), [0, 2, 0])
        
        result = compose(T1, T2)
        expected_t = [1, 2, 0]
        
        assert_array_almost_equal(result[:3, 3], expected_t)
    
    def test_compose_invalid_shape(self):
        """Test that invalid shapes raise errors."""
        with pytest.raises(ValueError):
            compose(np.eye(3), np.eye(4))


class TestInvert:
    """Tests for invert function."""
    
    def test_invert_identity(self, identity_transform):
        """Inverting identity should give identity."""
        result = invert(identity_transform)
        assert_array_almost_equal(result, identity_transform)
    
    def test_invert_compose_is_identity(self, sample_transform):
        """T @ T^-1 should give identity."""
        T_inv = invert(sample_transform)
        result = compose(sample_transform, T_inv)
        
        assert_array_almost_equal(result, np.eye(4), decimal=6)
    
    def test_invert_pure_translation(self):
        """Test inverting pure translation."""
        T = make_transform(np.eye(3), [1, 2, 3])
        T_inv = invert(T)
        
        expected_t = [-1, -2, -3]
        assert_array_almost_equal(T_inv[:3, 3], expected_t)


class TestApplyTransform:
    """Tests for apply_transform function."""
    
    def test_apply_identity(self, identity_transform):
        """Applying identity should not change points."""
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        
        result = apply_transform(identity_transform, points)
        assert_array_almost_equal(result, points)
    
    def test_apply_translation(self):
        """Test applying pure translation."""
        T = make_transform(np.eye(3), [1, 2, 3])
        points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        
        result = apply_transform(T, points)
        expected = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float64)
        
        assert_array_almost_equal(result, expected)
    
    def test_apply_single_point(self, sample_transform):
        """Test applying transform to a single point."""
        point = np.array([1, 0, 0], dtype=np.float64)
        
        result = apply_transform(sample_transform, point)
        
        assert result.shape == (1, 3)


class TestMakeDecompose:
    """Tests for make_transform and decompose_transform."""
    
    def test_roundtrip(self):
        """Test make/decompose roundtrip."""
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        t = np.array([1, 2, 3], dtype=np.float64)
        
        T = make_transform(R, t)
        R_out, t_out = decompose_transform(T)
        
        assert_array_almost_equal(R, R_out)
        assert_array_almost_equal(t, t_out)


class TestEulerConversion:
    """Tests for Euler angle conversion."""
    
    def test_identity_rotation(self):
        """Identity rotation should give zero angles."""
        R = np.eye(3, dtype=np.float64)
        
        angles = rotation_matrix_to_euler(R)
        
        assert_array_almost_equal(angles, [0, 0, 0], decimal=10)
    
    def test_euler_roundtrip(self):
        """Test Euler angle conversion roundtrip."""
        angles = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        
        R = euler_to_rotation_matrix(angles)
        angles_out = rotation_matrix_to_euler(R)
        
        assert_array_almost_equal(angles, angles_out, decimal=10)


class TestQuaternionConversion:
    """Tests for quaternion conversion."""
    
    def test_identity_rotation(self):
        """Identity rotation should give unit quaternion."""
        R = np.eye(3, dtype=np.float64)
        
        q = rotation_matrix_to_quaternion(R)
        
        # Unit quaternion: [1, 0, 0, 0]
        assert_array_almost_equal(q, [1, 0, 0, 0], decimal=10)
    
    def test_quaternion_roundtrip(self):
        """Test quaternion conversion roundtrip."""
        R = euler_to_rotation_matrix([0.1, 0.2, 0.3])
        
        q = rotation_matrix_to_quaternion(R)
        R_out = quaternion_to_rotation_matrix(q)
        
        assert_array_almost_equal(R, R_out, decimal=10)


class TestValidation:
    """Tests for validation functions."""
    
    def test_valid_rotation(self):
        """Test that valid rotation matrices pass validation."""
        R = euler_to_rotation_matrix([0.1, 0.2, 0.3])
        
        assert is_valid_rotation_matrix(R) is True
    
    def test_invalid_rotation_determinant(self):
        """Test that reflection matrices fail validation."""
        R = np.diag([1, 1, -1])  # Reflection
        
        assert is_valid_rotation_matrix(R) is False
    
    def test_valid_transform(self, sample_transform):
        """Test that valid transforms pass validation."""
        assert is_valid_transform(sample_transform) is True
    
    def test_invalid_transform_bottom_row(self):
        """Test that transforms with invalid bottom row fail."""
        T = np.eye(4)
        T[3, 0] = 1  # Invalid bottom row
        
        assert is_valid_transform(T) is False
