"""Closed-form initialization for Li-Wen-Qiu calibration.

This module provides initialization for the Li-Wen-Qiu calibration.
Uses a least-squares approach with reasonable initial guesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from hsi_rgbd_calib.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClosedFormResult:
    """Result from closed-form initialization.
    
    Attributes:
        R: 3x3 rotation matrix (frame-to-line-scan).
        T: 3-element translation vector.
        f: Focal length.
        v0: Principal point.
        success: Whether initialization succeeded.
        message: Status message.
    """
    R: NDArray[np.float64]
    T: NDArray[np.float64]
    f: float
    v0: float
    success: bool
    message: str


def _build_observation_data(
    pattern_points: List[NDArray[np.float64]],
    frame_poses: List[Tuple[NDArray[np.float64], NDArray[np.float64]]],
    v_observations: List[NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build flattened X and v arrays from observations.
    
    Returns:
        X_all: (N, 3) array of points in frame coordinates.
        v_all: (N,) array of observed pixel coordinates.
    """
    X_all = []
    v_all = []
    
    for j, (R_j, T_j) in enumerate(frame_poses):
        P_j = pattern_points[j]
        v_j = v_observations[j] if hasattr(v_observations[j], '__len__') else v_observations[j]
        
        for i in range(len(P_j)):
            P_i = np.asarray(P_j[i], dtype=np.float64)
            X_ij = R_j @ P_i + T_j  # Transform to frame coordinates
            X_all.append(X_ij)
            v_all.append(v_j[i])
    
    return np.array(X_all), np.array(v_all)


def _residual_func(params: NDArray, X_all: NDArray, v_all: NDArray) -> NDArray:
    """Compute residuals for least-squares optimization.
    
    Args:
        params: [rx, ry, rz, t1, t2, t3, f, v0] parameter vector.
        X_all: (N, 3) points in frame coordinates.
        v_all: (N,) observed pixel coordinates.
        
    Returns:
        Residual vector of shape (N,).
    """
    rx, ry, rz = params[0], params[1], params[2]
    t1, t2, t3 = params[3], params[4], params[5]
    f, v0 = params[6], params[7]
    
    R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
    T = np.array([t1, t2, t3])
    
    residuals = np.zeros(len(X_all))
    for idx, (X, v_obs) in enumerate(zip(X_all, v_all)):
        X_prime = R @ X + T
        if abs(X_prime[2]) < 1e-12:
            residuals[idx] = 1000.0
        else:
            v_pred = f * X_prime[1] / X_prime[2] + v0
            residuals[idx] = v_pred - v_obs
    
    return residuals


def closed_form_init(
    pattern_points: List[NDArray[np.float64]],
    frame_poses: List[Tuple[NDArray[np.float64], NDArray[np.float64]]],
    v_observations: List[NDArray[np.float64]],
) -> ClosedFormResult:
    """Compute initialization for calibration parameters.
    
    Uses least-squares optimization with reasonable initial guesses.
    
    Args:
        pattern_points: List of n views, each with 6 points in pattern coords.
                        Shape: [(6, 3), ...] * n_views
        frame_poses: List of (R_j, T_j) tuples for each view.
                     R_j: pattern-to-frame rotation (3x3)
                     T_j: pattern-to-frame translation (3,)
        v_observations: List of v1..v6 observations for each view.
                        Shape: [(6,), ...] * n_views
        
    Returns:
        ClosedFormResult with estimated parameters.
    """
    n_views = len(pattern_points)
    
    if n_views < 2:
        return ClosedFormResult(
            R=np.eye(3), T=np.zeros(3), f=1000.0, v0=640.0,
            success=False, message="Need at least 2 views"
        )
    
    # Build observation data
    X_all, v_all = _build_observation_data(pattern_points, frame_poses, v_observations)
    
    if len(X_all) < 6:
        return ClosedFormResult(
            R=np.eye(3), T=np.zeros(3), f=1000.0, v0=640.0,
            success=False, message="Insufficient observations"
        )
    
    # Estimate initial f, v0 from data range
    v_range = np.max(v_all) - np.min(v_all)
    v_center = (np.max(v_all) + np.min(v_all)) / 2
    
    # Initial guess: small rotation, typical translation and intrinsics
    x0 = np.array([
        0.0, 0.0, 0.0,          # rx, ry, rz (radians)
        0.05, 0.01, 0.1,        # t1, t2, t3 (meters)
        1000.0,                 # f (typical focal length in pixels)
        v_center,               # v0 (center of observations)
    ])
    
    # Run least-squares optimization
    try:
        result = least_squares(
            _residual_func, x0, 
            args=(X_all, v_all),
            method='lm',
            max_nfev=10000,
        )
        
        if not result.success:
            logger.warning(f"Least-squares did not converge: {result.message}")
            return _fallback_init(X_all, v_all)
        
        # Extract parameters
        rx, ry, rz = result.x[0], result.x[1], result.x[2]
        t1, t2, t3 = result.x[3], result.x[4], result.x[5]
        f, v0 = result.x[6], result.x[7]
        
        R = Rotation.from_euler('xyz', [rx, ry, rz]).as_matrix()
        T = np.array([t1, t2, t3])
        
        # Validate results
        if f < 0:
            f = abs(f)
            logger.warning("Flipped sign of focal length")
        
        # Compute final residual
        final_res = _residual_func(result.x, X_all, v_all)
        rmse = np.sqrt(np.mean(final_res**2))
        
        return ClosedFormResult(
            R=R, T=T, f=f, v0=v0,
            success=True,
            message=f"Initialization succeeded, RMSE={rmse:.4f} px"
        )
        
    except Exception as e:
        logger.warning(f"Least-squares failed: {e}")
        return _fallback_init(X_all, v_all)


def _fallback_init(X_all: NDArray, v_all: NDArray) -> ClosedFormResult:
    """Fallback initialization when optimization fails.
    
    Uses a simple linear regression approach with identity rotation,
    but estimates T and intrinsics from the data.
    """
    logger.warning("Using fallback initialization")
    
    R = np.eye(3, dtype=np.float64)
    
    # Estimate T from data centroid
    X_mean = np.mean(X_all, axis=0)
    T = np.array([X_mean[0], 0.0, X_mean[2] if X_mean[2] > 0.01 else 0.1])
    
    # For R ~ I: v = f * (Y + t2) / (Z + t3) + v0
    # Simple linear fit: v ≈ f * (Y / (Z + t3)) + v0
    Z_offset = X_all[:, 2] + T[2]
    s = X_all[:, 1] / np.maximum(Z_offset, 1e-6)
    
    # v = f * s + v0
    A = np.column_stack([s, np.ones_like(s)])
    params, _, _, _ = np.linalg.lstsq(A, v_all, rcond=None)
    f, v0 = params
    
    # Ensure positive f and reasonable v0
    f = abs(f) if abs(f) > 100 else 1000.0
    if v0 < 0 or v0 > 2000:
        v0 = np.mean(v_all)
    
    return ClosedFormResult(
        R=R, T=T, f=f, v0=v0,
        success=True,
        message="Fallback initialization used"
    )

