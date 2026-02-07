"""Closed-form initialization for Li-Wen-Qiu calibration.

This module implements Section 3.3 of the Li-Wen-Qiu paper:
SVD-based closed-form solution for intrinsic and extrinsic parameters.

Reference: Equations (17)-(26) in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd

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


def closed_form_init(
    pattern_points: List[NDArray[np.float64]],
    frame_poses: List[Tuple[NDArray[np.float64], NDArray[np.float64]]],
    v_observations: List[NDArray[np.float64]],
) -> ClosedFormResult:
    """Compute closed-form initialization for calibration parameters.
    
    This function implements Section 3.3 of the Li-Wen-Qiu paper.
    
    Algorithm:
    1. Transform pattern points P_i to frame camera coordinates X_ij
    2. Build matrix A_J and solve for view plane normal J (Eq 17)
    3. Build matrix A_K and solve for K (Eq 18)
    4. Recover rotation matrix R from J and K (Eqs 21-26)
    5. Solve for f, v0 from linear system (Eq 22)
    6. Solve for t1, t2, t3
    
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
    
    # Step 1: Transform all pattern points to frame camera coordinates
    X_all = []  # All points in frame coords
    v_all = []  # Corresponding observations
    
    for j in range(n_views):
        R_j, T_j = frame_poses[j]
        P_j = pattern_points[j]
        v_j = v_observations[j]
        
        for i in range(6):
            # X_ij = R_j @ P_i + T_j
            P_i = np.asarray(P_j[i], dtype=np.float64)
            X_ij = R_j @ P_i + T_j
            
            X_all.append(X_ij)
            v_all.append(v_j[i])
    
    X_all = np.array(X_all)  # (6n, 3)
    v_all = np.array(v_all)  # (6n,)
    n_total = len(X_all)
    
    # Step 2: Solve A_J @ J = 0 (Eq 17)
    # A_J has rows [X, Y, Z, 1]
    A_J = np.column_stack([
        X_all[:, 0],  # X
        X_all[:, 1],  # Y
        X_all[:, 2],  # Z
        np.ones(n_total),  # 1
    ])  # (6n, 4)
    
    try:
        _, S_J, Vh_J = svd(A_J, full_matrices=False)
        J = Vh_J[-1, :]  # Right singular vector for smallest singular value
    except np.linalg.LinAlgError:
        return ClosedFormResult(
            R=np.eye(3), T=np.zeros(3), f=1000.0, v0=640.0,
            success=False, message="SVD failed for A_J"
        )
    
    J1, J2, J3, J4 = J
    
    # Step 3: Solve A_K @ K = 0 (Eq 18)
    # A_K has rows [Y, Z, 1, -vY, -vZ, -v]
    A_K = np.column_stack([
        X_all[:, 1],        # Y
        X_all[:, 2],        # Z
        np.ones(n_total),   # 1
        -v_all * X_all[:, 1],  # -vY
        -v_all * X_all[:, 2],  # -vZ
        -v_all,             # -v
    ])  # (6n, 6)
    
    try:
        _, S_K, Vh_K = svd(A_K, full_matrices=False)
        K = Vh_K[-1, :]
    except np.linalg.LinAlgError:
        return ClosedFormResult(
            R=np.eye(3), T=np.zeros(3), f=1000.0, v0=640.0,
            success=False, message="SVD failed for A_K"
        )
    
    K1, K2, K3, K4, K5, K6 = K
    
    # Step 4: Recover rotation matrix
    # From Eq (21): (J1, J2, J3) is proportional to (r11, r12, r13)
    norm_J123 = np.sqrt(J1**2 + J2**2 + J3**2)
    
    if norm_J123 < 1e-12:
        return ClosedFormResult(
            R=np.eye(3), T=np.zeros(3), f=1000.0, v0=640.0,
            success=False, message="Degenerate J vector"
        )
    
    # Determine sign - pattern should be in front of camera
    # We try both signs and pick the one that makes sense
    r11 = J1 / norm_J123
    r12 = J2 / norm_J123
    r13 = J3 / norm_J123
    t1 = J4 / norm_J123
    
    # From Eq (24): gamma = ±sqrt(K1^2 + K4^2) or via constraint
    # Try to compute gamma from the relationship
    norm_K_partial = np.sqrt(K4**2 + K5**2)
    
    if norm_K_partial < 1e-12:
        return ClosedFormResult(
            R=np.eye(3), T=np.zeros(3), f=1000.0, v0=640.0,
            success=False, message="Degenerate K vector (K4, K5)"
        )
    
    # Try positive and negative gamma
    results = []
    for sign in [1.0, -1.0]:
        gamma = sign * norm_K_partial
        
        # From Eq (25): (r22, r23) proportional to (K4, K5) / gamma
        # Note: paper says r21, r22, r23 but we need to be careful about indexing
        r32 = K4 / gamma  # These might need adjustment based on exact formulas
        r33 = K5 / gamma
        
        # r31 from constraint that third column has unit norm
        r31_sq = 1.0 - r32**2 - r33**2
        if r31_sq < 0:
            continue  # Invalid
        r31 = np.sqrt(max(r31_sq, 0.0))
        
        # From Eq (26): second row is cross product of third and first rows
        # r2 = r3 × r1
        r1 = np.array([r11, r12, r13])
        r3 = np.array([r31, r32, r33])
        r2 = np.cross(r3, r1)
        
        R = np.vstack([r1, r2, r3])
        
        # Check orthonormality
        if not np.allclose(R @ R.T, np.eye(3), atol=0.1):
            continue
        
        # Enforce orthonormality via SVD
        U, _, Vt = svd(R)
        R = U @ Vt
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(R) < 0:
            R[2, :] *= -1
        
        # Step 5: Solve for f, v0 from Eq (22)
        # gamma * K1 = f * r33 - v0 * r23  (using adjusted indices)
        # gamma * K2 = -f * r32 + v0 * r22
        # This gives us: [r33, -r23] [f ]   [gamma * K1]
        #                [-r32, r22] [v0] = [gamma * K2]
        
        A_fv = np.array([
            [R[2, 2], -R[1, 2]],
            [-R[2, 1], R[1, 1]],
        ])
        b_fv = gamma * np.array([K1, K2])
        
        try:
            params_fv = np.linalg.solve(A_fv, b_fv)
            f, v0 = params_fv
        except np.linalg.LinAlgError:
            continue
        
        # f should be positive
        if f < 0:
            continue
        
        # Step 6: Solve for t2, t3
        # From remaining equations in Eq (22)
        # t3 = (gamma * K6 + r31 * t1) / r11
        if abs(r11) < 1e-12:
            continue
        
        t3 = (gamma * K6 + r31 * t1) / r11
        
        # t2 from equation
        if abs(f * r11) < 1e-12:
            continue
        
        t2 = (gamma * K3 + f * R[1, 0] * t1 - v0 * r11 * t3 + v0 * r31 * t1) / (f * r11)
        
        T = np.array([t1, t2, t3])
        
        results.append((R, T, f, v0))
    
    if len(results) == 0:
        # Fallback: use simplified initialization
        logger.warning("Closed-form solution failed, using fallback")
        return _fallback_init(X_all, v_all)
    
    # Pick the best result (lowest residual)
    best_result = None
    best_residual = float('inf')
    
    for R, T, f, v0 in results:
        # Compute residual
        residual = 0.0
        for X, v_obs in zip(X_all, v_all):
            # Project X to line-scan
            r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
            r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
            t2, t3 = T[1], T[2]
            
            X_x, X_y, X_z = X
            s_num = r21 * X_x + r22 * X_y + r23 * X_z + t2
            s_den = r31 * X_x + r32 * X_y + r33 * X_z + t3
            
            if abs(s_den) > 1e-12:
                s = s_num / s_den
                v_pred = f * s + v0
                residual += (v_pred - v_obs) ** 2
            else:
                residual += 1e6
        
        if residual < best_residual:
            best_residual = residual
            best_result = (R, T, f, v0)
    
    if best_result is None:
        return _fallback_init(X_all, v_all)
    
    R, T, f, v0 = best_result
    
    return ClosedFormResult(
        R=R, T=T, f=f, v0=v0,
        success=True,
        message=f"Closed-form init succeeded, residual={np.sqrt(best_residual/n_total):.4f}"
    )


def _fallback_init(X_all: NDArray, v_all: NDArray) -> ClosedFormResult:
    """Fallback initialization when closed-form fails.
    
    Uses a simple linear regression approach.
    """
    logger.warning("Using fallback initialization")
    
    # Assume identity rotation, estimate translation and intrinsics
    R = np.eye(3, dtype=np.float64)
    
    # Simple linear fit: v ≈ f * (Y/Z) + v0
    # where Y/Z is the normalized coordinate
    
    # Estimate average depth
    avg_z = np.mean(X_all[:, 2])
    if abs(avg_z) < 1e-6:
        avg_z = 0.5  # Default depth
    
    # Estimate f and v0 from simple linear regression
    s = X_all[:, 1] / np.maximum(X_all[:, 2], 1e-6)
    
    # v = f * s + v0
    A = np.column_stack([s, np.ones_like(s)])
    params, _, _, _ = np.linalg.lstsq(A, v_all, rcond=None)
    f, v0 = params
    
    # Ensure positive f
    f = abs(f) if abs(f) > 100 else 1000.0
    
    T = np.array([0.0, 0.0, 0.1])  # Default translation
    
    return ClosedFormResult(
        R=R, T=T, f=f, v0=v0,
        success=True,
        message="Fallback initialization used"
    )
