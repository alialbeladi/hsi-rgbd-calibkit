"""Diagnostic test for closed-form J/K colinearity.

This script verifies whether the DLT-computed J and K vectors
are colinear with ground-truth J_gt and K_gt at sigma=0.

Based on Li-Wen-Qiu paper equations (19) and (20).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import (
    simulate_views, NoiseConfig, get_default_ground_truth
)
from hsi_rgbd_calib.cal_method.li_wen_qiu.cross_ratio import (
    recover_pattern_points_from_observations,
)


def compute_gt_J(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Compute ground truth J vector from Eq. (19).
    
    J = [r11, r12, r13, t1]^T
    """
    return np.array([R[0, 0], R[0, 1], R[0, 2], T[0]])


def compute_gt_K(R: np.ndarray, T: np.ndarray, f: float, v0: float) -> np.ndarray:
    """Compute ground truth K vector from Eq. (20).
    
    K = [f*r33 - v0*r23,
         -f*r32 + v0*r22,
         f*r11*t2 - f*r21*t1 + v0*r11*t3 - v0*r31*t1,
         -r23,
         r22,
         r11*t3 - r31*t1]
    """
    r11, r12, r13 = R[0, 0], R[0, 1], R[0, 2]
    r21, r22, r23 = R[1, 0], R[1, 1], R[1, 2]
    r31, r32, r33 = R[2, 0], R[2, 1], R[2, 2]
    t1, t2, t3 = T[0], T[1], T[2]
    
    K = np.array([
        f * r33 - v0 * r23,                         # K[0]
        -f * r32 + v0 * r22,                        # K[1]
        f * r11 * t2 - f * r21 * t1 + v0 * r11 * t3 - v0 * r31 * t1,  # K[2]
        -r23,                                       # K[3]
        r22,                                        # K[4]
        r11 * t3 - r31 * t1,                        # K[5]
    ])
    return K


def build_A_J(X_all: np.ndarray, v_all: np.ndarray) -> np.ndarray:
    """Build the A_J matrix for DLT on J.
    
    From Eq. (17):
    X * r11 + Y * r12 + Z * r13 + t1 = 0
    
    So A_J has rows: [X_ij, Y_ij, Z_ij, 1]
    And J = null(A_J).
    """
    n = len(X_all)
    A = np.zeros((n, 4))
    for i in range(n):
        X, Y, Z = X_all[i]
        A[i] = [X, Y, Z, 1.0]
    return A


def build_A_K(X_all: np.ndarray, v_all: np.ndarray) -> np.ndarray:
    """Build the A_K matrix for DLT on K.
    
    From Eq. (18), the equation for v is:
    v = (f*Y + v0*Z + c3) / (Y*K[3] + Z*K[4] + K[5])
    
    Rearranged:
    v * (Y*K[3] + Z*K[4] + K[5]) = Y*K[0] + Z*K[1] + K[2]
    
    Y*K[0] + Z*K[1] + K[2] - v*Y*K[3] - v*Z*K[4] - v*K[5] = 0
    
    So A_K has rows: [Y_ij, Z_ij, 1, -v*Y_ij, -v*Z_ij, -v]
    """
    n = len(X_all)
    A = np.zeros((n, 6))
    for i in range(n):
        X, Y, Z = X_all[i]
        v = v_all[i]
        A[i] = [Y, Z, 1.0, -v * Y, -v * Z, -v]
    return A


def compute_J_from_svd(A_J: np.ndarray) -> np.ndarray:
    """Compute J as the nullspace of A_J using SVD."""
    U, S, Vh = np.linalg.svd(A_J)
    J = Vh[-1, :]  # Last row of Vh is the null vector
    return J


def compute_K_from_svd(A_K: np.ndarray) -> np.ndarray:
    """Compute K as the nullspace of A_K using SVD."""
    U, S, Vh = np.linalg.svd(A_K)
    K = Vh[-1, :]  # Last row of Vh is the null vector
    return K


def check_colinearity(v1: np.ndarray, v2: np.ndarray, name: str) -> bool:
    """Check if two vectors are colinear (up to scale)."""
    # Normalize both
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    # Check if parallel (dot product = ±1)
    dot = np.abs(np.dot(v1_norm, v2_norm))
    
    print(f"\n{name} Colinearity Check:")
    print(f"  v1 (SVD):   {v1}")
    print(f"  v2 (GT):    {v2}")
    print(f"  dot(norm):  {dot:.6f}")
    print(f"  Colinear:   {'YES' if dot > 0.999 else 'NO'}")
    
    # Also show scale factor
    # Find the scale such that v1 ≈ scale * v2
    idx = np.argmax(np.abs(v2))
    if abs(v2[idx]) > 1e-10:
        scale = v1[idx] / v2[idx]
        scaled_gt = scale * v2
        error = np.linalg.norm(v1 - scaled_gt)
        print(f"  Scale:      {scale:.6f}")
        print(f"  Error:      {error:.6e}")
    
    return dot > 0.999


def run_diagnostic():
    """Run the J/K colinearity diagnostic at sigma=0."""
    print("="*60)
    print("J/K COLINEARITY DIAGNOSTIC TEST")
    print("="*60)
    
    pattern = get_default_li_wen_qiu_pattern()
    gt = get_default_ground_truth()
    
    print(f"\nGround Truth Parameters:")
    print(f"  R = \n{gt.R}")
    print(f"  T = {gt.T}")
    print(f"  f = {gt.f}")
    print(f"  v0 = {gt.v0}")
    
    # Compute ground truth J and K
    J_gt = compute_gt_J(gt.R, gt.T)
    K_gt = compute_gt_K(gt.R, gt.T, gt.f, gt.v0)
    
    print(f"\nGround Truth J (Eq. 19): {J_gt}")
    print(f"Ground Truth K (Eq. 20): {K_gt}")
    
    # Simulate noiseless views
    sim = simulate_views(
        n_views=15,
        ground_truth=gt,
        noise_config=NoiseConfig(sigma_v=0.0),
        seed=42,
    )
    
    print(f"\nSimulated {len(sim.views)} noiseless views")
    
    # Recover pattern points and build observation data
    X_all = []
    v_all = []
    
    for view in sim.views:
        # Recover pattern points using cross-ratio
        recovered = recover_pattern_points_from_observations(
            v_obs=list(view.v_observations),
            wp1=pattern.wp1,
            wp2=pattern.wp2,
            pattern_lines=pattern.feature_lines,
        )
        
        # Transform to frame coordinates
        R_j = view.R_frame_pattern
        T_j = view.T_frame_pattern
        
        for i in range(6):
            P_i = np.array([recovered[i][0], recovered[i][1], 0.0])
            X_ij = R_j @ P_i + T_j  # Transform to frame coordinates
            X_all.append(X_ij)
            v_all.append(view.v_observations[i])
    
    X_all = np.array(X_all)
    v_all = np.array(v_all)
    
    print(f"\nBuilt {len(X_all)} observations")
    print(f"X_all range: X=[{X_all[:,0].min():.4f}, {X_all[:,0].max():.4f}], Y=[{X_all[:,1].min():.4f}, {X_all[:,1].max():.4f}], Z=[{X_all[:,2].min():.4f}, {X_all[:,2].max():.4f}]")
    print(f"v_all range: [{v_all.min():.1f}, {v_all.max():.1f}]")
    
    # Build A_J and A_K
    A_J = build_A_J(X_all, v_all)
    A_K = build_A_K(X_all, v_all)
    
    print(f"\nA_J shape: {A_J.shape}")
    print(f"A_K shape: {A_K.shape}")
    
    # Compute SVD singular values to check conditioning
    _, S_J, _ = np.linalg.svd(A_J)
    _, S_K, _ = np.linalg.svd(A_K)
    
    print(f"\nA_J singular values: {S_J}")
    print(f"A_K singular values: {S_K}")
    
    # Compute J and K from SVD
    J_svd = compute_J_from_svd(A_J)
    K_svd = compute_K_from_svd(A_K)
    
    # Check colinearity
    j_colinear = check_colinearity(J_svd, J_gt, "J")
    k_colinear = check_colinearity(K_svd, K_gt, "K")
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    if j_colinear and k_colinear:
        print("\n[OK] Both J and K are colinear with ground truth!")
        print("     => The problem is in Eq. (21)-(26) recovery logic.")
    elif not j_colinear:
        print("\n[FAIL] J is NOT colinear with ground truth!")
        print("       => Problem in A_J construction or frame transform.")
    elif not k_colinear:
        print("\n[FAIL] K is NOT colinear with ground truth!")
        print("       => Problem in A_K construction or column ordering.")
    
    return j_colinear, k_colinear


if __name__ == "__main__":
    run_diagnostic()
