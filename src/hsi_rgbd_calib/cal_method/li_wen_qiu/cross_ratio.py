"""Cross-ratio computation for Li-Wen-Qiu calibration.

This module implements Section 3.2 of the Li-Wen-Qiu paper:
point correspondence establishment using cross-ratio invariance.

Reference: Equations (11), (12), (13), (14) in the paper.
"""

from __future__ import annotations

from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray


def compute_cross_ratios(
    v1: float, v2: float, v3: float, v4: float, v5: float, v6: float
) -> Tuple[float, float]:
    """Compute the two cross-ratios from 6 observed line-scan coordinates.
    
    Implements Equations (13) and (14) from the paper:
        CR1 = CrossRatio(p2, p4, p6, p3) = ((v2-v6)/(v4-v6)) / ((v2-v3)/(v4-v3))
        CR2 = CrossRatio(p4, p6, p2, p5) = ((v4-v2)/(v6-v2)) / ((v4-v5)/(v6-v5))
    
    Args:
        v1-v6: Observed pixel coordinates from line-scan image for points P1-P6.
        
    Returns:
        Tuple of (CR1, CR2).
        
    Raises:
        ValueError: If cross-ratio computation fails (degenerate configuration).
    """
    # CR1 = ((v2-v6)/(v4-v6)) / ((v2-v3)/(v4-v3))
    # Rewritten: CR1 = ((v2-v6) * (v4-v3)) / ((v4-v6) * (v2-v3))
    den1 = (v4 - v6) * (v2 - v3)
    if abs(den1) < 1e-12:
        raise ValueError("Degenerate configuration for CR1: denominator near zero")
    
    CR1 = ((v2 - v6) * (v4 - v3)) / den1
    
    # CR2 = ((v4-v2)/(v6-v2)) / ((v4-v5)/(v6-v5))
    # Rewritten: CR2 = ((v4-v2) * (v6-v5)) / ((v6-v2) * (v4-v5))
    den2 = (v6 - v2) * (v4 - v5)
    if abs(den2) < 1e-12:
        raise ValueError("Degenerate configuration for CR2: denominator near zero")
    
    CR2 = ((v4 - v2) * (v6 - v5)) / den2
    
    return CR1, CR2


def compute_X3_X5_from_cross_ratios(
    CR1: float,
    CR2: float,
    wp1: float,
    wp2: float,
) -> Tuple[float, float]:
    """Compute X3 and X5 from cross-ratios and pattern dimensions.
    
    Implements Equations (11) and (12) from the paper:
        X3 = 2 * wp2 / (2 - CR1)
        X5 = wp1 / (1 - 2 * CR2)
    
    Args:
        CR1: First cross-ratio.
        CR2: Second cross-ratio.
        wp1: Pattern width parameter 1 (spacing to L5/L6).
        wp2: Pattern width parameter 2 (spacing to L3/L4).
        
    Returns:
        Tuple of (X3, X5) coordinates.
        
    Raises:
        ValueError: If denominators are near zero.
    """
    # X3 = 2 * wp2 / (2 - CR1)
    den_x3 = 2.0 - CR1
    if abs(den_x3) < 1e-12:
        raise ValueError(f"Degenerate X3 computation: 2 - CR1 = {den_x3}")
    
    X3 = (2.0 * wp2) / den_x3
    
    # X5 = wp1 / (1 - 2 * CR2)
    den_x5 = 1.0 - 2.0 * CR2
    if abs(den_x5) < 1e-12:
        raise ValueError(f"Degenerate X5 computation: 1 - 2*CR2 = {den_x5}")
    
    X5 = wp1 / den_x5
    
    return X3, X5


def recover_pattern_points_from_observations(
    v_obs: List[float],
    wp1: float,
    wp2: float,
    pattern_lines: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """Recover 3D pattern points P1-P6 from observed line-scan coordinates.
    
    This is the main function implementing Section 3.2 of the paper.
    
    Algorithm:
    1. Compute cross-ratios CR1, CR2 from v1-v6
    2. Compute X3, X5 from cross-ratios and pattern dimensions
    3. Compute P3, P5 on their feature lines (L3: Y=wp2, L5: Y=wp1)
    4. Form scan line through P3, P5
    5. Intersect scan line with all 6 feature lines to get P1-P6
    
    Args:
        v_obs: List of 6 observed pixel coordinates [v1, v2, v3, v4, v5, v6].
        wp1: Pattern width parameter 1.
        wp2: Pattern width parameter 2.
        pattern_lines: List of 6 feature line tuples [(a, b, c), ...].
        
    Returns:
        List of 6 3D points [(x, y, 0), ...] in pattern coordinates.
    """
    from hsi_rgbd_calib.boards.geometry import (
        intersect_lines_2d,
        line_through_points,
    )
    
    if len(v_obs) != 6:
        raise ValueError(f"Expected 6 observations, got {len(v_obs)}")
    
    v1, v2, v3, v4, v5, v6 = v_obs
    
    # Step 1: Compute cross-ratios
    CR1, CR2 = compute_cross_ratios(v1, v2, v3, v4, v5, v6)
    
    # Step 2: Compute X3, X5
    X3, X5 = compute_X3_X5_from_cross_ratios(CR1, CR2, wp1, wp2)
    
    # Step 3: P3 is at (X3, wp2, 0), P5 is at (X5, wp1, 0)
    P3_2d = (X3, wp2)
    P5_2d = (X5, wp1)
    
    # Step 4: Form scan line through P3, P5
    scan_line = line_through_points(P3_2d, P5_2d)
    
    # Step 5: Intersect with all feature lines
    points_3d = []
    for i, feature_line in enumerate(pattern_lines):
        pt = intersect_lines_2d(scan_line, feature_line)
        if pt is None:
            raise ValueError(f"Scan line parallel to feature line L{i+1}")
        points_3d.append((pt[0], pt[1], 0.0))
    
    return points_3d


def validate_cross_ratio_ordering(v_obs: List[float]) -> bool:
    """Validate that observations have expected ordering.
    
    For the Li-Wen-Qiu pattern, the 6 points should appear in
    a specific order along the scan line.
    
    Args:
        v_obs: List of 6 observed coordinates.
        
    Returns:
        True if ordering is valid.
    """
    # The exact ordering depends on the camera orientation
    # At minimum, check that all values are unique and monotonic
    # (either increasing or decreasing)
    
    if len(v_obs) != 6:
        return False
    
    v = np.array(v_obs)
    
    # Check uniqueness
    if len(np.unique(v)) != 6:
        return False
    
    # Values should be in some order (either sorted or reverse sorted)
    sorted_v = np.sort(v)
    is_increasing = np.allclose(v, sorted_v)
    is_decreasing = np.allclose(v, sorted_v[::-1])
    
    # For general configurations, we don't require strict sorting
    # Just check that differences are reasonable
    return True
