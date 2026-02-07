"""Test cross-ratio with NEW pattern indexing.

New pattern (matching paper's Table 2):
  L1: Y = 0
  L2: Y = wp2
  L3: Y = wp1
  L4: X = Y
  L5: X - Y = wp2
  L6: X - Y = wp1

So:
  P1 on L1: Y=0
  P2 on L2: Y=wp2  
  P3 on L3: Y=wp1
  P4 on L4: X=Y diagonal
  P5 on L5: X-Y=wp2 diagonal
  P6 on L6: X-Y=wp1 diagonal
"""

import numpy as np
from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern
from hsi_rgbd_calib.boards.geometry import intersect_lines_2d, line_through_points, cross_ratio_1d
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import simulate_views, NoiseConfig
from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    compute_scan_line_in_pattern, 
    compute_transform_pattern_to_linescan
)

# Get pattern
pattern = get_default_li_wen_qiu_pattern()
wp1, wp2 = pattern.wp1, pattern.wp2

print("=" * 60)
print("NEW PATTERN GEOMETRY")
print("=" * 60)
for name, fl in zip(pattern.line_names, pattern.feature_lines):
    print(f"  {name}: ({fl[0]:.3f})*X + ({fl[1]:.3f})*Y + ({fl[2]:.3f}) = 0")
print()

# Simulate one view
noise = NoiseConfig(sigma_v=0.0)
sim_result = simulate_views(n_views=1, noise_config=noise, seed=42)
gt = sim_result.ground_truth
view = sim_result.views[0]

# Get TRUE scan line
R0, T0 = compute_transform_pattern_to_linescan(
    view.R_frame_pattern, view.T_frame_pattern, gt.R, gt.T
)
scan_line = compute_scan_line_in_pattern(R0, T0)

print("TRUE PATTERN POINTS (from scan line):")
P_true = []
for i, fl in enumerate(pattern.feature_lines):
    pt = intersect_lines_2d(scan_line, fl)
    if pt:
        P_true.append(pt)
        print(f"  P{i+1}: X={pt[0]:.6f}, Y={pt[1]:.6f}")
    else:
        P_true.append((0, 0))
print()

# v observations
v = view.v_observations
print("OBSERVED v VALUES:")
for i in range(6):
    print(f"  v{i+1} = {v[i]:.4f}")
print()

# The cross-ratio is projective invariant
# So CR on pattern points = CR on observed v values
# 
# With new indexing:
# P1 (on L1, Y=0), P2 (on L2, Y=wp2), P3 (on L3, Y=wp1) are on horizontal lines
# P4 (on L4, X=Y), P5 (on L5, X-Y=wp2), P6 (on L6, X-Y=wp1) are on diagonal lines
#
# For the paper's approach, we need cross-ratios that relate to KNOWN geometry
# 
# Key insight: The paper uses cross-ratios of points P2, P3, P4, P6 (from old indexing)
# In OLD indexing: L3 was Y=wp2, L5 was Y=wp1
# In NEW indexing: L2 is Y=wp2, L3 is Y=wp1
#
# So the paper's "P3" (on Y=wp2 line) is now our P2
# And the paper's "P5" (on Y=wp1 line) is now our P3

print("MAPPING OLD -> NEW INDEXING:")
print("  Paper's P3 (Y=wp2 line) -> New P2")
print("  Paper's P5 (Y=wp1 line) -> New P3")
print()

# Paper's CR1 formula (Eq 13) uses P2, P4, P6, P3 (old indexing)
# P2 (old) = on L2 (X=Y diagonal) -> New P4
# P3 (old) = on L3 (Y=wp2) -> New P2
# P4 (old) = on L4 (X-Y=wp2 diagonal) -> New P5
# P6 (old) = on L6 (X-Y=wp1 diagonal) -> New P6

# So paper's CR1(P2, P4, P6, P3) becomes CR1(P4, P5, P6, P2) in new indexing!

# Let's compute:
print("CROSS-RATIO MAPPING:")
print("  Paper CR1(P2, P4, P6, P3) -> New CR1(P4, P5, P6, P2)")
print("  Paper CR2(P4, P6, P2, P5) -> New CR2(P5, P6, P4, P3)")
print()

# Compute using v values with NEW mapping
# New CR1 = CR(v4, v5, v6, v2)
def cr(a, b, c, d):
    """Cross-ratio (a,b;c,d) = ((a-c)/(b-c)) / ((a-d)/(b-d))"""
    return ((a - c) * (b - d)) / ((b - c) * (a - d))

CR1_new = cr(v[3], v[4], v[5], v[1])  # P4, P5, P6, P2
CR2_new = cr(v[4], v[5], v[3], v[2])  # P5, P6, P4, P3

print(f"NEW CR1 (v4, v5, v6, v2) = {CR1_new:.6f}")
print(f"NEW CR2 (v5, v6, v4, v3) = {CR2_new:.6f}")
print()

# Now apply the ORIGINAL X3, X5 formulas from paper (Eq 11, 12):
#   X3 = 2 * wp2 / (2 - CR1)
#   X5 = wp1 / (1 - 2 * CR2)
# 
# But in new indexing, X3 corresponds to X2 (Y=wp2 line) and X5 to X3 (Y=wp1)

X2_formula = (2.0 * wp2) / (2.0 - CR1_new)
X3_formula = wp1 / (1.0 - 2.0 * CR2_new)

print("RECOVERED X COORDINATES FROM CROSS-RATIO:")
print(f"  X2 (formula) = {X2_formula:.6f}  vs  TRUE X2 = {P_true[1][0]:.6f}")
print(f"  X3 (formula) = {X3_formula:.6f}  vs  TRUE X3 = {P_true[2][0]:.6f}")
print()

# Check if we're getting close
err_X2 = abs(X2_formula - P_true[1][0])
err_X3 = abs(X3_formula - P_true[2][0])
print(f"ERRORS: X2={err_X2:.6f}, X3={err_X3:.6f}")
