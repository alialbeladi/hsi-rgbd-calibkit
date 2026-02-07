"""Fix the cross-ratio recovery formula.

The correct cross-ratio formula CR(a,b,c,d) = ((a-c)/(b-c)) / ((a-d)/(b-d))
= ((a-c)*(b-d)) / ((b-c)*(a-d))

For CR(P1, P2, P4, P3) where y1=0, y2=wp2, y3=wp1, y4=unknown:
CR = ((y1-y4)*(y2-y3)) / ((y2-y4)*(y1-y3))
CR = ((0-y4)*(wp2-wp1)) / ((wp2-y4)*(0-wp1))
CR = ((-y4)*(wp2-wp1)) / ((wp2-y4)*(-wp1))
CR = (y4*(wp2-wp1)) / ((wp2-y4)*wp1)   [negatives cancel]
CR = (y4*(wp2-wp1)) / (wp1*wp2 - wp1*y4)

Solving for y4:
CR * (wp1*wp2 - wp1*y4) = y4*(wp2-wp1)
CR*wp1*wp2 - CR*wp1*y4 = y4*wp2 - y4*wp1
CR*wp1*wp2 = y4*wp2 - y4*wp1 + CR*wp1*y4
CR*wp1*wp2 = y4*(wp2 - wp1 + CR*wp1)
y4 = (CR*wp1*wp2) / (wp2 - wp1 + CR*wp1)
y4 = (CR*wp1*wp2) / (wp2 + wp1*(CR - 1))
"""

import numpy as np
from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern
from hsi_rgbd_calib.boards.geometry import intersect_lines_2d, cross_ratio_1d
from hsi_rgbd_calib.cal_method.li_wen_qiu.sim import simulate_views, NoiseConfig
from hsi_rgbd_calib.cal_method.li_wen_qiu.projection import (
    compute_scan_line_in_pattern, 
    compute_transform_pattern_to_linescan
)

pattern = get_default_li_wen_qiu_pattern()
wp1, wp2 = pattern.wp1, pattern.wp2

# Simulate
noise = NoiseConfig(sigma_v=0.0)
sim_result = simulate_views(n_views=1, noise_config=noise, seed=42)
gt = sim_result.ground_truth
view = sim_result.views[0]

# Get true pattern points
R0, T0 = compute_transform_pattern_to_linescan(
    view.R_frame_pattern, view.T_frame_pattern, gt.R, gt.T
)
scan_line = compute_scan_line_in_pattern(R0, T0)
P_true = [intersect_lines_2d(scan_line, fl) for fl in pattern.feature_lines]
v = view.v_observations

# Extract Y coordinates
y = [P_true[i][1] for i in range(6)]

print("=" * 60)
print("CROSS-RATIO POINT RECOVERY - CORRECTED")
print("=" * 60)
print()

# Step 1: Recover y4 using CR(P1, P2, P4, P3)
CR1 = cross_ratio_1d(v[0], v[1], v[3], v[2])  # Using v indices
print(f"CR1 = CR(v1, v2, v4, v3) = {CR1:.6f}")

# Corrected formula for y4:
# y4 = (CR1*wp1*wp2) / (wp2 + wp1*(CR1 - 1))
y4_recovered = (CR1 * wp1 * wp2) / (wp2 + wp1 * (CR1 - 1))
print(f"  y4 recovered = {y4_recovered:.6f}")
print(f"  y4 true      = {y[3]:.6f}")
print(f"  Error: {abs(y4_recovered - y[3]):.6f}")
print()

# Step 2: Recover y5 using a different cross-ratio
# We can use CR(P1, P5, P4, P2) which mixes known (P1, P2) and unknown (P4, P5)
# But we already know y4, so this becomes overdetermined
# Better: use CR that directly involves y5 with known points

# CR(P1, P2, P5, P3) = ((y1-y5)*(y2-y3)) / ((y2-y5)*(y1-y3))
# = ((-y5)*(wp2-wp1)) / ((wp2-y5)*(-wp1))
# = (y5*(wp2-wp1)) / ((wp2-y5)*wp1)
# Same form as y4 recovery!

CR2 = cross_ratio_1d(v[0], v[1], v[4], v[2])  # CR(v1, v2, v5, v3)
y5_recovered = (CR2 * wp1 * wp2) / (wp2 + wp1 * (CR2 - 1))
print(f"CR2 = CR(v1, v2, v5, v3) = {CR2:.6f}")
print(f"  y5 recovered = {y5_recovered:.6f}")
print(f"  y5 true      = {y[4]:.6f}")
print(f"  Error: {abs(y5_recovered - y[4]):.6f}")
print()

# Step 3: Recover y6 using CR(P1, P2, P6, P3)
CR3 = cross_ratio_1d(v[0], v[1], v[5], v[2])  # CR(v1, v2, v6, v3)
y6_recovered = (CR3 * wp1 * wp2) / (wp2 + wp1 * (CR3 - 1))
print(f"CR3 = CR(v1, v2, v6, v3) = {CR3:.6f}")
print(f"  y6 recovered = {y6_recovered:.6f}")
print(f"  y6 true      = {y[5]:.6f}")
print(f"  Error: {abs(y6_recovered - y[5]):.6f}")
print()

# Now with recovered y4, y5, y6 we can compute X4, X5, X6
# For L4 (X=Y): X4 = Y4
# For L5 (X-Y=wp2): X5 = Y5 + wp2
# For L6 (X-Y=wp1): X6 = Y6 + wp1

print("X COORDINATE RECOVERY:")
X4_recovered = y4_recovered  # X=Y on L4
X5_recovered = y5_recovered + wp2  # X-Y=wp2 on L5
X6_recovered = y6_recovered + wp1  # X-Y=wp1 on L6

print(f"  X4 recovered = {X4_recovered:.6f} (true = {P_true[3][0]:.6f})")
print(f"  X5 recovered = {X5_recovered:.6f} (true = {P_true[4][0]:.6f})")
print(f"  X6 recovered = {X6_recovered:.6f} (true = {P_true[5][0]:.6f})")
print()

# For the horizontal lines P1, P2, P3:
# We need to find their X coordinates too!
# P1, P4, P5, P6 are collinear (all on scan line)
# So we can use the scan line equation to find X1, X2, X3

# The scan line passes through (X4, Y4) and has some slope m
# m = Delta_Y / Delta_X between any two points

# Using P4 and P5: m = (Y4 - Y5) / (X4 - X5)
if abs(X4_recovered - X5_recovered) > 1e-10:
    m = (y4_recovered - y5_recovered) / (X4_recovered - X5_recovered)
else:
    m = float('inf')
print(f"Scan line slope m = {m:.6f}")

# X1 is where Y=0: X = X4 - m * Y4
X1_recovered = X4_recovered - m * y4_recovered
X2_recovered = X1_recovered + m * wp2
X3_recovered = X1_recovered + m * wp1

print(f"  X1 recovered = {X1_recovered:.6f} (true = {P_true[0][0]:.6f})")
print(f"  X2 recovered = {X2_recovered:.6f} (true = {P_true[1][0]:.6f})")
print(f"  X3 recovered = {X3_recovered:.6f} (true = {P_true[2][0]:.6f})")
