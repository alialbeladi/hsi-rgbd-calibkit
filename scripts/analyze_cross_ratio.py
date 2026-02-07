"""Cross-ratio analysis for Li-Wen-Qiu pattern."""

import numpy as np
from hsi_rgbd_calib.boards.li_wen_qiu_pattern import get_default_li_wen_qiu_pattern
from hsi_rgbd_calib.boards.geometry import intersect_lines_2d, line_through_points, cross_ratio_1d

pattern = get_default_li_wen_qiu_pattern()
wp1, wp2 = pattern.wp1, pattern.wp2

print("Testing cross-ratio invariants...")
print()

# Different scan lines
test_points = [
    ((0.03, 0), (0.05, wp1)),
    ((0.04, 0), (0.07, wp1)),
    ((0.05, 0), (0.06, wp1)),
    ((0.08, 0), (0.09, wp1)),
]

for start, end in test_points:
    scan_line = line_through_points(start, end)
    
    # Get intersections
    pts = []
    for fl in pattern.feature_lines:
        pt = intersect_lines_2d(scan_line, fl)
        pts.append(pt if pt else (0, 0))
    
    # Use Y-coordinate as 1D parameter
    y = [p[1] for p in pts]
    
    # Compute cross-ratios that paper might use
    # Try combinations based on paper's convention
    try:
        cr1 = cross_ratio_1d(y[1], y[3], y[5], y[2])  # P2, P4, P6, P3
        cr2 = cross_ratio_1d(y[3], y[5], y[1], y[4])  # P4, P6, P2, P5
        
        print(f"Scan {start} -> {end}:")
        print(f"  Y coords: y1={y[0]:.4f}, y2={y[1]:.4f}, y3={y[2]:.4f}")
        print(f"            y4={y[3]:.4f}, y5={y[4]:.4f}, y6={y[5]:.4f}")
        print(f"  CR(P2, P4, P6, P3) = {cr1:.6f}")
        print(f"  CR(P4, P6, P2, P5) = {cr2:.6f}")
    except Exception as e:
        print(f"Error: {e}")
    print()

print(f"Pattern: wp1={wp1}, wp2={wp2}")
print()

# From paper: CR1 and CR2 should be invariants that let us recover X2 and X3
# The formulas were:
#   X3 = 2 * wp2 / (2 - CR1)   <- from Eq 11
#   X5 = wp1 / (1 - 2 * CR2)   <- from Eq 12
#
# But now with our pattern indexing, P2 is at Y=wp2 and P3 is at Y=wp1
# So we might need X2 and X3 formulas

# Let's verify by computing what X2 SHOULD be for each scan line
print("Verifying X coordinates:")
for start, end in test_points:
    scan_line = line_through_points(start, end)
    pts = [intersect_lines_2d(scan_line, fl) for fl in pattern.feature_lines]
    
    print(f"Scan {start} -> {end}:")
    print(f"  P2 = ({pts[1][0]:.4f}, {pts[1][1]:.4f}) -> X2 = {pts[1][0]:.4f}")
    print(f"  P3 = ({pts[2][0]:.4f}, {pts[2][1]:.4f}) -> X3 = {pts[2][0]:.4f}")
