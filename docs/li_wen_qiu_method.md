# Li-Wen-Qiu Calibration Method

This document describes the Li-Wen-Qiu cross-ratio-based line-scan camera calibration method implemented in this toolkit.

## Reference

Li, Wen, Qiu (2016). *"Cross-ratio-based line scan camera calibration using a planar pattern."*  
Optical Engineering 55(1), 014104. DOI: [10.1117/1.OE.55.1.014104](https://doi.org/10.1117/1.OE.55.1.014104)

## Overview

The method calibrates a line-scan (pushbroom) camera that is rigidly coupled with a frame camera. It estimates:

- **Intrinsic parameters**: Focal length `f`, principal point `v0`, distortion `k`
- **Extrinsic parameters**: Rotation `R` and translation `T` between frame and line-scan cameras

## Calibration Pattern

The method uses a planar pattern with 6 feature lines:

```
        L2 (X=Y)
        /
       /  L4 (X=Y+wp2)
      /   /
L1 ─┬────┬────┬─── Y=0
    │     │     │
L3 ─┼────┼────┼─── Y=wp2
    │     │     │
L5 ─┴────┴────┴─── Y=wp1
           \
            L6 (X=Y+wp1)
```

Default dimensions: `wp1 = 100mm`, `wp2 = 50mm`

Pattern files: `configs/patterns/li_wen_qiu_pattern.yaml`

## Algorithm Stages

### Stage 1: Point Correspondence (Section 3.2)

For each view, given 6 observed pixel coordinates `v1...v6`:

1. Compute cross-ratios:
   - `CR1 = ((v2-v6)*(v4-v3)) / ((v4-v6)*(v2-v3))`
   - `CR2 = ((v4-v2)*(v6-v5)) / ((v6-v2)*(v4-v5))`

2. Compute X-coordinates of P3 and P5:
   - `X3 = 2*wp2 / (2 - CR1)`
   - `X5 = wp1 / (1 - 2*CR2)`

3. Form scan line through P3, P5 and intersect with all 6 feature lines

### Stage 2: Closed-Form Initialization (Section 3.3)

Given pattern points `P_i` and frame camera poses `(R_j, T_j)`:

1. Transform points to frame camera coordinates
2. Solve `A_J @ J = 0` via SVD for view plane normal
3. Solve `A_K @ K = 0` via SVD for rotation/translation mix
4. Recover rotation matrix R from J and K
5. Solve for f, v0, t1, t2, t3 from linear system

### Stage 3: Nonlinear Refinement (Section 3.4)

Optimize parameters `[f, v0, k, R, T]` using Nelder-Mead:

```python
cost = Σ_j Σ_i (v_ij - v̂_ij)²
```

Where `v̂_ij` is computed by:
1. Computing pattern-to-line-scan transform for view j
2. Intersecting scan plane with feature lines (updated P_i)
3. Projecting P_i using intrinsics with distortion

## Input Data Format

The calibration expects a session directory with:

```
session_dir/
├── session.yaml          # Session metadata
├── oak/
│   ├── intrinsics.yaml   # Frame camera intrinsics
│   └── poses/            # Per-view poses (R_j, T_j)
└── hsi/
    └── observations.yaml # Per-view v1..v6 observations
```

Or programmatically via `ViewObservation` objects:

```python
from hsi_rgbd_calib.cal_method.li_wen_qiu.backend import ViewObservation, LiWenQiuBackend

views = [
    ViewObservation(
        R_frame_pattern=R_j,     # 3x3 rotation
        T_frame_pattern=T_j,     # 3-element translation
        v_observations=v_obs,    # 6-element array [v1..v6]
    )
    for R_j, T_j, v_obs in zip(rotations, translations, observations)
]

backend = LiWenQiuBackend()
result = backend.estimate_from_observations(views)
```

## Validation via Simulation

The toolkit includes simulation utilities to validate the implementation:

```bash
# Run with noiseless data
python scripts/sim_run_li_wen_qiu.py --n-views 15 --noise 0.0

# Run with realistic noise
python scripts/sim_run_li_wen_qiu.py --n-views 20 --noise 0.2 --out report.json
```

### Acceptance Criteria

**Noiseless (n=10-20 views):**
- Rotation error < 1e-3 rad
- Translation error < 1e-6 m
- Focal length error < 1e-6 relative
- Principal point error < 1e-6 px

**Noisy (σ_v = 0.2 px):**
- Rotation error < 0.5 deg
- Translation error < 5 mm
- Focal length error < 1%
- Principal point error < 2 px

## Code Structure

```
src/hsi_rgbd_calib/cal_method/li_wen_qiu/
├── __init__.py       # Module exports
├── projection.py     # Line-scan projection model (Eq 4)
├── cross_ratio.py    # Cross-ratio computation (Sec 3.2)
├── closed_form.py    # SVD initialization (Sec 3.3)
├── nonlinear.py      # Nelder-Mead refinement (Sec 3.4)
├── backend.py        # Main LiWenQiuBackend class
└── sim.py            # Simulation utilities
```

## Notes

- The closed-form initialization assumes no distortion (k=0)
- Distortion is estimated during nonlinear refinement
- Pattern must be rigid and precisely manufactured
- Frame camera poses must be accurately estimated (e.g., via ChArUco)
