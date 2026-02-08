# Li-Wen-Qiu Calibration: Mathematical Theory and Implementation

This document provides the detailed mathematical derivations and implementation specifics for the Li-Wen-Qiu cross-ratio-based line-scan camera calibration method.

## Reference

Li, Wen, Qiu (2016). *"Cross-ratio-based line scan camera calibration using a planar pattern."*  
Optical Engineering 55(1), 014104. [DOI: 10.1117/1.OE.55.1.014104](https://doi.org/10.1117/1.OE.55.1.014104)

---

## 1. Camera Model

### 1.1 Line-Scan Projection Model

A line-scan (pushbroom) camera images a single row of pixels. The projection from a 3D point $P = (X, Y, Z)$ in camera coordinates to pixel coordinate $v$ is:

$$v = f \cdot \frac{Y}{Z} + v_0 + k \cdot Y \cdot \left(\frac{Y}{Z}\right)^2$$

Where:
- $f$ = focal length (pixels)
- $v_0$ = principal point (pixel offset)
- $k$ = radial distortion coefficient

**Implementation note**: The simplified model (no distortion) is used for closed-form initialization:
$$v = f \cdot \frac{Y}{Z} + v_0$$

**Code reference**: [`projection.py::project_to_linescan()`](../src/hsi_rgbd_calib/cal_method/li_wen_qiu/projection.py)

### 1.2 Coordinate Frames

Three coordinate frames are used:

1. **Pattern frame** $\{P\}$: Origin at pattern corner, X-Y in pattern plane, Z perpendicular
2. **Frame camera** $\{F\}$: Origin at OAK-RGB camera center
3. **Line-scan camera** $\{L\}$: Origin at HSI camera center

Transform chain:
$$P_L = R \cdot (R_j \cdot P_P + T_j) + T$$

Where:
- $(R_j, T_j)$ = pattern-to-frame transform for view $j$ (from ChArUco detection)
- $(R, T)$ = frame-to-linescan extrinsics (what we're solving for)

---

## 2. Pattern Design

### 2.1 Feature Line Geometry

The calibration pattern has 6 coplanar lines:

| Line | Equation | Description |
|------|----------|-------------|
| L1 | $Y = 0$ | Base horizontal |
| L2 | $Y = w_{p2}$ | Middle horizontal |
| L3 | $Y = w_{p1}$ | Top horizontal |
| L4 | $X = Y$ | Base diagonal |
| L5 | $X - Y = w_{p2}$ | Middle diagonal |
| L6 | $X - Y = w_{p1}$ | Top diagonal |

Default: $w_{p1} = 0.1$ m (100mm), $w_{p2} = 0.05$ m (50mm)

### 2.2 Why These Lines?

The key insight is that:
1. **Horizontal lines** (L1, L2, L3) have known Y-coordinates: 0, $w_{p2}$, $w_{p1}$
2. **Diagonal lines** (L4, L5, L6) allow cross-ratio computation

When a scan line intersects these 6 lines, we get 6 pixel observations $v_1, \ldots, v_6$.

---

## 3. Cross-Ratio Point Recovery

### 3.1 The Cross-Ratio Invariant

The cross-ratio of four collinear points $A, B, C, D$ is defined as:

$$CR(A, B, C, D) = \frac{(A - C)(B - D)}{(B - C)(A - D)}$$

**Critical property**: Cross-ratio is preserved under projective transformation. Therefore:
$$CR(v_1, v_2, v_N, v_3) = CR(Y_1, Y_2, Y_N, Y_3)$$

where $v_i$ are pixel coordinates and $Y_i$ are the Y-coordinates of intersection points.

### 3.2 Recovering Unknown Y-Coordinates

For points $P_1, P_2, P_3$ on horizontal lines:
- $Y_1 = 0$
- $Y_2 = w_{p2}$  
- $Y_3 = w_{p1}$

For points $P_4, P_5, P_6$ on diagonal lines, we need to find their Y-coordinates.

**Derivation**: Given $Y_1 = 0$, $Y_2 = w_{p2}$, $Y_3 = w_{p1}$, and observed $v_i$, compute:

$$CR = CR(v_1, v_2, v_N, v_3) = \frac{(v_1 - v_N)(v_2 - v_3)}{(v_2 - v_N)(v_1 - v_3)}$$

Then solve for $Y_N$ using:

$$CR = \frac{(0 - Y_N)(w_{p2} - w_{p1})}{(w_{p2} - Y_N)(0 - w_{p1})} = \frac{-Y_N \cdot (w_{p2} - w_{p1})}{(w_{p2} - Y_N) \cdot (-w_{p1})}$$

Rearranging:

$$Y_N = \frac{CR \cdot w_{p1} \cdot w_{p2}}{w_{p2} + w_{p1} \cdot (CR - 1)}$$

### 3.3 From Y to (X, Y) Points

Once we have $Y_N$ for diagonal lines, we get full coordinates:
- L4 ($X = Y$): Point is $(Y_N, Y_N)$
- L5 ($X - Y = w_{p2}$): Point is $(Y_N + w_{p2}, Y_N)$
- L6 ($X - Y = w_{p1}$): Point is $(Y_N + w_{p1}, Y_N)$

**Code reference**: [`cross_ratio.py::recover_pattern_points_from_observations()`](../src/hsi_rgbd_calib/cal_method/li_wen_qiu/cross_ratio.py)

### 3.4 Implementation Details

> [!IMPORTANT]
> **Point ordering matters!** The cross-ratio uses the ordering $(v_1, v_2, v_N, v_3)$ where:
> - $v_1$ corresponds to L1 ($Y=0$)
> - $v_2$ corresponds to L2 ($Y=w_{p2}$)
> - $v_3$ corresponds to L3 ($Y=w_{p1}$)
> - $v_N$ is the diagonal line being recovered

The implementation maintains strict index correspondence between line indices and observation indices.

---

## 4. Closed-Form Initialization (DLT)

### 4.1 The Constraint Equations

From the projection equation, for each observation:

$$v_{ij} = f \cdot \frac{Y'_{ij}}{Z'_{ij}} + v_0$$

Where $P'_{ij} = R \cdot P_{ij}^{(F)} + T$ and $P_{ij}^{(F)} = R_j \cdot P_i + T_j$.

Rearranging:

$$v_{ij} \cdot Z'_{ij} = f \cdot Y'_{ij} + v_0 \cdot Z'_{ij}$$

### 4.2 Building the DLT System

**Matrix A_J** (Eq. 17): The constraint that all points lie in the slit plane:

For each observation, the slit plane constraint gives:
$$X'_{ij} = r_{11} X_{ij} + r_{12} Y_{ij} + r_{13} Z_{ij} + t_1 = 0$$

This defines a linear system:
$$A_J \cdot J = 0$$

where $J = [r_{11}, r_{12}, r_{13}, t_1]^T$ and $A_J$ has rows $[X_{ij}, Y_{ij}, Z_{ij}, 1]$.

**Matrix A_K** (Eq. 18): From the projection equation:

$$Y \cdot K_0 + Z \cdot K_1 + K_2 - v \cdot Y \cdot K_3 - v \cdot Z \cdot K_4 - v \cdot K_5 = 0$$

Where:
$$K = [f r_{33} - v_0 r_{23}, \; -f r_{32} + v_0 r_{22}, \; c_3, \; -r_{23}, \; r_{22}, \; r_{11} t_3 - r_{31} t_1]$$

### 4.3 SVD Solution

We solve for $J$ and $K$ as the null space of $A_J$ and $A_K$:

```python
_, S_J, Vh_J = np.linalg.svd(A_J)
J = Vh_J[-1, :]  # Last row = null space

_, S_K, Vh_K = np.linalg.svd(A_K)
K = Vh_K[-1, :]  # Last row = null space
```

> [!NOTE]
> The smallest singular value should be near zero. If $\sigma_{min} / \sigma_{max} > 0.01$, the system is poorly conditioned.

### 4.4 Recovering R, T, f, v₀ from J and K

**Step 1: Normalize J** (Eq. 21)

$$\alpha = \pm \frac{1}{\|J_{:3}\|}$$

$$r_1 = [r_{11}, r_{12}, r_{13}] = \alpha \cdot J_{:3}$$
$$t_1 = \alpha \cdot J_3$$

**Step 2: Recover r₂ from K** (Eq. 22-25)

From K:
- $K_3 = \gamma \cdot (-r_{23})$
- $K_4 = \gamma \cdot r_{22}$

Using orthogonality $r_1 \cdot r_2 = 0$ and $\|r_2\| = 1$:

$$r_{21} = -\frac{r_{12} K_4 - r_{13} K_3}{r_{11} \cdot \gamma}$$

$$\gamma = \sqrt{\frac{(r_{12} K_4 - r_{13} K_3)^2}{r_{11}^2} + K_3^2 + K_4^2}$$

**Step 3: Recover r₃** (Eq. 26)

$$r_3 = r_1 \times r_2$$

**Step 4: Recover f, v₀**

Solve the 2×2 system from $K_0, K_1$:

$$\begin{bmatrix} r_{33} & -r_{23} \\ -r_{32} & r_{22} \end{bmatrix} \begin{bmatrix} f \\ v_0 \end{bmatrix} = \begin{bmatrix} K_0 / \gamma \\ K_1 / \gamma \end{bmatrix}$$

**Step 5: Recover T**

$$t_3 = \frac{K_5 / \gamma + r_{31} t_1}{r_{11}}$$
$$t_2 = \text{(derived from } K_2 \text{)}$$

### 4.5 Sign Ambiguity Resolution

> [!IMPORTANT]
> **The SVD gives J and K only up to sign.** There are 4 possible sign combinations:
> - $\alpha = \pm 1/\|J_{:3}\|$
> - $\gamma = \pm \sqrt{\ldots}$

**Implementation approach**: Try all 4 combinations, filter by:
1. `t3 > 0` (chirality: pattern in front of camera)
2. `det(R) = +1` (proper rotation)
3. Reasonable `f > 100` (not degenerate)

Then **select best by reprojection RMSE**:

```python
for R, T, f, v0 in candidates:
    rmse = compute_reprojection_error(R, T, f, v0, X_all, v_all)
    if rmse < best_rmse:
        best = (R, T, f, v0)
```

**Code reference**: [`closed_form.py::_recover_parameters_from_JK()`](../src/hsi_rgbd_calib/cal_method/li_wen_qiu/closed_form.py)

---

## 5. Nonlinear Refinement

### 5.1 Cost Function

$$\text{cost} = \sum_j \sum_i \left( v_{ij} - \hat{v}_{ij}(\theta) \right)^2$$

Where $\theta = [r_x, r_y, r_z, t_1, t_2, t_3, f, v_0, k]$ and $\hat{v}_{ij}$ is the predicted pixel coordinate.

### 5.2 Re-computing Pattern Points

> [!WARNING]
> **The pattern points P_i change during optimization!**

When R, T change, the scan line geometry changes, so we must re-intersect the scan line with feature lines at each iteration:

```python
def compute_residuals(params, views):
    R, T, f, v0, k = unpack(params)
    residuals = []
    
    for view in views:
        # Compute scan line in pattern coordinates
        scan_line = compute_scan_line(R, T, view.R_j, view.T_j)
        
        # Intersect with feature lines to get updated P_i
        P_i = [intersect(scan_line, L_i) for L_i in feature_lines]
        
        # Project and compute residual
        for i, P in enumerate(P_i):
            v_pred = project(P, R, T, f, v0, k, view.R_j, view.T_j)
            residuals.append(v_pred - view.v_obs[i])
    
    return residuals
```

### 5.3 Optimization Method

We use Nelder-Mead (derivative-free) because:
1. The re-intersection step is not differentiable
2. Robust to local minima near the good closed-form init
3. Converges well for this 9-parameter problem

**Code reference**: [`nonlinear.py::refine_parameters()`](../src/hsi_rgbd_calib/cal_method/li_wen_qiu/nonlinear.py)

---

## 6. Numerical Stability Considerations

### 6.1 Cross-Ratio Degenerate Cases

The cross-ratio becomes unstable when:
- Two v observations are nearly equal (denominator → 0)
- The scan line is nearly parallel to a feature line

**Mitigation**: Check for minimum separation of v values:
```python
min_sep = min(|v_i - v_j| for all i ≠ j)
if min_sep < threshold:
    # Use fallback or skip this view
```

### 6.2 DLT Conditioning

The A_J and A_K matrices can be poorly conditioned if:
- Views are too similar (lack of geometric diversity)
- Points are clustered (poor spatial distribution)

**Check**: Monitor smallest singular value ratio $\sigma_{min}/\sigma_{max}$.

### 6.3 Sign Selection Edge Cases

The reprojection-based sign selection fails if:
- Multiple sign combinations give similar RMSE (very rare)
- All combinations violate chirality (bad data)

**Fallback**: Return identity + default intrinsics with `success=False`.

---

## 7. Testing and Validation

### 7.1 Stress Test Results

At various noise levels (100 views each):

| σ (px) | Init Rot Err (°) | Final Rot Err (°) | Final f Err (%) |
|--------|------------------|-------------------|-----------------|
| 0.0    | 0.0000          | 0.0000            | 0.0000          |
| 0.1    | 0.1441          | 0.0031            | 0.0011          |
| 0.2    | 0.2841          | 0.0061            | 0.0022          |
| 0.5    | 0.6785          | 0.0153            | 0.0055          |
| 1.0    | 1.2472          | 0.0305            | 0.0111          |

### 7.2 Key Verification Commands

```bash
# Test closed-form at σ=0
python scripts/test_closed_form.py

# Extended stress test
python scripts/extended_stress_test.py

# Full pipeline test
python -m pytest tests/test_li_wen_qiu_end_to_end.py -v
```

---

## 8. Code Structure Summary

| Module | Purpose | Key Functions |
|--------|---------|--------------|
| `projection.py` | Line-scan camera projection | `project_to_linescan()`, `compute_scan_line_in_pattern()` |
| `cross_ratio.py` | Cross-ratio point recovery | `compute_cross_ratio()`, `recover_pattern_points_from_observations()` |
| `closed_form.py` | DLT initialization | `_build_A_J()`, `_build_A_K()`, `_recover_parameters_from_JK()` |
| `nonlinear.py` | Refinement | `refine_parameters()`, `_residual_func()` |
| `backend.py` | Main orchestration | `LiWenQiuBackend.estimate_from_observations()` |
| `sim.py` | Simulation | `simulate_views()`, `get_default_ground_truth()` |
