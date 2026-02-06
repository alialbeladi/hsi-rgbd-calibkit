# Validation Guide

This guide explains how to validate calibration results and interpret quality metrics.

## Overview

Validation assesses calibration quality through:

1. **Reprojection Error** - How accurately the model predicts image coordinates
2. **Repeatability** - Consistency across multiple calibration runs
3. **Sanity Checks** - Physical plausibility of estimated parameters

## Running Validation

```bash
hsi-rgbd-calib validate \
    --session ./my_session \
    --rig ./output/rig.yaml \
    --out ./output
```

Output:
```
output/
└── validation_report.json
```

## Validation Metrics

### Reprojection Error

Measures how well the calibrated model predicts pixel locations.

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| RMSE (px) | < 0.5 | 0.5 - 1.0 | > 1.0 |
| Median (px) | < 0.4 | 0.4 - 0.8 | > 0.8 |
| Max (px) | < 2.0 | 2.0 - 5.0 | > 5.0 |

#### Interpreting Reprojection Error

- **< 0.5 px RMSE**: Excellent calibration, suitable for precision mapping
- **0.5 - 1.0 px RMSE**: Good calibration, suitable for most applications
- **> 1.0 px RMSE**: Consider recalibrating with more data or checking for issues

### Repeatability

Measures consistency across multiple calibration runs.

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Translation std (mm) | < 2 | 2 - 5 | > 5 |
| Rotation std (deg) | < 0.2 | 0.2 - 0.5 | > 0.5 |

To test repeatability, run calibration multiple times with different subsets of data.

### Slow-Motion Check

For HSI pushbroom cameras, we assume quasi-static acquisition. This check validates that assumption.

```
Motion per line = velocity / FPS
```

| Scenario | Motion/Line | Status |
|----------|-------------|--------|
| 0.05 m/s at 100 FPS | 0.5 mm | ✓ Good |
| 0.1 m/s at 100 FPS | 1.0 mm | ⚠ Marginal |
| 0.2 m/s at 100 FPS | 2.0 mm | ✗ Too fast |

### Extrinsic Sanity Check

Validates that estimated extrinsics are physically plausible:

- **Translation**: Should be within expected mounting offset (typically < 0.5m)
- **Rotation**: Should be within expected alignment (typically < 45°)

## Validation Report Structure

```json
{
  "session_path": "/path/to/session",
  "rig_yaml_path": "/path/to/rig.yaml",
  "validation_summary": {
    "reprojection_rmse_px": 0.423,
    "median_abs_error_px": 0.356,
    "max_error_px": 1.205
  },
  "extrinsic_sanity_check": {
    "passed": true,
    "translation_magnitude_m": 0.108,
    "rotation_angle_deg": 1.23
  },
  "slow_motion_check": {
    "passed": true,
    "max_motion_per_line_mm": 0.5,
    "threshold_mm": 1.0
  }
}
```

## Troubleshooting Poor Calibration

### High Reprojection Error

1. **Check target quality** - Is the target flat? Well-printed?
2. **Check lighting** - Are there reflections or shadows?
3. **Check data quantity** - More views improve accuracy
4. **Check target coverage** - Target should cover full field of view

### Poor Repeatability

1. **Check mechanical stability** - Is the rig rigidly mounted?
2. **Check data diversity** - Use varied target poses
3. **Check for outliers** - Enable RANSAC in configuration

### Failed Sanity Checks

1. **Extrinsic bounds** - Review mounting configuration
2. **Slow motion** - Reduce rig velocity or increase FPS
