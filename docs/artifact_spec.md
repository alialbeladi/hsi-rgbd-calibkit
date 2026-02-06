# Calibration Artifact Specification

This document specifies the `rig.yaml` calibration artifact format.

## Overview

The `rig.yaml` file contains all calibration information needed for downstream HSI-RGBD fusion and mapping applications.

## Schema Version

Current version: **1.0**

## Top-Level Structure

```yaml
artifact_version: "1.0"
created_at: "2024-01-15T10:30:00.123456"
frames: {...}
oak: {...}
hsi: {...}
extrinsics: {...}
validation_summary: {...}
session_path: "/path/to/session"  # optional
notes: ""  # optional
```

## Field Definitions

### artifact_version

- **Type**: string
- **Required**: Yes
- **Description**: Schema version for compatibility checking

### created_at

- **Type**: string (ISO 8601 timestamp)
- **Required**: Yes
- **Description**: Artifact creation timestamp

### frames

Defines coordinate frame conventions.

```yaml
frames:
  rig:
    name: rig
    description: "Rig reference frame"
    convention: "X: Right, Y: Down, Z: Forward"
  oak_rgb:
    name: oak_rgb
    description: "OAK-D RGB camera optical frame"
    convention: "X: Right, Y: Down, Z: Forward"
  oak_depth:
    name: oak_depth
    description: "OAK-D depth camera optical frame"
    convention: "X: Right, Y: Down, Z: Forward"
  hsi:
    name: hsi
    description: "HSI line-scan camera slit frame"
    convention: "X: Along slit, Y: Cross-track, Z: Forward"
```

### oak

OAK camera calibration data.

```yaml
oak:
  rgb_intrinsics:
    fx: 1000.0          # Focal length X (pixels)
    fy: 1000.0          # Focal length Y (pixels)
    cx: 640.0           # Principal point X (pixels)
    cy: 360.0           # Principal point Y (pixels)
    width: 1280         # Image width (pixels)
    height: 720         # Image height (pixels)
    distortion_model: radtan
    distortion_coeffs: [k1, k2, p1, p2, k3]
  
  depth_alignment:
    aligned_to_rgb: true
    output_width: 1280
    output_height: 720
    stereo_mode: standard
    subpixel_enabled: true
    lr_check_enabled: true
    notes: ""
  
  device_serial: "14110250AB"
  device_model: "OAK-D S2"
```

### hsi

HSI camera calibration data.

```yaml
hsi:
  slit_intrinsics:
    focal_length_slit: 1150.0    # Focal length along slit (pixels)
    principal_point_u0: 640.0    # Principal point (pixels)
    slit_width: 1280             # Slit width (pixels)
    distortion_model: none
    distortion_coeffs: []
  
  wavelengths_nm: null           # List of wavelengths or null
  wavelengths_file: null         # Path to wavelengths file
  num_bands: 224
  integration_time_us: 5000.0
```

### extrinsics

Transformation between cameras.

```yaml
extrinsics:
  # 4x4 transformation matrix (row-major)
  # Transforms points from HSI frame to OAK RGB frame
  T_oakrgb_hsi:
    - [r11, r12, r13, tx]
    - [r21, r22, r23, ty]
    - [r31, r32, r33, tz]
    - [0.0, 0.0, 0.0, 1.0]
  
  uncertainty:
    translation_std_m: 0.002     # Optional
    rotation_std_deg: 0.15       # Optional
  
  method: li_wen_qiu
  notes: ""
```

### validation_summary

Calibration quality metrics.

```yaml
validation_summary:
  reprojection_rmse_px: 0.423
  median_abs_error_px: 0.356
  max_error_px: 1.205
  repeatability_translation_mm: null
  repeatability_rotation_deg: null
  num_validation_points: 156
  slow_motion_check_passed: true
  slow_motion_notes: ""
```

## Example Complete Artifact

```yaml
artifact_version: "1.0"
created_at: "2024-01-15T10:30:00.123456"

frames:
  rig:
    name: rig
    description: "Rig reference frame"
    convention: "X: Right, Y: Down, Z: Forward"
  oak_rgb:
    name: oak_rgb
    description: "OAK-D RGB camera optical frame"
    convention: "X: Right, Y: Down, Z: Forward"
  oak_depth:
    name: oak_depth
    description: "OAK-D depth camera optical frame"
    convention: "X: Right, Y: Down, Z: Forward"
  hsi:
    name: hsi
    description: "HSI line-scan camera slit frame"
    convention: "X: Along slit, Y: Cross-track, Z: Forward"

oak:
  rgb_intrinsics:
    fx: 1000.0
    fy: 1000.0
    cx: 640.0
    cy: 360.0
    width: 1280
    height: 720
    distortion_model: radtan
    distortion_coeffs: [0.05, -0.08, 0.0, 0.0, 0.02]
  depth_alignment:
    aligned_to_rgb: true
    output_width: 1280
    output_height: 720
    stereo_mode: standard
    subpixel_enabled: true
    lr_check_enabled: true
    notes: ""
  device_serial: "14110250AB"
  device_model: "OAK-D S2"

hsi:
  slit_intrinsics:
    focal_length_slit: 1150.0
    principal_point_u0: 640.0
    slit_width: 1280
    distortion_model: none
    distortion_coeffs: []
  wavelengths_nm: null
  wavelengths_file: null
  num_bands: 224
  integration_time_us: 5000.0

extrinsics:
  T_oakrgb_hsi:
    - [0.9998, 0.0175, -0.0087, 0.052]
    - [-0.0174, 0.9998, 0.0052, -0.018]
    - [0.0088, -0.0050, 0.9999, 0.095]
    - [0.0, 0.0, 0.0, 1.0]
  uncertainty:
    translation_std_m: 0.002
    rotation_std_deg: 0.15
  method: li_wen_qiu
  notes: ""

validation_summary:
  reprojection_rmse_px: 0.423
  median_abs_error_px: 0.356
  max_error_px: 1.205
  repeatability_translation_mm: null
  repeatability_rotation_deg: null
  num_validation_points: 156
  slow_motion_check_passed: true
  slow_motion_notes: "Motion 0.5mm/line within 1.0mm threshold"

session_path: "/data/calibration/session_2024_01_15"
notes: "Production calibration for field deployment"
```

## Loading in Python

```python
from hsi_rgbd_calib.io import load_rig_yaml

artifact = load_rig_yaml("rig.yaml")

# Access extrinsic matrix
T = artifact.extrinsics.to_matrix()  # numpy 4x4 array

# Access RGB intrinsics
K = artifact.oak.rgb_intrinsics.to_camera_matrix()  # numpy 3x3 array
```
