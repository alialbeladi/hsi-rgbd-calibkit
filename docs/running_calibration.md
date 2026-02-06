# Running Calibration

This guide explains how to run HSI-RGBD calibration from data acquisition to artifact generation.

## Overview

The calibration workflow consists of:

1. **Data Acquisition** - Capture synchronized HSI and RGB images of a calibration target
2. **Session Preparation** - Organize data into the expected folder structure
3. **Calibration** - Run the calibration method to estimate extrinsics and intrinsics
4. **Validation** - Verify calibration quality
5. **Export** - Generate the final `rig.yaml` artifact

## Session Folder Structure

A calibration session must follow this structure:

```
my_session/
├── session.yaml              # Session metadata (required)
├── oak/
│   ├── intrinsics.yaml       # OAK RGB camera intrinsics (required)
│   ├── depth_config.yaml     # Depth alignment config (optional)
│   └── images/               # Sample RGB images (optional)
│       ├── frame_0000.png
│       └── ...
├── hsi/
│   ├── metadata.yaml         # HSI camera metadata (required)
│   └── cubes/                # HSI cube data (optional)
│       └── ...
└── cal_method/               # Precomputed calibration (optional)
    └── cal_method_output.yaml
```

## Step 1: Prepare session.yaml

```yaml
name: my_calibration_session
description: "Calibration of HSI rig on 2024-01-15"
created_at: "2024-01-15T10:30:00Z"

equipment:
  hsi_camera:
    model: "Specim FX10"
    serial: "FX10-001"
  rgb_camera:
    model: "OAK-D S2"
    serial: "14110250AB"

acquisition:
  num_frames: 50
  duration_seconds: 30.0
  target_type: charuco_6x9
```

## Step 2: Prepare OAK Intrinsics

Create `oak/intrinsics.yaml` with your camera's calibrated intrinsics:

```yaml
device:
  serial: "14110250AB"
  model: "OAK-D S2"

rgb:
  fx: 1000.0
  fy: 1000.0
  cx: 640.0
  cy: 360.0
  width: 1280
  height: 720
  distortion_model: radtan
  distortion_coeffs: [0.05, -0.08, 0.0, 0.0, 0.02]
```

## Step 3: Prepare HSI Metadata

Create `hsi/metadata.yaml`:

```yaml
slit_width: 1280
num_bands: 224
fps: 100.0
integration_time_us: 5000.0
estimated_velocity_mps: 0.05
```

## Step 4: Run Calibration

### Using Stub Backend (Precomputed Results)

If you have precomputed results from an external tool:

1. Place results in `cal_method/cal_method_output.yaml`
2. Run:

```bash
hsi-rgbd-calib calibrate \
    --session ./my_session \
    --config configs/li_wen_qiu.yaml \
    --out ./output
```

### Using Python Backend (Placeholder)

The Python backend contains placeholder implementation. For production, use the stub backend or implement the TODO sections.

## Step 5: Check Output

After calibration, you'll have:

```
output/
├── rig.yaml              # Main calibration artifact
└── calibration_report.json  # Detailed report
```

## Step 6: Validate

```bash
hsi-rgbd-calib validate \
    --session ./my_session \
    --rig ./output/rig.yaml \
    --out ./output
```

## Dry Run Mode

Preview what will happen without making changes:

```bash
hsi-rgbd-calib calibrate \
    --session ./my_session \
    --config configs/li_wen_qiu.yaml \
    --out ./output \
    --dry-run
```

## Quick Start with Sample Data

Test the workflow using the included sample session:

```bash
cd hsi-rgbd-calibkit

# Run calibration
hsi-rgbd-calib calibrate \
    --session datasets/sample_session \
    --config configs/li_wen_qiu.yaml \
    --out output

# Validate
hsi-rgbd-calib validate \
    --session datasets/sample_session \
    --rig output/rig.yaml \
    --out output
```
