# HSI-RGBD Calibration Kit

[![CI](https://github.com/your-org/hsi-rgbd-calibkit/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/hsi-rgbd-calibkit/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-quality calibration toolkit for a rigid pushbroom hyperspectral (HSI) line-scan camera with an OAK-D S2 RGB-D camera rig.

## Overview

This repository provides tools to:

1. **Calibrate** the extrinsic transformation between an HSI line-scan camera and an OAK-D S2 RGB camera
2. **Estimate** HSI slit intrinsics using the Li-Wen-Qiu line-scan + frame camera calibration method
3. **Validate** calibration quality with reprojection error and repeatability metrics
4. **Export** a reproducible calibration artifact (`rig.yaml`) for downstream mapping

### Key Features

- Adopts the **Li-Wen-Qiu method** for line-scan + frame camera calibration
- Uses **OAK-D S2** for RGB intrinsics and depth alignment configuration
- Produces a **standardized `rig.yaml`** artifact for reproducible downstream use
- Includes **quantitative validation** (reprojection RMSE, repeatability, slow-motion sanity checks)
- Provides both **stub** and **Python** backends for the calibration method

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/hsi-rgbd-calibkit.git
cd hsi-rgbd-calibkit

# Install in development mode
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare a Calibration Session

Create a session folder with the required structure:

```
my_session/
├── session.yaml          # Session metadata
├── oak/
│   ├── intrinsics.yaml   # OAK RGB camera intrinsics
│   └── depth_config.yaml # Depth alignment configuration
├── hsi/
│   └── metadata.yaml     # HSI cube metadata
└── cal_method/           # Optional: precomputed calibration results
    └── cal_method_output.yaml
```

See [docs/running_calibration.md](docs/running_calibration.md) for detailed session structure.

### 2. Run Calibration

```bash
# Using the sample session
hsi-rgbd-calib calibrate \
    --session datasets/sample_session \
    --config configs/li_wen_qiu.yaml \
    --out output/

# Dry-run to see what would happen
hsi-rgbd-calib calibrate \
    --session datasets/sample_session \
    --config configs/li_wen_qiu.yaml \
    --out output/ \
    --dry-run
```

### 3. Validate Results

```bash
hsi-rgbd-calib validate \
    --session datasets/sample_session \
    --rig output/rig.yaml \
    --out output/
```

### 4. Export Artifact

```bash
hsi-rgbd-calib export \
    --session datasets/sample_session \
    --rig output/rig.yaml \
    --out output/
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `hsi-rgbd-calib calibrate` | Run calibration and generate `rig.yaml` |
| `hsi-rgbd-calib validate` | Validate calibration with metrics |
| `hsi-rgbd-calib export` | Export calibration artifact and report |

Use `--help` on any command for detailed options:

```bash
hsi-rgbd-calib --help
hsi-rgbd-calib calibrate --help
```

## Calibration Artifact (`rig.yaml`)

The output `rig.yaml` contains:

- **Artifact metadata**: version, timestamp
- **Frame definitions**: rig, oak_rgb, oak_depth, hsi
- **OAK intrinsics**: RGB camera matrix, distortion, depth alignment config
- **HSI intrinsics**: slit parameters, wavelengths
- **Extrinsics**: `T_oakrgb_hsi` as 4×4 matrix with optional uncertainty
- **Validation summary**: reprojection RMSE, repeatability metrics

See [docs/artifact_spec.md](docs/artifact_spec.md) for the complete specification.

## Documentation

- [Setup Guide](docs/setup.md)
- [Running Calibration](docs/running_calibration.md)
- [Validation Guide](docs/validation.md)
- [Calibration Targets](docs/targets.md)
- [Artifact Specification](docs/artifact_spec.md)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=hsi_rgbd_calib
```

## Project Structure

```
hsi-rgbd-calibkit/
├── src/hsi_rgbd_calib/     # Main package
│   ├── common/             # SE(3) transforms, logging, frame conventions
│   ├── io/                 # Artifact schema, import/export
│   ├── oak/                # OAK intrinsics and depth config handling
│   ├── boards/             # Calibration target helpers
│   ├── cal_method/         # Calibration method (Li-Wen-Qiu) wrapper
│   ├── metrics/            # Validation metrics
│   └── cli.py              # CLI entrypoint
├── scripts/                # Standalone scripts
├── configs/                # Configuration files
├── assets/                 # Calibration target assets
├── datasets/               # Sample data
├── docs/                   # Documentation
└── tests/                  # Unit tests
```

## References

This toolkit adopts the Li-Wen-Qiu line-scan + frame camera calibration method:

> Li, J., Wen, S., & Qiu, S. (Year). *Line-scan camera calibration using a frame camera*.

## License

MIT License - see [LICENSE](LICENSE) for details.
