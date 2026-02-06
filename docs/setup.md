# Setup Guide

This guide covers installation and environment setup for the HSI-RGBD Calibration Kit.

## Requirements

- Python 3.10 or later
- Git
- (Optional) OpenCV with ArUco support for target detection

## Installation

### From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/your-org/hsi-rgbd-calibkit.git
cd hsi-rgbd-calibkit

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### From PyPI (Future)

```bash
pip install hsi-rgbd-calibkit
```

## Verify Installation

```bash
# Check CLI is available
hsi-rgbd-calib --help

# Check version
hsi-rgbd-calib --version

# Run tests
pytest
```

## Dependencies

The toolkit uses the following core dependencies:

| Package | Purpose |
|---------|---------|
| numpy | Array operations, transforms |
| opencv-python | Image processing, ArUco detection |
| pyyaml | Configuration and artifact files |
| scipy | Optimization, spatial transforms |
| rich | CLI logging and formatting |

Development dependencies:

| Package | Purpose |
|---------|---------|
| pytest | Unit testing |
| pytest-cov | Code coverage |

## OAK-D Setup (Optional)

If you plan to extract intrinsics directly from a connected OAK-D camera:

```bash
pip install depthai
```

However, for calibration purposes, you typically provide intrinsics via YAML files rather than connecting to the camera.

## Troubleshooting

### Import Errors

If you see import errors, ensure you installed in development mode:

```bash
pip install -e .
```

### OpenCV Issues

For ChArUco detection, you need OpenCV with ArUco support:

```bash
pip install opencv-contrib-python
```

### Path Issues on Windows

When specifying paths on Windows, use forward slashes or raw strings:

```bash
hsi-rgbd-calib calibrate --session ./datasets/sample_session ...
```
