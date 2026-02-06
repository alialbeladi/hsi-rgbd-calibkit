#!/usr/bin/env python3
"""Calibration script - thin wrapper around library calibration function.

This script provides a standalone way to run calibration without the CLI.

Usage:
    python scripts/calibrate.py --session <path> --config <config.yaml> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hsi_rgbd_calib.cli import cmd_calibrate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run HSI-RGBD calibration",
    )
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to calibration session directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to calibration configuration file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    from hsi_rgbd_calib.common.logging import setup_logging
    import logging
    
    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)
    
    return cmd_calibrate(args)


if __name__ == "__main__":
    sys.exit(main())
