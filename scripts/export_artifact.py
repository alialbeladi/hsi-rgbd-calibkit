#!/usr/bin/env python3
"""Export script - thin wrapper around library export function.

Usage:
    python scripts/export_artifact.py --session <path> --rig <rig.yaml> --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hsi_rgbd_calib.cli import cmd_export


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export HSI-RGBD calibration artifacts",
    )
    parser.add_argument(
        "--session",
        type=Path,
        required=True,
        help="Path to calibration session directory",
    )
    parser.add_argument(
        "--rig",
        type=Path,
        required=True,
        help="Path to rig.yaml calibration artifact",
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
    
    return cmd_export(args)


if __name__ == "__main__":
    sys.exit(main())
