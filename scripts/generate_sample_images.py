#!/usr/bin/env python3
"""Generate sample placeholder images for the sample session.

This script creates minimal placeholder images for testing without
requiring actual camera data.

Usage:
    python scripts/generate_sample_images.py --out datasets/sample_session
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


def create_checkerboard_image(
    width: int = 640,
    height: int = 480,
    square_size: int = 40,
) -> np.ndarray:
    """Create a simple checkerboard pattern image."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                image[y:y+square_size, x:x+square_size] = [200, 200, 200]
            else:
                image[y:y+square_size, x:x+square_size] = [50, 50, 50]
    
    return image


def create_gradient_image(
    width: int = 1280,
    height: int = 1,
) -> np.ndarray:
    """Create a simple gradient image (for HSI line simulation)."""
    gradient = np.linspace(0, 255, width).astype(np.uint8)
    image = np.tile(gradient, (height, 1))
    return image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate sample placeholder images",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/sample_session"),
        help="Output session directory",
    )
    
    args = parser.parse_args()
    output_dir = args.out.resolve()
    
    try:
        import cv2
    except ImportError:
        print("OpenCV not available, skipping image generation")
        print("Images are optional - the toolkit works with metadata only")
        return 0
    
    # Create directories
    (output_dir / "oak" / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "hsi" / "lines").mkdir(parents=True, exist_ok=True)
    
    # Generate OAK sample images
    print("Generating OAK sample images...")
    for i in range(3):
        img = create_checkerboard_image()
        # Add some variation
        noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite(str(output_dir / "oak" / "images" / f"frame_{i:04d}.png"), img)
    
    # Generate HSI sample lines
    print("Generating HSI sample lines...")
    for i in range(5):
        line = create_gradient_image()
        cv2.imwrite(str(output_dir / "hsi" / "lines" / f"line_{i:04d}.png"), line)
    
    print(f"Generated sample images in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
