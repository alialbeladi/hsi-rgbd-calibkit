#!/usr/bin/env python3
"""Generate ChArUco boards as vector PDFs for A1 paper.
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("Requires opencv-python and numpy.  pip install opencv-python numpy")

try:
    from fpdf import FPDF
except ImportError:
    sys.exit("Requires fpdf2 for PDF output.  pip install fpdf2")

# A1 dimensions in mm
A1_W, A1_H = 841.0, 594.0

def generate_marker_bits(marker_id, dictionary_id, size_bits=6):
    """Return a bitmask for the given marker ID."""
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    # OpenCV's generateImageMarker includes the black border bits.
    # For a 6x6 marker, it returns 8x8 (6 data + 2 border).
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_bits + 2)
    return img < 128  # True for black

def draw_marker(pdf, marker_id, x, y, size_mm, dictionary_id, size_bits=6):
    """Draw an ArUco marker at (x, y) with given size in mm."""
    bits = generate_marker_bits(marker_id, dictionary_id, size_bits)
    cells = size_bits + 2
    cell_size = size_mm / cells
    
    pdf.set_fill_color(0, 0, 0)
    for r in range(cells):
        for c in range(cells):
            if bits[r, c]:
                pdf.rect(x + c * cell_size, y + r * cell_size, cell_size, cell_size, 'F')

def build_charuco_pdf(output_path, rows, cols, square_size, marker_size, dictionary_id):
    """Create a PDF with a ChArUco board centered on A1."""
    board_w = cols * square_size
    board_h = rows * square_size
    
    if board_w > A1_W or board_h > A1_H:
        print(f"Warning: Board size ({board_w}x{board_h}mm) exceeds A1 dimensions ({A1_W}x{A1_H}mm)")
        
    start_x = (A1_W - board_w) / 2
    start_y = (A1_H - board_h) / 2
    
    pdf = FPDF(orientation='L', unit='mm', format=(A1_H, A1_W))
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)
    
    # Draw squares
    marker_idx = 0
    for r in range(rows):
        for c in range(cols):
            x = start_x + c * square_size
            y = start_y + r * square_size
            
            # (0,0) is black. (r+c)%2 == 0 is black.
            if (r + c) % 2 == 0:
                pdf.set_fill_color(0, 0, 0)
                pdf.rect(x, y, square_size, square_size, 'F')
            else:
                # White square - place marker
                # Marker is centered in the square
                m_offset = (square_size - marker_size) / 2
                draw_marker(pdf, marker_idx, x + m_offset, y + m_offset, marker_size, dictionary_id)
                marker_idx += 1
                
    pdf.output(str(output_path))
    print(f"Generated: {output_path}")
    print(f"  Board: {cols}x{rows} grid")
    print(f"  Square size: {square_size}mm, Marker size: {marker_size}mm")
    print(f"  Total size: {board_w}x{board_h}mm")

def main():
    parser = argparse.ArgumentParser(description="Generate ChArUco A1 PDF")
    parser.add_argument("--rows", type=int, default=7, help="Number of rows")
    parser.add_argument("--cols", type=int, default=11, help="Number of columns")
    parser.add_argument("--square-size", type=float, default=70.0, help="Square side in mm")
    parser.add_argument("--marker-size", type=float, default=50.0, help="Marker side in mm")
    parser.add_argument("--output", type=str, default="charuco_a1.pdf", help="Output filename")
    parser.add_argument("--dict", type=str, default="6X6_250", help="ArUco dictionary")
    
    args = parser.parse_args()
    
    dict_map = {
        "4X4_50": cv2.aruco.DICT_4X4_50,
        "5X5_100": cv2.aruco.DICT_5X5_100,
        "6X6_250": cv2.aruco.DICT_6X6_250,
        "7X7_1000": cv2.aruco.DICT_7X7_1000,
    }
    
    if args.dict not in dict_map:
        sys.exit(f"Unknown dictionary: {args.dict}")
        
    build_charuco_pdf(args.output, args.rows, args.cols, args.square_size, args.marker_size, dict_map[args.dict])

if __name__ == "__main__":
    main()
