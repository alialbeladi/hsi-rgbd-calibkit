#!/usr/bin/env python3
"""Generate a PDF calibration target combining ArUco markers with
Li-Wen-Qiu right-triangle calibration pattern (Fig. 2 of the paper).

The pattern consists of 3 filled black right-angled triangles stacked
vertically, whose edges form 6 feature lines (L1-L6):
  - L1, L3, L5: horizontal edges (tops of the 3 triangles)
  - L2, L4, L6: diagonal edges (hypotenuses, all parallel)

Layout: A4 landscape (297 x 210 mm), 4 ArUco markers in the corners,
triangular pattern centred.

Pattern coordinate system (matching the paper):
  - Origin at left end of L1
  - X axis points RIGHT
  - Y axis points DOWN

Outputs:
  combined_target.pdf   — print-ready PDF at exact A4 dimensions
  combined_target.yaml  — machine-readable coordinates (pattern frame, mm)

Requirements:
    pip install fpdf2 opencv-python numpy pyyaml

Usage:
    python scripts/generate_calibration_target.py
    python scripts/generate_calibration_target.py --hp 50 --wp 220
"""

import argparse
import json
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

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ═══════════════════════════════════════════════════════════════════════
# Defaults
# ═══════════════════════════════════════════════════════════════════════
PAGE_W, PAGE_H = 297.0, 210.0          # A4 landscape (mm)
PAGE_MARGIN = 5.0                       # page edge → marker edge
MARKER_SIZE = 20.0                      # ArUco marker side (mm)
MARKER_IDS = [0, 1, 2, 3]              # TL, TR, BL, BR
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250
DATA_BITS, BORDER_BITS = 6, 1
CELLS = DATA_BITS + 2 * BORDER_BITS    # 8

HP_DEFAULT = 60.0   # triangle section height (mm)
WP_DEFAULT = 240.0  # pattern width (mm)


# ═══════════════════════════════════════════════════════════════════════
# ArUco helpers
# ═══════════════════════════════════════════════════════════════════════
def _marker_bits(mid: int) -> np.ndarray:
    """Return CELLS×CELLS bool array (True = black cell)."""
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    img = cv2.aruco.generateImageMarker(d, mid, CELLS)
    return img < 128


def _draw_marker(pdf: FPDF, mid: int, x: float, y: float, size: float):
    """Draw an ArUco marker at (x, y) with given size in mm."""
    bits = _marker_bits(mid)
    cell = size / CELLS
    # White background
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(x, y, size, size, 'F')
    # Black cells
    pdf.set_fill_color(0, 0, 0)
    for r in range(CELLS):
        for c in range(CELLS):
            if bits[r, c]:
                pdf.rect(x + c * cell, y + r * cell, cell, cell, 'F')


# ═══════════════════════════════════════════════════════════════════════
# PDF builder
# ═══════════════════════════════════════════════════════════════════════
def build_pdf(hp: float, wp: float, margin: float, msz: float) -> FPDF:
    pat_h = 3.0 * hp                           # total pattern height
    pat_x = (PAGE_W - wp) / 2.0                # pattern left edge
    pat_y = (PAGE_H - pat_h) / 2.0             # pattern top edge

    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=False)

    # ── White background ──────────────────────────────────────────
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(0, 0, PAGE_W, PAGE_H, 'F')

    # ── 3 black right-angled triangles ────────────────────────────
    # Each triangle i (0-indexed):
    #   Top-left:  (0,  i*hp)      — on L_{2i+1}
    #   Top-right: (wp, i*hp)      — tip of triangle
    #   Bot-left:  (0, (i+1)*hp)   — right-angle vertex
    # Hypotenuse (L_{2i+2}) goes from bot-left to top-right.
    pdf.set_fill_color(0, 0, 0)
    for i in range(3):
        y_top = pat_y + i * hp
        y_bot = pat_y + (i + 1) * hp
        x_left = pat_x
        x_right = pat_x + wp
        # Draw filled triangle using polygon
        pdf.polygon(
            [(x_left, y_top), (x_right, y_top), (x_left, y_bot)],
            style='F'
        )

    # ── ArUco markers in corners ──────────────────────────────────
    positions = [
        (margin, margin),                                       # TL
        (PAGE_W - margin - msz, margin),                        # TR
        (margin, PAGE_H - margin - msz),                        # BL
        (PAGE_W - margin - msz, PAGE_H - margin - msz),        # BR
    ]
    for mid, (mx, my) in zip(MARKER_IDS, positions):
        _draw_marker(pdf, mid, mx, my, msz)

    return pdf


# ═══════════════════════════════════════════════════════════════════════
# Config builder (YAML/JSON with all coordinates)
# ═══════════════════════════════════════════════════════════════════════
def build_config(hp: float, wp: float, margin: float, msz: float) -> dict:
    pat_h = 3.0 * hp
    pat_x = (PAGE_W - wp) / 2.0
    pat_y = (PAGE_H - pat_h) / 2.0
    # Pattern origin in page coords (top-left of L1)
    ox = pat_x
    oy = pat_y

    # Marker corners in pattern coords (Y-down, origin at L1 left end)
    positions = [
        (margin, margin),
        (PAGE_W - margin - msz, margin),
        (margin, PAGE_H - margin - msz),
        (PAGE_W - margin - msz, PAGE_H - margin - msz),
    ]
    markers = {}
    for mid, (sx, sy) in zip(MARKER_IDS, positions):
        svg_corners = [(sx, sy), (sx+msz, sy), (sx+msz, sy+msz), (sx, sy+msz)]
        pat_corners = [[round(cx - ox, 2), round(cy - oy, 2)]
                       for cx, cy in svg_corners]
        markers[mid] = {"corners_mm": pat_corners, "size_mm": msz}

    # Feature line equations  ax + by + c = 0  (pattern coords, Y-down)
    feature_lines = {
        "L1": {"eq": [0, 1, 0],              "type": "horizontal", "y": 0},
        "L2": {"eq": [hp, wp, -hp*wp],        "type": "diagonal",
               "from": [0, hp], "to": [wp, 0]},
        "L3": {"eq": [0, 1, -hp],             "type": "horizontal", "y": hp},
        "L4": {"eq": [hp, wp, -2*hp*wp],      "type": "diagonal",
               "from": [0, 2*hp], "to": [wp, hp]},
        "L5": {"eq": [0, 1, -2*hp],           "type": "horizontal", "y": 2*hp},
        "L6": {"eq": [hp, wp, -3*hp*wp],      "type": "diagonal",
               "from": [0, 3*hp], "to": [wp, 2*hp]},
    }

    # 9 intersection points (horizontal ∩ diagonal)
    intersections = {}
    for hi, hy in [("L1", 0), ("L3", hp), ("L5", 2*hp)]:
        for di, d_idx in [("L2", 1), ("L4", 2), ("L6", 3)]:
            x = (d_idx * hp - hy) * wp / hp
            if 0 <= x <= wp:
                intersections[f"{hi}∩{di}"] = [round(x, 2), round(hy, 2)]

    return {
        "description": (
            "Combined calibration target: ArUco + Li-Wen-Qiu right-triangle pattern. "
            "Coordinates in pattern frame (mm). "
            "Origin at left end of L1. X right, Y down."
        ),
        "page": {"width_mm": PAGE_W, "height_mm": PAGE_H, "orientation": "landscape"},
        "pattern_origin_on_page_mm": {"x": round(ox, 2), "y": round(oy, 2)},
        "aruco": {
            "dictionary": "DICT_6X6_250",
            "marker_size_mm": msz,
            "marker_ids": MARKER_IDS,
            "markers": markers,
        },
        "li_wen_qiu": {
            "hp_mm": hp,
            "wp_mm": wp,
            "feature_lines": feature_lines,
            "intersection_points_mm": intersections,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", type=Path,
                    default=Path("assets/calibration_targets"))
    ap.add_argument("--hp", type=float, default=HP_DEFAULT,
                    help=f"Triangle section height in mm (default: {HP_DEFAULT})")
    ap.add_argument("--wp", type=float, default=WP_DEFAULT,
                    help=f"Pattern width in mm (default: {WP_DEFAULT})")
    ap.add_argument("--marker-size", type=float, default=MARKER_SIZE,
                    help=f"ArUco marker side in mm (default: {MARKER_SIZE})")
    ap.add_argument("--margin", type=float, default=PAGE_MARGIN,
                    help=f"Page margin in mm (default: {PAGE_MARGIN})")
    args = ap.parse_args()

    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pdf_path = out / "combined_target.pdf"
    cfg_path = out / "combined_target.yaml"

    # ── Generate PDF ──────────────────────────────────────────────
    pdf = build_pdf(args.hp, args.wp, args.margin, args.marker_size)
    pdf.output(str(pdf_path))
    print(f"[OK] PDF    -> {pdf_path}")

    # ── Generate config ───────────────────────────────────────────
    config = build_config(args.hp, args.wp, args.margin, args.marker_size)
    if HAS_YAML:
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False,
                      allow_unicode=True)
    else:
        cfg_path = cfg_path.with_suffix(".json")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    print(f"[OK] Config -> {cfg_path}")

    print(f"\n  hp={args.hp}mm  wp={args.wp}mm  "
          f"pattern: {args.wp:.0f}x{3*args.hp:.0f}mm  "
          f"markers: {len(MARKER_IDS)}x{args.marker_size:.0f}mm")
    print("  Print at 100% scale on A4 landscape. Measure after printing!")


if __name__ == "__main__":
    main()
