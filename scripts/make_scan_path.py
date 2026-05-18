#!/usr/bin/env python3
"""
Generate a scan path as an (N, 3) numpy array and save it to a .npy file.

Usage:
    micromamba run -n RobotArm python scripts/make_scan_path.py --shape zigzag
    micromamba run -n RobotArm python scripts/make_scan_path.py --shape circle
    micromamba run -n RobotArm python scripts/make_scan_path.py --shape spiral

Then run the sim with the generated path:
    micromamba run -n RobotArm python scripts/start_sim.py --scan --scan-path results/scan_path.npy
"""
from __future__ import annotations

import argparse
import os

import numpy as np

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "results")

# Busbar geometry (must match start_sim.py constants)
SCAN_X_M      = 0.176   # fixed EE X during scan (busbar near face 0.169 m + 7 mm offset)
SCAN_HEIGHT_M = 0.247   # fixed EE Z during scan (busbar top 0.165 m + 82 mm clearance)
BUSBAR_Y      = 0.0     # busbar centre Y
HALF          = 0.126   # half-length of busbar (252 mm / 2)


def zigzag(rows: int = 5, cols: int = 8) -> np.ndarray:
    """Raster scan: sweep X, step Y, reverse X on alternate rows."""
    y_vals = np.linspace(BUSBAR_Y - HALF, BUSBAR_Y + HALF, rows)
    x_vals = np.linspace(SCAN_X_M - 0.015, SCAN_X_M + 0.015, cols)
    pts = []
    for i, y in enumerate(y_vals):
        xs = x_vals if i % 2 == 0 else x_vals[::-1]
        pts.extend([[x, y, SCAN_HEIGHT_M] for x in xs])
    return np.array(pts)


def circle(n: int = 60, radius: float = 0.06) -> np.ndarray:
    """Circle centred above busbar centre."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.array([
        [SCAN_X_M + radius * np.cos(a),
         BUSBAR_Y + radius * np.sin(a),
         SCAN_HEIGHT_M]
        for a in angles
    ])


def spiral(turns: int = 3, n: int = 120, max_radius: float = 0.06) -> np.ndarray:
    """Outward spiral centred above busbar centre."""
    angles = np.linspace(0, turns * 2 * np.pi, n)
    radii  = np.linspace(0.005, max_radius, n)
    return np.array([
        [SCAN_X_M + r * np.cos(a),
         BUSBAR_Y + r * np.sin(a),
         SCAN_HEIGHT_M]
        for r, a in zip(radii, angles)
    ])


SHAPES = {"zigzag": zigzag, "circle": circle, "spiral": spiral}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a scan path .npy file")
    parser.add_argument("--shape", choices=list(SHAPES), default="zigzag",
                        help="Path shape to generate (default: zigzag)")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, "scan_path.npy"),
                        help="Output .npy file path")
    args = parser.parse_args()

    waypoints = SHAPES[args.shape]()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, waypoints)

    print(f"[make_scan_path] Shape   : {args.shape}")
    print(f"[make_scan_path] Points  : {len(waypoints)}")
    print(f"[make_scan_path] X range : {waypoints[:,0].min():.3f} → {waypoints[:,0].max():.3f} m")
    print(f"[make_scan_path] Y range : {waypoints[:,1].min():.3f} → {waypoints[:,1].max():.3f} m")
    print(f"[make_scan_path] Z range : {waypoints[:,2].min():.3f} → {waypoints[:,2].max():.3f} m")
    print(f"[make_scan_path] Saved   : {args.output}")
    print()
    print("Run the sim with:")
    print(f"  micromamba run -n RobotArm python scripts/start_sim.py --scan --scan-path {args.output}")


if __name__ == "__main__":
    main()
