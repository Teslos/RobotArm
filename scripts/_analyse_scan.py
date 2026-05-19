"""Find reachable scan waypoints: best X line covering Y=-0.126..+0.126 at Z~0.162."""
import numpy as np

d = np.load("results/workspace.npz")
grid  = d["grid_points"]
reach = d["reachable"]
jpos  = d["joint_positions"]

pts = grid[reach]
jp  = jpos[reach]

Z_TARGET = 0.162
Z_TOL    = 0.030
Y_HALF   = 0.126

# Restrict to Z band
mz = np.abs(pts[:, 2] - Z_TARGET) < Z_TOL
band = pts[mz];  band_j = jp[mz]
print(f"Reachable pts in Z band ({Z_TARGET-Z_TOL:.2f}–{Z_TARGET+Z_TOL:.2f} m): {len(band)}")

# For each candidate X, check whether scan start and end are reachable (< 25 mm away)
print("\nX vs nearest-pt distance to ideal scan line [X, ±0.126, 0.162]:")
xs = np.unique(np.round(band[:, 0], 3))
for x in xs:
    mx = np.abs(band[:, 0] - x) < 0.02
    sub = band[mx]
    if len(sub) == 0: continue
    d_start = np.min(np.linalg.norm(sub - [x, -Y_HALF, Z_TARGET], axis=1))
    d_ctr   = np.min(np.linalg.norm(sub - [x,      0, Z_TARGET], axis=1))
    d_end   = np.min(np.linalg.norm(sub - [x, +Y_HALF, Z_TARGET], axis=1))
    print(f"  X={x:.3f}: Δ_start={d_start*1000:.0f}mm  Δ_ctr={d_ctr*1000:.0f}mm  Δ_end={d_end*1000:.0f}mm  ({len(sub)} pts)")

# Pick the X with minimum worst-case distance across all three waypoints
best_x, best_worst = None, 1e9
for x in xs:
    mx = np.abs(band[:, 0] - x) < 0.02
    sub = band[mx]
    if len(sub) == 0: continue
    worst = max(
        np.min(np.linalg.norm(sub - [x, -Y_HALF, Z_TARGET], axis=1)),
        np.min(np.linalg.norm(sub - [x,      0,  Z_TARGET], axis=1)),
        np.min(np.linalg.norm(sub - [x, +Y_HALF, Z_TARGET], axis=1)),
    )
    if worst < best_worst:
        best_worst = worst
        best_x = x

print(f"\nBest X: {best_x:.3f} m  (worst waypoint gap {best_worst*1000:.0f} mm)")
mx = np.abs(band[:, 0] - best_x) < 0.02
sub = band[mx];  sub_j = jp[mz][np.abs(band[:, 0] - best_x) < 0.02]
for label, y_tgt in [("centre", 0.0), ("scan start", -Y_HALF), ("scan end", +Y_HALF)]:
    dists = np.linalg.norm(sub - [best_x, y_tgt, Z_TARGET], axis=1)
    i = np.argmin(dists)
    p = sub[i];  j = np.degrees(sub_j[i])
    print(f"  {label}: pos={np.round(p,3).tolist()}  dist={dists[i]*1000:.0f}mm  joints={np.round(j,1).tolist()}")
