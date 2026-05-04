#!/usr/bin/env python3
"""
IK-based grid search over the busbar region.

Runs LulaKinematicsSolver on a 3-D Cartesian grid, saves reachability
results to an .npz file, and prints a summary + best approach point.

Usage:
    micromamba run -n RobotArm python scripts/map_workspace.py
    micromamba run -n RobotArm python scripts/map_workspace.py --output results/workspace.npz
    micromamba run -n RobotArm python scripts/map_workspace.py --nx 10 --ny 10 --nz 6

Prerequisites:
    assets/mecharm_270/rmpflow/robot_descriptor.yaml  (LULA robot description)
    assets/mecharm_270/mecharm_270.urdf
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR  = os.path.join(REPO_ROOT, "assets")
URDF_PATH   = os.path.join(ASSETS_DIR, "mecademic_description", "urdf", "meca500r3.urdf")
ROBOT_DESC  = os.path.join(ASSETS_DIR, "mecademic_description", "rmpflow", "robot_descriptor.yaml")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
sys.path.insert(0, REPO_ROOT)


def _build_solver():
    try:
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    except ImportError:
        from omni.isaac.motion_generation.lula import LulaKinematicsSolver  # type: ignore[no-redef]

    if not os.path.exists(ROBOT_DESC):
        raise FileNotFoundError(
            f"LULA robot descriptor not found: {ROBOT_DESC}\n"
            "Generate it with Isaac Sim's LULA robot description generator, or\n"
            "copy it from the RMPFlow asset pack for the mechArm 270-Pi."
        )
    return LulaKinematicsSolver(
        robot_description_path=ROBOT_DESC,
        urdf_path=URDF_PATH,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IK-based workspace grid search")
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "workspace.npz"),
                        help="Output .npz path (default: results/workspace.npz)")
    parser.add_argument("--nx", type=int, default=8, help="Grid points along X (default 8)")
    parser.add_argument("--ny", type=int, default=8, help="Grid points along Y (default 8)")
    parser.add_argument("--nz", type=int, default=5, help="Grid points along Z (default 5)")
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.10, 0.35],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--y-range", nargs=2, type=float, default=[-0.25, 0.25],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--z-range", nargs=2, type=float, default=[0.10, 0.35],
                        metavar=("MIN", "MAX"))
    args = parser.parse_args()

    # SimulationApp must be created before any omni.* imports
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})

    import numpy as np
    from exts.robot_arm.config import GridSearchCfg
    from exts.robot_arm.tasks.workspace_mapper import WorkspaceMapper

    cfg = GridSearchCfg(
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
    )

    print(f"[workspace] Grid: {cfg.nx}×{cfg.ny}×{cfg.nz} = {cfg.nx * cfg.ny * cfg.nz} points")
    print(f"[workspace] X {cfg.x_range}, Y {cfg.y_range}, Z {cfg.z_range}")

    solver = _build_solver()
    mapper = WorkspaceMapper(ik_solver=solver)
    wmap = mapper.run(cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    wmap.save(args.output)
    print(f"[workspace] Saved to {args.output}")

    # Print the best approach point toward the busbar centre
    # Busbar at (0.15, 0.0, 0.03), top surface z=0.08 m → 10 cm above = 0.18 m
    busbar_centre = np.array([0.15, 0.0, 0.18])
    try:
        ee_pos, joints = wmap.best_approach(busbar_centre)
        print(f"[workspace] Best approach point : {ee_pos.tolist()}")
        print(f"[workspace] Joint angles (rad)  : {joints.tolist()}")
    except RuntimeError as e:
        print(f"[workspace] {e}")

    app.close()


if __name__ == "__main__":
    main()
