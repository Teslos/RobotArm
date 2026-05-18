#!/usr/bin/env python3
"""
Direct control of the physical Meca500 R3 via the official mecademicpy SDK.

Install SDK:
    pip install mecademicpy

Usage:
    python scripts/meca500_real.py --home-only
    python scripts/meca500_real.py --demo
    python scripts/meca500_real.py --scan
    python scripts/meca500_real.py --scan --scan-path results/scan_path.npy
    python scripts/meca500_real.py --scan --ip 192.168.0.100

==============================================================================
SAFETY INSTRUCTIONS — READ BEFORE OPERATING THE MECA500 R3
==============================================================================

PRE-OPERATION CHECKLIST
  [ ] Emergency stop (e-stop) button is within arm's reach and functional.
  [ ] The robot work envelope (≈500 mm sphere) is completely clear of people,
      tools, cables, and the busbar fixture before starting.
  [ ] The robot base is bolted down or clamped firmly to the bench.
  [ ] The busbar fixture is secured so it cannot shift during scanning.
  [ ] You have read the Meca500 R3 User Manual, especially §5 (Safety) and
      §7 (Installation and Setup).
  [ ] A second person is present, or you have informed someone that the arm
      will be moving unattended.

DURING OPERATION
  [ ] Keep your hand near the e-stop at all times during the first run.
  [ ] Do NOT lean into the work envelope while the arm is powered.
  [ ] If the arm moves unexpectedly, press e-stop first — investigate after.
  [ ] If you hear grinding, clicking, or unusual motor noise, stop immediately.
  [ ] Joint velocity is limited to 20 % of maximum in this script. Do not
      raise JOINT_VEL_PCT above 30 % without additional collision checking.
  [ ] Cartesian scan velocity is limited to 20 mm/s. Do not exceed 50 mm/s
      near the busbar surface.

JOINT LIMITS (hardware hard stops — software limits are tighter by 5°)
  J1: ±175°   J2: −70° → +90°   J3: −135° → +70°
  J4: ±170°   J5: ±115°          J6: ±180°

SCAN GEOMETRY (relative to robot base, same frame as the sim)
  Busbar position: X=190 mm, Z centre=86.25 mm (bottom at 65 mm, top at 107.5 mm)
                  Rotated 90° around Y — wide face (100×404 mm) now horizontal.
  Approach pose : joints = [−2.4°, −2.0°, 52.3°, −2.2°, −32.1°, −37.6°]
                  EE position: [180mm, −6mm, 180mm] (72mm above busbar top).
  Hover height  : EE Z ≈ 188 mm  (≈80 mm above busbar top surface at 107.5 mm)
  Scan X        : 190 mm          (directly above busbar centre)
  Scan Y range  : −126 mm → +126 mm  (252 mm busbar length)

AFTER OPERATION
  [ ] The script deactivates the robot automatically in the finally block.
  [ ] Once deactivated, the joints are unpowered — support the arm if needed.
  [ ] Power off the robot controller before touching the arm mechanically.

==============================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# ── mecademicpy ────────────────────────────────────────────────────────────────
try:
    from mecademicpy.robot import Robot
except ImportError as exc:
    sys.exit(
        f"[ERROR] mecademicpy not installed: {exc}\n"
        "  Install with:  pip install mecademicpy"
    )

# ==============================================================================
# SAFETY CONSTANTS — change these only if you know what you are doing
# ==============================================================================

ROBOT_IP: str = "192.168.0.100"   # default Meca500 IP
ROBOT_PORT: int = 10000            # default SDK port

# Velocity limits — conservative for first-time use
JOINT_VEL_PCT: float = 20.0       # % of max joint velocity  (1–100)
CART_LIN_VEL_MM_S: float = 20.0   # mm/s Cartesian linear    (robot max ~150)
CART_ANG_VEL_DEG_S: float = 45.0  # deg/s Cartesian angular

WAIT_IDLE_TIMEOUT_S: float = 60.0  # per-move timeout

# Software joint limits — 5° tighter than hardware hard stops on every axis
SW_LIMITS_DEG: List[Tuple[float, float]] = [
    (-170.0,  170.0),   # J1
    ( -65.0,   85.0),   # J2
    (-130.0,   65.0),   # J3
    (-165.0,  165.0),   # J4
    (-110.0,  110.0),   # J5
    (-175.0,  175.0),   # J6
]

# Cartesian workspace safety box (mm, in robot base frame)
# Reject any Cartesian target outside this box before sending to robot.
WS_X: Tuple[float, float] = (  50.0,  350.0)
WS_Y: Tuple[float, float] = (-250.0,  250.0)
WS_Z: Tuple[float, float] = (  20.0,  400.0)

# ==============================================================================
# SCAN GEOMETRY — matches simulation constants (metres → mm)
# ==============================================================================

DEMO_JOINT_DEG: List[float] = [-2.4, -2.0, 52.3, -2.2, -32.1, -37.6]

SCAN_X_MM: float = 190.0      # EE X during scan — directly above busbar centre
SCAN_Z_MM: float = 188.0      # EE hover height — busbar top 107.5 mm + 80 mm clearance
BUSBAR_Y_MM: float = 0.0      # busbar centre Y
HALF_SCAN_MM: float = 126.0   # half of 252 mm busbar length
N_SCAN_POINTS: int = 40       # waypoints along the sweep


# ==============================================================================
# Safety helpers
# ==============================================================================

def check_joint_limits(joints_deg: List[float]) -> None:
    """Raise ValueError if any joint violates the software limits."""
    for i, (j, (lo, hi)) in enumerate(zip(joints_deg, SW_LIMITS_DEG), start=1):
        if not (lo <= j <= hi):
            raise ValueError(
                f"J{i} = {j:.2f}° violates software limit [{lo}°, {hi}°]"
            )


def check_workspace(x: float, y: float, z: float) -> None:
    """Raise ValueError if the Cartesian target is outside the safety box."""
    def _chk(v, lo, hi, axis):
        if not (lo <= v <= hi):
            raise ValueError(
                f"{axis} = {v:.1f} mm outside workspace [{lo}, {hi}] mm"
            )
    _chk(x, *WS_X, "X")
    _chk(y, *WS_Y, "Y")
    _chk(z, *WS_Z, "Z")


def safe_move_joints(robot: Robot, joints_deg: List[float]) -> None:
    """Check limits, then issue MoveJoints and wait for completion."""
    check_joint_limits(joints_deg)
    robot.MoveJoints(*joints_deg)
    robot.WaitIdle(timeout=WAIT_IDLE_TIMEOUT_S)
    _check_robot_error(robot)


def safe_move_lin(
    robot: Robot,
    x: float, y: float, z: float,
    alpha: float, beta: float, gamma: float,
) -> None:
    """Check workspace bounds, then issue MoveLin and wait for completion."""
    check_workspace(x, y, z)
    robot.MoveLin(x, y, z, alpha, beta, gamma)
    robot.WaitIdle(timeout=WAIT_IDLE_TIMEOUT_S)
    _check_robot_error(robot)


def _check_robot_error(robot: Robot) -> None:
    """Raise RuntimeError if the robot is in an error state."""
    info = robot.GetRobotInfo()
    if info is None:
        return
    # mecademicpy ≥2.x: RobotInfo has error_status
    if hasattr(info, "error_status") and info.error_status:
        raise RuntimeError(
            f"[ROBOT ERROR] error_status={info.error_status} — "
            "call robot.ResetError() after investigating."
        )


# ==============================================================================
# Connection / activation
# ==============================================================================

def connect_and_activate(ip: str) -> Robot:
    """
    Connect to the robot, clear any prior error, activate servo drives,
    and home the arm.  Returns the connected, homed Robot instance.

    Homing moves the arm to the joint-zero reference position using the
    integrated mechanical limit switches — keep the workspace clear.
    """
    print(f"[robot] Connecting to Meca500 at {ip}:{ROBOT_PORT} …")
    robot = Robot()
    robot.Connect(address=ip, enable_synchronous_mode=True)
    print("[robot] Connected.")

    # Clear any latent error from a previous session
    try:
        robot.ResetError()
        robot.ResumeMotion()
    except Exception:
        pass

    _check_robot_error(robot)

    print("[robot] Activating servo drives …")
    robot.ActivateRobot()

    print("[robot] Homing … (arm will move — ensure workspace is clear)")
    robot.Home()
    print("[robot] Homing complete.")

    # Apply velocity limits
    robot.SetJointVel(JOINT_VEL_PCT)
    robot.SetCartLinVel(CART_LIN_VEL_MM_S)
    robot.SetCartAngVel(CART_ANG_VEL_DEG_S)
    print(
        f"[robot] Velocity limits: joint={JOINT_VEL_PCT}%  "
        f"linear={CART_LIN_VEL_MM_S} mm/s  angular={CART_ANG_VEL_DEG_S} deg/s"
    )

    return robot


def safe_shutdown(robot: Robot) -> None:
    """Deactivate drives and disconnect cleanly — always call in finally block."""
    try:
        robot.DeactivateRobot()
        print("[robot] Drives deactivated.")
    except Exception as exc:
        print(f"[robot] DeactivateRobot warning: {exc}")
    try:
        robot.Disconnect()
        print("[robot] Disconnected.")
    except Exception as exc:
        print(f"[robot] Disconnect warning: {exc}")


# ==============================================================================
# Operation modes
# ==============================================================================

def run_home_only(robot: Robot) -> None:
    """Activate, home, then immediately deactivate — used to verify connection."""
    print("[mode] Home-only: connection and homing verified. Done.")


def run_demo(robot: Robot) -> None:
    """
    Move to the busbar approach pose using validated joint targets from the
    Isaac Sim IK grid search (DEMO_JOINT_DEG).
    """
    print(f"[demo] Moving to approach pose: {DEMO_JOINT_DEG} deg")
    safe_move_joints(robot, DEMO_JOINT_DEG)

    pose = robot.GetPose()
    joints = robot.GetJoints()
    print(f"[demo] Reached.  Joints (deg): {joints}")
    print(f"[demo] Pose (mm, deg)        : {pose}")
    print("[demo] Arm is above busbar centre. Press Ctrl+C to continue shutdown.")
    try:
        time.sleep(5.0)
    except KeyboardInterrupt:
        pass


def run_scan(robot: Robot, scan_path_file: Optional[str] = None) -> None:
    """
    Phase 1 — Approach: move to busbar hover pose (joint space, validated IK).
    Phase 2 — Sweep  : linear scan along the busbar at constant height.

    If scan_path_file is given, loads an (N, 3) array of XYZ waypoints in mm.
    Otherwise performs the default 252 mm linear Y sweep.

    The EE orientation (alpha, beta, gamma) is read from the robot after the
    approach move, so it is always consistent with the physical configuration.
    """
    # ── Phase 1: approach ────────────────────────────────────────────────────
    print(f"[scan] Phase 1 — Approach: moving to {DEMO_JOINT_DEG} deg")
    safe_move_joints(robot, DEMO_JOINT_DEG)

    # Read actual EE pose to lock in orientation for linear moves
    pose = robot.GetPose()
    print(f"[scan] Approach pose: {pose}")

    if pose is None or len(pose) < 6:
        raise RuntimeError(
            "[scan] GetPose() returned unexpected value — cannot lock EE orientation."
        )
    ee_x, ee_y, ee_z, ee_alpha, ee_beta, ee_gamma = (float(v) for v in pose[:6])
    print(
        f"[scan] EE position  : x={ee_x:.1f} y={ee_y:.1f} z={ee_z:.1f} mm\n"
        f"[scan] EE orientation: α={ee_alpha:.1f}° β={ee_beta:.1f}° γ={ee_gamma:.1f}°"
    )

    # ── Build scan waypoints ─────────────────────────────────────────────────
    if scan_path_file is not None:
        waypoints = np.load(scan_path_file)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            raise ValueError(
                f"scan_path file must be (N, 3), got shape {waypoints.shape}"
            )
        # Convert from metres (sim convention) to mm if values look like metres
        if waypoints[:, :3].max() < 5.0:
            waypoints = waypoints * 1000.0
            print(f"[scan] Converted waypoints from m → mm")
        print(f"[scan] Loaded {len(waypoints)}-point path from {scan_path_file}")
    else:
        y_vals = np.linspace(
            BUSBAR_Y_MM - HALF_SCAN_MM,
            BUSBAR_Y_MM + HALF_SCAN_MM,
            N_SCAN_POINTS,
        )
        waypoints = np.column_stack([
            np.full(N_SCAN_POINTS, SCAN_X_MM),
            y_vals,
            np.full(N_SCAN_POINTS, SCAN_Z_MM),
        ])
        print(
            f"[scan] Default linear sweep: {N_SCAN_POINTS} points  "
            f"Y: {y_vals[0]:.0f} → {y_vals[-1]:.0f} mm"
        )

    # Validate all waypoints against workspace bounds before starting
    print("[scan] Pre-validating all waypoints against workspace limits …")
    for i, (wx, wy, wz) in enumerate(waypoints):
        try:
            check_workspace(wx, wy, wz)
        except ValueError as exc:
            raise ValueError(f"Waypoint {i}: {exc}") from exc
    print(f"[scan] All {len(waypoints)} waypoints OK.")

    # ── Phase 2: sweep ───────────────────────────────────────────────────────
    print(f"[scan] Phase 2 — Sweep: executing {len(waypoints)} Cartesian moves …")
    for i, (wx, wy, wz) in enumerate(waypoints):
        pct = (i + 1) / len(waypoints) * 100
        print(
            f"[scan] Waypoint {i+1:3d}/{len(waypoints)}  "
            f"x={wx:.1f} y={wy:.1f} z={wz:.1f} mm  ({pct:.0f}%)"
        )
        robot.MoveLin(wx, wy, wz, ee_alpha, ee_beta, ee_gamma)

    # Wait for the last move to finish
    robot.WaitIdle(timeout=WAIT_IDLE_TIMEOUT_S)
    _check_robot_error(robot)

    final_pose = robot.GetPose()
    print(f"[scan] Scan complete. Final pose: {final_pose}")

    # Return to safe approach pose
    print("[scan] Returning to approach pose …")
    safe_move_joints(robot, DEMO_JOINT_DEG)
    print("[scan] Done.")


# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Meca500 R3 direct robot control — read safety instructions at top of file"
    )
    parser.add_argument("--ip", default=ROBOT_IP, help=f"Robot IP (default: {ROBOT_IP})")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--home-only", action="store_true",
        help="Connect, home, then deactivate — for connection testing"
    )
    mode.add_argument(
        "--demo", action="store_true",
        help="Move to busbar approach pose and hold for 5 s"
    )
    mode.add_argument(
        "--scan", action="store_true",
        help=f"Approach busbar then sweep {HALF_SCAN_MM*2:.0f} mm along Y"
    )
    parser.add_argument(
        "--scan-path", metavar="FILE.npy",
        help="(N,3) array of EE XYZ waypoints in mm (or m — auto-detected)"
    )
    args = parser.parse_args()

    # ── Print safety reminder ────────────────────────────────────────────────
    print("=" * 70)
    print("MECA500 R3 REAL ROBOT CONTROL")
    print("=" * 70)
    print("  >> Ensure workspace is clear before proceeding.")
    print(f"  >> Joint velocity cap : {JOINT_VEL_PCT} %")
    print(f"  >> Cartesian vel cap  : {CART_LIN_VEL_MM_S} mm/s")
    print("  >> Press Ctrl+C at any time to pause and deactivate.")
    print("=" * 70)
    input("  Press ENTER to continue or Ctrl+C to abort … ")

    robot: Optional[Robot] = None
    try:
        robot = connect_and_activate(args.ip)

        if args.home_only:
            run_home_only(robot)
        elif args.demo:
            run_demo(robot)
        elif args.scan:
            if args.scan_path and not os.path.exists(args.scan_path):
                raise FileNotFoundError(f"scan-path file not found: {args.scan_path}")
            run_scan(robot, scan_path_file=args.scan_path)

    except KeyboardInterrupt:
        print("\n[robot] Keyboard interrupt — pausing motion and shutting down …")
        if robot is not None:
            try:
                robot.PauseMotion()
                robot.ClearMotion()
            except Exception:
                pass
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    finally:
        if robot is not None:
            safe_shutdown(robot)


if __name__ == "__main__":
    main()
