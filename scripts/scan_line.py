#!/usr/bin/env python3
"""
Single-line scan on the physical Meca500 R3, synchronised with INT_Monitor.

Replaces three scripts from ``C:\\EmpaDaten\\INT_Monitor\\py``:

=====================================  ======================================
Original                               Equivalent invocation
=====================================  ======================================
``Width X And Z Measurment .py``       ``--axis x --n-points 22 --step 0.5
                                         --axis-pair xz --settle 1.0
                                         --verify-up 40``
``UPDATE mit SYNCRO.py``               ``--axis x --n-points 10 --step -5
                                         --axis-pair xz --settle 0.5
                                         --verify-up 20 --verify-down 20``
``Y and Z Measurment.py``              ``--axis y --n-points 120 --step 1
                                         --axis-pair yz --settle 1.0
                                         --verify-up 40``
=====================================  ======================================

Usage::

    micromamba run -n RobotArm python scripts/scan_line.py --dry-run
    micromamba run -n RobotArm python scripts/scan_line.py \
        --axis y --n-points 120 --step 1 --axis-pair yz

Note on step signs: the originals combined a negative ``STEP`` constant with
``target = start - i * STEP``, so the arm travelled the *opposite* way to what
the constant suggested.  Here ``--step`` is the signed travel per point, full
stop.

==============================================================================
SAFETY - READ scripts/meca500_real.py FOR THE FULL CHECKLIST
==============================================================================
  * Homing moves the arm. Clear the work envelope first.
  * --dry-run validates the geometry without contacting the robot.
  * --verify-up / --verify-down replay the post-scan Z excursions the original
    scripts performed; both default to 0 (no excursion).
  * When the scan finishes the arm returns to its start pose and stays there,
    activated and connected; --disconnect-when-done deactivates and disconnects
    instead.  An error or a Ctrl+C always ends in a full shutdown.
==============================================================================
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# `metrology` is imported as a top-level package rather than as
# `exts.robot_arm.metrology`, because exts/robot_arm/__init__.py pulls in
# Isaac Sim.  Driving the physical arm does not need a simulator installed.
sys.path.insert(0, os.path.join(REPO_ROOT, "exts", "robot_arm"))

from metrology import (  # noqa: E402
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_ROBOT_IP,
    AxisPair,
    InstrumentLink,
    PointLog,
    RobotSession,
    ScanSettings,
    VelocityLimits,
    check_waypoints,
    connect_robot,
    line_scan,
    run_scan,
    serve,
    write_point_file,
)
from metrology.limits import DEFAULT_WORKSPACE, LimitViolation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="INT_Monitor-synchronised single-line scan (mm, degrees)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    line = parser.add_argument_group("line geometry")
    line.add_argument("--axis", choices=("x", "y", "z"), default="x",
                      help="Axis the line runs along")
    line.add_argument("--n-points", type=int, default=22,
                      help="Number of measurement points, including the origin")
    line.add_argument("--step", type=float, default=0.5, metavar="MM",
                      help="Signed travel per point")
    line.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "Z"),
                      help="Line origin in mm; default is the arm's current pose")

    motion = parser.add_argument_group("motion")
    motion.add_argument("--settle", type=float, default=1.0, metavar="SECONDS",
                        help="Dwell after each move, before triggering")
    motion.add_argument("--dwell", type=float, default=0.0, metavar="SECONDS",
                        help="Extra hold after each acquisition")
    motion.add_argument("--move-pose", action="store_true",
                        help="Use joint-interpolated MovePose instead of MoveLin")
    motion.add_argument("--verify-up", type=float, default=0.0, metavar="MM",
                        help="After the scan, rise this far in Z and return")
    motion.add_argument("--verify-down", type=float, default=0.0, metavar="MM",
                        help="After the scan, drop this far in Z and return")
    motion.add_argument("--joint-vel", type=float, default=20.0, metavar="PCT",
                        help="Joint velocity cap, %% of maximum")
    motion.add_argument("--cart-vel", type=float, default=20.0, metavar="MM_S",
                        help="Cartesian linear velocity cap")

    net = parser.add_argument_group("instrument link")
    net.add_argument("--host", default=DEFAULT_HOST, help="TCP bind address")
    net.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port")
    net.add_argument("--axis-pair", default="xz",
                     choices=[pair.name.lower() for pair in AxisPair],
                     help="Coordinate pair streamed in each point label")
    net.add_argument("--response-timeout", type=float, default=30.0,
                     metavar="SECONDS",
                     help="Per-point wait for the instrument's 'done'")
    net.add_argument("--accept-timeout", type=float, default=None,
                     metavar="SECONDS",
                     help="Give up if no instrument connects in time")

    io_group = parser.add_argument_group("output")
    io_group.add_argument("--output", default=os.path.join(RESULTS_DIR, "line_scan.csv"),
                          help="CSV log of measured points")
    io_group.add_argument("--point-files", metavar="DIR",
                          help="Also write the legacy one-file-per-point text "
                               "dumps into DIR")

    parser.add_argument("--ip", default=DEFAULT_ROBOT_IP, help="Robot IP address")
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate and validate the waypoints, save them to "
                             "a .npy next to --output, and exit without moving")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the interactive safety confirmation")
    parser.add_argument("--disconnect-when-done", action="store_true",
                        help="Deactivate the drives and disconnect after a "
                             "successful scan. By default the arm just returns "
                             "to its start pose and stays activated and homed, "
                             "so the next scan needs no re-homing")
    return parser


def describe(path, origin) -> None:
    print(f"[line] Origin    : x={origin[0]:.2f} y={origin[1]:.2f} z={origin[2]:.2f} mm")
    print(f"[line] Waypoints : {len(path)}")
    for axis, (low, high) in zip("XYZ", path.extent_mm):
        print(f"[line] {axis} range   : {low:.2f} -> {high:.2f} mm")


def run_verification_moves(session: RobotSession, home_pose, up_mm: float, down_mm: float) -> None:
    """Replay the post-scan Z excursions, each relative to the recorded pose.

    Offsets are applied to ``home_pose`` rather than to the live position, so a
    partially completed excursion cannot leave the arm somewhere unexpected.
    A zero offset is skipped instead of issuing a move to where the arm already
    is, which is what the originals did with their ``LOWER_OFFSET = 0``.
    """
    if up_mm:
        print(f"[line] Verification: +{up_mm:.1f} mm in Z")
        session.move_lin_offset(home_pose, dz=up_mm)
        session.move_lin(home_pose)
    if down_mm:
        print(f"[line] Verification: -{down_mm:.1f} mm in Z")
        session.move_lin_offset(home_pose, dz=-down_mm)
        session.move_lin(home_pose)


def main() -> int:
    args = build_parser().parse_args()

    axis_pair = AxisPair.parse(args.axis_pair)
    settings = ScanSettings(
        settle_s=args.settle,
        dwell_s=args.dwell,
        response_timeout_s=args.response_timeout,
        use_move_lin=not args.move_pose,
    )

    if args.dry_run:
        origin = args.origin or [190.0, 0.0, 188.0]
        path = line_scan(origin, args.axis, args.n_points, args.step)
        describe(path, origin)
        try:
            check_waypoints(path.points, DEFAULT_WORKSPACE)
        except LimitViolation as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            print("[ERROR] Adjust the line, or widen DEFAULT_WORKSPACE in "
                  "exts/robot_arm/metrology/limits.py if the box is wrong.",
                  file=sys.stderr)
            return 1
        print("[line] All waypoints are inside the workspace box.")
        npy_path = os.path.splitext(args.output)[0] + "_waypoints.npy"
        os.makedirs(os.path.dirname(os.path.abspath(npy_path)), exist_ok=True)
        np.save(npy_path, path.points)
        print(f"[line] Waypoints saved: {npy_path}")
        print("[line] Dry run complete - the robot was not contacted.")
        return 0

    print("=" * 72)
    print("MECA500 R3 LINE SCAN - the arm will home and then move.")
    print(f"  Line : {args.n_points} point(s), {args.step} mm apart along {args.axis}")
    print(f"  Caps : joint {args.joint_vel} %, linear {args.cart_vel} mm/s")
    print("=" * 72)
    if not args.yes:
        try:
            input("  Press ENTER to continue, or Ctrl+C to abort ... ")
        except KeyboardInterrupt:
            print("\n[line] Aborted before connecting.")
            return 1

    session = None
    exit_code = 0
    scan_completed = False
    try:
        session = connect_robot(args.ip)
        session.activate_and_home()
        session.apply_velocity_limits(
            VelocityLimits(joint_pct=args.joint_vel, cart_lin_mm_s=args.cart_vel)
        )

        def handle(conn, _addr) -> None:
            nonlocal scan_completed
            link = InstrumentLink(
                conn, axis_pair,
                response_timeout_s=args.response_timeout,
            )
            banner = link.initialize()
            if banner:
                print(f"[instrument] Init data: "
                      f"{banner.decode('utf-8', errors='ignore').strip()}")

            home_pose = session.get_pose()
            origin = args.origin or home_pose[:3]
            orientation = home_pose[3:]
            path = line_scan(origin, args.axis, args.n_points, args.step)
            describe(path, origin)

            with PointLog(args.output) as log:
                result = run_scan(
                    session, link, path, orientation,
                    settings=settings, log=log,
                )

            print(f"[line] {result.n_points} point(s) in {result.duration_s:.1f} s")
            print(f"[line] Log: {args.output}")

            if args.point_files:
                for record in result.records:
                    write_point_file(
                        args.point_files, record.index, result.n_points,
                        record.label,
                    )
                print(f"[line] Per-point files: {args.point_files}")

            run_verification_moves(session, home_pose, args.verify_up, args.verify_down)
            print("[line] Returning to the start pose ...")
            session.move_lin(home_pose)
            scan_completed = True

        serve(
            handle, host=args.host, port=args.port,
            max_connections=1, accept_timeout_s=args.accept_timeout,
        )

    except KeyboardInterrupt:
        print("\n[line] Interrupted - pausing motion.")
        if session is not None:
            for step in ("PauseMotion", "ClearMotion"):
                try:
                    getattr(session.robot, step)()
                except Exception:  # noqa: BLE001 - best effort during abort
                    pass
        exit_code = 130
    except Exception as exc:  # noqa: BLE001 - top-level CLI reporting
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        if session is not None:
            if scan_completed and not args.disconnect_when_done:
                # Measurement done: the arm is back at its start pose and stays
                # activated and homed.  Nothing is deactivated or disconnected,
                # so the next run can skip homing.  Pass --disconnect-when-done
                # to power the drives down instead.
                print("[line] Done - arm parked at the start pose, drives still "
                      "active (not disconnected).")
            else:
                session.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
