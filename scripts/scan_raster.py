#!/usr/bin/env python3
"""
2-D raster scan on the physical Meca500 R3, synchronised with INT_Monitor.

Replaces three near-duplicate scripts from ``C:\\EmpaDaten\\INT_Monitor\\py``:

======================================  =====================================
Original                                Equivalent invocation
======================================  =====================================
``Any Lines in Y Direaction             ``--fast-axis y --slow-axis x
Scanning.py``                             --n-fast 55 --n-slow 45
                                          --step-fast 2 --step-slow 2``
``Finall  Rectangular Mesurment.py``    ``--fast-axis x --slow-axis y
                                          --n-fast 100 --n-slow 100
                                          --step-fast 1 --step-slow 1``
``RECTANGULAR _Pattern_Meca2            ``--fast-axis x --slow-axis y
Wsafty.py``                               --n-fast 10 --n-slow 10
                                          --step-fast 2 --step-slow 2
                                          --hop-height 2 --settle 1.0``
======================================  =====================================

Usage::

    micromamba run -n RobotArm python scripts/scan_raster.py --dry-run
    micromamba run -n RobotArm python scripts/scan_raster.py \
        --fast-axis y --slow-axis x --n-fast 55 --n-slow 45 \
        --step-fast 2 --step-slow 2 --axis-pair xy

How it runs
-----------
1. Connects to the robot, activates, homes, applies conservative velocity caps.
2. Opens a TCP server and waits for the INT_Monitor client (one connection).
3. Records the arm's current pose as the raster origin, unless --home or
   --origin override it.
4. Builds the full absolute waypoint list and validates it against the
   workspace box *before moving*.
5. Measures every waypoint, appending to a CSV as it goes.
6. Returns to the home pose (--home, else the pose it started from) and stops
   there: the drives stay activated and
   the connection is left open, so a follow-up scan needs no re-homing.  Pass
   ``--disconnect-when-done`` to deactivate and disconnect instead.  An error
   or a Ctrl+C always ends in a full shutdown.

==============================================================================
SAFETY - READ scripts/meca500_real.py FOR THE FULL CHECKLIST
==============================================================================
  * Homing moves the arm. Clear the work envelope first.
  * Keep a hand near the e-stop for the first run of any new raster.
  * --dry-run prints and saves the waypoints without touching the robot; use it
    to sanity-check a new grid before powering the arm.
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
    ScanSettings,
    VelocityLimits,
    check_waypoints,
    connect_robot,
    raster_grid,
    resolve_home_pose,
    run_scan,
    serve,
    write_point_file,
)
from metrology.limits import (  # noqa: E402
    DEFAULT_WORKSPACE,
    LimitViolation,
    check_pose,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="INT_Monitor-synchronised 2-D raster scan (mm, degrees)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    grid = parser.add_argument_group("raster geometry")
    grid.add_argument("--fast-axis", choices=("x", "y", "z"), default="x",
                      help="Axis stepped within a scan line")
    grid.add_argument("--slow-axis", choices=("x", "y", "z"), default="y",
                      help="Axis stepped between scan lines")
    grid.add_argument("--n-fast", type=int, default=10,
                      help="Waypoints per line")
    grid.add_argument("--n-slow", type=int, default=10,
                      help="Number of lines")
    grid.add_argument("--step-fast", type=float, default=2.0, metavar="MM",
                      help="Spacing within a line; negative reverses direction")
    grid.add_argument("--step-slow", type=float, default=2.0, metavar="MM",
                      help="Spacing between lines; negative reverses direction")
    grid.add_argument("--serpentine", action="store_true",
                      help="Reverse alternate lines instead of rewinding to the "
                           "line start (halves travel, no long return move)")
    grid.add_argument("--origin", type=float, nargs=3, metavar=("X", "Y", "Z"),
                      help="Raster origin in mm; default is the arm's current "
                           "pose at scan start")

    motion = parser.add_argument_group("motion")
    motion.add_argument("--hop-height", type=float, default=0.0, metavar="MM",
                        help="Lift while travelling between lines (0 = none)")
    motion.add_argument("--settle", type=float, default=0.1, metavar="SECONDS",
                        help="Dwell after each move, before triggering")
    motion.add_argument("--dwell", type=float, default=0.0, metavar="SECONDS",
                        help="Extra hold after each acquisition")
    motion.add_argument("--move-pose", action="store_true",
                        help="Use joint-interpolated MovePose instead of MoveLin")
    motion.add_argument("--joint-vel", type=float, default=20.0, metavar="PCT",
                        help="Joint velocity cap, %% of maximum")
    motion.add_argument("--cart-vel", type=float, default=20.0, metavar="MM_S",
                        help="Cartesian linear velocity cap")

    net = parser.add_argument_group("instrument link")
    net.add_argument("--host", default=DEFAULT_HOST, help="TCP bind address")
    net.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP port")
    net.add_argument("--axis-pair", default="xy",
                     choices=[pair.name.lower() for pair in AxisPair],
                     help="Coordinate pair streamed in each point label")
    net.add_argument("--response-timeout", type=float, default=30.0,
                     metavar="SECONDS",
                     help="Per-point wait for the instrument's 'done'")
    net.add_argument("--accept-timeout", type=float, default=None,
                     metavar="SECONDS",
                     help="Give up if no instrument connects in time")

    io_group = parser.add_argument_group("output")
    io_group.add_argument("--output", default=os.path.join(RESULTS_DIR, "raster_scan.csv"),
                          help="CSV log of measured points")
    io_group.add_argument("--point-files", metavar="DIR",
                          help="Also write the legacy one-file-per-point text "
                               "dumps into DIR")

    parser.add_argument("--home", type=float, nargs="+", metavar="V",
                        help="Pose the arm returns to when the scan ends: "
                             "X Y Z, or X Y Z ALPHA BETA GAMMA to pin the "
                             "orientation too. Default: wherever the arm "
                             "stands when the scan starts. Also the default "
                             "--origin")
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
    extent = path.extent_mm
    print(f"[raster] Origin      : x={origin[0]:.2f} y={origin[1]:.2f} z={origin[2]:.2f} mm")
    print(f"[raster] Waypoints   : {len(path)} in {len(path.line_starts)} line(s)")
    for axis, (low, high) in zip("XYZ", extent):
        print(f"[raster] {axis} range     : {low:.2f} -> {high:.2f} mm")


def main() -> int:
    args = build_parser().parse_args()

    if args.fast_axis == args.slow_axis:
        print("[ERROR] --fast-axis and --slow-axis must differ.", file=sys.stderr)
        return 2

    if args.home is not None and len(args.home) not in (3, 6):
        print("[ERROR] --home takes 3 values (X Y Z) or 6 "
              "(X Y Z ALPHA BETA GAMMA), got "
              f"{len(args.home)}.", file=sys.stderr)
        return 2

    axis_pair = AxisPair.parse(args.axis_pair)
    settings = ScanSettings(
        settle_s=args.settle,
        dwell_s=args.dwell,
        hop_height_mm=args.hop_height,
        response_timeout_s=args.response_timeout,
        use_move_lin=not args.move_pose,
    )

    # -- dry run: geometry only, no hardware ---------------------------------
    if args.dry_run:
        home = args.home or [190.0, 0.0, 188.0]
        origin = args.origin or home[:3]
        path = raster_grid(
            origin, args.n_fast, args.n_slow, args.step_fast, args.step_slow,
            fast_axis=args.fast_axis, slow_axis=args.slow_axis,
            serpentine=args.serpentine,
        )
        describe(path, origin)
        try:
            DEFAULT_WORKSPACE.check(*home[:3], context="--home")
            check_waypoints(path.points, DEFAULT_WORKSPACE)
        except LimitViolation as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            print("[ERROR] Adjust the grid, or widen DEFAULT_WORKSPACE in "
                  "exts/robot_arm/metrology/limits.py if the box is wrong.",
                  file=sys.stderr)
            return 1
        print("[raster] All waypoints are inside the workspace box.")
        npy_path = os.path.splitext(args.output)[0] + "_waypoints.npy"
        os.makedirs(os.path.dirname(os.path.abspath(npy_path)), exist_ok=True)
        np.save(npy_path, path.points)
        print(f"[raster] Waypoints saved: {npy_path}")
        print(f"[raster] Home pose    : "
              f"x={home[0]:.2f} y={home[1]:.2f} z={home[2]:.2f} mm")
        print("[raster] Dry run complete - the robot was not contacted.")
        return 0

    print("=" * 72)
    print("MECA500 R3 RASTER SCAN - the arm will home and then move.")
    print(f"  Grid : {args.n_slow} line(s) x {args.n_fast} point(s)")
    print(f"  Step : {args.step_fast} mm along {args.fast_axis}, "
          f"{args.step_slow} mm along {args.slow_axis}")
    print(f"  Caps : joint {args.joint_vel} %, linear {args.cart_vel} mm/s")
    print("=" * 72)
    if not args.yes:
        try:
            input("  Press ENTER to continue, or Ctrl+C to abort ... ")
        except KeyboardInterrupt:
            print("\n[raster] Aborted before connecting.")
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

            home_pose = resolve_home_pose(args.home, session.get_pose())
            check_pose(home_pose, DEFAULT_WORKSPACE, context="--home")
            origin = args.origin or home_pose[:3]
            orientation = home_pose[3:]
            path = raster_grid(
                origin, args.n_fast, args.n_slow, args.step_fast, args.step_slow,
                fast_axis=args.fast_axis, slow_axis=args.slow_axis,
                serpentine=args.serpentine,
            )
            describe(path, origin)

            with PointLog(args.output) as log:
                result = run_scan(
                    session, link, path, orientation,
                    settings=settings, log=log,
                )

            print(f"[raster] {result.n_points} point(s) in {result.duration_s:.1f} s")
            print(f"[raster] Log: {args.output}")

            if args.point_files:
                for record in result.records:
                    write_point_file(
                        args.point_files, record.index, result.n_points,
                        record.label,
                    )
                print(f"[raster] Per-point files: {args.point_files}")

            print("[raster] Returning to the start pose ...")
            session.move_lin(home_pose)
            scan_completed = True

        serve(
            handle, host=args.host, port=args.port,
            max_connections=1, accept_timeout_s=args.accept_timeout,
        )

    except KeyboardInterrupt:
        print("\n[raster] Interrupted - pausing motion.")
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
                print("[raster] Done - arm parked at the start pose, drives still "
                      "active (not disconnected).")
            else:
                session.shutdown()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
