"""
Scan executor: drive the arm along a :class:`~.patterns.ScanPath` and take one
INT_Monitor acquisition at every waypoint.

This is the piece all four original scanning scripts open-coded, each with its
own copy of the handshake, its own bug set and its own idea of what a "line"
is.  Keeping it in one place means a fix lands everywhere.

Behaviour worth knowing
-----------------------
* Every waypoint is validated against the workspace box *before the first move*,
  so a bad step size aborts on the PC instead of halfway into the fixture.
* Positions written to the log are read back from the robot, not assumed from
  the commanded target.  ``RECTANGULAR _Pattern_Meca2 Wsafty.py`` derived its
  recorded coordinates from a counter (``point_idx % 10 * d_x``), so its data
  file claimed a perfect grid while the arm was drifting 2 mm per line.
* The safety hop, when enabled, brackets the *travel between lines*: up at the
  end of a line, across at height, down at the next line start.  The originals
  moved up and straight back down without travelling in between, which
  protected nothing.
* Any instrument failure aborts the scan and propagates, instead of being
  printed and then reported as a success.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from .instrument import InstrumentLink
from .limits import check_waypoints
from .patterns import ScanPath, attach_orientation
from .recording import PointLog, PointRecord
from .robot_session import RobotSession

__all__ = ["ScanSettings", "ScanResult", "run_scan"]


@dataclass
class ScanSettings:
    """Knobs for :func:`run_scan`."""

    settle_s: float = 0.1
    """Dwell after the move completes, before the instrument is triggered."""

    dwell_s: float = 0.0
    """Extra hold after the acquisition, for slow sensors."""

    hop_height_mm: float = 0.0
    """Lift applied while travelling between scan lines.  0 disables hopping."""

    response_timeout_s: Optional[float] = None
    """Per-point ``done`` timeout; ``None`` uses the link's own default."""

    use_move_lin: bool = True
    """Straight-line Cartesian moves; ``False`` uses joint-interpolated moves."""

    progress_every: int = 1
    """Print a progress line every N waypoints; 0 silences progress output."""


@dataclass
class ScanResult:
    """What a completed scan produced."""

    records: List[PointRecord]
    duration_s: float

    @property
    def n_points(self) -> int:
        return len(self.records)

    @property
    def poses(self) -> np.ndarray:
        """``(N, 6)`` array of the poses actually reached."""
        if not self.records:
            return np.empty((0, 6))
        return np.array([record.pose for record in self.records], dtype=float)


def _hop_poses(
    poses: np.ndarray,
    path: ScanPath,
    hop_height_mm: float,
) -> np.ndarray:
    """The lift/traverse poses :func:`run_scan` will visit between lines."""
    if not hop_height_mm:
        return np.empty((0, 6))
    rows = []
    for start in path.line_starts:
        if start == 0:
            continue
        for index in (start - 1, start):
            pose = poses[index].copy()
            pose[2] += hop_height_mm
            rows.append(pose)
    return np.array(rows) if rows else np.empty((0, 6))


def run_scan(
    session: RobotSession,
    link: InstrumentLink,
    path: ScanPath,
    orientation_deg: Sequence[float],
    *,
    settings: Optional[ScanSettings] = None,
    log: Optional[PointLog] = None,
    on_point: Optional[Callable[[PointRecord], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ScanResult:
    """Execute ``path`` and measure at every waypoint.

    Parameters
    ----------
    session:
        Connected, homed :class:`~.robot_session.RobotSession`.
    link:
        Initialised :class:`~.instrument.InstrumentLink`.
    path:
        Absolute waypoints, in millimetres, base frame.
    orientation_deg:
        ``[alpha, beta, gamma]`` held constant for the whole scan — normally
        read back from the robot after the approach move so it matches the real
        arm configuration.
    log:
        Optional already-opened :class:`~.recording.PointLog`.

    Raises
    ------
    LimitViolation
        If any waypoint (including hop poses) leaves the workspace box.
    InstrumentError
        If the instrument disconnects or misses its response deadline.
    """
    settings = settings or ScanSettings()
    poses = attach_orientation(path, orientation_deg)

    # Validate the travel poses too, not just the measurement poses.  Only the
    # poses a hop actually visits are checked, so a scan that never hops is not
    # rejected for a clearance it will never use.
    check_waypoints(poses, session.workspace)
    hop_poses = _hop_poses(poses, path, settings.hop_height_mm)
    if len(hop_poses):
        check_waypoints(hop_poses, session.workspace)

    print(
        f"[scan] {len(poses)} waypoints, "
        f"{len(path.line_starts)} line(s); all inside the workspace box."
    )

    move = session.move_lin if settings.use_move_lin else session.move_pose
    records: List[PointRecord] = []
    started = clock()

    for index, target in enumerate(poses):
        line = path.line_of(index)

        if settings.hop_height_mm and path.is_line_start(index) and index > 0:
            # Lift at the end of the previous line, travel across, come down.
            previous = list(poses[index - 1])
            session.move_lin_offset(previous, dz=settings.hop_height_mm)
            session.move_lin_offset(target, dz=settings.hop_height_mm)

        move(list(target))
        if settings.settle_s > 0:
            sleep(settings.settle_s)

        reached = session.get_pose()
        label = link.axis_pair.format_point(reached)
        acquire_s = link.measure_point(label, settings.response_timeout_s)

        record = PointRecord(
            index=index,
            line=line,
            label=label,
            pose=reached,
            acquire_s=acquire_s,
            timestamp_s=clock() - started,
        )
        records.append(record)
        if log is not None:
            log.append(record)
        if on_point is not None:
            on_point(record)

        if settings.progress_every and index % settings.progress_every == 0:
            print(
                f"[scan] {index + 1:>5}/{len(poses)}  line {line + 1}  "
                f"{label}  acquired in {acquire_s:.2f} s"
            )

        if settings.dwell_s > 0:
            sleep(settings.dwell_s)

    return ScanResult(records=records, duration_s=clock() - started)
