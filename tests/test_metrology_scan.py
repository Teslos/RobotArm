"""
Integration tests for the scan executor, using a stub robot and a fake
instrument link.  No sockets, no hardware, no sleeping.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from exts.robot_arm.metrology.instrument import AxisPair, InstrumentTimeout
from exts.robot_arm.metrology.limits import LimitViolation, WorkspaceBox
from exts.robot_arm.metrology.patterns import line_scan, raster_grid
from exts.robot_arm.metrology.recording import PointLog
from exts.robot_arm.metrology.robot_session import RobotSession
from exts.robot_arm.metrology.scan import ScanSettings, run_scan

ORIGIN = [190.0, 0.0, 188.0]
ORIENTATION = [-180.0, 0.0, 90.0]
BOX = WorkspaceBox(
    x_min=100.0, x_max=300.0,
    y_min=-150.0, y_max=150.0,
    z_min=100.0, z_max=300.0,
)


class TrackingRobot:
    """Stub robot that actually moves: GetRtCartPos reflects the last target."""

    def __init__(self, start=None):
        self.pose = list(start or (ORIGIN + ORIENTATION))
        self.moves = []

    def MoveLin(self, *pose):
        self.moves.append(("MoveLin", tuple(pose)))
        self.pose = list(pose)

    def MovePose(self, *pose):
        self.moves.append(("MovePose", tuple(pose)))
        self.pose = list(pose)

    def WaitIdle(self, timeout=None):
        pass

    def GetRtCartPos(self):
        return list(self.pose)

    @property
    def move_targets(self):
        return [move[1] for move in self.moves]


class FakeLink:
    """Instrument stub: records labels and hands back canned acquire times."""

    def __init__(self, axis_pair=AxisPair.XY, acquire_times=None, fail_at=None):
        self.axis_pair = axis_pair
        self.labels = []
        self.timeouts = []
        self._times = itertools.cycle(acquire_times or [0.5])
        self._fail_at = fail_at

    def measure_point(self, label, timeout_s=None):
        if self._fail_at is not None and len(self.labels) == self._fail_at:
            raise InstrumentTimeout("instrument went quiet")
        self.labels.append(label)
        self.timeouts.append(timeout_s)
        return next(self._times)


def make_session(robot=None):
    return RobotSession(robot or TrackingRobot(), workspace=BOX, verbose=False)


def noop_sleep(_seconds):
    pass


def run(session, link, path, **kwargs):
    kwargs.setdefault("settings", ScanSettings(settle_s=0.0, progress_every=0))
    kwargs.setdefault("sleep", noop_sleep)
    return run_scan(session, link, path, ORIENTATION, **kwargs)


# ── Happy path ───────────────────────────────────────────────────────────────

def test_scan_visits_every_waypoint_in_order():
    robot = TrackingRobot()
    link = FakeLink()
    path = raster_grid(ORIGIN, n_fast=3, n_slow=2, step_fast_mm=1.0, step_slow_mm=2.0)

    result = run(make_session(robot), link, path)

    assert result.n_points == 6
    np.testing.assert_allclose(result.poses[:, :3], path.points)


def test_scan_sends_six_argument_moves_with_the_locked_orientation():
    robot = TrackingRobot()
    path = line_scan(ORIGIN, "x", 3, 1.0)

    run(make_session(robot), FakeLink(), path)

    assert all(len(target) == 6 for target in robot.move_targets)
    assert all(list(target[3:]) == ORIENTATION for target in robot.move_targets)


def test_scan_labels_use_the_measured_pose_not_the_commanded_one():
    """RECTANGULAR _Pattern_Meca2 Wsafty.py logged `point_idx % 10 * d_x`
    instead of the real position, so its files hid the 2 mm/line drift."""
    class DriftingRobot(TrackingRobot):
        def MoveLin(self, *pose):
            drifted = list(pose)
            drifted[0] += 0.4          # the arm lands slightly off
            super().MoveLin(*drifted)

    link = FakeLink()
    path = line_scan(ORIGIN, "x", 2, 1.0)
    run(make_session(DriftingRobot()), link, path)

    assert link.labels == ["x190.4 y0.0", "x191.4 y0.0"]


def test_scan_records_line_numbers():
    link = FakeLink()
    path = raster_grid(ORIGIN, n_fast=2, n_slow=3, step_fast_mm=1.0, step_slow_mm=1.0)
    result = run(make_session(), link, path)
    assert [record.line for record in result.records] == [0, 0, 1, 1, 2, 2]


def test_scan_records_acquire_times_and_monotonic_timestamps():
    link = FakeLink(acquire_times=[0.25, 0.75])
    result = run(make_session(), link, line_scan(ORIGIN, "x", 4, 1.0))
    assert [r.acquire_s for r in result.records] == [0.25, 0.75, 0.25, 0.75]
    stamps = [r.timestamp_s for r in result.records]
    assert stamps == sorted(stamps)


def test_scan_forwards_the_response_timeout():
    link = FakeLink()
    settings = ScanSettings(settle_s=0.0, response_timeout_s=7.5, progress_every=0)
    run(make_session(), link, line_scan(ORIGIN, "x", 2, 1.0), settings=settings)
    assert link.timeouts == [7.5, 7.5]


def test_scan_uses_the_configured_axis_pair_for_labels():
    link = FakeLink(axis_pair=AxisPair.YZ)
    run(make_session(), link, line_scan(ORIGIN, "y", 2, 5.0))
    assert link.labels == ["y0.0 z188.0", "y5.0 z188.0"]


def test_scan_can_use_move_pose_instead_of_move_lin():
    robot = TrackingRobot()
    settings = ScanSettings(settle_s=0.0, use_move_lin=False, progress_every=0)
    run(make_session(robot), FakeLink(), line_scan(ORIGIN, "x", 2, 1.0),
        settings=settings)
    assert {move[0] for move in robot.moves} == {"MovePose"}


# ── Safety hop ───────────────────────────────────────────────────────────────

def test_hop_brackets_the_travel_between_lines():
    """The originals moved up and straight back down without travelling."""
    robot = TrackingRobot()
    settings = ScanSettings(settle_s=0.0, hop_height_mm=5.0, progress_every=0)
    path = raster_grid(ORIGIN, n_fast=2, n_slow=2, step_fast_mm=1.0, step_slow_mm=3.0)

    run(make_session(robot), FakeLink(), path, settings=settings)

    z_values = [target[2] for target in robot.move_targets]
    # 2 measurement points, lift + traverse at height, then 2 more points.
    assert z_values == [188.0, 188.0, 193.0, 193.0, 188.0, 188.0]
    lift, traverse = robot.move_targets[2], robot.move_targets[3]
    assert lift[0] == pytest.approx(191.0)     # above the end of line 0
    assert traverse[1] == pytest.approx(3.0)   # above the start of line 1


def test_no_hop_moves_when_hop_height_is_zero():
    robot = TrackingRobot()
    path = raster_grid(ORIGIN, n_fast=2, n_slow=2, step_fast_mm=1.0, step_slow_mm=3.0)
    run(make_session(robot), FakeLink(), path)
    assert len(robot.moves) == 4


def test_hop_is_validated_against_the_workspace_before_any_move():
    robot = TrackingRobot()
    settings = ScanSettings(settle_s=0.0, hop_height_mm=200.0, progress_every=0)
    path = raster_grid(ORIGIN, n_fast=2, n_slow=2, step_fast_mm=1.0, step_slow_mm=1.0)

    with pytest.raises(LimitViolation, match="Z ="):
        run(make_session(robot), FakeLink(), path, settings=settings)
    assert robot.moves == []


# ── Pre-flight validation ────────────────────────────────────────────────────

def test_a_waypoint_outside_the_box_aborts_before_the_first_move():
    """A mistyped step size must fail on the PC, not inside the fixture."""
    robot = TrackingRobot()
    path = line_scan(ORIGIN, "y", 200, 1.0)   # runs well past y_max = 150

    with pytest.raises(LimitViolation, match="Waypoint"):
        run(make_session(robot), FakeLink(), path)
    assert robot.moves == []


# ── Failure propagation ──────────────────────────────────────────────────────

def test_an_instrument_failure_aborts_the_scan():
    """Finall Rectangular Mesurment.py ignored the failure return value and
    printed 'Complete!' anyway."""
    robot = TrackingRobot()
    link = FakeLink(fail_at=2)

    with pytest.raises(InstrumentTimeout):
        run(make_session(robot), link, line_scan(ORIGIN, "x", 5, 1.0))
    assert len(link.labels) == 2


def test_points_measured_before_a_failure_are_already_on_disk(tmp_path):
    path_csv = tmp_path / "scan.csv"
    link = FakeLink(fail_at=2)
    with PointLog(str(path_csv)) as log:
        with pytest.raises(InstrumentTimeout):
            run(make_session(), link, line_scan(ORIGIN, "x", 5, 1.0), log=log)

    lines = path_csv.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3          # header + 2 measured points


# ── Callbacks and results ────────────────────────────────────────────────────

def test_on_point_callback_receives_every_record():
    seen = []
    run(make_session(), FakeLink(), line_scan(ORIGIN, "x", 3, 1.0),
        on_point=seen.append)
    assert [record.index for record in seen] == [0, 1, 2]


def test_result_poses_shape_and_empty_case():
    result = run(make_session(), FakeLink(), line_scan(ORIGIN, "x", 3, 1.0))
    assert result.poses.shape == (3, 6)
    assert result.duration_s >= 0.0


def test_settle_and_dwell_are_applied_per_point():
    slept = []
    settings = ScanSettings(settle_s=0.2, dwell_s=0.3, progress_every=0)
    run_scan(
        make_session(), FakeLink(), line_scan(ORIGIN, "x", 2, 1.0), ORIENTATION,
        settings=settings, sleep=slept.append,
    )
    assert slept == [0.2, 0.3, 0.2, 0.3]


# -- End-to-end over a real socket -------------------------------------------

def test_scan_over_a_real_socket_with_the_real_protocol(tmp_path):
    """Full pipeline: robot stub + genuine InstrumentLink on a socketpair.

    Exercises the actual wire format, the drain/trigger/wait handshake and the
    CSV log together, which is what the original scripts open-coded per file.
    """
    import socket
    import threading

    from exts.robot_arm.metrology.instrument import END_TOKEN, InstrumentLink

    pc, instrument = socket.socketpair()
    robot = TrackingRobot()
    path = raster_grid(ORIGIN, n_fast=3, n_slow=2, step_fast_mm=1.0, step_slow_mm=2.0)
    n_points = len(path)
    received = []
    stop = threading.Event()

    def instrument_side():
        """Answer every trigger with 'done', the way INT_Monitor does."""
        buffer = b""
        instrument.settimeout(5.0)
        while len(received) < n_points and not stop.is_set():
            try:
                chunk = instrument.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buffer += chunk
            while b"\r\n" in buffer:
                raw, _, buffer = buffer.partition(b"\r\n")
                line = raw.decode()
                if line == "1":                 # trigger
                    instrument.sendall(END_TOKEN)
                elif line:
                    received.append(line)

    thread = threading.Thread(target=instrument_side)
    thread.start()
    try:
        link = InstrumentLink(
            pc, AxisPair.XY,
            command_gap_s=0.0, response_timeout_s=5.0,
            sleep=lambda _s: None,
        )
        csv_path = tmp_path / "scan.csv"
        with PointLog(str(csv_path)) as log:
            result = run_scan(
                make_session(robot), link, path, ORIENTATION,
                settings=ScanSettings(settle_s=0.0, progress_every=0),
                log=log, sleep=noop_sleep,
            )
    finally:
        stop.set()
        thread.join(timeout=5.0)
        pc.close()
        instrument.close()

    assert result.n_points == n_points
    assert received == [
        f"x{x:.1f} y{y:.1f}" for x, y, _z in path.points
    ]
    assert len(csv_path.read_text(encoding="utf-8").strip().splitlines()) == n_points + 1


def test_a_single_line_scan_is_not_rejected_for_unused_hop_clearance():
    """No line change means no hop, so hop clearance must not be demanded."""
    robot = TrackingRobot()
    settings = ScanSettings(settle_s=0.0, hop_height_mm=500.0, progress_every=0)
    run(make_session(robot), FakeLink(), line_scan(ORIGIN, "x", 3, 1.0),
        settings=settings)
    assert len(robot.moves) == 3


def test_hop_validation_only_covers_the_poses_it_visits():
    """A hop that clears the box only at the line boundary must still abort."""
    robot = TrackingRobot()
    settings = ScanSettings(settle_s=0.0, hop_height_mm=120.0, progress_every=0)
    path = raster_grid(ORIGIN, n_fast=2, n_slow=2, step_fast_mm=1.0, step_slow_mm=1.0)
    with pytest.raises(LimitViolation, match="Z ="):
        run(make_session(robot), FakeLink(), path, settings=settings)
    assert robot.moves == []
