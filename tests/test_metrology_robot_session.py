"""
Unit tests for RobotSession — driven by a stub that records every SDK call,
so no mecademicpy install and no hardware is needed.
"""
from __future__ import annotations

import pytest

from exts.robot_arm.metrology.limits import LimitViolation, WorkspaceBox
from exts.robot_arm.metrology.robot_session import (
    RobotSession,
    VelocityLimits,
    normalize_pose,
)

HOME = [190.0, 0.0, 188.0, -180.0, 0.0, 90.0]
SMALL_BOX = WorkspaceBox(
    x_min=100.0, x_max=300.0,
    y_min=-100.0, y_max=100.0,
    z_min=100.0, z_max=300.0,
)


class FakeRobot:
    """Records calls; mimics the handful of mecademicpy methods used."""

    def __init__(self, pose=None, fail_on=()):
        self.calls = []
        self.pose = list(pose if pose is not None else HOME)
        self.fail_on = set(fail_on)

    def _record(self, name, *args, **kwargs):
        if name in self.fail_on:
            raise RuntimeError(f"{name} boom")
        self.calls.append((name, args, kwargs))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args, **kwargs):
            return self._record(name, *args, **kwargs)

        return method

    def GetRtCartPos(self):
        self.calls.append(("GetRtCartPos", (), {}))
        return self.pose

    def names(self):
        return [call[0] for call in self.calls]

    def args_of(self, name):
        return [call[1] for call in self.calls if call[0] == name]


def make_session(robot=None, **kwargs):
    kwargs.setdefault("workspace", SMALL_BOX)
    kwargs.setdefault("verbose", False)
    return RobotSession(robot or FakeRobot(), **kwargs)


# ── normalize_pose ───────────────────────────────────────────────────────────

def test_normalize_pose_returns_a_fresh_list():
    source = list(HOME)
    result = normalize_pose(source)
    result[0] = -1.0
    assert source[0] == HOME[0]


def test_normalize_pose_accepts_a_tuple():
    assert normalize_pose(tuple(HOME)) == HOME


def test_normalize_pose_rejects_none():
    with pytest.raises(LimitViolation, match="no Cartesian pose"):
        normalize_pose(None)


def test_normalize_pose_rejects_short_pose():
    """'Unexpected robot pose' was checked in only one of the six originals."""
    with pytest.raises(LimitViolation, match="6-value pose"):
        normalize_pose([1.0, 2.0, 3.0])


def test_normalize_pose_rejects_non_numeric():
    with pytest.raises(LimitViolation, match="Unreadable robot pose"):
        normalize_pose(["a"] * 6)


# ── Lifecycle ────────────────────────────────────────────────────────────────

def test_activate_and_home_sequence():
    robot = FakeRobot()
    make_session(robot).activate_and_home()
    names = robot.names()
    assert names[-3:] == ["ActivateRobot", "Home", "WaitHomed"]


def test_activate_and_home_tolerates_a_failing_reset():
    robot = FakeRobot(fail_on={"ResetError"})
    make_session(robot).activate_and_home()
    assert "ActivateRobot" in robot.names()


def test_apply_velocity_limits_forwards_all_three_caps():
    robot = FakeRobot()
    make_session(robot).apply_velocity_limits(VelocityLimits(10.0, 5.0, 30.0))
    assert robot.args_of("SetJointVel") == [(10.0,)]
    assert robot.args_of("SetCartLinVel") == [(5.0,)]
    assert robot.args_of("SetCartAngVel") == [(30.0,)]


@pytest.mark.parametrize("kwargs", [
    {"joint_pct": 0.0}, {"joint_pct": 101.0},
    {"cart_lin_mm_s": 0.0}, {"cart_ang_deg_s": -1.0},
])
def test_velocity_limits_validate(kwargs):
    with pytest.raises(ValueError):
        VelocityLimits(**kwargs)


def test_shutdown_never_raises_and_cannot_mask_the_real_error():
    """Two originals had a bare `finally: robot.DeactivateRobot()`."""
    robot = FakeRobot(fail_on={"DeactivateRobot", "Disconnect"})
    make_session(robot).shutdown()  # must not raise


def test_context_manager_shuts_down_even_on_error():
    robot = FakeRobot()
    with pytest.raises(ValueError):
        with make_session(robot):
            raise ValueError("scan failed")
    assert "DeactivateRobot" in robot.names()
    assert "Disconnect" in robot.names()


# ── Motion ───────────────────────────────────────────────────────────────────

def test_move_lin_splats_six_arguments():
    """The originals passed the pose as a single list -> TypeError in the SDK."""
    robot = FakeRobot()
    make_session(robot).move_lin(HOME)
    assert robot.args_of("MoveLin") == [tuple(HOME)]


def test_move_pose_splats_six_arguments():
    robot = FakeRobot()
    make_session(robot).move_pose(HOME)
    assert robot.args_of("MovePose") == [tuple(HOME)]


def test_moves_wait_for_idle():
    robot = FakeRobot()
    make_session(robot).move_lin(HOME)
    assert robot.names()[-1] == "WaitIdle"


def test_wait_idle_passes_the_timeout():
    robot = FakeRobot()
    make_session(robot, wait_timeout_s=12.0).move_lin(HOME)
    assert robot.args_of("WaitIdle") == [()]
    assert {"timeout": 12.0} in [c[2] for c in robot.calls if c[0] == "WaitIdle"]


def test_wait_idle_falls_back_when_timeout_is_unsupported():
    class OldSdkRobot(FakeRobot):
        def WaitIdle(self, *args, **kwargs):
            if kwargs:
                raise TypeError("WaitIdle() takes no keyword arguments")
            self.calls.append(("WaitIdle", (), {}))

    robot = OldSdkRobot()
    make_session(robot).move_lin(HOME)
    assert "WaitIdle" in robot.names()


def test_move_lin_rejects_a_target_outside_the_box():
    robot = FakeRobot()
    with pytest.raises(LimitViolation, match="MoveLin target"):
        make_session(robot).move_lin([1000.0, 0.0, 200.0, 0.0, 0.0, 0.0])
    assert "MoveLin" not in robot.names()


def test_move_joints_rejects_a_target_outside_the_soft_limits():
    robot = FakeRobot()
    joints = [0.0, 0.0, 69.0, 0.0, 0.0, 0.0]
    with pytest.raises(LimitViolation, match="J3"):
        make_session(robot).move_joints(joints)
    assert "MoveJoints" not in robot.names()


def test_move_joints_splats():
    robot = FakeRobot()
    joints = [-2.4, -2.0, 52.3, -2.2, -32.1, -37.6]
    make_session(robot).move_joints(joints)
    assert robot.args_of("MoveJoints") == [tuple(joints)]


def test_move_lin_offset_is_relative_to_the_given_pose_not_the_live_one():
    """Repeated offsets must not accumulate, unlike MoveLinRelWrf chains."""
    robot = FakeRobot()
    session = make_session(robot)
    for _ in range(3):
        session.move_lin_offset(HOME, dz=10.0)
    expected = tuple([HOME[0], HOME[1], HOME[2] + 10.0, *HOME[3:]])
    assert robot.args_of("MoveLin") == [expected] * 3


def test_move_lin_offset_returns_the_commanded_pose():
    session = make_session()
    assert session.move_lin_offset(HOME, dx=1.0, dy=-2.0, dz=3.0) == [
        191.0, -2.0, 191.0, -180.0, 0.0, 90.0
    ]


def test_move_lin_offset_validates_the_offset_target():
    robot = FakeRobot()
    with pytest.raises(LimitViolation, match="Z ="):
        make_session(robot).move_lin_offset(HOME, dz=500.0)
    assert "MoveLin" not in robot.names()


# ── State ────────────────────────────────────────────────────────────────────

def test_get_pose_does_not_hand_back_sdk_owned_state():
    robot = FakeRobot()
    session = make_session(robot)
    pose = session.get_pose()
    pose[0] = -1.0
    assert session.get_pose()[0] == HOME[0]


def test_get_orientation():
    assert make_session().get_orientation() == HOME[3:]
