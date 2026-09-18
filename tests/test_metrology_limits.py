"""Unit tests for the motion-safety limits — no robot required."""
from __future__ import annotations

import pytest

from exts.robot_arm.metrology.limits import (
    DEFAULT_WORKSPACE,
    HARD_JOINT_LIMITS_DEG,
    POSE_LENGTH,
    SOFT_JOINT_LIMITS_DEG,
    SOFT_LIMIT_MARGIN_DEG,
    LimitViolation,
    WorkspaceBox,
    check_joint_limits,
    check_pose,
    check_waypoints,
)

INSIDE = (190.0, 0.0, 188.0)


# ── WorkspaceBox ─────────────────────────────────────────────────────────────

def test_box_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="min < max"):
        WorkspaceBox(x_min=10, x_max=10, y_min=-1, y_max=1, z_min=0, z_max=1)


def test_contains_is_inclusive_on_both_ends():
    box = DEFAULT_WORKSPACE
    assert box.contains(box.x_min, box.y_min, box.z_min)
    assert box.contains(box.x_max, box.y_max, box.z_max)
    assert not box.contains(box.x_min - 1e-6, 0.0, 100.0)


@pytest.mark.parametrize("point,axis", [
    ((10.0, 0.0, 188.0), "X"),
    ((190.0, 900.0, 188.0), "Y"),
    ((190.0, 0.0, 5.0), "Z"),
])
def test_check_names_the_offending_axis(point, axis):
    with pytest.raises(LimitViolation, match=rf"\b{axis} ="):
        DEFAULT_WORKSPACE.check(*point)


def test_check_includes_context_prefix():
    with pytest.raises(LimitViolation, match="Waypoint 7"):
        DEFAULT_WORKSPACE.check(0.0, 0.0, 0.0, context="Waypoint 7")


def test_bounds_ordering():
    assert DEFAULT_WORKSPACE.bounds == (
        (DEFAULT_WORKSPACE.x_min, DEFAULT_WORKSPACE.x_max),
        (DEFAULT_WORKSPACE.y_min, DEFAULT_WORKSPACE.y_max),
        (DEFAULT_WORKSPACE.z_min, DEFAULT_WORKSPACE.z_max),
    )


# ── Joint limits ─────────────────────────────────────────────────────────────

def test_soft_limits_sit_inside_hard_limits_on_every_axis():
    assert len(SOFT_JOINT_LIMITS_DEG) == len(HARD_JOINT_LIMITS_DEG) == 6
    for (soft_lo, soft_hi), (hard_lo, hard_hi) in zip(
        SOFT_JOINT_LIMITS_DEG, HARD_JOINT_LIMITS_DEG
    ):
        assert soft_lo == pytest.approx(hard_lo + SOFT_LIMIT_MARGIN_DEG)
        assert soft_hi == pytest.approx(hard_hi - SOFT_LIMIT_MARGIN_DEG)
        assert hard_lo < soft_lo < soft_hi < hard_hi


def test_hard_limits_match_the_datasheet():
    # CLAUDE.md: J1 +-175, J2 -70..90, J3 -135..70, J4 +-170, J5 +-115, J6 +-180
    assert HARD_JOINT_LIMITS_DEG == (
        (-175.0, 175.0),
        (-70.0, 90.0),
        (-135.0, 70.0),
        (-170.0, 170.0),
        (-115.0, 115.0),
        (-180.0, 180.0),
    )


def test_check_joint_limits_accepts_the_documented_approach_pose():
    # DEMO_JOINT_DEG from scripts/meca500_real.py
    check_joint_limits([-2.4, -2.0, 52.3, -2.2, -32.1, -37.6])


def test_check_joint_limits_reports_the_joint_number():
    joints = [0.0] * 6
    joints[2] = 69.0  # inside the hard stop (70), outside the soft one (65)
    with pytest.raises(LimitViolation, match="J3"):
        check_joint_limits(joints)


def test_check_joint_limits_rejects_wrong_length():
    with pytest.raises(LimitViolation, match="Expected 6 joint angles"):
        check_joint_limits([0.0, 0.0, 0.0])


# ── Poses and waypoints ──────────────────────────────────────────────────────

def test_check_pose_ignores_orientation():
    check_pose([*INSIDE, 999.0, -999.0, 720.0])


def test_check_pose_rejects_short_pose():
    with pytest.raises(LimitViolation, match=f"{POSE_LENGTH}-value pose"):
        check_pose(list(INSIDE))


def test_check_waypoints_accepts_triples_and_poses():
    assert check_waypoints([INSIDE, (*INSIDE, 0.0, 0.0, 0.0)]) == 2


def test_check_waypoints_reports_the_failing_index():
    waypoints = [INSIDE, INSIDE, (190.0, 0.0, 1000.0)]
    with pytest.raises(LimitViolation, match="Waypoint 2"):
        check_waypoints(waypoints)


def test_check_waypoints_rejects_odd_lengths():
    with pytest.raises(LimitViolation, match="expected 3 or 6 values"):
        check_waypoints([(1.0, 2.0)])


def test_check_waypoints_on_empty_input():
    assert check_waypoints([]) == 0
