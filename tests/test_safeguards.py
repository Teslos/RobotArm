"""Unit tests for SafeguardManager — mocks out Isaac Sim Articulation."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from exts.robot_arm.config import RobotArmCfg, SafeguardCfg
from exts.robot_arm.safeguards import SafeguardManager


def _make_robot(position=(0.0, 0.0, 0.0), num_dof=6):
    robot = MagicMock()
    robot.num_dof = num_dof
    robot.get_world_pose.return_value = (np.array(position), np.array([1, 0, 0, 0]))
    return robot


def _make_cfg(drift_check_interval=10, max_drift_m=1e-4):
    cfg = RobotArmCfg()
    cfg.safeguard = SafeguardCfg(
        drift_check_interval=drift_check_interval,
        max_drift_m=max_drift_m,
    )
    return cfg


class TestTeleportGuard:
    def test_zeros_velocities_and_efforts_on_teleport(self):
        robot = _make_robot()
        mgr = SafeguardManager(robot, cfg=_make_cfg())

        new_pos = np.array([1.0, 0.0, 0.5])
        new_ori = np.array([1.0, 0.0, 0.0, 0.0])
        mgr.safe_teleport(new_pos, new_ori)

        robot.set_world_pose.assert_called_once_with(
            position=new_pos, orientation=new_ori
        )
        robot.set_joint_velocities.assert_called_once()
        robot.set_joint_efforts.assert_called_once()
        np.testing.assert_array_equal(
            robot.set_joint_velocities.call_args[0][0], np.zeros(6)
        )
        np.testing.assert_array_equal(
            robot.set_joint_efforts.call_args[0][0], np.zeros(6)
        )

    def test_teleport_calls_set_world_pose_before_zeroing(self):
        robot = _make_robot()
        mgr = SafeguardManager(robot, cfg=_make_cfg())
        call_order = []
        robot.set_world_pose.side_effect = lambda **_: call_order.append("pose")
        robot.set_joint_velocities.side_effect = lambda _: call_order.append("vel")
        robot.set_joint_efforts.side_effect = lambda _: call_order.append("eff")

        mgr.safe_teleport(np.zeros(3), np.array([1, 0, 0, 0]))
        assert call_order == ["pose", "vel", "eff"]


class TestOriginRecentering:
    def test_no_snap_when_within_tolerance(self):
        robot = _make_robot(position=(0.0, 0.0, 0.0))
        cfg = _make_cfg(drift_check_interval=5, max_drift_m=1e-4)
        mgr = SafeguardManager(robot, cfg=cfg)

        # Drift less than threshold
        robot.get_world_pose.return_value = (
            np.array([0.00005, 0.0, 0.0]),
            np.array([1, 0, 0, 0]),
        )
        # Reset call count from __init__
        robot.set_world_pose.reset_mock()

        for _ in range(5):
            mgr.step()

        robot.set_world_pose.assert_not_called()

    def test_snaps_base_when_drift_exceeds_threshold(self):
        robot = _make_robot(position=(0.0, 0.0, 0.0))
        cfg = _make_cfg(drift_check_interval=5, max_drift_m=1e-4)
        mgr = SafeguardManager(robot, cfg=cfg)

        # Drift exceeds 0.1 mm
        robot.get_world_pose.return_value = (
            np.array([0.001, 0.0, 0.0]),
            np.array([1, 0, 0, 0]),
        )
        robot.set_world_pose.reset_mock()

        for _ in range(5):
            mgr.step()

        robot.set_world_pose.assert_called_once()
        snap_pos = robot.set_world_pose.call_args[1]["position"]
        np.testing.assert_array_almost_equal(snap_pos, [0.0, 0.0, 0.0])
