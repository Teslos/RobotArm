"""Unit tests for config.py — no Isaac Sim required."""
import math
import pytest

from exts.robot_arm.config import JointCfg, RobotArmCfg, PhysicsCfg, SceneCfg


def test_joint_cfg_critical_damping():
    j = JointCfg(stiffness=400.0)
    assert math.isclose(j.damping, 2.0 * math.sqrt(400.0))


def test_joint_cfg_different_stiffness():
    j = JointCfg(stiffness=100.0)
    assert math.isclose(j.damping, 2.0 * math.sqrt(100.0))


def test_robot_arm_cfg_defaults():
    cfg = RobotArmCfg()
    assert cfg.physics.solver == "TGS"
    assert math.isclose(cfg.physics.dt, 1.0 / 60.0)
    assert cfg.physics.up_axis == "Z"
    assert cfg.articulation.fixed_base is True
    assert cfg.articulation.joint_friction == 0.05
    assert cfg.articulation.joint_damping == 1.0
    assert cfg.contact.contact_offset == 0.02
    assert cfg.contact.rest_offset == 0.00
    assert cfg.sensor.render_lag_frames == 1
    assert cfg.safeguard.drift_check_interval == 10_000
    assert cfg.safeguard.max_drift_m == 1e-4
    assert cfg.controller.rmpflow_timeout_s == 5.0
    assert cfg.controller.stall_velocity_threshold == 1e-3
    assert isinstance(cfg.scene, SceneCfg)
    assert cfg.scene.robot_prim_path == "/World/mecharm_270"
