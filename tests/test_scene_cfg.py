"""Unit tests for SceneCfg — no Isaac Sim required."""
import pytest
from exts.robot_arm.config import SceneCfg, RobotArmCfg


class TestSceneCfgDefaults:
    def test_prim_paths(self):
        sc = SceneCfg()
        assert sc.robot_prim_path == "/World/mecharm_270"
        assert sc.busbar_prim_path == "/World/busbar"

    def test_robot_position_at_origin(self):
        sc = SceneCfg()
        assert sc.robot_position == (0.0, 0.0, 0.0)

    def test_busbar_position_within_reach(self):
        sc = SceneCfg()
        x, y, z = sc.busbar_position
        reach = (x**2 + y**2 + z**2) ** 0.5
        assert reach <= 0.27, "Default busbar position must be within 270 mm reach"

    def test_usd_paths_are_strings(self):
        sc = SceneCfg()
        assert isinstance(sc.robot_usd_path, str)
        assert isinstance(sc.busbar_usd_path, str)

    def test_usd_paths_contain_expected_filenames(self):
        sc = SceneCfg()
        assert "mecharm_270" in sc.robot_usd_path
        assert "busbar" in sc.busbar_usd_path

    def test_ccd_link_names_contains_link6(self):
        sc = SceneCfg()
        assert "link6" in sc.ccd_link_names  # lowercase — matches mechArm 270-Pi URDF


class TestSceneCfgOverrides:
    def test_busbar_position_override(self):
        sc = SceneCfg(busbar_position=(0.3, 0.1, 0.0))
        assert sc.busbar_position == (0.3, 0.1, 0.0)

    def test_prim_path_override(self):
        sc = SceneCfg(robot_prim_path="/World/myrobot")
        assert sc.robot_prim_path == "/World/myrobot"

    def test_ccd_link_names_override(self):
        sc = SceneCfg(ccd_link_names=("CustomLink",))
        assert sc.ccd_link_names == ("CustomLink",)


class TestRobotArmCfgScene:
    def test_robot_arm_cfg_has_scene_field(self):
        cfg = RobotArmCfg()
        assert isinstance(cfg.scene, SceneCfg)

    def test_scene_field_uses_correct_defaults(self):
        cfg = RobotArmCfg()
        assert cfg.scene.robot_prim_path == "/World/mecharm_270"
