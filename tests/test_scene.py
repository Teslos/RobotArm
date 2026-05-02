"""Unit tests for build_scene and _load_busbar — mocks out Isaac Sim."""
import pytest
from unittest.mock import MagicMock, patch, call

from exts.robot_arm.config import RobotArmCfg, SceneCfg


def _make_cfg():
    return RobotArmCfg()


@patch("exts.robot_arm.scene.load_mecharm_270")
@patch("exts.robot_arm.scene.build_world")
class TestBuildScene:
    def test_returns_three_tuple(self, mock_build_world, mock_load_robot):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_world.stage.GetPrimAtPath.return_value.IsValid.return_value = True
        mock_build_world.return_value = mock_world

        result = build_scene(_make_cfg())

        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_build_world_called_once_with_cfg(self, mock_build_world, mock_load_robot):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_world.stage.GetPrimAtPath.return_value.IsValid.return_value = True
        mock_build_world.return_value = mock_world

        cfg = _make_cfg()
        build_scene(cfg)

        mock_build_world.assert_called_once_with(cfg)

    def test_load_mecharm_called_with_world_and_cfg(self, mock_build_world, mock_load_robot):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_world.stage.GetPrimAtPath.return_value.IsValid.return_value = True
        mock_build_world.return_value = mock_world

        cfg = _make_cfg()
        build_scene(cfg)

        mock_load_robot.assert_called_once_with(mock_world, cfg)

    def test_world_is_first_element(self, mock_build_world, mock_load_robot):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_world.stage.GetPrimAtPath.return_value.IsValid.return_value = True
        mock_build_world.return_value = mock_world

        world, _, _ = build_scene(_make_cfg())

        assert world is mock_world

    def test_robot_is_second_element(self, mock_build_world, mock_load_robot):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_world.stage.GetPrimAtPath.return_value.IsValid.return_value = True
        mock_build_world.return_value = mock_world

        _, robot, _ = build_scene(_make_cfg())

        assert robot is mock_load_robot.return_value

    def test_default_cfg_when_none_passed(self, mock_build_world, mock_load_robot):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_world.stage.GetPrimAtPath.return_value.IsValid.return_value = True
        mock_build_world.return_value = mock_world

        result = build_scene(None)

        assert len(result) == 3


@patch("exts.robot_arm.scene.load_mecharm_270")
@patch("exts.robot_arm.scene.build_world")
@patch("exts.robot_arm.scene.add_reference_to_stage")
class TestLoadBusbar:
    def test_add_reference_called_for_busbar(
        self, mock_add_ref, mock_build_world, mock_load_robot
    ):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_world.stage.GetPrimAtPath.return_value.IsValid.return_value = True
        mock_build_world.return_value = mock_world

        cfg = _make_cfg()
        build_scene(cfg)

        busbar_call = [
            c for c in mock_add_ref.call_args_list
            if c[1].get("prim_path") == cfg.scene.busbar_prim_path
        ]
        assert len(busbar_call) == 1
        assert busbar_call[0][1]["usd_path"] == cfg.scene.busbar_usd_path

    def test_busbar_prim_retrieved_after_load(
        self, mock_add_ref, mock_build_world, mock_load_robot
    ):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_world.stage.GetPrimAtPath.return_value.IsValid.return_value = True
        mock_build_world.return_value = mock_world

        cfg = _make_cfg()
        build_scene(cfg)

        mock_world.stage.GetPrimAtPath.assert_called_with(cfg.scene.busbar_prim_path)

    def test_invalid_busbar_prim_raises_runtime_error(
        self, mock_add_ref, mock_build_world, mock_load_robot
    ):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_world.stage.GetPrimAtPath.return_value.IsValid.return_value = False
        mock_build_world.return_value = mock_world

        with pytest.raises(RuntimeError, match="Busbar prim not found"):
            build_scene(_make_cfg())

    def test_busbar_prim_is_third_element(
        self, mock_add_ref, mock_build_world, mock_load_robot
    ):
        from exts.robot_arm.scene import build_scene

        mock_world = MagicMock()
        mock_prim = MagicMock()
        mock_prim.IsValid.return_value = True
        mock_world.stage.GetPrimAtPath.return_value = mock_prim
        mock_build_world.return_value = mock_world

        _, _, busbar = build_scene(_make_cfg())

        assert busbar is mock_prim
