"""Unit tests for load_mecharm_270 — mocks out Isaac Sim."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

from exts.robot_arm.config import RobotArmCfg, SceneCfg


def _make_world():
    world = MagicMock()
    world.scene = MagicMock()
    return world


def _make_cfg(robot_prim_path="/World/mecharm_270", ccd_link_names=("Link6", "end_effector")):
    cfg = RobotArmCfg()
    cfg.scene = SceneCfg(
        robot_prim_path=robot_prim_path,
        ccd_link_names=ccd_link_names,
    )
    return cfg


@patch("exts.robot_arm.robots.mecharm.configure_articulation")
@patch("exts.robot_arm.robots.mecharm.Articulation")
@patch("exts.robot_arm.robots.mecharm.add_reference_to_stage")
class TestLoadMecharm270:
    def test_add_reference_called_with_correct_args(
        self, mock_add_ref, mock_artic, mock_configure
    ):
        from exts.robot_arm.robots.mecharm import load_mecharm_270

        cfg = _make_cfg()
        load_mecharm_270(_make_world(), cfg)

        mock_add_ref.assert_called_once_with(
            usd_path=cfg.scene.robot_usd_path,
            prim_path=cfg.scene.robot_prim_path,
        )

    def test_articulation_constructed_at_correct_prim_path(
        self, mock_add_ref, mock_artic, mock_configure
    ):
        from exts.robot_arm.robots.mecharm import load_mecharm_270

        cfg = _make_cfg(robot_prim_path="/World/testbot")
        load_mecharm_270(_make_world(), cfg)

        mock_artic.assert_called_once()
        kwargs = mock_artic.call_args
        assert kwargs[1]["prim_path"] == "/World/testbot"

    def test_articulation_named_mecharm_270(
        self, mock_add_ref, mock_artic, mock_configure
    ):
        from exts.robot_arm.robots.mecharm import load_mecharm_270

        load_mecharm_270(_make_world(), _make_cfg())
        assert mock_artic.call_args[1]["name"] == "mecharm_270"

    def test_world_scene_add_called(
        self, mock_add_ref, mock_artic, mock_configure
    ):
        from exts.robot_arm.robots.mecharm import load_mecharm_270

        world = _make_world()
        load_mecharm_270(world, _make_cfg())

        world.scene.add.assert_called_once_with(mock_artic.return_value)

    def test_configure_articulation_called(
        self, mock_add_ref, mock_artic, mock_configure
    ):
        from exts.robot_arm.robots.mecharm import load_mecharm_270

        cfg = _make_cfg()
        load_mecharm_270(_make_world(), cfg)

        mock_configure.assert_called_once()
        args, kwargs = mock_configure.call_args
        assert args[0] is mock_artic.return_value
        assert args[1] is cfg

    def test_ccd_link_names_forwarded(
        self, mock_add_ref, mock_artic, mock_configure
    ):
        from exts.robot_arm.robots.mecharm import load_mecharm_270

        cfg = _make_cfg(ccd_link_names=("TestLink",))
        load_mecharm_270(_make_world(), cfg)

        _, kwargs = mock_configure.call_args
        assert kwargs["ccd_link_names"] == ("TestLink",)

    def test_returns_articulation_instance(
        self, mock_add_ref, mock_artic, mock_configure
    ):
        from exts.robot_arm.robots.mecharm import load_mecharm_270

        result = load_mecharm_270(_make_world(), _make_cfg())
        assert result is mock_artic.return_value

    def test_default_cfg_used_when_none_passed(
        self, mock_add_ref, mock_artic, mock_configure
    ):
        from exts.robot_arm.robots.mecharm import load_mecharm_270

        load_mecharm_270(_make_world(), None)

        mock_add_ref.assert_called_once()
        usd_path = mock_add_ref.call_args[1]["usd_path"]
        assert isinstance(usd_path, str) and len(usd_path) > 0
