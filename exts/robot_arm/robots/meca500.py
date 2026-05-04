"""Meca500 R3 loader: USD reference + articulation setup."""
from __future__ import annotations

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage

from ..articulation import configure_articulation
from ..config import RobotArmCfg


def load_meca500(world: World, cfg: RobotArmCfg | None = None) -> Articulation:
    """
    Load the Mecademic Meca500 R3 USD onto the stage, wrap it as an
    Articulation, apply physics drives and constraints, and register it
    with the scene.

    NOTE: Call world.reset() after this function before accessing
    dof_names or num_dof — Isaac Sim populates those during initialization.
    """
    if cfg is None:
        cfg = RobotArmCfg()
    sc = cfg.scene

    add_reference_to_stage(usd_path=sc.robot_usd_path, prim_path=sc.robot_prim_path)

    robot = Articulation(
        prim_path=sc.robot_prim_path,
        name="meca500r3",
        position=list(sc.robot_position),
    )
    world.scene.add(robot)

    configure_articulation(robot, cfg, ccd_link_names=tuple(sc.ccd_link_names))

    return robot
