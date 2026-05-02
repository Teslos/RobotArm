"""Scene assembly: World + mechArm 270-Pi + busbar."""
from __future__ import annotations

from typing import Any, Tuple

from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Gf, UsdGeom

from .config import RobotArmCfg
from .robots.mecharm import load_mecharm_270
from .world import build_world


def build_scene(cfg: RobotArmCfg | None = None) -> Tuple[World, Articulation, Any]:
    """
    Build the complete Isaac Sim scene.

    Returns:
        (world, robot, busbar_prim) — all registered/positioned, ready for
        world.reset() to finalize articulation initialization.
    """
    if cfg is None:
        cfg = RobotArmCfg()

    world = build_world(cfg)
    robot = load_mecharm_270(world, cfg)
    busbar = _load_busbar(world, cfg)

    return world, robot, busbar


def _load_busbar(world: World, cfg: RobotArmCfg) -> Any:
    """
    Reference the busbar USD onto the stage as static geometry and position it.

    The busbar is a measurement target only — no RigidBodyAPI, just a static
    collider. Position is set via a direct UsdGeom translate op (spec §4.3).
    Returns the raw USD prim; callers can call world.stage.GetPrimAtPath(
    cfg.scene.busbar_prim_path) to retrieve it later.
    """
    sc = cfg.scene

    add_reference_to_stage(usd_path=sc.busbar_usd_path, prim_path=sc.busbar_prim_path)

    prim = world.stage.GetPrimAtPath(sc.busbar_prim_path)
    if not prim.IsValid():
        raise RuntimeError(
            f"Busbar prim not found at '{sc.busbar_prim_path}' after load. "
            "Verify busbar_usd_path and that the USD was converted correctly."
        )

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*sc.busbar_position))

    return prim
