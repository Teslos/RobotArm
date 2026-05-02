"""World / stage initialisation: TGS solver, 60 Hz, Z-up, collision groups."""
from __future__ import annotations

import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from pxr import Gf, PhysxSchema, UsdPhysics

from .config import RobotArmCfg


def build_world(cfg: RobotArmCfg | None = None) -> World:
    """Create and configure the Isaac Sim World according to spec."""
    if cfg is None:
        cfg = RobotArmCfg()

    stage_utils.set_stage_up_axis(cfg.physics.up_axis)

    world = World(physics_dt=cfg.physics.dt, rendering_dt=cfg.physics.dt)
    world.scene.add_default_ground_plane()

    _configure_physics_scene(world, cfg)
    _configure_collision_groups(world)

    return world


def _configure_physics_scene(world: World, cfg: RobotArmCfg) -> None:
    stage = world.stage
    physics_scene_path = "/World/physicsScene"

    # Ensure a physics scene prim exists
    scene_prim = stage.GetPrimAtPath(physics_scene_path)
    if not scene_prim.IsValid():
        physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
    else:
        physics_scene = UsdPhysics.Scene.Get(stage, physics_scene_path)

    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(*cfg.physics.gravity[:3]))
    physics_scene.CreateGravityMagnitudeAttr().Set(
        Gf.Vec3f(*cfg.physics.gravity).GetLength()
    )

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
    # TGS is more stable than PGS for position-driven actuators
    physx_scene.CreateSolverTypeAttr().Set(cfg.physics.solver)
    physx_scene.CreateTimeStepsPerSecondAttr().Set(
        int(round(1.0 / cfg.physics.dt))
    )


def _configure_collision_groups(world: World) -> None:
    """
    Group A: robot links  — self-collision disabled for adjacent links.
    Group B: environment  — full collision with Group A.
    """
    stage = world.stage

    robot_group_path = "/World/CollisionGroups/RobotGroup"
    env_group_path = "/World/CollisionGroups/EnvGroup"

    for path in (robot_group_path, env_group_path):
        if not stage.GetPrimAtPath(path).IsValid():
            stage.DefinePrim(path, "PhysicsCollisionGroup")

    robot_group = UsdPhysics.CollisionGroup.Get(stage, robot_group_path)
    env_group = UsdPhysics.CollisionGroup.Get(stage, env_group_path)

    if robot_group and env_group:
        # Robot links collide with environment but not with adjacent self links
        robot_group.GetFilteredGroupsRel().AddTarget(robot_group_path)
