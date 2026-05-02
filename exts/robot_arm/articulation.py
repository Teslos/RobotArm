"""Articulation setup: position drives, hard limits, CCD, friction/damping."""
from __future__ import annotations

from typing import List

import numpy as np
from omni.isaac.core.articulations import Articulation
from pxr import PhysxSchema, UsdPhysics

from .config import RobotArmCfg


CCD_LINK_NAMES = ("end_effector", "sensor_arm")


def configure_articulation(
    robot: Articulation,
    cfg: RobotArmCfg | None = None,
    ccd_link_names: tuple[str, ...] = CCD_LINK_NAMES,
) -> None:
    """Apply all spec-mandated articulation settings to an already-added robot."""
    if cfg is None:
        cfg = RobotArmCfg()

    stage = robot.prim.GetStage()
    art_prim = robot.prim

    _set_fixed_base(art_prim, cfg.articulation.fixed_base)
    _apply_joint_drives(robot, cfg)
    _apply_contact_offsets(art_prim, stage, cfg)
    _enable_ccd_on_links(art_prim, stage, ccd_link_names)


def _set_fixed_base(art_prim, fixed: bool) -> None:
    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(art_prim)
    physx_art.CreateFixedBaseAttr().Set(fixed)


def _apply_joint_drives(robot: Articulation, cfg: RobotArmCfg) -> None:
    stage = robot.prim.GetStage()
    dof_names = robot.dof_names

    for dof_name in dof_names:
        joint_path = _find_joint_prim_path(robot, dof_name)
        if not joint_path:
            continue

        joint_prim = stage.GetPrimAtPath(joint_path)

        # Position drive
        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive_api.CreateTypeAttr().Set("force")
        drive_api.CreateTargetPositionAttr().Set(0.0)
        drive_api.CreateStiffnessAttr().Set(cfg.joint.stiffness)
        drive_api.CreateDampingAttr().Set(cfg.joint.damping)

        # Hard joint limits (soft_limit disabled by not applying LimitSoftAPI)
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        physx_joint.CreateJointFrictionAttr().Set(cfg.articulation.joint_friction)
        physx_joint.CreateDampingAttr().Set(cfg.articulation.joint_damping)


def _apply_contact_offsets(art_prim, stage, cfg: RobotArmCfg) -> None:
    for prim in stage.TraverseAll():
        if not prim.GetPath().HasPrefix(art_prim.GetPath()):
            continue
        if PhysxSchema.PhysxCollisionAPI.CanApply(prim):
            col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col_api.CreateContactOffsetAttr().Set(cfg.contact.contact_offset)
            col_api.CreateRestOffsetAttr().Set(cfg.contact.rest_offset)


def _enable_ccd_on_links(art_prim, stage, link_names: tuple[str, ...]) -> None:
    """Enable CCD on thin/fast-moving links to prevent tunnelling."""
    for prim in stage.TraverseAll():
        if not prim.GetPath().HasPrefix(art_prim.GetPath()):
            continue
        if any(name in prim.GetName() for name in link_names):
            rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb_api.CreateEnableCCDAttr().Set(True)


def _find_joint_prim_path(robot: Articulation, dof_name: str) -> str | None:
    stage = robot.prim.GetStage()
    for prim in stage.TraverseAll():
        if prim.GetPath().HasPrefix(robot.prim.GetPath()):
            if prim.GetName() == dof_name and prim.IsA(UsdPhysics.RevoluteJoint):
                return str(prim.GetPath())
    return None


def set_joint_position_targets(robot: Articulation, targets: List[float]) -> None:
    """Write joint position targets via direct USD attribute writes (spec §4.3)."""
    stage = robot.prim.GetStage()
    dof_names = robot.dof_names

    for dof_name, target in zip(dof_names, targets):
        joint_path = _find_joint_prim_path(robot, dof_name)
        if not joint_path:
            continue
        prim = stage.GetPrimAtPath(joint_path)
        drive_api = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive_api:
            drive_api.GetTargetPositionAttr().Set(float(target))
