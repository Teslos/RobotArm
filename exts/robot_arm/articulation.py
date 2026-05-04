"""Articulation setup: position drives, hard limits, CCD, friction/damping."""
from __future__ import annotations

from typing import List

import numpy as np
from omni.isaac.core.articulations import Articulation
from pxr import PhysxSchema, UsdPhysics

from .config import RobotArmCfg


CCD_LINK_NAMES = ("end_effector", "sensor_arm")

# Both typenames appear depending on Isaac Sim / URDF importer version
_REVOLUTE_TYPENAMES = frozenset({"RevoluteJoint", "PhysicsRevoluteJoint"})


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
    _apply_link_densities(art_prim, stage, cfg.articulation.link_density)


def _set_fixed_base(art_prim, fixed: bool) -> None:
    # In Isaac Sim 4.5 / PhysX 5, fixedBase is set by the URDF importer
    # (fix_base=True -> world fixed joint).  PhysxArticulationAPI no longer
    # exposes CreateFixedBaseAttr, so we just ensure the API is applied.
    PhysxSchema.PhysxArticulationAPI.Apply(art_prim)


def _apply_joint_drives(robot: Articulation, cfg: RobotArmCfg) -> None:
    """
    Set PD gains and drive type via USD DriveAPI on each revolute joint.

    UsdPhysics.DriveAPI stiffness/damping are the PhysX-level Kp/Kd gains.
    _find_joint_prim_path searches the full stage (no prefix filter) so it
    works regardless of where Isaac Sim's URDF importer places the joint prims.
    """
    stage = robot.prim.GetStage()
    found = 0
    for dof_name in robot.dof_names:
        joint_path = _find_joint_prim_path(stage, dof_name)
        if not joint_path:
            print(f"[articulation] WARNING: joint prim not found for DOF '{dof_name}'")
            continue
        joint_prim = stage.GetPrimAtPath(joint_path)
        drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        drive_api.CreateTypeAttr().Set("force")
        drive_api.CreateStiffnessAttr().Set(cfg.joint.stiffness)
        drive_api.CreateDampingAttr().Set(cfg.joint.damping)
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint_prim)
        physx_joint.CreateJointFrictionAttr().Set(cfg.articulation.joint_friction)
        found += 1
    print(f"[articulation] PD drives configured on {found}/{len(robot.dof_names)} joints")


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


def _find_joint_prim_path(stage, dof_name: str) -> str | None:
    """
    Search the entire stage for a revolute joint prim whose name matches dof_name.

    Does NOT filter by robot path prefix — the robot's prim hierarchy after USD
    referencing can be at an unexpected depth depending on Isaac Sim version and
    URDF importer.  Accepts both 'RevoluteJoint' and 'PhysicsRevoluteJoint'
    typenames; falls back to any prim with an angular DriveAPI already applied.
    """
    for prim in stage.TraverseAll():
        if prim.GetName() != dof_name:
            continue
        if prim.GetTypeName() in _REVOLUTE_TYPENAMES or prim.IsA(UsdPhysics.RevoluteJoint):
            return str(prim.GetPath())
        if UsdPhysics.DriveAPI.Get(prim, "angular"):
            return str(prim.GetPath())
    return None


def _apply_link_densities(art_prim, stage, density: float) -> None:
    """
    Apply uniform material density to every rigid body link under the articulation.
    PhysX uses density x convex-hull volume to compute mass and inertia tensor
    automatically (spec §1.1 density-first approach).
    Only applied when the link has no explicit mass already set.
    """
    for prim in stage.TraverseAll():
        if not prim.GetPath().HasPrefix(art_prim.GetPath()):
            continue
        if not PhysxSchema.PhysxRigidBodyAPI.CanApply(prim):
            continue
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        existing_mass = mass_api.GetMassAttr()
        if not existing_mass or not existing_mass.HasAuthoredValue():
            mass_api.CreateDensityAttr().Set(density)


def set_joint_position_targets(robot: Articulation, targets: List[float]) -> None:
    """
    Command joint position targets via Isaac Sim's ArticulationController.

    Uses apply_action() which handles DOF indexing and unit conventions
    internally (targets in radians, consistent with Isaac Sim convention).
    This replaces direct USD DriveAPI attribute writes which are fragile
    across URDF importer stage layout variations.
    """
    try:
        from isaacsim.core.api.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.utils.types import ArticulationAction  # type: ignore[no-redef]
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=np.array(targets, dtype=float))
    )
