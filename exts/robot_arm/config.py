"""Typed configuration dataclasses mirroring config/robot_arm.yaml."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import math
import os

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets")


@dataclass
class PhysicsCfg:
    solver: str = "TGS"
    dt: float = 1.0 / 60.0
    gravity: List[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    up_axis: str = "Z"


@dataclass
class ArticulationCfg:
    fixed_base: bool = True
    joint_friction: float = 0.05
    joint_damping: float = 1.0
    # Density applied to all rigid body links (kg/m³).
    # PhysX computes mass + inertia tensor from convex hull volume × density.
    # 2700 = aluminium (Meca500 R3 construction).
    link_density: float = 2700.0


@dataclass
class ContactCfg:
    contact_offset: float = 0.02
    rest_offset: float = 0.00


@dataclass
class JointCfg:
    stiffness: float = 400.0
    damping: float = field(init=False)

    def __post_init__(self) -> None:
        # Critical damping: D = 2 * sqrt(K)
        self.damping = 2.0 * math.sqrt(self.stiffness)


@dataclass
class SensorCfg:
    render_lag_frames: int = 1


@dataclass
class SafeguardCfg:
    drift_check_interval: int = 10_000
    max_drift_m: float = 1e-4


@dataclass
class ControllerCfg:
    rmpflow_timeout_s: float = 5.0
    stall_velocity_threshold: float = 1e-3


@dataclass
class SceneCfg:
    robot_usd_path: str = field(
        default_factory=lambda: os.path.normpath(
            os.path.join(_ASSETS_DIR, "mecademic_description", "urdf", "meca500r3.usd")
        )
    )
    busbar_usd_path: str = field(
        default_factory=lambda: os.path.normpath(
            os.path.join(_ASSETS_DIR, "busbar", "busbar.usd")
        )
    )
    robot_prim_path: str = "/World/meca500r3"
    busbar_prim_path: str = "/World/busbar"
    robot_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    busbar_position: Tuple[float, float, float] = (0.190, 0.0, 0.08625)
    ccd_link_names: Tuple[str, ...] = field(
        default_factory=lambda: ("meca_axis_6_link",)  # EE link from Meca500 R3 URDF
    )


@dataclass
class GridSearchCfg:
    """Configuration for IK-based workspace grid search."""
    # Cartesian search volume (metres), centred loosely over the busbar
    x_range: Tuple[float, float] = (0.10, 0.35)
    y_range: Tuple[float, float] = (-0.25, 0.25)
    z_range: Tuple[float, float] = (0.10, 0.35)
    # Grid resolution in each axis
    nx: int = 8
    ny: int = 8
    nz: int = 5
    # EE orientations to attempt per grid point: (qw, qx, qy, qz) or None
    # None = position-only IK (no orientation constraint) — maximises reachable fraction.
    # Add explicit orientations (e.g. "pointing down") to filter for approach angle.
    orientations: Tuple[Optional[Tuple[float, float, float, float]], ...] = field(
        default_factory=lambda: (None,)
    )


@dataclass
class RobotArmCfg:
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    articulation: ArticulationCfg = field(default_factory=ArticulationCfg)
    contact: ContactCfg = field(default_factory=ContactCfg)
    joint: JointCfg = field(default_factory=JointCfg)
    sensor: SensorCfg = field(default_factory=SensorCfg)
    safeguard: SafeguardCfg = field(default_factory=SafeguardCfg)
    controller: ControllerCfg = field(default_factory=ControllerCfg)
    scene: SceneCfg = field(default_factory=SceneCfg)
