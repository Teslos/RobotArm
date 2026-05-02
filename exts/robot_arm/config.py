"""Typed configuration dataclasses mirroring config/robot_arm.yaml."""
from dataclasses import dataclass, field
from typing import List
import math


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
class RobotArmCfg:
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    articulation: ArticulationCfg = field(default_factory=ArticulationCfg)
    contact: ContactCfg = field(default_factory=ContactCfg)
    joint: JointCfg = field(default_factory=JointCfg)
    sensor: SensorCfg = field(default_factory=SensorCfg)
    safeguard: SafeguardCfg = field(default_factory=SafeguardCfg)
    controller: ControllerCfg = field(default_factory=ControllerCfg)
