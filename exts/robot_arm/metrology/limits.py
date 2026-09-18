"""
Motion-safety limits for the *physical* Meca500 R3.

Units here are the robot's native ones — **millimetres and degrees** — unlike
the Isaac Sim side of this repo, which works in metres and radians.

Every Cartesian target handed to the robot is validated against
:data:`DEFAULT_WORKSPACE` and every joint target against
:data:`SOFT_JOINT_LIMITS_DEG` before it leaves the PC.  The original
``INT_Monitor`` scripts performed no validation at all: a mistyped step size
drove the arm straight into the fixture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

__all__ = [
    "LimitViolation",
    "WorkspaceBox",
    "HARD_JOINT_LIMITS_DEG",
    "SOFT_JOINT_LIMITS_DEG",
    "SOFT_LIMIT_MARGIN_DEG",
    "DEFAULT_WORKSPACE",
    "check_joint_limits",
    "check_pose",
    "check_waypoints",
]

POSE_LENGTH = 6
"""A Mecademic Cartesian pose is ``[x, y, z, alpha, beta, gamma]``."""


class LimitViolation(ValueError):
    """Raised when a joint or Cartesian target falls outside its safe range."""


# ── Joint limits ─────────────────────────────────────────────────────────────

HARD_JOINT_LIMITS_DEG: Tuple[Tuple[float, float], ...] = (
    (-175.0, 175.0),   # J1
    (-70.0, 90.0),     # J2
    (-135.0, 70.0),    # J3
    (-170.0, 170.0),   # J4
    (-115.0, 115.0),   # J5
    (-180.0, 180.0),   # J6
)
"""Mechanical hard stops of the Meca500 R3, per the user manual."""

SOFT_LIMIT_MARGIN_DEG: float = 5.0
"""How far inside the hard stops the software limits sit."""

SOFT_JOINT_LIMITS_DEG: Tuple[Tuple[float, float], ...] = tuple(
    (lo + SOFT_LIMIT_MARGIN_DEG, hi - SOFT_LIMIT_MARGIN_DEG)
    for lo, hi in HARD_JOINT_LIMITS_DEG
)
"""Hard stops shrunk by :data:`SOFT_LIMIT_MARGIN_DEG` on both ends."""


# ── Cartesian workspace ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkspaceBox:
    """Axis-aligned Cartesian keep-in volume, in millimetres, base frame."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        for axis, lo, hi in (
            ("x", self.x_min, self.x_max),
            ("y", self.y_min, self.y_max),
            ("z", self.z_min, self.z_max),
        ):
            if lo >= hi:
                raise ValueError(
                    f"WorkspaceBox {axis} bounds must satisfy min < max, "
                    f"got [{lo}, {hi}]"
                )

    @property
    def bounds(self) -> Tuple[Tuple[float, float], ...]:
        """``((x_min, x_max), (y_min, y_max), (z_min, z_max))``."""
        return (
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (self.z_min, self.z_max),
        )

    def contains(self, x: float, y: float, z: float) -> bool:
        """True when ``(x, y, z)`` lies inside the box (bounds inclusive)."""
        return all(
            lo <= value <= hi
            for value, (lo, hi) in zip((x, y, z), self.bounds)
        )

    def check(self, x: float, y: float, z: float, *, context: str = "") -> None:
        """Raise :class:`LimitViolation` if ``(x, y, z)`` is outside the box."""
        prefix = f"{context}: " if context else ""
        for name, value, (lo, hi) in zip("XYZ", (x, y, z), self.bounds):
            if not lo <= value <= hi:
                raise LimitViolation(
                    f"{prefix}{name} = {value:.3f} mm is outside the safe "
                    f"workspace [{lo:.1f}, {hi:.1f}] mm"
                )


DEFAULT_WORKSPACE = WorkspaceBox(
    x_min=50.0, x_max=350.0,
    y_min=-250.0, y_max=250.0,
    z_min=20.0, z_max=400.0,
)
"""Conservative keep-in box around the busbar fixture (mirrors
``scripts/meca500_real.py``)."""


# ── Validators ───────────────────────────────────────────────────────────────

def check_joint_limits(
    joints_deg: Sequence[float],
    limits: Sequence[Tuple[float, float]] = SOFT_JOINT_LIMITS_DEG,
) -> None:
    """Raise :class:`LimitViolation` if any joint angle is out of range."""
    if len(joints_deg) != len(limits):
        raise LimitViolation(
            f"Expected {len(limits)} joint angles, got {len(joints_deg)}"
        )
    for index, (angle, (lo, hi)) in enumerate(zip(joints_deg, limits), start=1):
        if not lo <= angle <= hi:
            raise LimitViolation(
                f"J{index} = {angle:.3f}° is outside the software limit "
                f"[{lo:.1f}°, {hi:.1f}°]"
            )


def check_pose(
    pose: Sequence[float],
    workspace: WorkspaceBox = DEFAULT_WORKSPACE,
    *,
    context: str = "",
) -> None:
    """Validate a six-value Cartesian pose against ``workspace``.

    Only the translation is bounded; orientation is unconstrained.
    """
    if len(pose) != POSE_LENGTH:
        raise LimitViolation(
            f"Expected a {POSE_LENGTH}-value pose "
            f"[x, y, z, alpha, beta, gamma], got {list(pose)!r}"
        )
    workspace.check(pose[0], pose[1], pose[2], context=context)


def check_waypoints(
    waypoints: Iterable[Sequence[float]],
    workspace: WorkspaceBox = DEFAULT_WORKSPACE,
) -> int:
    """Validate every waypoint *before* the first move command is sent.

    Accepts ``(x, y, z)`` triples or full six-value poses.  Returns the number
    of waypoints checked so callers can log it.
    """
    count = 0
    for index, waypoint in enumerate(waypoints):
        if len(waypoint) not in (3, POSE_LENGTH):
            raise LimitViolation(
                f"Waypoint {index}: expected 3 or {POSE_LENGTH} values, "
                f"got {list(waypoint)!r}"
            )
        workspace.check(
            waypoint[0], waypoint[1], waypoint[2],
            context=f"Waypoint {index}",
        )
        count += 1
    return count
