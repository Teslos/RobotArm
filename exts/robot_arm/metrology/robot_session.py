"""
Thin, safety-checked wrapper around a ``mecademicpy`` ``Robot``.

The wrapper is duck-typed: it never imports ``mecademicpy`` at module scope, so
the module (and everything that depends on it) imports and tests fine on a
machine without the SDK.  :func:`connect_robot` does the real import lazily.

Fixes relative to the original scripts
--------------------------------------
* **Pose arguments are splatted.**  ``UPDATE mit SYNCRO.py``,
  ``Width X And Z Measurment .py`` and ``Y and Z Measurment.py`` all called
  ``robot.MovePose(pose)`` with a *list*; the SDK expects six positional
  arguments, so every one of those calls raises ``TypeError`` at runtime.
* **Poses are never "called".**  ``Width X And Z Measurment .py`` did
  ``start_pose = home_pose()`` on a list — an immediate ``TypeError``.
* **Shutdown cannot mask the real error.**  Two scripts had a bare
  ``finally: robot.DeactivateRobot()``; if ``Connect`` failed, the deactivate
  raised and buried the original exception.  :meth:`RobotSession.shutdown`
  swallows and reports its own failures instead, and its ``deactivate`` /
  ``disconnect`` steps can each be skipped so a finished scan can leave the
  arm powered and homed.
* **Every target is bounds-checked** against the workspace box and the software
  joint limits before it is sent.
* **Waits are bounded** by ``wait_timeout_s`` rather than blocking forever.
"""
from __future__ import annotations

from typing import Any, List, Sequence

from .limits import (
    DEFAULT_WORKSPACE,
    POSE_LENGTH,
    SOFT_JOINT_LIMITS_DEG,
    LimitViolation,
    WorkspaceBox,
    check_joint_limits,
    check_pose,
)

__all__ = [
    "DEFAULT_ROBOT_IP",
    "VelocityLimits",
    "RobotSession",
    "connect_robot",
    "normalize_pose",
]

DEFAULT_ROBOT_IP = "192.168.0.100"
DEFAULT_WAIT_TIMEOUT_S = 60.0


class VelocityLimits:
    """Conservative motion caps applied right after activation."""

    def __init__(
        self,
        joint_pct: float = 20.0,
        cart_lin_mm_s: float = 20.0,
        cart_ang_deg_s: float = 45.0,
    ) -> None:
        if not 0.0 < joint_pct <= 100.0:
            raise ValueError(f"joint_pct must be in (0, 100], got {joint_pct}")
        if cart_lin_mm_s <= 0.0 or cart_ang_deg_s <= 0.0:
            raise ValueError("Cartesian velocity caps must be positive")
        self.joint_pct = float(joint_pct)
        self.cart_lin_mm_s = float(cart_lin_mm_s)
        self.cart_ang_deg_s = float(cart_ang_deg_s)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"VelocityLimits(joint_pct={self.joint_pct}, "
            f"cart_lin_mm_s={self.cart_lin_mm_s}, "
            f"cart_ang_deg_s={self.cart_ang_deg_s})"
        )


def normalize_pose(pose: Any) -> List[float]:
    """Coerce whatever the SDK returned into a plain 6-float list.

    ``GetRtCartPos`` has returned lists, tuples and numpy arrays across SDK
    versions.  Callers that mutated the result in place (as the originals did)
    were also mutating SDK-owned state; this always hands back a fresh list.
    """
    if pose is None:
        raise LimitViolation("Robot returned no Cartesian pose (got None)")
    try:
        values = [float(v) for v in pose]
    except (TypeError, ValueError) as exc:
        raise LimitViolation(f"Unreadable robot pose: {pose!r}") from exc
    if len(values) != POSE_LENGTH:
        raise LimitViolation(
            f"Expected a {POSE_LENGTH}-value pose "
            f"[x, y, z, alpha, beta, gamma], got {values!r}"
        )
    return values


class RobotSession:
    """Safety-checked facade over a connected Meca500.

    The ``robot`` argument only has to provide the handful of methods used
    below, which keeps the class testable with a stub.
    """

    def __init__(
        self,
        robot: Any,
        *,
        workspace: WorkspaceBox = DEFAULT_WORKSPACE,
        joint_limits: Sequence[Sequence[float]] = SOFT_JOINT_LIMITS_DEG,
        wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
        verbose: bool = True,
    ) -> None:
        self.robot = robot
        self.workspace = workspace
        self.joint_limits = joint_limits
        self.wait_timeout_s = wait_timeout_s
        self.verbose = verbose

    # -- plumbing ------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[robot] {message}")

    def _wait_idle(self) -> None:
        try:
            self.robot.WaitIdle(timeout=self.wait_timeout_s)
        except TypeError:
            # Older SDK builds expose WaitIdle() without a timeout keyword.
            self.robot.WaitIdle()

    # -- lifecycle -----------------------------------------------------------

    def activate_and_home(self) -> None:
        """Clear latent errors, power the drives and home the arm."""
        for step in ("ResetError", "ResumeMotion"):
            method = getattr(self.robot, step, None)
            if method is None:
                continue
            try:
                method()
            except Exception as exc:  # noqa: BLE001 - best effort recovery
                self._log(f"{step} skipped: {exc}")

        self._log("Activating servo drives ...")
        self.robot.ActivateRobot()
        self._log("Homing (the arm will move - keep the workspace clear) ...")
        self.robot.Home()
        self.robot.WaitHomed()
        self._log("Homed and standing by.")

    def apply_velocity_limits(self, limits: VelocityLimits) -> None:
        """Push the velocity caps to the controller."""
        self.robot.SetJointVel(limits.joint_pct)
        self.robot.SetCartLinVel(limits.cart_lin_mm_s)
        self.robot.SetCartAngVel(limits.cart_ang_deg_s)
        self._log(
            f"Velocity caps: joint={limits.joint_pct}%  "
            f"linear={limits.cart_lin_mm_s} mm/s  "
            f"angular={limits.cart_ang_deg_s} deg/s"
        )

    def shutdown(self, *, deactivate: bool = True, disconnect: bool = True) -> None:
        """Deactivate and/or disconnect.  Never raises.

        Both steps are opt-out so a caller can finish a scan without powering
        the arm down: ``shutdown(deactivate=False, disconnect=False)`` leaves
        the drives energised and the arm homed, which is what the scan scripts
        do after a successful run (the arm only has to be re-homed once the
        drives have actually been deactivated).
        """
        steps = []
        if deactivate:
            steps.append("DeactivateRobot")
        if disconnect:
            steps.append("Disconnect")
        for step in steps:
            method = getattr(self.robot, step, None)
            if method is None:
                continue
            try:
                method()
                self._log(f"{step} OK.")
            except Exception as exc:  # noqa: BLE001 - must not mask the real error
                self._log(f"{step} failed during shutdown: {exc}")

    def __enter__(self) -> "RobotSession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    # -- state ---------------------------------------------------------------

    def get_pose(self) -> List[float]:
        """Current Cartesian pose as a fresh ``[x, y, z, alpha, beta, gamma]``."""
        return normalize_pose(self.robot.GetRtCartPos())

    def get_orientation(self) -> List[float]:
        """Current EE orientation ``[alpha, beta, gamma]`` in degrees."""
        return self.get_pose()[3:]

    # -- motion --------------------------------------------------------------

    def move_joints(self, joints_deg: Sequence[float]) -> None:
        """Validate against the software joint limits, then ``MoveJoints``."""
        check_joint_limits(joints_deg, self.joint_limits)
        self.robot.MoveJoints(*[float(j) for j in joints_deg])
        self._wait_idle()

    def move_lin(self, pose: Sequence[float]) -> None:
        """Validate the workspace box, then a straight-line Cartesian move."""
        values = [float(v) for v in pose]
        check_pose(values, self.workspace, context="MoveLin target")
        self.robot.MoveLin(*values)
        self._wait_idle()

    def move_pose(self, pose: Sequence[float]) -> None:
        """Validate the workspace box, then a joint-interpolated pose move."""
        values = [float(v) for v in pose]
        check_pose(values, self.workspace, context="MovePose target")
        self.robot.MovePose(*values)
        self._wait_idle()

    def move_lin_offset(
        self,
        pose: Sequence[float],
        *,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
    ) -> List[float]:
        """Linear move to ``pose`` shifted by an absolute offset.

        Returns the pose actually commanded.  Offsets are applied to the given
        pose rather than to the robot's live position, so repeated calls cannot
        accumulate drift.
        """
        target = [float(v) for v in pose]
        target[0] += dx
        target[1] += dy
        target[2] += dz
        self.move_lin(target)
        return target


def connect_robot(
    ip: str = DEFAULT_ROBOT_IP,
    *,
    workspace: WorkspaceBox = DEFAULT_WORKSPACE,
    wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
    verbose: bool = True,
) -> RobotSession:
    """Import ``mecademicpy``, connect to ``ip`` and wrap it in a session.

    The SDK import happens here so the rest of this package stays importable
    without it installed.
    """
    try:
        from mecademicpy.robot import Robot
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "mecademicpy is not installed. Install it with:\n"
            "    micromamba run -n RobotArm pip install mecademicpy"
        ) from exc

    robot = Robot()
    if verbose:
        print(f"[robot] Connecting to Meca500 at {ip} ...")
    robot.Connect(address=ip, enable_synchronous_mode=True)
    if verbose:
        print("[robot] Connected.")
    return RobotSession(
        robot,
        workspace=workspace,
        wait_timeout_s=wait_timeout_s,
        verbose=verbose,
    )
