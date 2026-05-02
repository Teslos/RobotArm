"""Edge-case safeguards: teleport guard, drift re-centering (spec §5.1-5.2)."""
from __future__ import annotations

import numpy as np
from omni.isaac.core.articulations import Articulation
from pxr import Gf

from .config import RobotArmCfg


class SafeguardManager:
    """
    Owns two safeguards:
      - Teleportation guard  (§5.1): zeros velocities/efforts on any manual translate.
      - Origin re-centering  (§5.2): snaps base if it drifts > max_drift_m after N steps.
    """

    def __init__(self, robot: Articulation, cfg: RobotArmCfg | None = None) -> None:
        self._robot = robot
        self._cfg = cfg or RobotArmCfg()
        self._step_count: int = 0
        self._origin: np.ndarray = self._read_base_position()

    # ------------------------------------------------------------------
    # Per-step call
    # ------------------------------------------------------------------

    def step(self) -> None:
        self._step_count += 1
        sg = self._cfg.safeguard
        if self._step_count % sg.drift_check_interval == 0:
            self._check_origin_drift()

    # ------------------------------------------------------------------
    # Teleportation guard (§5.1)
    # ------------------------------------------------------------------

    def safe_teleport(self, position: np.ndarray, orientation: np.ndarray) -> None:
        """Move the robot base then immediately zero all velocities and efforts."""
        self._robot.set_world_pose(position=position, orientation=orientation)
        n_dof = self._robot.num_dof
        self._robot.set_joint_velocities(np.zeros(n_dof))
        self._robot.set_joint_efforts(np.zeros(n_dof))

    # ------------------------------------------------------------------
    # Origin drift re-centering (§5.2)
    # ------------------------------------------------------------------

    def _read_base_position(self) -> np.ndarray:
        pos, _ = self._robot.get_world_pose()
        return np.array(pos)

    def _check_origin_drift(self) -> None:
        current = self._read_base_position()
        drift = np.linalg.norm(current - self._origin)
        if drift > self._cfg.safeguard.max_drift_m:
            # Silently re-snap — no exception raised
            self._robot.set_world_pose(
                position=self._origin,
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # identity quaternion
            )
