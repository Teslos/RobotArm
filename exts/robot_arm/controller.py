"""RMPFlow controller wrapper with stall monitor and gripper bypass (spec §2, §5.3-5.4)."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
from omni.isaac.core.articulations import Articulation

from .articulation import set_joint_position_targets
from .config import RobotArmCfg


def _import_rmpflow():
    try:
        from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
    except ImportError:
        from omni.isaac.motion_generation import ArticulationMotionPolicy, RmpFlow  # type: ignore[no-redef]
    return RmpFlow, ArticulationMotionPolicy


class RobotArmController:
    """
    Wraps RMPFlow with:
      - 60 Hz update locked to physics step (spec §2.3).
      - Stall monitor that cancels the goal if velocity < threshold
        for longer than timeout_s (spec §5.3).
      - Gripper treated as kinematic probe — no grasping physics (spec §5.4).
    """

    def __init__(
        self,
        robot: Articulation,
        robot_description_path: str,
        rmpflow_config_path: str,
        urdf_path: str,
        end_effector_frame_name: str,
        cfg: RobotArmCfg | None = None,
    ) -> None:
        self._robot = robot
        self._cfg = cfg or RobotArmCfg()

        RmpFlow, ArticulationMotionPolicy = _import_rmpflow()
        self._rmpflow = RmpFlow(
            robot_description_path=robot_description_path,
            rmpflow_config_path=rmpflow_config_path,
            urdf_path=urdf_path,
            end_effector_frame_name=end_effector_frame_name,
            maximum_substep_size=self._cfg.physics.dt,
        )
        self._motion_policy = ArticulationMotionPolicy(
            robot, self._rmpflow, self._cfg.physics.dt
        )

        self._goal_active: bool = False
        self._goal_start_time: float = 0.0
        self._target_position: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_end_effector_target(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        self._rmpflow.set_end_effector_target(
            target_position=position,
            target_orientation=orientation,
        )
        self._goal_active = True
        self._goal_start_time = time.monotonic()
        self._target_position = position

    def step(self) -> None:
        """Call once per 60 Hz physics step."""
        if not self._goal_active:
            return

        self._rmpflow.update_world()
        action = self._motion_policy.get_next_articulation_action()

        # Write targets via direct USD attribute writes (spec §4.3)
        set_joint_position_targets(self._robot, action.joint_positions.tolist())

        if self._is_stalled():
            self._cancel_goal()

    def reset(self) -> None:
        """Hard Home: zeros RMPFlow buffers alongside articulation state (spec §4.2)."""
        self._rmpflow.reset()
        self._goal_active = False
        self._target_position = None
        n_dof = self._robot.num_dof
        home = np.zeros(n_dof)
        set_joint_position_targets(self._robot, home.tolist())
        self._robot.set_joint_velocities(home)
        self._robot.set_joint_efforts(home)

    # ------------------------------------------------------------------
    # Stall detection (§5.3)
    # ------------------------------------------------------------------

    def _is_stalled(self) -> bool:
        elapsed = time.monotonic() - self._goal_start_time
        if elapsed < self._cfg.controller.rmpflow_timeout_s:
            return False
        vels = self._robot.get_joint_velocities()
        return bool(np.all(np.abs(vels) < self._cfg.controller.stall_velocity_threshold))

    def _cancel_goal(self) -> None:
        self._rmpflow.set_end_effector_target(
            target_position=self._robot.get_world_pose()[0],
        )
        self._goal_active = False
