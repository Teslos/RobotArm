"""IK-based grid search for reachability analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..config import GridSearchCfg


@dataclass
class WorkspaceMap:
    """Results of an IK-based grid search."""
    grid_points: np.ndarray      # (N, 3) Cartesian XYZ, metres
    reachable: np.ndarray        # (N,) bool
    joint_positions: np.ndarray  # (N, n_dof), NaN where unreachable
    n_dof: int

    @property
    def reachable_points(self) -> np.ndarray:
        """(M, 3) positions that passed IK + joint-limit checks."""
        return self.grid_points[self.reachable]

    @property
    def reachable_fraction(self) -> float:
        return float(self.reachable.sum()) / max(len(self.reachable), 1)

    def reachable_joint_positions(self) -> np.ndarray:
        """(M, n_dof) IK solutions for reachable points only."""
        return self.joint_positions[self.reachable]

    def best_approach(self, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ee_position, joint_angles) for the reachable point nearest target."""
        pts = self.reachable_points
        if len(pts) == 0:
            raise RuntimeError("No reachable points in WorkspaceMap.")
        idx = int(np.argmin(np.linalg.norm(pts - target, axis=1)))
        # Map back to full-grid index
        full_idx = int(np.where(self.reachable)[0][idx])
        return self.grid_points[full_idx], self.joint_positions[full_idx]

    def summary(self) -> str:
        total = len(self.grid_points)
        n = int(self.reachable.sum())
        return f"WorkspaceMap: {n}/{total} reachable ({self.reachable_fraction * 100:.1f}%)"

    def save(self, path: str) -> None:
        np.savez(
            path,
            grid_points=self.grid_points,
            reachable=self.reachable,
            joint_positions=self.joint_positions,
            n_dof=np.array(self.n_dof),
        )

    @classmethod
    def load(cls, path: str) -> "WorkspaceMap":
        d = np.load(path)
        return cls(
            grid_points=d["grid_points"],
            reachable=d["reachable"],
            joint_positions=d["joint_positions"],
            n_dof=int(d["n_dof"]),
        )


class WorkspaceMapper:
    """
    Evaluates IK on a 3-D Cartesian grid and classifies each point as
    reachable or unreachable.

    The IK solver is dependency-injected, so this class is fully testable
    without Isaac Sim.  In production, pass a LulaKinematicsSolver instance.

    Warm-starting: after each successful IK solve, the joint solution is used
    as the initial guess for the next grid point.  Neighbouring points tend to
    have similar configurations, so this cuts solver iterations significantly.
    """

    def __init__(
        self,
        ik_solver: Any,
        ee_frame: str = "end_effector",
        lower_limits: Optional[np.ndarray] = None,
        upper_limits: Optional[np.ndarray] = None,
    ) -> None:
        self._solver = ik_solver
        self._ee_frame = ee_frame
        self._lower = lower_limits
        self._upper = upper_limits

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        cfg: GridSearchCfg,
        seed: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> WorkspaceMap:
        """Evaluate IK for every grid point and return a WorkspaceMap."""
        points = make_grid(cfg)
        n = len(points)
        n_dof = self._get_n_dof()

        reachable = np.zeros(n, dtype=bool)
        joint_pos = np.full((n, n_dof), np.nan)
        warm = seed.copy() if seed is not None else None

        log_every = max(1, n // 20)
        for i, pt in enumerate(points):
            if verbose and i % log_every == 0:
                print(f"[workspace] {i}/{n} points evaluated …")
            sol = self._try_orientations(pt, cfg.orientations, warm)
            if sol is not None:
                reachable[i] = True
                joint_pos[i] = sol
                warm = sol  # warm-start the next grid point from last success

        result = WorkspaceMap(
            grid_points=points,
            reachable=reachable,
            joint_positions=joint_pos,
            n_dof=n_dof,
        )
        if verbose:
            print(f"[workspace] {result.summary()}")
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_orientations(
        self,
        position: np.ndarray,
        orientations: Sequence,
        warm: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        for ori in orientations:
            ori_arr = np.array(ori, dtype=float) if ori is not None else None
            try:
                action, ok = self._call_ik(position, ori_arr, warm)
            except Exception:
                continue
            # Isaac Sim 4.5 returns np.ndarray; older API returns ArticulationAction
            jpos = action if isinstance(action, np.ndarray) else np.asarray(action.joint_positions, dtype=float)
            if ok and self._limits_ok(jpos):
                return jpos
        return None

    def _call_ik(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray],
        warm: Optional[np.ndarray],
    ):
        """Call solver with optional frame_name (Isaac Sim 4.5+) and warm-start."""
        kwargs: dict = {"target_position": position}
        if orientation is not None:
            kwargs["target_orientation"] = orientation
        if warm is not None:
            try:
                return self._solver.compute_inverse_kinematics(
                    self._ee_frame, **kwargs, warm_start=warm
                )
            except TypeError:
                pass  # solver doesn't accept frame_name or warm_start
        try:
            return self._solver.compute_inverse_kinematics(self._ee_frame, **kwargs)
        except TypeError:
            # older API without frame_name
            return self._solver.compute_inverse_kinematics(**kwargs)

    def _limits_ok(self, jpos: np.ndarray) -> bool:
        if self._lower is not None and np.any(jpos < self._lower):
            return False
        if self._upper is not None and np.any(jpos > self._upper):
            return False
        return True

    def _get_n_dof(self) -> int:
        for attr in ("num_dofs", "num_dof", "num_cspace_coords"):
            if hasattr(self._solver, attr):
                return int(getattr(self._solver, attr))
        # get_cspace_position_limits returns (lower, upper) arrays of length n_dof
        if hasattr(self._solver, "get_cspace_position_limits"):
            try:
                lower, _ = self._solver.get_cspace_position_limits()
                return len(lower)
            except Exception:
                pass
        # probe via a dummy IK call — try new API (frame_name) then old API
        for args in [(self._ee_frame,), ()]:
            try:
                action, _ = self._solver.compute_inverse_kinematics(
                    *args, target_position=np.array([0.2, 0.0, 0.2])
                )
                jpos = action if isinstance(action, np.ndarray) else action.joint_positions
                return len(jpos)
            except TypeError:
                continue
            except Exception as exc:
                print(f"[workspace] n_dof probe failed (args={args}): {exc}")
                continue
        raise RuntimeError("Cannot determine n_dof from IK solver.")


# ------------------------------------------------------------------
# Grid helper (module-level for direct test access)
# ------------------------------------------------------------------

def make_grid(cfg: GridSearchCfg) -> np.ndarray:
    """Return (N, 3) array of Cartesian grid points."""
    xs = np.linspace(cfg.x_range[0], cfg.x_range[1], cfg.nx)
    ys = np.linspace(cfg.y_range[0], cfg.y_range[1], cfg.ny)
    zs = np.linspace(cfg.z_range[0], cfg.z_range[1], cfg.nz)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
