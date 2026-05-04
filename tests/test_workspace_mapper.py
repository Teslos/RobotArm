"""Unit tests for WorkspaceMapper — no Isaac Sim required."""
from __future__ import annotations

import numpy as np
import pytest

from exts.robot_arm.config import GridSearchCfg
from exts.robot_arm.tasks.workspace_mapper import (
    WorkspaceMap,
    WorkspaceMapper,
    make_grid,
)

N_DOF = 6


# ── Stub IK solvers ──────────────────────────────────────────────────────────

class _FakeAction:
    def __init__(self, jpos):
        self.joint_positions = np.array(jpos, dtype=float)


class _AlwaysReachSolver:
    """IK always succeeds; returns a fixed non-zero solution."""
    num_dofs = N_DOF

    def compute_inverse_kinematics(self, target_position, target_orientation=None):
        return _FakeAction([0.1] * N_DOF), True


class _NeverReachSolver:
    """IK always fails."""
    num_dofs = N_DOF

    def compute_inverse_kinematics(self, target_position, target_orientation=None):
        return _FakeAction([0.0] * N_DOF), False


class _HighZSolver:
    """IK succeeds only for points with z > 0.15."""
    num_dofs = N_DOF

    def compute_inverse_kinematics(self, target_position, target_orientation=None):
        ok = target_position[2] > 0.15
        return _FakeAction([0.1] * N_DOF if ok else [0.0] * N_DOF), ok


class _WarmStartAwareSolver:
    """Records whether warm_start_position was passed."""
    num_dofs = N_DOF
    warm_calls: int = 0
    cold_calls: int = 0

    def compute_inverse_kinematics(
        self, target_position, target_orientation=None, warm_start_position=None
    ):
        if warm_start_position is not None:
            _WarmStartAwareSolver.warm_calls += 1
        else:
            _WarmStartAwareSolver.cold_calls += 1
        return _FakeAction([0.1] * N_DOF), True


class _NoWarmStartSolver:
    """Raises TypeError if warm_start_position is passed (simulates older API)."""
    num_dofs = N_DOF

    def compute_inverse_kinematics(self, target_position, target_orientation=None):
        return _FakeAction([0.1] * N_DOF), True


# ── make_grid ────────────────────────────────────────────────────────────────

def test_make_grid_total_points():
    cfg = GridSearchCfg(nx=3, ny=4, nz=2)
    pts = make_grid(cfg)
    assert pts.shape == (3 * 4 * 2, 3)


def test_make_grid_bounds():
    cfg = GridSearchCfg(
        x_range=(0.1, 0.3), y_range=(0.0, 0.2), z_range=(0.05, 0.25),
        nx=5, ny=5, nz=5,
    )
    pts = make_grid(cfg)
    assert pts[:, 0].min() == pytest.approx(0.1)
    assert pts[:, 0].max() == pytest.approx(0.3)
    assert pts[:, 1].min() == pytest.approx(0.0)
    assert pts[:, 1].max() == pytest.approx(0.2)
    assert pts[:, 2].min() == pytest.approx(0.05)
    assert pts[:, 2].max() == pytest.approx(0.25)


def test_make_grid_single_point():
    cfg = GridSearchCfg(nx=1, ny=1, nz=1, x_range=(0.2, 0.2), y_range=(0.0, 0.0), z_range=(0.15, 0.15))
    pts = make_grid(cfg)
    assert pts.shape == (1, 3)
    assert pts[0] == pytest.approx([0.2, 0.0, 0.15])


# ── WorkspaceMapper.run ──────────────────────────────────────────────────────

def test_all_reachable():
    cfg = GridSearchCfg(nx=2, ny=2, nz=2)
    mapper = WorkspaceMapper(_AlwaysReachSolver())
    wmap = mapper.run(cfg, verbose=False)
    assert wmap.reachable.all()
    assert wmap.reachable_fraction == pytest.approx(1.0)
    assert wmap.reachable_points.shape == (8, 3)
    assert not np.isnan(wmap.joint_positions).any()


def test_none_reachable():
    cfg = GridSearchCfg(nx=2, ny=2, nz=2)
    mapper = WorkspaceMapper(_NeverReachSolver())
    wmap = mapper.run(cfg, verbose=False)
    assert not wmap.reachable.any()
    assert wmap.reachable_fraction == pytest.approx(0.0)
    assert np.isnan(wmap.joint_positions).all()


def test_partial_reachable():
    # nz=4 → z values at 0.10, ~0.167, ~0.233, 0.30 — 3 above 0.15
    cfg = GridSearchCfg(z_range=(0.10, 0.30), nx=2, ny=2, nz=4)
    mapper = WorkspaceMapper(_HighZSolver())
    wmap = mapper.run(cfg, verbose=False)
    assert 0.0 < wmap.reachable_fraction < 1.0


def test_n_dof_from_attribute():
    cfg = GridSearchCfg(nx=2, ny=2, nz=2)
    mapper = WorkspaceMapper(_AlwaysReachSolver())
    wmap = mapper.run(cfg, verbose=False)
    assert wmap.n_dof == N_DOF


def test_n_dof_from_probe():
    """n_dof discovered by probing when solver has no attribute."""
    class _NoAttrSolver:
        def compute_inverse_kinematics(self, target_position, target_orientation=None):
            return _FakeAction([0.0] * N_DOF), True

    cfg = GridSearchCfg(nx=1, ny=1, nz=1)
    mapper = WorkspaceMapper(_NoAttrSolver())
    wmap = mapper.run(cfg, verbose=False)
    assert wmap.n_dof == N_DOF


# ── Joint limit filtering ────────────────────────────────────────────────────

def test_joint_limits_pass():
    lower = np.full(N_DOF, -1.0)
    upper = np.full(N_DOF,  1.0)
    mapper = WorkspaceMapper(_AlwaysReachSolver(), lower_limits=lower, upper_limits=upper)
    wmap = mapper.run(GridSearchCfg(nx=2, ny=2, nz=2), verbose=False)
    assert wmap.reachable.all()  # solution [0.1]*6 is within [-1, 1]


def test_joint_limits_reject():
    lower = np.full(N_DOF, -1.0)
    upper = np.full(N_DOF,  0.05)  # 0.1 > 0.05, so solution is rejected
    mapper = WorkspaceMapper(_AlwaysReachSolver(), lower_limits=lower, upper_limits=upper)
    wmap = mapper.run(GridSearchCfg(nx=2, ny=2, nz=2), verbose=False)
    assert not wmap.reachable.any()


# ── Warm-starting ────────────────────────────────────────────────────────────

def test_warm_start_propagates():
    _WarmStartAwareSolver.warm_calls = 0
    _WarmStartAwareSolver.cold_calls = 0
    cfg = GridSearchCfg(nx=2, ny=2, nz=2)
    mapper = WorkspaceMapper(_WarmStartAwareSolver())
    mapper.run(cfg, verbose=False)
    # First call is cold (no seed), subsequent calls use the last good solution
    assert _WarmStartAwareSolver.cold_calls == 1
    assert _WarmStartAwareSolver.warm_calls == 7  # 8 points total - 1 cold


def test_warm_start_fallback_on_typeerror():
    """Solver that doesn't accept warm_start_position should not raise."""
    cfg = GridSearchCfg(nx=2, ny=2, nz=2)
    mapper = WorkspaceMapper(_NoWarmStartSolver())
    wmap = mapper.run(cfg, verbose=False)
    assert wmap.reachable.all()


# ── WorkspaceMap helpers ─────────────────────────────────────────────────────

def test_best_approach():
    cfg = GridSearchCfg(nx=3, ny=3, nz=3)
    mapper = WorkspaceMapper(_AlwaysReachSolver())
    wmap = mapper.run(cfg, verbose=False)
    target = np.array([0.225, 0.0, 0.225])
    ee_pos, joints = wmap.best_approach(target)
    # Nearest point should be within half a grid cell
    assert np.linalg.norm(ee_pos - target) < 0.15
    assert joints.shape == (N_DOF,)


def test_best_approach_raises_when_empty():
    cfg = GridSearchCfg(nx=2, ny=2, nz=2)
    mapper = WorkspaceMapper(_NeverReachSolver())
    wmap = mapper.run(cfg, verbose=False)
    with pytest.raises(RuntimeError, match="No reachable points"):
        wmap.best_approach(np.array([0.2, 0.0, 0.2]))


def test_reachable_joint_positions_shape():
    cfg = GridSearchCfg(nx=3, ny=3, nz=3)
    mapper = WorkspaceMapper(_AlwaysReachSolver())
    wmap = mapper.run(cfg, verbose=False)
    jpos = wmap.reachable_joint_positions()
    assert jpos.shape == (27, N_DOF)


# ── Persistence ──────────────────────────────────────────────────────────────

def test_save_load_roundtrip(tmp_path):
    cfg = GridSearchCfg(nx=2, ny=2, nz=2)
    mapper = WorkspaceMapper(_AlwaysReachSolver())
    original = mapper.run(cfg, verbose=False)

    path = str(tmp_path / "workspace.npz")
    original.save(path)
    loaded = WorkspaceMap.load(path)

    np.testing.assert_array_equal(loaded.grid_points, original.grid_points)
    np.testing.assert_array_equal(loaded.reachable, original.reachable)
    np.testing.assert_array_almost_equal(loaded.joint_positions, original.joint_positions)
    assert loaded.n_dof == original.n_dof


def test_summary_string():
    cfg = GridSearchCfg(nx=2, ny=2, nz=2)
    mapper = WorkspaceMapper(_AlwaysReachSolver())
    wmap = mapper.run(cfg, verbose=False)
    s = wmap.summary()
    assert "8/8" in s
    assert "100.0%" in s
