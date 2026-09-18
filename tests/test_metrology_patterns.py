"""Unit tests for the scan-pattern geometry — pure arithmetic, no hardware."""
from __future__ import annotations

import numpy as np
import pytest

from exts.robot_arm.metrology.patterns import (
    ScanPath,
    attach_orientation,
    axis_index,
    line_scan,
    raster_grid,
)

ORIGIN = [190.0, 0.0, 188.0]
POSE = [190.0, 0.0, 188.0, -180.0, 0.0, 90.0]


# ── axis_index ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("axis,expected", [
    ("x", 0), ("Y", 1), (" z ", 2),
])
def test_axis_index(axis, expected):
    assert axis_index(axis) == expected


def test_axis_index_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown axis"):
        axis_index("w")


# ── ScanPath ─────────────────────────────────────────────────────────────────

def test_scanpath_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        ScanPath(points=np.zeros((4, 2)))


def test_scanpath_rejects_out_of_range_line_start():
    with pytest.raises(ValueError, match="outside the path"):
        ScanPath(points=np.zeros((2, 3)), line_starts=(0, 5))


def test_scanpath_line_of_and_is_line_start():
    path = ScanPath(points=np.zeros((6, 3)), line_starts=(0, 2, 4))
    assert [path.line_of(i) for i in range(6)] == [0, 0, 1, 1, 2, 2]
    assert [path.is_line_start(i) for i in range(6)] == [
        True, False, True, False, True, False
    ]


def test_scanpath_line_of_rejects_bad_index():
    path = ScanPath(points=np.zeros((2, 3)))
    with pytest.raises(IndexError):
        path.line_of(2)


def test_scanpath_extent():
    points = np.array([[0.0, 1.0, 2.0], [5.0, -1.0, 7.0]])
    np.testing.assert_allclose(
        ScanPath(points=points).extent_mm,
        [[0.0, 5.0], [-1.0, 1.0], [2.0, 7.0]],
    )


# ── line_scan ────────────────────────────────────────────────────────────────

def test_line_scan_starts_at_the_origin():
    path = line_scan(ORIGIN, "x", 4, 0.5)
    np.testing.assert_allclose(path.points[0], ORIGIN)


def test_line_scan_spacing_and_direction():
    path = line_scan(ORIGIN, "y", 5, -1.0)
    np.testing.assert_allclose(path.points[:, 1], [0.0, -1.0, -2.0, -3.0, -4.0])
    # The other axes stay put.
    assert np.all(path.points[:, 0] == ORIGIN[0])
    assert np.all(path.points[:, 2] == ORIGIN[2])


def test_line_scan_accepts_a_six_value_pose_and_drops_orientation():
    path = line_scan(POSE, "z", 3, 2.0)
    assert path.points.shape == (3, 3)
    np.testing.assert_allclose(path.points[-1], [190.0, 0.0, 192.0])


def test_line_scan_single_point():
    path = line_scan(ORIGIN, "x", 1, 10.0)
    assert len(path) == 1


def test_line_scan_rejects_zero_points():
    with pytest.raises(ValueError, match="n_points must be >= 1"):
        line_scan(ORIGIN, "x", 0, 1.0)


def test_line_scan_does_not_alias_the_origin():
    origin = list(ORIGIN)
    path = line_scan(origin, "x", 3, 1.0)
    path.points[0, 0] = -999.0
    assert origin == ORIGIN


# ── raster_grid ──────────────────────────────────────────────────────────────

def test_raster_shape_and_line_starts():
    path = raster_grid(ORIGIN, n_fast=4, n_slow=3, step_fast_mm=1.0, step_slow_mm=2.0)
    assert path.points.shape == (12, 3)
    assert path.line_starts == (0, 4, 8)


def test_raster_every_line_starts_at_the_same_fast_coordinate():
    """The bug class from RECTANGULAR _Pattern_Meca2 Wsafty.py.

    That script stepped 10 x 2.0 mm along X but rewound only 18 mm, so line n
    began 2n mm off.  Absolute waypoints cannot drift.
    """
    n_fast, n_slow, step = 10, 10, 2.0
    path = raster_grid(ORIGIN, n_fast, n_slow, step, 2.0)
    starts = path.points[list(path.line_starts), 0]
    np.testing.assert_allclose(starts, ORIGIN[0])


def test_raster_line_travel_matches_step_times_count():
    """Guards the hard-coded 100 mm rewind in Finall  Rectangular Mesurment.py."""
    n_fast, step = 7, 1.5
    path = raster_grid(ORIGIN, n_fast, 2, step, 1.0)
    first_line = path.points[:n_fast, 0]
    assert first_line[-1] - first_line[0] == pytest.approx((n_fast - 1) * step)


def test_raster_slow_axis_advances_once_per_line():
    path = raster_grid(ORIGIN, n_fast=3, n_slow=4, step_fast_mm=1.0, step_slow_mm=2.5)
    y_per_line = [path.points[start, 1] for start in path.line_starts]
    np.testing.assert_allclose(y_per_line, [0.0, 2.5, 5.0, 7.5])
    # Y is constant within a line.
    for start in path.line_starts:
        np.testing.assert_allclose(path.points[start:start + 3, 1], path.points[start, 1])


def test_raster_serpentine_reverses_odd_lines_only():
    path = raster_grid(
        ORIGIN, n_fast=3, n_slow=2, step_fast_mm=1.0, step_slow_mm=1.0,
        serpentine=True,
    )
    np.testing.assert_allclose(path.points[0:3, 0], [190.0, 191.0, 192.0])
    np.testing.assert_allclose(path.points[3:6, 0], [192.0, 191.0, 190.0])


def test_raster_serpentine_covers_the_same_point_set_as_the_plain_raster():
    plain = raster_grid(ORIGIN, 4, 3, 1.0, 2.0)
    snake = raster_grid(ORIGIN, 4, 3, 1.0, 2.0, serpentine=True)
    key = lambda points: sorted(map(tuple, np.round(points, 6)))  # noqa: E731
    assert key(plain.points) == key(snake.points)


def test_raster_custom_axes():
    path = raster_grid(
        ORIGIN, n_fast=2, n_slow=2, step_fast_mm=1.0, step_slow_mm=3.0,
        fast_axis="y", slow_axis="z",
    )
    np.testing.assert_allclose(
        path.points,
        [[190.0, 0.0, 188.0],
         [190.0, 1.0, 188.0],
         [190.0, 0.0, 191.0],
         [190.0, 1.0, 191.0]],
    )


def test_raster_rejects_identical_axes():
    with pytest.raises(ValueError, match="must differ"):
        raster_grid(ORIGIN, 2, 2, 1.0, 1.0, fast_axis="x", slow_axis="x")


@pytest.mark.parametrize("n_fast,n_slow", [(0, 2), (2, 0), (-1, 3)])
def test_raster_rejects_empty_grids(n_fast, n_slow):
    with pytest.raises(ValueError, match=">= 1"):
        raster_grid(ORIGIN, n_fast, n_slow, 1.0, 1.0)


def test_raster_reproduces_the_y_direction_scanner_geometry():
    """55 points at 2 mm along Y, 45 lines at 2 mm along X."""
    path = raster_grid(
        ORIGIN, n_fast=55, n_slow=45, step_fast_mm=2.0, step_slow_mm=2.0,
        fast_axis="y", slow_axis="x",
    )
    assert len(path) == 55 * 45
    np.testing.assert_allclose(path.extent_mm[1], [0.0, 108.0])   # Y: 54 * 2
    np.testing.assert_allclose(path.extent_mm[0], [190.0, 278.0])  # X: 44 * 2


# ── attach_orientation ───────────────────────────────────────────────────────

def test_attach_orientation_broadcasts():
    path = line_scan(ORIGIN, "x", 3, 1.0)
    poses = attach_orientation(path, [-180.0, 0.0, 90.0])
    assert poses.shape == (3, 6)
    np.testing.assert_allclose(poses[:, 3:], np.tile([-180.0, 0.0, 90.0], (3, 1)))
    np.testing.assert_allclose(poses[:, :3], path.points)


def test_attach_orientation_rejects_wrong_length():
    path = line_scan(ORIGIN, "x", 2, 1.0)
    with pytest.raises(ValueError, match="3 angles"):
        attach_orientation(path, [0.0, 0.0])
