"""
Scan-pattern geometry for the INT_Monitor measurement rigs.

Everything here is pure arithmetic on millimetres — no sockets, no robot, no
Isaac Sim — so it is cheap to test and safe to eyeball before a run.

Why absolute waypoints
----------------------
The original scripts drove the arm with chained *relative* moves
(``MoveLinRelWrf``) and returned to the start of each line with a hard-coded
distance.  Two of them got that distance wrong:

* ``RECTANGULAR _Pattern_Meca2 Wsafty.py`` stepped ``10 x 2.0 mm = 20 mm``
  along X but rewound only ``18 mm``, so every line started 2 mm further out —
  a 20 mm skew by the end of a 10-line scan.
* ``Finall  Rectangular Mesurment.py`` rewound a literal ``100 mm`` that only
  happens to match ``N_points_per_line * d_x`` for the values checked in; the
  stale comment still claimed ``20 mm``.

Generating absolute positions from a single recorded origin removes the class
of bug entirely: no move depends on the accumulated result of previous moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "AXIS_INDEX",
    "ScanPath",
    "axis_index",
    "line_scan",
    "raster_grid",
    "attach_orientation",
]

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def axis_index(axis: str) -> int:
    """Map an axis letter to its index in a Cartesian pose."""
    try:
        return AXIS_INDEX[axis.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unknown axis {axis!r}; expected one of {sorted(AXIS_INDEX)}"
        ) from None


@dataclass(frozen=True)
class ScanPath:
    """An ordered set of absolute XYZ waypoints plus its line structure.

    Attributes
    ----------
    points:
        ``(N, 3)`` array of absolute positions in millimetres, base frame.
    line_starts:
        Indices into :attr:`points` at which a new scan line begins.  Callers
        use these to insert a safety hop between lines instead of hopping in
        place, which is what the original scripts did (they moved up by
        ``safety_z`` and straight back down without travelling in between, so
        the hop protected nothing — and in ``Finall  Rectangular Mesurment.py``
        ``safety_z`` was ``0`` anyway, making it a no-op).
    """

    points: np.ndarray
    line_starts: Tuple[int, ...] = field(default=(0,))

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"ScanPath.points must have shape (N, 3), got {points.shape}"
            )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "line_starts", tuple(int(i) for i in self.line_starts))
        for index in self.line_starts:
            if not 0 <= index < len(points):
                raise ValueError(
                    f"line_starts index {index} is outside the path "
                    f"(0..{len(points) - 1})"
                )

    def __len__(self) -> int:
        return len(self.points)

    def is_line_start(self, index: int) -> bool:
        """True when waypoint ``index`` opens a new scan line."""
        return index in set(self.line_starts)

    def line_of(self, index: int) -> int:
        """Zero-based line number containing waypoint ``index``."""
        if not 0 <= index < len(self.points):
            raise IndexError(index)
        return sum(1 for start in self.line_starts if start <= index) - 1

    @property
    def extent_mm(self) -> np.ndarray:
        """``(3, 2)`` array of per-axis ``[min, max]``."""
        return np.column_stack([self.points.min(axis=0), self.points.max(axis=0)])


def _as_origin(origin: Sequence[float]) -> np.ndarray:
    origin = np.asarray(origin, dtype=float)
    if origin.shape not in ((3,), (6,)):
        raise ValueError(
            f"origin must be a 3- or 6-value pose, got shape {origin.shape}"
        )
    return origin[:3].copy()


def line_scan(
    origin: Sequence[float],
    axis: str,
    n_points: int,
    step_mm: float,
) -> ScanPath:
    """``n_points`` collinear waypoints starting at ``origin``.

    The first waypoint *is* ``origin`` — the original line scanners measured
    ``start + i * step`` for ``i`` in ``0..n-1`` too, but the rectangular
    scanners moved first and measured second, silently dropping the origin from
    the data set.  One convention for both is less surprising.

    A negative ``step_mm`` scans backwards.  Note that
    ``Width X And Z Measurment .py`` and ``Y and Z Measurment.py`` combined a
    negative ``STEP`` with ``target = start - i * STEP``, so the arm travelled
    in the *opposite* direction to the one the constant suggested; here the
    sign of ``step_mm`` is the direction of travel.
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    base = _as_origin(origin)
    index = axis_index(axis)

    points = np.tile(base, (n_points, 1))
    points[:, index] = base[index] + np.arange(n_points) * float(step_mm)
    return ScanPath(points=points, line_starts=(0,))


def raster_grid(
    origin: Sequence[float],
    n_fast: int,
    n_slow: int,
    step_fast_mm: float,
    step_slow_mm: float,
    *,
    fast_axis: str = "x",
    slow_axis: str = "y",
    serpentine: bool = False,
) -> ScanPath:
    """A 2-D raster of ``n_slow`` lines of ``n_fast`` waypoints each.

    Parameters
    ----------
    origin:
        Pose of the first waypoint (3- or 6-value); only XYZ is used.
    n_fast, n_slow:
        Waypoints per line, and number of lines.
    step_fast_mm, step_slow_mm:
        Spacing within a line and between lines.  Signs set the direction.
    fast_axis, slow_axis:
        Which Cartesian axes the two loops run along.  Must differ.
    serpentine:
        When True, alternate lines are traversed in reverse (boustrophedon),
        which halves the travel and removes the long rewind move entirely.
        When False every line starts at the same fast-axis coordinate, matching
        the original "return to line start" behaviour.
    """
    if n_fast < 1 or n_slow < 1:
        raise ValueError(
            f"n_fast and n_slow must both be >= 1, got {n_fast} and {n_slow}"
        )
    fast = axis_index(fast_axis)
    slow = axis_index(slow_axis)
    if fast == slow:
        raise ValueError(
            f"fast_axis and slow_axis must differ, both are {fast_axis!r}"
        )

    base = _as_origin(origin)
    fast_offsets = np.arange(n_fast) * float(step_fast_mm)

    rows = []
    line_starts = []
    for line in range(n_slow):
        line_starts.append(len(rows))
        offsets = fast_offsets if (not serpentine or line % 2 == 0) else fast_offsets[::-1]
        for offset in offsets:
            point = base.copy()
            point[fast] = base[fast] + offset
            point[slow] = base[slow] + line * float(step_slow_mm)
            rows.append(point)

    return ScanPath(points=np.array(rows, dtype=float), line_starts=tuple(line_starts))


def attach_orientation(
    path: ScanPath,
    orientation_deg: Sequence[float],
) -> np.ndarray:
    """Combine a path with a fixed EE orientation into ``(N, 6)`` poses.

    ``orientation_deg`` is ``[alpha, beta, gamma]``, normally read back from the
    robot after the approach move so it always matches the real configuration.
    """
    orientation = np.asarray(orientation_deg, dtype=float)
    if orientation.shape != (3,):
        raise ValueError(
            f"orientation_deg must hold 3 angles, got shape {orientation.shape}"
        )
    return np.column_stack([path.points, np.tile(orientation, (len(path), 1))])
