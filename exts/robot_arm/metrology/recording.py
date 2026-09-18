"""
Measurement-point logging.

The original scripts wrote one tiny text file per point, named
``data_{n:02d}.txt`` / ``point{n:02d}.txt``.  Two problems followed from that:

* With ``N_lines * N_points_per_line = 10 000`` points, ``:02d`` stops padding
  after point 99, so a directory listing sorts ``data_100`` before ``data_99``.
  :func:`point_filename` pads to the width of the real total instead.
* The per-point files each held a single line that was already on stdout, and
  nothing recorded *when* a point was taken or how long the instrument needed.

The primary output here is therefore one CSV per scan, appended as the scan
runs so a crash keeps everything measured so far.  Per-point text files remain
available for instrument-side tooling that expects them.
"""
from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field
from typing import IO, Iterable, List, Optional, Sequence

__all__ = [
    "PointRecord",
    "PointLog",
    "point_filename",
    "write_point_file",
]

CSV_FIELDS = (
    "index",
    "line",
    "label",
    "x_mm",
    "y_mm",
    "z_mm",
    "alpha_deg",
    "beta_deg",
    "gamma_deg",
    "acquire_s",
    "timestamp_s",
)


def point_filename(index: int, total: int, prefix: str = "point", suffix: str = ".txt") -> str:
    """``point0042.txt`` — zero-padded to the width the full scan needs."""
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    width = len(str(total - 1)) if total > 1 else 1
    return f"{prefix}{index:0{width}d}{suffix}"


@dataclass
class PointRecord:
    """One measured point: where the arm actually was, and what it cost."""

    index: int
    line: int
    label: str
    pose: Sequence[float] = field(default=(0.0,) * 6)
    acquire_s: float = 0.0
    timestamp_s: float = 0.0

    def __post_init__(self) -> None:
        pose = [float(v) for v in self.pose]
        if len(pose) != 6:
            raise ValueError(
                f"PointRecord.pose needs 6 values [x, y, z, alpha, beta, "
                f"gamma], got {pose!r}"
            )
        self.pose = pose

    def as_row(self) -> dict:
        """Flatten to the CSV column layout."""
        x, y, z, alpha, beta, gamma = self.pose
        return {
            "index": self.index,
            "line": self.line,
            "label": self.label,
            "x_mm": f"{x:.4f}",
            "y_mm": f"{y:.4f}",
            "z_mm": f"{z:.4f}",
            "alpha_deg": f"{alpha:.4f}",
            "beta_deg": f"{beta:.4f}",
            "gamma_deg": f"{gamma:.4f}",
            "acquire_s": f"{self.acquire_s:.3f}",
            "timestamp_s": f"{self.timestamp_s:.3f}",
        }

    def as_dict(self) -> dict:  # pragma: no cover - convenience
        return asdict(self)


class PointLog:
    """Append-as-you-go CSV writer for :class:`PointRecord`.

    Use as a context manager so the file is flushed and closed even when the
    scan aborts::

        with PointLog("results/scan.csv") as log:
            log.append(record)
    """

    def __init__(self, path: str, *, overwrite: bool = True) -> None:
        self.path = path
        self._overwrite = overwrite
        self._handle: Optional[IO[str]] = None
        self._writer: Optional[csv.DictWriter] = None
        self.records: List[PointRecord] = []

    def open(self) -> "PointLog":
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        exists = os.path.exists(self.path) and not self._overwrite
        mode = "a" if exists else "w"
        self._handle = open(self.path, mode, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(CSV_FIELDS))
        if not exists:
            self._writer.writeheader()
        self._handle.flush()
        return self

    def append(self, record: PointRecord) -> None:
        """Write one record and flush, so a crash keeps the measured points."""
        if self._writer is None or self._handle is None:
            raise RuntimeError("PointLog.open() must be called before append()")
        self._writer.writerow(record.as_row())
        self._handle.flush()
        self.records.append(record)

    def extend(self, records: Iterable[PointRecord]) -> None:
        for record in records:
            self.append(record)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None

    def __enter__(self) -> "PointLog":
        return self.open()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.records)


def write_point_file(
    directory: str,
    index: int,
    total: int,
    text: str,
    *,
    prefix: str = "point",
) -> str:
    """Write the legacy one-line-per-point text file; returns the path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, point_filename(index, total, prefix=prefix))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.rstrip("\n") + "\n")
    return path
