"""
Collect INT_Monitor per-point FFT text files into one table per version.

Directory layout produced by the instrument::

    <root>/<coordinate folder>/<something>_FFT/<something>_FFT_<version>.txt

where *coordinate folder* encodes the measurement position, e.g. ``Y12.5`` or
``x-3.0 y7.5``.  Each text file has a header line and a data line whose first
few whitespace-separated fields are metadata, followed by flat
``freq mag phase`` triplets.

Fixes relative to ``txt to excel.py``
-------------------------------------
* **Negative coordinates parse.**  The old pattern was ``[yY]([\\d.]+)``, which
  cannot match the minus sign, so every folder on the negative side of the
  origin sorted as ``0.0`` and the rows came out in the wrong order — silently,
  because a failed match returned ``0.0`` rather than raising.
* **Malformed coordinates no longer crash the sort.**  ``[\\d.]+`` happily
  matches ``12.5.3``, and the ``float()`` that followed raised ``ValueError``
  from inside ``list.sort``, aborting the whole export.  Unparseable values are
  now reported and the file is skipped.
* **``get_z_value`` actually read X.**  Its docstring promised Z, its regex
  matched ``[xX]``, and nothing ever called it.  Replaced by
  :func:`parse_coordinates`, which returns every axis it finds.
* **Coordinates survive into the output.**  The old label was uppercased into
  one opaque string; numeric ``x_mm`` / ``y_mm`` / ``z_mm`` columns are emitted
  alongside it so the sheet can be sorted and plotted.
* **Deterministic ordering.**  ``glob`` order is arbitrary; rows are sorted by
  ``(y, x, z, folder name)``.
* **No pandas dependency.**  The workbook is written with ``openpyxl``
  directly, and CSV output needs nothing beyond the standard library.
"""
from __future__ import annotations

import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "COORDINATE_PATTERN",
    "DEFAULT_VERSIONS",
    "FftRow",
    "parse_coordinates",
    "coordinate_sort_key",
    "find_coordinate_folder",
    "parse_triplets",
    "read_fft_file",
    "collect_fft_files",
    "build_rows",
    "rows_to_table",
    "write_csv",
    "write_xlsx",
]

COORDINATE_PATTERN = re.compile(r"(?<![A-Za-z0-9])([xyzXYZ])\s*(-?\d+(?:\.\d+)?)")
"""Matches ``x12.5``, ``Y-3``, ``z 0.25`` but not the ``x`` inside ``max12``."""

DEFAULT_VERSIONS: Tuple[str, ...] = ("V0", "V1", "V2")
DEFAULT_GLOB = os.path.join("*", "*_FFT", "*_FFT_{version}.txt")
DEFAULT_ENCODING = "cp1252"
DEFAULT_DATA_LINE = 1        # zero-based: the line after the header
DEFAULT_SKIP_FIELDS = 3      # leading metadata fields on the data line


class FftParseError(ValueError):
    """A file could not be turned into a row."""


def parse_coordinates(name: str) -> Dict[str, float]:
    """Extract every ``<axis><value>`` pair from a folder or file name.

    Later occurrences win, so ``"run_x1 x2"`` yields ``{"x": 2.0}``.  Returns an
    empty dict when the name carries no coordinates — callers decide whether
    that is fatal, instead of silently getting ``0.0`` as the old code did.
    """
    coordinates: Dict[str, float] = {}
    for axis, value in COORDINATE_PATTERN.findall(name):
        coordinates[axis.lower()] = float(value)
    return coordinates


def coordinate_sort_key(
    coordinates: Dict[str, float],
    label: str = "",
    order: Sequence[str] = ("y", "x", "z"),
) -> Tuple:
    """Sort key over the requested axes, with the label breaking ties.

    Missing axes sort *after* present ones rather than pretending to be ``0.0``.
    """
    key: List[object] = []
    for axis in order:
        if axis in coordinates:
            key.extend((0, coordinates[axis]))
        else:
            key.extend((1, 0.0))
    key.append(label)
    return tuple(key)


def find_coordinate_folder(path: str) -> str:
    """Innermost path component that carries coordinates; ``""`` if none."""
    for part in reversed(os.path.normpath(path).split(os.sep)):
        if parse_coordinates(part):
            return part
    return ""


def parse_triplets(
    data_line: str,
    skip_fields: int = DEFAULT_SKIP_FIELDS,
) -> List[Tuple[float, float, float]]:
    """Split a data line into ``(freq, magnitude, phase)`` triplets.

    Non-numeric tokens after ``skip_fields`` are dropped, matching the original
    behaviour.  A trailing partial triplet is discarded.
    """
    if skip_fields < 0:
        raise ValueError(f"skip_fields must be >= 0, got {skip_fields}")

    values: List[float] = []
    for token in data_line.split()[skip_fields:]:
        try:
            values.append(float(token))
        except ValueError:
            continue

    usable = len(values) - (len(values) % 3)
    return [
        (values[i], values[i + 1], values[i + 2])
        for i in range(0, usable, 3)
    ]


def read_fft_file(
    path: str,
    *,
    encoding: str = DEFAULT_ENCODING,
    data_line: int = DEFAULT_DATA_LINE,
    skip_fields: int = DEFAULT_SKIP_FIELDS,
) -> List[Tuple[float, float, float]]:
    """Read one instrument text file and return its FFT triplets."""
    with open(path, "r", encoding=encoding, errors="ignore") as handle:
        lines = handle.read().splitlines()

    if len(lines) <= data_line:
        raise FftParseError(
            f"{path}: expected at least {data_line + 1} lines, got {len(lines)}"
        )

    triplets = parse_triplets(lines[data_line], skip_fields=skip_fields)
    if not triplets:
        raise FftParseError(f"{path}: no complete freq/mag/phase triplets found")
    return triplets


@dataclass
class FftRow:
    """One measurement point's spectrum, ready to become a sheet row."""

    label: str
    coordinates: Dict[str, float]
    triplets: List[Tuple[float, float, float]]
    source: str = ""

    def as_dict(self) -> Dict[str, object]:
        """Flat column mapping: metadata first, then ``Freq_n/Mag_n/Phase_n``."""
        row: Dict[str, object] = {"Coordinate": self.label}
        for axis in ("x", "y", "z"):
            row[f"{axis}_mm"] = self.coordinates.get(axis)
        for bin_index, (freq, mag, phase) in enumerate(self.triplets):
            row[f"Freq_{bin_index}"] = freq
            row[f"Mag_{bin_index}"] = mag
            row[f"Phase_{bin_index}"] = phase
        return row


def collect_fft_files(
    root: str,
    version: str,
    pattern: str = DEFAULT_GLOB,
) -> List[str]:
    """Glob the per-version FFT files under ``root`` in a stable order."""
    matches = glob.glob(os.path.join(root, pattern.format(version=version)))
    return sorted(matches)


def build_rows(
    paths: Iterable[str],
    *,
    encoding: str = DEFAULT_ENCODING,
    data_line: int = DEFAULT_DATA_LINE,
    skip_fields: int = DEFAULT_SKIP_FIELDS,
    sort_order: Sequence[str] = ("y", "x", "z"),
    on_error: Optional[Callable[[str, Exception], None]] = None,
) -> List[FftRow]:
    """Turn FFT files into sorted :class:`FftRow` objects.

    Files that cannot be parsed are skipped; ``on_error(path, exception)`` is
    called for each so the CLI can report them rather than swallow them.
    """
    rows: List[FftRow] = []
    for path in paths:
        try:
            label = find_coordinate_folder(path)
            if not label:
                raise FftParseError(
                    f"{path}: no coordinate folder (expected a path component "
                    "like 'Y12.5')"
                )
            coordinates = parse_coordinates(label)
            triplets = read_fft_file(
                path,
                encoding=encoding,
                data_line=data_line,
                skip_fields=skip_fields,
            )
        except (OSError, ValueError) as exc:
            if on_error is not None:
                on_error(path, exc)
            continue
        rows.append(
            FftRow(
                label=label,
                coordinates=coordinates,
                triplets=triplets,
                source=path,
            )
        )

    rows.sort(key=lambda row: coordinate_sort_key(row.coordinates, row.label, sort_order))
    return rows


def rows_to_table(rows: Sequence[FftRow]) -> Tuple[List[str], List[List[object]]]:
    """Build ``(header, table)`` with a union of all columns, order preserved.

    Points with different bin counts are padded with ``None`` rather than
    truncated, so a short spectrum never silently drops another point's data.
    """
    header: List[str] = []
    seen = set()
    dicts = [row.as_dict() for row in rows]
    for row_dict in dicts:
        for column in row_dict:
            if column not in seen:
                seen.add(column)
                header.append(column)

    table = [[row_dict.get(column) for column in header] for row_dict in dicts]
    return header, table


def write_csv(path: str, rows: Sequence[FftRow]) -> int:
    """Write one CSV file; returns the number of data rows written."""
    header, table = rows_to_table(rows)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header or ["Info"])
        if table:
            writer.writerows(table)
        else:
            writer.writerow(["No data found"])
    return len(table)


def write_xlsx(path: str, sheets: Dict[str, Sequence[FftRow]]) -> int:
    """Write one sheet per key; returns the total number of data rows.

    An empty mapping (or one whose sheets are all empty) still produces a
    readable workbook with an ``Info`` sheet, as the original did.
    """
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "openpyxl is required for .xlsx output. Install it with:\n"
            "    micromamba run -n RobotArm pip install openpyxl\n"
            "or export with --format csv instead."
        ) from exc

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    workbook = Workbook()
    workbook.remove(workbook.active)

    total = 0
    for name, rows in sheets.items():
        if not rows:
            continue
        header, table = rows_to_table(rows)
        sheet = workbook.create_sheet(title=name[:31])
        sheet.append(header)
        for row in table:
            sheet.append(row)
        total += len(table)

    if not workbook.sheetnames:
        sheet = workbook.create_sheet(title="Info")
        sheet.append(["Info"])
        sheet.append([f"No data found for: {', '.join(sheets) or '(no versions)'}"])

    workbook.save(path)
    return total
