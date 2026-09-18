#!/usr/bin/env python3
"""
Collect INT_Monitor per-point FFT text files into one workbook.

Replaces ``C:\\EmpaDaten\\INT_Monitor\\py\\txt to excel.py``, which had the
measurement root, the version list, the glob depth and the output name all
hard-coded, sorted rows with a regex that could not see a minus sign, and
crashed the whole export on a single malformed folder name.

Usage::

    micromamba run -n RobotArm python scripts/export_fft_to_excel.py \
        --root "C:/EmpaDaten/Data_Folder/mess38"

    micromamba run -n RobotArm python scripts/export_fft_to_excel.py \
        --root C:/EmpaDaten/Data_Folder/mess38 --versions V0 V1 V2 V3 \
        --format csv --output results/mess38

Expected layout under ``--root``::

    <root>/<coordinate folder>/<name>_FFT/<name>_FFT_<version>.txt

The coordinate folder is any path component carrying ``x``/``y``/``z`` values,
e.g. ``Y12.5`` or ``x-3.0 y7.5``.  Its numeric values become ``x_mm``/``y_mm``/
``z_mm`` columns so the sheet can be sorted and plotted directly.

``--format xlsx`` needs ``openpyxl``; ``--format csv`` needs nothing beyond the
standard library and writes one file per version.
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# See scripts/scan_raster.py: imported top-level so Isaac Sim is not required.
sys.path.insert(0, os.path.join(REPO_ROOT, "exts", "robot_arm"))

from metrology.fft_export import (  # noqa: E402
    DEFAULT_DATA_LINE,
    DEFAULT_ENCODING,
    DEFAULT_GLOB,
    DEFAULT_SKIP_FIELDS,
    DEFAULT_VERSIONS,
    build_rows,
    collect_fft_files,
    write_csv,
    write_xlsx,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge INT_Monitor FFT text files into a workbook",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True,
                        help="Measurement root folder to scan")
    parser.add_argument("--versions", nargs="+", default=list(DEFAULT_VERSIONS),
                        help="Version suffixes to export, one sheet/file each")
    parser.add_argument("--output", default=None,
                        help="Output .xlsx path, or the base path for --format "
                             "csv; defaults to <root>/Master_FFT_All_Versions")
    parser.add_argument("--format", choices=("xlsx", "csv"), default="xlsx",
                        help="xlsx writes one workbook, csv writes one file "
                             "per version")
    parser.add_argument("--glob", default=DEFAULT_GLOB,
                        help="Glob relative to --root; '{version}' is substituted")
    parser.add_argument("--encoding", default=DEFAULT_ENCODING,
                        help="Text encoding of the instrument files")
    parser.add_argument("--data-line", type=int, default=DEFAULT_DATA_LINE,
                        metavar="N",
                        help="Zero-based index of the line holding the spectrum")
    parser.add_argument("--skip-fields", type=int, default=DEFAULT_SKIP_FIELDS,
                        metavar="N",
                        help="Leading metadata fields to drop from the data line")
    parser.add_argument("--sort-by", nargs="+", default=["y", "x", "z"],
                        choices=("x", "y", "z"),
                        help="Coordinate axes used to order the rows")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any file had to be skipped")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not os.path.isdir(args.root):
        print(f"[ERROR] --root is not a directory: {args.root}", file=sys.stderr)
        return 2

    default_base = os.path.join(args.root, "Master_FFT_All_Versions")
    output = args.output or default_base

    skipped: list[str] = []

    def report(path: str, exc: Exception) -> None:
        skipped.append(path)
        print(f"[warn] Skipped {path}: {exc}", file=sys.stderr)

    sheets = {}
    for version in args.versions:
        paths = collect_fft_files(args.root, version, args.glob)
        rows = build_rows(
            paths,
            encoding=args.encoding,
            data_line=args.data_line,
            skip_fields=args.skip_fields,
            sort_order=args.sort_by,
            on_error=report,
        )
        sheets[version] = rows
        print(f"[fft] {version}: {len(paths)} file(s) found, {len(rows)} row(s) usable")

    if args.format == "csv":
        total = 0
        for version, rows in sheets.items():
            path = f"{os.path.splitext(output)[0]}_{version}.csv"
            total += write_csv(path, rows)
            print(f"[fft] Wrote {path}")
    else:
        path = output if output.lower().endswith(".xlsx") else output + ".xlsx"
        try:
            total = write_xlsx(path, sheets)
        except ImportError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        print(f"[fft] Wrote {path}")

    print(f"[fft] {total} data row(s) exported, {len(skipped)} file(s) skipped")

    if skipped and args.strict:
        print("[ERROR] --strict: some files could not be parsed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
