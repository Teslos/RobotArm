"""Unit tests for measurement-point logging."""
from __future__ import annotations

import csv
import os

import pytest

from exts.robot_arm.metrology.recording import (
    CSV_FIELDS,
    PointLog,
    PointRecord,
    point_filename,
    write_point_file,
)

POSE = [190.0, -12.5, 188.0, -180.0, 0.0, 90.0]


# ── point_filename ───────────────────────────────────────────────────────────

def test_point_filename_pads_to_the_scan_size():
    """`data_{n:02d}` broke sorting past point 99 in the 10 000-point raster."""
    assert point_filename(7, 10_000) == "point0007.txt"
    assert point_filename(9_999, 10_000) == "point9999.txt"


def test_point_filename_padding_keeps_lexicographic_order_numeric():
    names = [point_filename(i, 150) for i in (9, 10, 99, 100, 149)]
    assert names == sorted(names)


def test_point_filename_small_scans():
    assert point_filename(0, 1) == "point0.txt"
    assert point_filename(5, 10) == "point5.txt"


def test_point_filename_prefix_and_suffix():
    assert point_filename(3, 100, prefix="data_", suffix=".dat") == "data_03.dat"


@pytest.mark.parametrize("index,total", [(-1, 10), (0, 0)])
def test_point_filename_validates(index, total):
    with pytest.raises(ValueError):
        point_filename(index, total)


# ── PointRecord ──────────────────────────────────────────────────────────────

def test_record_rejects_a_short_pose():
    with pytest.raises(ValueError, match="6 values"):
        PointRecord(index=0, line=0, label="x1 y2", pose=[1.0, 2.0, 3.0])


def test_record_row_has_every_csv_column():
    record = PointRecord(index=3, line=1, label="x1.0 y2.0", pose=POSE,
                         acquire_s=1.25, timestamp_s=9.5)
    assert set(record.as_row()) == set(CSV_FIELDS)


def test_record_row_values():
    record = PointRecord(index=3, line=1, label="x190.0 y-12.5", pose=POSE,
                         acquire_s=1.25, timestamp_s=9.5)
    row = record.as_row()
    assert row["index"] == 3
    assert row["line"] == 1
    assert float(row["x_mm"]) == pytest.approx(190.0)
    assert float(row["y_mm"]) == pytest.approx(-12.5)
    assert float(row["gamma_deg"]) == pytest.approx(90.0)
    assert float(row["acquire_s"]) == pytest.approx(1.25)


def test_record_coerces_pose_to_floats():
    record = PointRecord(index=0, line=0, label="", pose=(1, 2, 3, 4, 5, 6))
    assert record.pose == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# ── PointLog ─────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_log_writes_a_header_and_rows(tmp_path):
    path = tmp_path / "scan.csv"
    with PointLog(str(path)) as log:
        for i in range(3):
            log.append(PointRecord(index=i, line=0, label=f"x{i}", pose=POSE))
    rows = read_csv(path)
    assert [row["index"] for row in rows] == ["0", "1", "2"]
    assert list(rows[0]) == list(CSV_FIELDS)


def test_log_creates_missing_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "scan.csv"
    with PointLog(str(path)) as log:
        log.append(PointRecord(index=0, line=0, label="x0", pose=POSE))
    assert os.path.exists(path)


def test_log_flushes_so_an_aborted_scan_keeps_its_points(tmp_path):
    """The point of appending: a crash must not cost the measured points."""
    path = tmp_path / "scan.csv"
    log = PointLog(str(path)).open()
    log.append(PointRecord(index=0, line=0, label="x0", pose=POSE))
    # No close() — simulate the process dying here.
    assert len(read_csv(path)) == 1


def test_log_overwrites_by_default(tmp_path):
    path = tmp_path / "scan.csv"
    for _ in range(2):
        with PointLog(str(path)) as log:
            log.append(PointRecord(index=0, line=0, label="x0", pose=POSE))
    assert len(read_csv(path)) == 1


def test_log_can_append_to_an_existing_file(tmp_path):
    path = tmp_path / "scan.csv"
    with PointLog(str(path)) as log:
        log.append(PointRecord(index=0, line=0, label="x0", pose=POSE))
    with PointLog(str(path), overwrite=False) as log:
        log.append(PointRecord(index=1, line=0, label="x1", pose=POSE))
    rows = read_csv(path)
    assert [row["index"] for row in rows] == ["0", "1"]


def test_log_append_before_open_is_an_error(tmp_path):
    log = PointLog(str(tmp_path / "scan.csv"))
    with pytest.raises(RuntimeError, match="open"):
        log.append(PointRecord(index=0, line=0, label="x0", pose=POSE))


def test_log_extend_and_len(tmp_path):
    path = tmp_path / "scan.csv"
    records = [PointRecord(index=i, line=0, label=f"x{i}", pose=POSE) for i in range(4)]
    with PointLog(str(path)) as log:
        log.extend(records)
        assert len(log) == 4


def test_log_close_is_idempotent(tmp_path):
    log = PointLog(str(tmp_path / "scan.csv")).open()
    log.close()
    log.close()


# ── write_point_file ─────────────────────────────────────────────────────────

def test_write_point_file_content_and_name(tmp_path):
    path = write_point_file(str(tmp_path), 7, 1000, "x190.0 y-12.5")
    assert os.path.basename(path) == "point007.txt"
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == "x190.0 y-12.5\n"


def test_write_point_file_does_not_double_the_newline(tmp_path):
    path = write_point_file(str(tmp_path), 0, 10, "x1.0 y2.0\n")
    with open(path, encoding="utf-8") as handle:
        assert handle.read() == "x1.0 y2.0\n"
