"""Unit tests for the FFT text-to-table export."""
from __future__ import annotations

import os

import pytest

from exts.robot_arm.metrology.fft_export import (
    FftParseError,
    FftRow,
    build_rows,
    collect_fft_files,
    coordinate_sort_key,
    find_coordinate_folder,
    parse_coordinates,
    parse_triplets,
    read_fft_file,
    rows_to_table,
    write_csv,
    write_xlsx,
)

HEADER = "Index Time Label Freq Mag Phase"


def make_fft_tree(root, folders, version="V0", triplets=3):
    """Build <root>/<folder>/run_FFT/run_FFT_<version>.txt for each folder."""
    for i, folder in enumerate(folders):
        directory = os.path.join(str(root), folder, "run_FFT")
        os.makedirs(directory, exist_ok=True)
        values = " ".join(
            f"{100.0 * (k + 1)} {i + k / 10:.3f} {0.5 * k}"
            for k in range(triplets)
        )
        path = os.path.join(directory, f"run_FFT_{version}.txt")
        with open(path, "w", encoding="cp1252") as handle:
            handle.write(HEADER + "\n")
            handle.write(f"{i} 0.0 tag {values}\n")
    return str(root)


# ── parse_coordinates ────────────────────────────────────────────────────────

def test_parse_coordinates_basic():
    assert parse_coordinates("x12.5 y3") == {"x": 12.5, "y": 3.0}


def test_parse_coordinates_handles_negatives():
    """`[yY]([\\d.]+)` could not match a minus sign, so negatives read as 0.0."""
    assert parse_coordinates("Y-12.5") == {"y": -12.5}
    assert parse_coordinates("x-3.0 y-7.5") == {"x": -3.0, "y": -7.5}


def test_parse_coordinates_is_case_insensitive():
    assert parse_coordinates("Z4") == {"z": 4.0}


def test_parse_coordinates_allows_a_space_after_the_axis():
    assert parse_coordinates("y 2.25") == {"y": 2.25}


def test_parse_coordinates_ignores_axis_letters_inside_words():
    assert parse_coordinates("max12_run") == {}
    assert parse_coordinates("mess38") == {}


def test_parse_coordinates_returns_empty_rather_than_zero():
    """The old helpers returned 0.0 for 'not found', which sorted silently."""
    assert parse_coordinates("no_coords_here") == {}


def test_parse_coordinates_tolerates_a_malformed_value():
    """`[\\d.]+` matched '12.5.3' and then float() blew up inside list.sort."""
    assert parse_coordinates("y12.5.3") == {"y": 12.5}


def test_parse_coordinates_last_occurrence_wins():
    assert parse_coordinates("x1 x2") == {"x": 2.0}


# ── sorting ──────────────────────────────────────────────────────────────────

def test_sort_key_orders_negatives_before_positives():
    labels = ["Y5", "Y-10", "Y0", "Y-2.5"]
    ordered = sorted(labels, key=lambda name: coordinate_sort_key(parse_coordinates(name), name))
    assert ordered == ["Y-10", "Y-2.5", "Y0", "Y5"]


def test_sort_key_puts_missing_axes_last():
    with_y = coordinate_sort_key({"y": 999.0}, "a")
    without = coordinate_sort_key({}, "a")
    assert with_y < without


def test_sort_key_breaks_ties_on_the_label():
    a = coordinate_sort_key({"y": 1.0}, "aaa")
    b = coordinate_sort_key({"y": 1.0}, "bbb")
    assert a < b


def test_sort_key_respects_a_custom_axis_order():
    coords = {"x": 1.0, "y": 2.0}
    assert coordinate_sort_key(coords, order=("x",))[1] == 1.0
    assert coordinate_sort_key(coords, order=("y",))[1] == 2.0


# ── find_coordinate_folder ───────────────────────────────────────────────────

def test_find_coordinate_folder_picks_the_innermost_match():
    path = os.path.join("root", "Y10", "Y20", "run_FFT", "run_FFT_V0.txt")
    assert find_coordinate_folder(path) == "Y20"


def test_find_coordinate_folder_returns_empty_when_absent():
    assert find_coordinate_folder(os.path.join("root", "run", "a.txt")) == ""


# ── parse_triplets ───────────────────────────────────────────────────────────

def test_parse_triplets_skips_metadata_fields():
    assert parse_triplets("0 0.0 tag 100 1.5 0.25") == [(100.0, 1.5, 0.25)]


def test_parse_triplets_drops_a_trailing_partial_triplet():
    assert parse_triplets("a b c 1 2 3 4 5") == [(1.0, 2.0, 3.0)]


def test_parse_triplets_ignores_non_numeric_tokens():
    assert parse_triplets("a b c 1 NaNtoken 2 3") == [(1.0, 2.0, 3.0)]


def test_parse_triplets_with_zero_skip():
    assert parse_triplets("1 2 3", skip_fields=0) == [(1.0, 2.0, 3.0)]


def test_parse_triplets_on_a_short_line():
    assert parse_triplets("0 0.0 tag") == []


def test_parse_triplets_rejects_negative_skip():
    with pytest.raises(ValueError, match="skip_fields"):
        parse_triplets("1 2 3", skip_fields=-1)


# ── read_fft_file ────────────────────────────────────────────────────────────

def test_read_fft_file(tmp_path):
    root = make_fft_tree(tmp_path, ["Y1.0"], triplets=2)
    path = os.path.join(root, "Y1.0", "run_FFT", "run_FFT_V0.txt")
    triplets = read_fft_file(path)
    assert len(triplets) == 2
    assert triplets[0][0] == pytest.approx(100.0)


def test_read_fft_file_rejects_a_file_without_a_data_line(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text(HEADER + "\n", encoding="cp1252")
    with pytest.raises(FftParseError, match="at least 2 lines"):
        read_fft_file(str(path))


def test_read_fft_file_rejects_a_data_line_without_triplets(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text(HEADER + "\n0 0.0 tag\n", encoding="cp1252")
    with pytest.raises(FftParseError, match="no complete freq/mag/phase"):
        read_fft_file(str(path))


# ── collect + build_rows ─────────────────────────────────────────────────────

def test_collect_fft_files_finds_every_version_file(tmp_path):
    root = make_fft_tree(tmp_path, ["Y1.0", "Y2.0", "Y-3.0"])
    assert len(collect_fft_files(root, "V0")) == 3
    assert collect_fft_files(root, "V9") == []


def test_build_rows_sorts_by_coordinate_including_negatives(tmp_path):
    root = make_fft_tree(tmp_path, ["Y5.0", "Y-10.0", "Y0.0", "Y-2.5"])
    rows = build_rows(collect_fft_files(root, "V0"))
    assert [row.coordinates["y"] for row in rows] == [-10.0, -2.5, 0.0, 5.0]


def test_build_rows_reports_and_skips_unparseable_files(tmp_path):
    root = make_fft_tree(tmp_path, ["Y1.0"])
    bad_dir = os.path.join(root, "Y2.0", "run_FFT")
    os.makedirs(bad_dir)
    with open(os.path.join(bad_dir, "run_FFT_V0.txt"), "w", encoding="cp1252") as handle:
        handle.write("only a header\n")

    problems = []
    rows = build_rows(
        collect_fft_files(root, "V0"),
        on_error=lambda path, exc: problems.append(path),
    )
    assert len(rows) == 1
    assert len(problems) == 1


def test_build_rows_skips_files_without_a_coordinate_folder(tmp_path):
    directory = os.path.join(str(tmp_path), "plain", "run_FFT")
    os.makedirs(directory)
    with open(os.path.join(directory, "run_FFT_V0.txt"), "w", encoding="cp1252") as handle:
        handle.write(HEADER + "\n0 0.0 tag 1 2 3\n")

    problems = []
    rows = build_rows(
        collect_fft_files(str(tmp_path), "V0"),
        on_error=lambda path, exc: problems.append(str(exc)),
    )
    assert rows == []
    assert "no coordinate folder" in problems[0]


def test_build_rows_on_an_empty_input():
    assert build_rows([]) == []


# ── rows_to_table ────────────────────────────────────────────────────────────

def test_row_dict_exposes_numeric_coordinate_columns():
    row = FftRow(label="Y-12.5", coordinates={"y": -12.5}, triplets=[(1.0, 2.0, 3.0)])
    as_dict = row.as_dict()
    assert as_dict["Coordinate"] == "Y-12.5"
    assert as_dict["y_mm"] == -12.5
    assert as_dict["x_mm"] is None
    assert (as_dict["Freq_0"], as_dict["Mag_0"], as_dict["Phase_0"]) == (1.0, 2.0, 3.0)


def test_rows_to_table_unions_columns_and_pads_short_rows():
    """A point with fewer bins must not truncate the others."""
    short = FftRow(label="Y1", coordinates={"y": 1.0}, triplets=[(1.0, 2.0, 3.0)])
    long = FftRow(label="Y2", coordinates={"y": 2.0},
                  triplets=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    header, table = rows_to_table([short, long])
    assert "Freq_1" in header
    assert table[0][header.index("Freq_1")] is None
    assert table[1][header.index("Freq_1")] == 4.0


def test_rows_to_table_empty():
    assert rows_to_table([]) == ([], [])


# ── writers ──────────────────────────────────────────────────────────────────

def test_write_csv_round_trip(tmp_path):
    import csv as csv_module

    rows = [FftRow(label="Y1", coordinates={"y": 1.0}, triplets=[(1.0, 2.0, 3.0)])]
    path = tmp_path / "out" / "v0.csv"
    assert write_csv(str(path), rows) == 1
    with open(path, newline="", encoding="utf-8") as handle:
        table = list(csv_module.DictReader(handle))
    assert table[0]["Coordinate"] == "Y1"
    assert table[0]["y_mm"] == "1.0"


def test_write_csv_with_no_rows_still_writes_a_readable_file(tmp_path):
    path = tmp_path / "empty.csv"
    assert write_csv(str(path), []) == 0
    assert path.read_text(encoding="utf-8").strip().splitlines() == [
        "Info", "No data found"
    ]


def test_write_xlsx_creates_one_sheet_per_version(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    rows = [FftRow(label="Y1", coordinates={"y": 1.0}, triplets=[(1.0, 2.0, 3.0)])]
    path = tmp_path / "book.xlsx"
    assert write_xlsx(str(path), {"V0": rows, "V1": rows, "V2": []}) == 2

    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["V0", "V1"]
    assert book["V0"].cell(row=1, column=1).value == "Coordinate"
    assert book["V0"].cell(row=2, column=1).value == "Y1"


def test_write_xlsx_with_no_data_writes_an_info_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "book.xlsx"
    assert write_xlsx(str(path), {"V0": [], "V1": []}) == 0
    book = openpyxl.load_workbook(path)
    assert book.sheetnames == ["Info"]
