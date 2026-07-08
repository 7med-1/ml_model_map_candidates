from __future__ import annotations

from room_matcher.cleaning import clean_room_matching_csv, normalize_room_name


def test_normalize_room_name_collapses_common_variants() -> None:
    assert normalize_room_name("Room, 2 Queen Beds, Non Smoking") == "room 2 queen beds nonsmoking"
    assert normalize_room_name("1 King Bed - Accessible Nonsmoking Room") == "1 king bed accessible nonsmoking room"


def test_clean_room_matching_csv_deduplicates_and_counts(tmp_path) -> None:
    input_csv = tmp_path / "room_matching.csv"
    output_csv = tmp_path / "clean.csv"
    sqlite_path = tmp_path / "clean.sqlite3"
    report_path = tmp_path / "report.json"
    input_csv.write_text(
        "\n".join(
            [
                "nuitee_room_name,provider_room_name",
                'Standard 2 Queen Room,"Room, 2 Queen Beds, Non Smoking"',
                'Standard 2 Queen Room,"Room, 2 Queen Beds, Non Smoking"',
                'Standard King Room (Accessible),1 King Bed - Accessible Nonsmoking Room',
                ",missing provider",
            ]
        ),
        encoding="utf-8",
    )

    stats = clean_room_matching_csv(
        input_csv,
        output_csv,
        sqlite_path,
        report_path,
    )

    assert stats.raw_rows == 4
    assert stats.dropped_rows == 1
    assert stats.unique_pairs == 2

    cleaned = output_csv.read_text(encoding="utf-8")
    assert "pair_count" in cleaned
    assert ",2,1" in cleaned


def test_clean_room_matching_csv_respects_row_limit(tmp_path) -> None:
    input_csv = tmp_path / "room_matching.csv"
    output_csv = tmp_path / "clean.csv"
    sqlite_path = tmp_path / "clean.sqlite3"
    report_path = tmp_path / "report.json"
    input_csv.write_text(
        "\n".join(
            [
                "nuitee_room_name,provider_room_name",
                "Room A,Candidate A",
                "Room B,Candidate B",
                "Room C,Candidate C",
            ]
        ),
        encoding="utf-8",
    )

    stats = clean_room_matching_csv(
        input_csv,
        output_csv,
        sqlite_path,
        report_path,
        max_rows=2,
    )

    assert stats.raw_rows == 2
    assert stats.processed_row_limit == 2
    cleaned = output_csv.read_text(encoding="utf-8")
    assert "Room C" not in cleaned
