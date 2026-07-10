from __future__ import annotations

from room_matcher.cleaning import clean_room_matching_csv, normalize_room_name


def test_normalize_room_name_collapses_common_variants() -> None:
    assert normalize_room_name("Room, 2 Queen Beds, Non Smoking") == "room 2 queen bed nonsmoking"
    assert normalize_room_name("1 King Bed - Accessible Nonsmoking Room") == "1 king bed accessible nonsmoking room"
    assert normalize_room_name("STD1K RM NSMK") == "standard 1 king room nonsmoking"
    assert normalize_room_name("double room meerblick") == "double room water_view"


def test_normalize_room_name_maps_multilingual_room_terms() -> None:
    assert normalize_room_name("Habitacion doble balcon vista al mar") == "room double balcony water_view"
    assert (
        normalize_room_name("Premium-Zimmer, 1 King-Bett, Nichtraucher, Balkon")
        == "premium room 1 king bed nonsmoking balcony"
    )
    assert (
        normalize_room_name(
            "Quarto Standard, 2 camas queen-size, nao fumadores, vista para a piscina"
        )
        == "room standard 2 bed queen nonsmoking pool_view"
    )


def test_normalize_room_name_maps_rate_plan_noise_to_canonical_tokens() -> None:
    assert (
        normalize_room_name("Superior Room, Bed & Breakfast, Non Refundable")
        == "superior room breakfast_included nonrefundable"
    )
    assert (
        normalize_room_name(
            "superior king room - non-refundable - breakfast included in the price : Breakfast Continental"
        )
        == "superior king room nonrefundable breakfast_included"
    )


def test_normalize_room_name_maps_compact_view_and_bed_codes() -> None:
    assert normalize_room_name("1k runway vw mob acc tub") == "1 king runway_view accessible tub"
    assert normalize_room_name("coronado bridge vw kng bed") == "coronado bridge_view king bed"
    assert normalize_room_name("appt 2 qn rm bayview nsmk") == "apartment 2 queen room water_view nonsmoking"


def test_normalize_room_name_handles_raw_abbreviation_examples_from_dataset_sample() -> None:
    assert (
        normalize_room_name("STUDIO STE 1 QN MOBIL ACCESS TUB NOSMK")
        == "studio suite 1 queen accessible tub nonsmoking"
    )
    assert (
        normalize_room_name("1 queen bednon smkg (package)")
        == "1 queen bed nonsmoking package"
    )
    assert (
        normalize_room_name("1 KNG EXE LVL MOBILITY/HEARING ACCESS RI SHWR")
        == "1 king executive level accessible rollin_shower"
    )
    assert (
        normalize_room_name("room, 1 king bed, accessible (roll-in shower)")
        == "room 1 king bed accessible rollin_shower"
    )
    assert (
        normalize_room_name("1 Kng Stndrd Hstrc Alexa Twr Cnl Strt Vw")
        == "1 king standard historic alexa tower canal street view"
    )


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
