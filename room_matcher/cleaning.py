from __future__ import annotations

import csv
import json
import re
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path


RAW_ROOM_NAME_COLUMN = "nuitee_room_name"
RAW_PROVIDER_ROOM_COLUMN = "provider_room_name"
CLEANED_FIELDNAMES = [
    "room_name",
    "candidate_room",
    "room_name_normalized",
    "candidate_room_normalized",
    "pair_count",
    "provider_room_ambiguity",
]

SPACE_PATTERN = re.compile(r"\s+")
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
NORMALIZATION_PATTERNS = (
    (re.compile(r"\bnon[\s-]?smoking\b"), "nonsmoking"),
    (re.compile(r"\bno[\s-]?smoking\b"), "nonsmoking"),
    (re.compile(r"\bn[\s-]?smk\b"), "nonsmoking"),
    (re.compile(r"\bsmk\b"), "smoking"),
    (re.compile(r"\bste\b"), "suite"),
    (re.compile(r"\brm\b"), "room"),
    (re.compile(r"\bdbl\b"), "double"),
    (re.compile(r"\btwn\b"), "twin"),
    (re.compile(r"\bsgl\b"), "single"),
    (re.compile(r"\bkg\b"), "king"),
    (re.compile(r"\bqn\b"), "queen"),
)
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


@dataclass(slots=True)
class CleaningStats:
    raw_rows: int
    dropped_rows: int
    unique_pairs: int
    ambiguous_provider_rooms: int
    output_csv: str
    sqlite_path: str
    processed_row_limit: int | None = None

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def normalize_room_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = text.replace("&", " and ")

    for word, digit in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)

    for pattern, replacement in NORMALIZATION_PATTERNS:
        text = pattern.sub(replacement, text)

    text = NON_ALNUM_PATTERN.sub(" ", text)
    return SPACE_PATTERN.sub(" ", text).strip()


def load_cleaning_stats(report_path: str | Path) -> CleaningStats:
    with Path(report_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return CleaningStats(**payload)


def iter_clean_rows(cleaned_csv_path: str | Path):
    with Path(cleaned_csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def clean_room_matching_csv(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    sqlite_path: str | Path,
    report_path: str | Path,
    *,
    batch_size: int = 10_000,
    max_rows: int | None = None,
) -> CleaningStats:
    input_csv = Path(input_csv_path)
    output_csv = Path(output_csv_path)
    sqlite_file = Path(sqlite_path)
    report_file = Path(report_path)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sqlite_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    if sqlite_file.exists():
        sqlite_file.unlink()

    connection = sqlite3.connect(sqlite_file)
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute(
            """
            CREATE TABLE room_pairs (
                room_name TEXT NOT NULL,
                candidate_room TEXT NOT NULL,
                room_name_normalized TEXT NOT NULL,
                candidate_room_normalized TEXT NOT NULL,
                pair_count INTEGER NOT NULL,
                PRIMARY KEY (room_name, candidate_room)
            )
            """
        )

        insert_sql = """
            INSERT INTO room_pairs (
                room_name,
                candidate_room,
                room_name_normalized,
                candidate_room_normalized,
                pair_count
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(room_name, candidate_room) DO UPDATE
            SET pair_count = pair_count + excluded.pair_count
        """

        raw_rows = 0
        dropped_rows = 0
        batch: list[tuple[str, str, str, str, int]] = []

        with input_csv.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected_columns = {RAW_ROOM_NAME_COLUMN, RAW_PROVIDER_ROOM_COLUMN}
            if not reader.fieldnames or not expected_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"Input CSV must contain columns: {sorted(expected_columns)}"
                )

            for row in reader:
                raw_rows += 1
                room_name = (row.get(RAW_ROOM_NAME_COLUMN) or "").strip()
                candidate_room = (row.get(RAW_PROVIDER_ROOM_COLUMN) or "").strip()
                if not room_name or not candidate_room:
                    dropped_rows += 1
                else:
                    batch.append(
                        (
                            room_name,
                            candidate_room,
                            normalize_room_name(room_name),
                            normalize_room_name(candidate_room),
                            1,
                        )
                    )
                    if len(batch) >= batch_size:
                        connection.executemany(insert_sql, batch)
                        batch.clear()

                if max_rows is not None and raw_rows >= max_rows:
                    break

            if batch:
                connection.executemany(insert_sql, batch)

        connection.commit()

        unique_pairs = connection.execute(
            "SELECT COUNT(*) FROM room_pairs"
        ).fetchone()[0]
        ambiguous_provider_rooms = connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT candidate_room
                FROM room_pairs
                GROUP BY candidate_room
                HAVING COUNT(DISTINCT room_name) > 1
            )
            """
        ).fetchone()[0]

        export_cursor = connection.execute(
            """
            SELECT
                room_pairs.room_name,
                room_pairs.candidate_room,
                room_pairs.room_name_normalized,
                room_pairs.candidate_room_normalized,
                room_pairs.pair_count,
                provider_stats.provider_room_ambiguity
            FROM room_pairs
            JOIN (
                SELECT
                    candidate_room,
                    COUNT(DISTINCT room_name) AS provider_room_ambiguity
                FROM room_pairs
                GROUP BY candidate_room
            ) AS provider_stats
                ON provider_stats.candidate_room = room_pairs.candidate_room
            ORDER BY room_pairs.room_name, room_pairs.candidate_room
            """
        )

        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CLEANED_FIELDNAMES)
            writer.writeheader()
            for row in export_cursor:
                writer.writerow(
                    {
                        "room_name": row[0],
                        "candidate_room": row[1],
                        "room_name_normalized": row[2],
                        "candidate_room_normalized": row[3],
                        "pair_count": row[4],
                        "provider_room_ambiguity": row[5],
                    }
                )

        stats = CleaningStats(
            raw_rows=raw_rows,
            dropped_rows=dropped_rows,
            unique_pairs=unique_pairs,
            ambiguous_provider_rooms=ambiguous_provider_rooms,
            output_csv=str(output_csv),
            sqlite_path=str(sqlite_file),
            processed_row_limit=max_rows,
        )

        with report_file.open("w", encoding="utf-8") as handle:
            json.dump(stats.to_dict(), handle, indent=2, sort_keys=True)

        return stats
    finally:
        connection.close()
