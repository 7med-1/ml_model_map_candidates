from __future__ import annotations

import argparse

from room_matcher.cleaning import clean_room_matching_csv
from room_matcher.paths import (
    BASELINE_CLEANING_REPORT_PATH,
    BASELINE_CLEAN_CSV_PATH,
    BASELINE_SQLITE_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the room matching CSV into a deduplicated training dataset.",
    )
    parser.add_argument("--input-csv", default="room_matching.csv")
    parser.add_argument("--output-csv", default=str(BASELINE_CLEAN_CSV_PATH))
    parser.add_argument("--sqlite-path", default=str(BASELINE_SQLITE_PATH))
    parser.add_argument("--report-path", default=str(BASELINE_CLEANING_REPORT_PATH))
    parser.add_argument("--max-clean-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = clean_room_matching_csv(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        sqlite_path=args.sqlite_path,
        report_path=args.report_path,
        max_rows=args.max_clean_rows,
    )
    print("Clean CSV:", stats.output_csv)
    print("SQLite DB:", stats.sqlite_path)
    print("Raw rows processed:", stats.raw_rows)
    print("Unique pairs:", stats.unique_pairs)


if __name__ == "__main__":
    main()
