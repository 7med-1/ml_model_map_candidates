from __future__ import annotations

import csv
import json
import re
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from room_matcher.progress import print_status


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
NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9_]+")
COMPACT_CODE_PATTERNS = (
    (re.compile(r"\bstdsd\b"), " standard "),
    (re.compile(r"\bstds\b"), " standard "),
    (re.compile(r"\bstd\s*1k\b"), " standard 1 king room "),
    (re.compile(r"\bstd\s*2k\b"), " standard 2 king room "),
    (re.compile(r"\bstd\s*1q\b"), " standard 1 queen room "),
    (re.compile(r"\bstd\s*2q\b"), " standard 2 queen room "),
    (re.compile(r"\bstd\s*1d\b"), " standard 1 double room "),
    (re.compile(r"\bstd\s*2d\b"), " standard 2 double room "),
    (re.compile(r"\b(\d+)\s*kng\b"), r" \1 king "),
    (re.compile(r"\b(\d+)\s*kg\b"), r" \1 king "),
    (re.compile(r"\b(\d+)\s*qn\b"), r" \1 queen "),
    (re.compile(r"\b(\d+)\s*dbl\b"), r" \1 double "),
    (re.compile(r"\b(\d+)\s*db\b"), r" \1 double "),
    (re.compile(r"\b(\d+)\s*twn\b"), r" \1 twin "),
    (re.compile(r"\b(\d+)\s*sgl\b"), r" \1 single "),
    (re.compile(r"\b(\d+)\s*rm\b"), r" \1 room "),
    (re.compile(r"\b(\d+)\s*k\b"), r" \1 king "),
    (re.compile(r"\b(\d+)\s*q\b"), r" \1 queen "),
)
PHRASE_NORMALIZATION_PATTERNS = (
    (re.compile(r"([a-z])non\s+(?:smkg|smk|smok|smoke|smoking)\b"), r"\1 nonsmoking "),
    (re.compile(r"\bnon[\s-]?smoking\b"), " nonsmoking "),
    (re.compile(r"\bno[\s-]?smoking\b"), " nonsmoking "),
    (re.compile(r"\bnon[\s-]?smok\b"), " nonsmoking "),
    (re.compile(r"\bno[\s-]?smok\b"), " nonsmoking "),
    (re.compile(r"\bno[\s-]?smoke\b"), " nonsmoking "),
    (re.compile(r"\bn[\s-]?smk\b"), " nonsmoking "),
    (re.compile(r"\bn/?s\b"), " nonsmoking "),
    (re.compile(r"\bnsmk\b"), " nonsmoking "),
    (re.compile(r"\bnon[\s-]?fumeurs?\b"), " nonsmoking "),
    (re.compile(r"\bnichtraucher\b"), " nonsmoking "),
    (re.compile(r"\brauchfrei\b"), " nonsmoking "),
    (re.compile(r"\bsin\s+humo\b"), " nonsmoking "),
    (re.compile(r"\bno\s+fumadores?\b"), " nonsmoking "),
    (re.compile(r"\bpara\s+no\s+fumadores?\b"), " nonsmoking "),
    (re.compile(r"\bnao\s+fumadores?\b"), " nonsmoking "),
    (re.compile(r"\bsmk\b"), " smoking "),
    (re.compile(r"\bfumeurs?\b"), " smoking "),
    (re.compile(r"\bfumadores?\b"), " smoking "),
    (re.compile(r"\braucher\b"), " smoking "),
    (re.compile(r"\bwheelchair[\s-]?accessible\b"), " accessible "),
    (re.compile(r"\bmobility[\s-]?accessible\b"), " accessible "),
    (re.compile(r"\bhearing[\s-]?accessible\b"), " accessible "),
    (re.compile(r"\bbarrier[\s-]?free\b"), " accessible "),
    (re.compile(r"\bbarrierefrei\b"), " accessible "),
    (re.compile(r"\bmobility/hearing\s+access\b"), " accessible "),
    (re.compile(r"\bmobility\s+access\b"), " accessible "),
    (re.compile(r"\bhearing\s+access\b"), " accessible "),
    (re.compile(r"\broll[\s-]?in\s+sh(?:ower|wr)\b"), " rollin_shower "),
    (re.compile(r"\bri[\s/-]?shwr\b"), " rollin_shower "),
    (re.compile(r"\bri[\s/-]?shower\b"), " rollin_shower "),
    (re.compile(r"\bsea[\s-]?view\b"), " water_view "),
    (re.compile(r"\bocean[\s-]?view\b"), " water_view "),
    (re.compile(r"\bbay[\s-]?view\b"), " water_view "),
    (re.compile(r"\bvue\s+mer\b"), " water_view "),
    (re.compile(r"\bmeerblick\b"), " water_view "),
    (re.compile(r"\bseitenblick\s+auf\s+das\s+meer\b"), " water_view "),
    (re.compile(r"\bblick\s+auf\s+das\s+meer\b"), " water_view "),
    (re.compile(r"\bvistas?\s+(?:al\s+|a\s+la\s+|a\s+|para\s+a\s+)?mar\b"), " water_view "),
    (re.compile(r"\bpool[\s-]?view\b"), " pool_view "),
    (re.compile(r"\bpoolvw\b"), " pool_view "),
    (re.compile(r"\bswimming\s+pool\s+view\b"), " pool_view "),
    (re.compile(r"\bvue\s+piscine\b"), " pool_view "),
    (re.compile(r"\bvistas?\s+(?:al\s+|a\s+la\s+|a\s+|para\s+a\s+)?piscina\b"), " pool_view "),
    (re.compile(r"\bgarden[\s-]?view\b"), " garden_view "),
    (re.compile(r"\bvue\s+jardin\b"), " garden_view "),
    (re.compile(r"\bgartenblick\b"), " garden_view "),
    (re.compile(r"\bcity[\s-]?view\b"), " city_view "),
    (re.compile(r"\bstadtblick\b"), " city_view "),
    (re.compile(r"\brunway[\s-]?view\b"), " runway_view "),
    (re.compile(r"\bbridge[\s-]?view\b"), " bridge_view "),
    (re.compile(r"\bmountain[\s-]?view\b"), " mountain_view "),
    (re.compile(r"\batlas\s+mountains?\s+view\b"), " mountain_view "),
    (re.compile(r"\bbergblick\b"), " mountain_view "),
    (re.compile(r"\broom[\s-]?only\b"), " room_only "),
    (re.compile(r"\baccommodation[\s-]?only\b"), " room_only "),
    (re.compile(r"\bbed\s*(?:and|&)\s*breakfast\b"), " breakfast_included "),
    (re.compile(r"\bbreakfast\s+included\b"), " breakfast_included "),
    (re.compile(r"\bfree\s+breakfast\b"), " breakfast_included "),
    (re.compile(r"\bcontinental\s+breakfast\b"), " breakfast_included "),
    (re.compile(r"\bbreakfast\s+continental\b"), " breakfast_included "),
    (re.compile(r"\bfull\s+breakfast\b"), " breakfast_included "),
    (re.compile(r"\bpetit\s+dejeuner\b"), " breakfast_included "),
    (re.compile(r"\bbreakfast_included\s+in\s+the\s+price\b"), " breakfast_included "),
    (re.compile(r"\bhalf[\s-]?board\b"), " half_board "),
    (re.compile(r"\bdemi[\s-]?pension\b"), " half_board "),
    (re.compile(r"\bmedia\s+pension\b"), " half_board "),
    (re.compile(r"\bfull[\s-]?board\b"), " full_board "),
    (re.compile(r"\bpension\s+complete\b"), " full_board "),
    (re.compile(r"\bpension\s+completa\b"), " full_board "),
    (re.compile(r"\ball[\s-]?inclusive\b"), " all_inclusive "),
    (re.compile(r"\btodo\s+incluido\b"), " all_inclusive "),
    (re.compile(r"\bnon[\s-]?refundable\b"), " nonrefundable "),
    (re.compile(r"\bnrf\b"), " nonrefundable "),
    (re.compile(r"\bnon\s+remboursable\b"), " nonrefundable "),
    (re.compile(r"\bno\s+reembolsable\b"), " nonrefundable "),
    (re.compile(r"\bnicht\s+erstattbar\b"), " nonrefundable "),
    (re.compile(r"\b(?:balcony|balconies|balconey|balc|balcon|balcn|balcone|balcón|balkon|balkony|balcao|balcão)\b"), " balcony "),
)
TOKEN_NORMALIZATION_MAP = {
    "rm": "room",
    "rms": "room",
    "rooms": "room",
    "zimmer": "room",
    "zimmern": "room",
    "chambre": "room",
    "chambres": "room",
    "habitacion": "room",
    "habitaciones": "room",
    "quarto": "room",
    "quartos": "room",
    "camera": "room",
    "camara": "room",
    "ste": "suite",
    "suites": "suite",
    "studios": "studio",
    "bdrm": "bedroom",
    "bdrms": "bedroom",
    "apt": "apartment",
    "appt": "apartment",
    "apartamento": "apartment",
    "appartement": "apartment",
    "beds": "bed",
    "bett": "bed",
    "betten": "bed",
    "lit": "bed",
    "lits": "bed",
    "cama": "bed",
    "camas": "bed",
    "kg": "king",
    "kng": "king",
    "kingsize": "king",
    "kingbed": "king bed",
    "qn": "queen",
    "queensize": "queen",
    "queenbed": "queen bed",
    "dbl": "double",
    "db": "double",
    "doble": "double",
    "doppel": "double",
    "fullbed": "double bed",
    "twn": "twin",
    "twinbed": "twin bed",
    "sgl": "single",
    "einzel": "single",
    "individual": "single",
    "sofabed": "sofa bed",
    "balcon": "balcony",
    "balcone": "balcony",
    "balcones": "balcony",
    "balkon": "balcony",
    "bal": "balcony",
    "balc": "balcony",
    "terrasse": "terrace",
    "terraza": "terrace",
    "terraco": "terrace",
    "vue": "view",
    "vista": "view",
    "vistas": "view",
    "vw": "view",
    "blick": "view",
    "meer": "sea",
    "mer": "sea",
    "mar": "sea",
    "ocean": "sea",
    "piscina": "pool",
    "piscine": "pool",
    "jardin": "garden",
    "garten": "garden",
    "ciudad": "city",
    "stadt": "city",
    "montana": "mountain",
    "montagne": "mountain",
    "mountains": "mountain",
    "estandar": "standard",
    "standart": "standard",
    "std": "standard",
    "stndrd": "standard",
    "stnd": "standard",
    "delux": "deluxe",
    "delx": "deluxe",
    "dlx": "deluxe",
    "exe": "executive",
    "nichtraucher": "nonsmoking",
    "rauchfrei": "nonsmoking",
    "nsmoking": "nonsmoking",
    "nosmok": "nonsmoking",
    "nonsmok": "nonsmoking",
    "nosmoke": "nonsmoking",
    "nonsmoke": "nonsmoking",
    "nosmkg": "nonsmoking",
    "nonsmkg": "nonsmoking",
    "nosmokg": "nonsmoking",
    "nonsmokg": "nonsmoking",
    "nosmk": "nonsmoking",
    "ns": "nonsmoking",
    "fumador": "smoking",
    "fumadores": "smoking",
    "fumeur": "smoking",
    "fumeurs": "smoking",
    "smkg": "smoking",
    "barrierefrei": "accessible",
    "accesible": "accessible",
    "acc": "accessible",
    "access": "accessible",
    "mob": "accessible",
    "mobil": "accessible",
    "mobility": "accessible",
    "hear": "accessible",
    "hearing": "accessible",
    "hstrc": "historic",
    "twr": "tower",
    "cnl": "canal",
    "strt": "street",
    "lvl": "level",
    "shwr": "shower",
    "poolvw": "pool_view",
    "seavw": "water_view",
    "seaview": "water_view",
    "oceanvw": "water_view",
    "oceanview": "water_view",
    "bayvw": "water_view",
    "bayview": "water_view",
    "lakeview": "water_view",
    "meerblick": "water_view",
    "oceanfront": "water_view",
    "seafront": "water_view",
    "cityvw": "city_view",
    "gardenvw": "garden_view",
    "runwayview": "runway_view",
    "runwayvw": "runway_view",
    "bridgeview": "bridge_view",
    "bridgevw": "bridge_view",
}
FILLER_TOKENS = {
    "a",
    "al",
    "am",
    "an",
    "auf",
    "da",
    "das",
    "de",
    "del",
    "dem",
    "den",
    "der",
    "des",
    "do",
    "dos",
    "du",
    "el",
    "im",
    "in",
    "la",
    "las",
    "los",
    "of",
    "para",
    "price",
    "por",
    "rate",
    "the",
    "w",
    "with",
}
TOKEN_SEQUENCE_NORMALIZATION_MAP = {
    ("king", "size"): "king",
    ("queen", "size"): "queen",
    ("double", "size"): "double",
    ("swimming", "pool"): "pool",
    ("sea", "view"): "water_view",
    ("view", "sea"): "water_view",
    ("pool", "view"): "pool_view",
    ("view", "pool"): "pool_view",
    ("bay", "view"): "water_view",
    ("view", "bay"): "water_view",
    ("garden", "view"): "garden_view",
    ("view", "garden"): "garden_view",
    ("city", "view"): "city_view",
    ("view", "city"): "city_view",
    ("runway", "view"): "runway_view",
    ("view", "runway"): "runway_view",
    ("bridge", "view"): "bridge_view",
    ("view", "bridge"): "bridge_view",
    ("mountain", "view"): "mountain_view",
    ("view", "mountain"): "mountain_view",
    ("side", "sea", "view"): "water_view",
    ("partial", "sea", "view"): "water_view",
    ("view", "swimming", "pool"): "pool_view",
    ("swimming", "pool", "view"): "pool_view",
}
TOKEN_SEQUENCE_LENGTHS = sorted(
    {len(key) for key in TOKEN_SEQUENCE_NORMALIZATION_MAP},
    reverse=True,
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

    text = _apply_patterns(text, COMPACT_CODE_PATTERNS)
    text = _apply_patterns(text, PHRASE_NORMALIZATION_PATTERNS)

    text = NON_ALNUM_PATTERN.sub(" ", text)
    tokens: list[str] = []
    for token in SPACE_PATTERN.sub(" ", text).strip().split():
        replacement = TOKEN_NORMALIZATION_MAP.get(token, token)
        tokens.extend(part for part in replacement.split() if part)
    tokens = [token for token in tokens if token and token not in FILLER_TOKENS]
    tokens = _merge_token_sequences(tokens)
    tokens = _collapse_adjacent_duplicates(tokens)
    return " ".join(tokens)


def _apply_patterns(
    text: str,
    patterns: tuple[tuple[re.Pattern[str], str], ...],
) -> str:
    for pattern, replacement in patterns:
        text = pattern.sub(replacement, text)
    return text


def _merge_token_sequences(tokens: list[str]) -> list[str]:
    merged = list(tokens)
    changed = True

    # Merge view and room-code sequences into single canonical tokens.
    while changed:
        changed = False
        collapsed: list[str] = []
        index = 0
        while index < len(merged):
            replacement = None
            consumed = 1
            for length in TOKEN_SEQUENCE_LENGTHS:
                sequence = tuple(merged[index : index + length])
                replacement = TOKEN_SEQUENCE_NORMALIZATION_MAP.get(sequence)
                if replacement is not None:
                    consumed = length
                    break
            if replacement is None:
                collapsed.append(merged[index])
                index += 1
                continue

            collapsed.append(replacement)
            index += consumed
            changed = True
        merged = collapsed

    return merged


def _collapse_adjacent_duplicates(tokens: list[str]) -> list[str]:
    collapsed: list[str] = []
    for token in tokens:
        if collapsed and collapsed[-1] == token:
            continue
        collapsed.append(token)
    return collapsed


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
    progress_every_rows: int = 250_000,
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

    print_status(f"Cleaning raw CSV: {input_csv}")
    print_status(f"Writing cleaned dataset to: {output_csv}")

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

                if progress_every_rows and raw_rows % progress_every_rows == 0:
                    print_status(f"Processed {raw_rows:,} raw rows")

                if max_rows is not None and raw_rows >= max_rows:
                    break

            if batch:
                connection.executemany(insert_sql, batch)

        connection.commit()
        print_status("Finished reading raw rows, aggregating unique pairs")

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

        print_status("Exporting cleaned rows to CSV")
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

        print_status(
            "Cleaning finished: "
            f"{stats.raw_rows:,} raw rows, "
            f"{stats.unique_pairs:,} unique pairs, "
            f"{stats.ambiguous_provider_rooms:,} ambiguous provider rooms"
        )
        return stats
    finally:
        connection.close()
