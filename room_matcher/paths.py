from __future__ import annotations

import shutil
from pathlib import Path


ARTIFACTS_ROOT = Path("artifacts")
BASELINE_ARTIFACTS_DIR = ARTIFACTS_ROOT / "baseline"

BASELINE_MODEL_PATH = BASELINE_ARTIFACTS_DIR / "room_matcher.joblib"

LEGACY_BASELINE_MODEL_PATH = ARTIFACTS_ROOT / "room_matcher.joblib"

BASELINE_CLEAN_CSV_PATH = BASELINE_ARTIFACTS_DIR / "room_matching_clean.csv"
BASELINE_SQLITE_PATH = BASELINE_ARTIFACTS_DIR / "room_matching_clean.sqlite3"

LEGACY_BASELINE_CLEAN_CSV_PATH = ARTIFACTS_ROOT / "room_matching_clean.csv"
LEGACY_BASELINE_SQLITE_PATH = ARTIFACTS_ROOT / "room_matching_clean.sqlite3"

REPORTS_ROOT = Path("reports")
BASELINE_CLEANING_REPORT_PATH = REPORTS_ROOT / "baseline_cleaning_summary.json"


def sync_legacy_baseline_artifacts() -> list[tuple[Path, Path]]:
    synced_paths: list[tuple[Path, Path]] = []
    legacy_pairs = [
        (LEGACY_BASELINE_MODEL_PATH, BASELINE_MODEL_PATH),
        (LEGACY_BASELINE_CLEAN_CSV_PATH, BASELINE_CLEAN_CSV_PATH),
        (LEGACY_BASELINE_SQLITE_PATH, BASELINE_SQLITE_PATH),
    ]

    for source_path, target_path in legacy_pairs:
        if not source_path.exists():
            continue
        if target_path.exists() and target_path.stat().st_mtime >= source_path.stat().st_mtime:
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        synced_paths.append((source_path, target_path))

    return synced_paths
