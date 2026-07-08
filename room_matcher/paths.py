from __future__ import annotations

from pathlib import Path


ARTIFACTS_ROOT = Path("artifacts")
BASELINE_ARTIFACTS_DIR = ARTIFACTS_ROOT / "baseline"
HF_ARTIFACTS_DIR = ARTIFACTS_ROOT / "hf"

BASELINE_MODEL_PATH = BASELINE_ARTIFACTS_DIR / "room_matcher.joblib"
HF_MODEL_PATH = HF_ARTIFACTS_DIR / "room_matcher"

BASELINE_CLEAN_CSV_PATH = BASELINE_ARTIFACTS_DIR / "room_matching_clean.csv"
BASELINE_SQLITE_PATH = BASELINE_ARTIFACTS_DIR / "room_matching_clean.sqlite3"
HF_CLEAN_CSV_PATH = HF_ARTIFACTS_DIR / "room_matching_clean.csv"
HF_SQLITE_PATH = HF_ARTIFACTS_DIR / "room_matching_clean.sqlite3"

REPORTS_ROOT = Path("reports")
BASELINE_CLEANING_REPORT_PATH = REPORTS_ROOT / "baseline_cleaning_summary.json"
HF_CLEANING_REPORT_PATH = REPORTS_ROOT / "hf_cleaning_summary.json"

