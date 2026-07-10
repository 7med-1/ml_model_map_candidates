from __future__ import annotations

import os
from pathlib import Path

from room_matcher.paths import (
    BASELINE_MODEL_PATH,
    LEGACY_BASELINE_MODEL_PATH,
    sync_legacy_baseline_artifacts,
)


def test_sync_legacy_baseline_artifacts_copies_newer_legacy_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_MODEL_PATH.write_text("older-model", encoding="utf-8")
    os.utime(BASELINE_MODEL_PATH, (100, 100))

    LEGACY_BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_BASELINE_MODEL_PATH.write_text("legacy-model", encoding="utf-8")
    os.utime(LEGACY_BASELINE_MODEL_PATH, (200, 200))

    synced_paths = sync_legacy_baseline_artifacts()

    assert (LEGACY_BASELINE_MODEL_PATH, BASELINE_MODEL_PATH) in synced_paths
    assert BASELINE_MODEL_PATH.read_text(encoding="utf-8") == "legacy-model"


def test_sync_legacy_baseline_artifacts_keeps_newer_baseline_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_MODEL_PATH.write_text("baseline-model", encoding="utf-8")
    os.utime(BASELINE_MODEL_PATH, (200, 200))

    LEGACY_BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_BASELINE_MODEL_PATH.write_text("legacy-model", encoding="utf-8")
    os.utime(LEGACY_BASELINE_MODEL_PATH, (100, 100))

    synced_paths = sync_legacy_baseline_artifacts()

    assert synced_paths == []
    assert BASELINE_MODEL_PATH.read_text(encoding="utf-8") == "baseline-model"
