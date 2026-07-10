from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import room_matcher.api as api
from room_matcher.paths import BASELINE_MODEL_PATH, LEGACY_BASELINE_MODEL_PATH


class DummyModel:
    def predict_matches(
        self,
        room_name: str,
        candidate_rooms: list[str],
        *,
        threshold: float | None = None,
    ) -> dict[str, object]:
        return {
            "room_name": room_name,
            "threshold": 0.5 if threshold is None else threshold,
            "matched_rooms": [candidate_rooms[0]],
            "scored_candidates": [
                {"candidate_room": candidate_rooms[0], "score": 0.91},
                {"candidate_room": candidate_rooms[1], "score": 0.12},
                {"candidate_room": candidate_rooms[2], "score": 0.11},
                {"candidate_room": candidate_rooms[3], "score": 0.10},
                {"candidate_room": candidate_rooms[4], "score": 0.09},
            ],
        }


def test_predict_endpoint_returns_matches(monkeypatch) -> None:
    monkeypatch.setattr(api, "get_model_bundle", lambda: (DummyModel(), {}))
    request = api.PredictionRequest(
        room_name="Standard King Room",
        candidate_rooms=[
            "Room, 1 King Bed, Non Smoking",
            "Suite, 2 Queen Beds, Non Smoking",
            "Room, 2 Double Beds",
            "Deluxe Ocean View",
            "Accessible Studio",
        ],
    )

    response = api.predict(request)

    assert response.room_name == "Standard King Room"
    assert response.matched_rooms == ["Room, 1 King Bed, Non Smoking"]
    assert response.scored_candidates[0].score == 0.91


def test_get_model_path_rejects_non_baseline_model_type(monkeypatch) -> None:
    monkeypatch.setenv("ROOM_MATCHER_MODEL_TYPE", "transformer")
    with pytest.raises(ValueError, match="baseline"):
        api.get_model_path()


def test_health_reports_invalid_model_type(monkeypatch) -> None:
    monkeypatch.setenv("ROOM_MATCHER_MODEL_TYPE", "transformer")
    payload = api.health()

    assert payload["status"] == "invalid_model_type"
    assert payload["model_type"] == "transformer"


def test_get_model_bundle_syncs_legacy_baseline_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROOM_MATCHER_MODEL_TYPE", raising=False)
    monkeypatch.delenv("ROOM_MATCHER_MODEL_PATH", raising=False)

    LEGACY_BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_BASELINE_MODEL_PATH.write_text("legacy-model", encoding="utf-8")

    loaded_paths: list[Path] = []

    def fake_load(path: Path) -> tuple[DummyModel, dict[str, object]]:
        loaded_paths.append(path)
        return DummyModel(), {}

    monkeypatch.setattr(api.RoomMatcherModel, "load", staticmethod(fake_load))
    api.get_model_bundle.cache_clear()
    try:
        model, metadata = api.get_model_bundle()
    finally:
        api.get_model_bundle.cache_clear()

    assert isinstance(model, DummyModel)
    assert metadata == {}
    assert BASELINE_MODEL_PATH.exists()
    assert loaded_paths == [BASELINE_MODEL_PATH]


def test_prediction_request_rejects_short_candidate_lists() -> None:
    with pytest.raises(ValidationError):
        api.PredictionRequest(
            room_name="Standard King Room",
            candidate_rooms=[
                "Room A",
                "Room B",
                "Room C",
                "Room D",
            ],
        )
