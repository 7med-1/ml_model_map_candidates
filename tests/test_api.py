from __future__ import annotations

import pytest
from pydantic import ValidationError

import room_matcher.api as api


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


def test_get_model_path_uses_hf_default(monkeypatch) -> None:
    monkeypatch.setenv("ROOM_MATCHER_MODEL_TYPE", "hf")
    assert str(api.get_model_path()) == "artifacts/hf/room_matcher"


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
