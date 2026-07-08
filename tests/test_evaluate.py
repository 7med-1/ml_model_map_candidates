from __future__ import annotations

from room_matcher.evaluate import OverlapRoomMatcher


def test_overlap_room_matcher_prefers_better_match() -> None:
    model = OverlapRoomMatcher()
    results = model.predict_scores(
        "Standard King Room",
        [
            "Suite, 2 Queen Beds, Non Smoking",
            "Room, 1 King Bed, Non Smoking",
        ],
    )

    assert results[0]["candidate_room"] == "Room, 1 King Bed, Non Smoking"
