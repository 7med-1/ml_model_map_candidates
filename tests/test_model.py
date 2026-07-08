from __future__ import annotations

from room_matcher.model import ScoredScenario, evaluate_scored_scenarios, tune_threshold


def test_evaluate_scored_scenarios_counts_exact_matches() -> None:
    scenarios = [
        ScoredScenario(
            room_name="Standard King Room",
            actual_matches=["Room, 1 King Bed, Non Smoking"],
            scored_candidates=[
                {"candidate_room": "Room, 1 King Bed, Non Smoking", "score": 0.93},
                {"candidate_room": "Suite, 2 Queen Beds, Non Smoking", "score": 0.10},
            ],
        )
    ]

    metrics = evaluate_scored_scenarios(scenarios, threshold=0.5)

    assert metrics["scenario_count"] == 1
    assert metrics["exact_match_rate"] == 1.0
    assert metrics["f1_mean"] == 1.0


def test_tune_threshold_picks_best_f1() -> None:
    scenarios = [
        ScoredScenario(
            room_name="Standard Queen Room",
            actual_matches=["Room, 1 Queen Bed, Non Smoking"],
            scored_candidates=[
                {"candidate_room": "Room, 1 Queen Bed, Non Smoking", "score": 0.80},
                {"candidate_room": "Room, 1 Queen Bed, Smoking", "score": 0.45},
            ],
        )
    ]

    threshold, _results = tune_threshold(scenarios, candidate_thresholds=[0.4, 0.5, 0.7])

    assert threshold in {0.5, 0.7}
