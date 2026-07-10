from __future__ import annotations

import random

from room_matcher.cleaning import normalize_room_name
from room_matcher.model import (
    CandidateRecord,
    ScoredScenario,
    adjust_prediction_score,
    build_room_profile,
    build_token_index,
    is_candidate_compatible_for_live_match,
    evaluate_scored_scenarios,
    sample_negative_candidates,
    summarize_pair_features,
    tune_threshold,
)


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


def test_summarize_pair_features_flags_structural_conflicts() -> None:
    query_profile = build_room_profile(normalized=normalize_room_name("STDSD QN RM MEERBLICK NSMK"))
    candidate_profile = build_room_profile(normalized=normalize_room_name("STD KG RM POOLVW SMK"))

    features = summarize_pair_features(query_profile, candidate_profile)

    assert features["bed_type_conflict"] == 1.0
    assert features["view_conflict"] == 1.0
    assert features["smoking_conflict"] == 1.0
    assert features["bed_count_conflict"] == 0.0


def test_summarize_pair_features_flags_room_class_conflict() -> None:
    query_profile = build_room_profile(normalized=normalize_room_name("Standard Queen Room"))
    candidate_profile = build_room_profile(normalized=normalize_room_name("Superior Queen Room"))

    features = summarize_pair_features(query_profile, candidate_profile)

    assert features["room_class_conflict"] == 1.0


def test_sample_negative_candidates_prefers_hard_conflicts() -> None:
    provider_pool = [
        CandidateRecord(
            candidate_room="Standard Queen Room with Sea View Non Smoking",
            candidate_room_normalized=normalize_room_name("Standard Queen Room with Sea View Non Smoking"),
        ),
        CandidateRecord(
            candidate_room="Standard King Room with Sea View Non Smoking",
            candidate_room_normalized=normalize_room_name("Standard King Room with Sea View Non Smoking"),
        ),
        CandidateRecord(
            candidate_room="Standard Queen Room with Pool View Non Smoking",
            candidate_room_normalized=normalize_room_name("Standard Queen Room with Pool View Non Smoking"),
        ),
        CandidateRecord(
            candidate_room="Standard Double Room with City View Non Smoking",
            candidate_room_normalized=normalize_room_name("Standard Double Room with City View Non Smoking"),
        ),
        CandidateRecord(
            candidate_room="Apartment, 2 Bedrooms, Capacity 6",
            candidate_room_normalized=normalize_room_name("Apartment, 2 Bedrooms, Capacity 6"),
        ),
    ]
    token_index = build_token_index(provider_pool)

    negatives = sample_negative_candidates(
        normalize_room_name("STDSD QN RM MEERBLICK NSMK"),
        positive_candidates={"Standard Queen Room with Sea View Non Smoking"},
        provider_pool=provider_pool,
        token_index=token_index,
        rng=random.Random(42),
        count=3,
    )

    negative_rooms = {record.candidate_room for record in negatives}
    assert "Standard King Room with Sea View Non Smoking" in negative_rooms
    assert "Standard Queen Room with Pool View Non Smoking" in negative_rooms
    assert "Standard Double Room with City View Non Smoking" in negative_rooms


def test_adjust_prediction_score_penalizes_explicit_conflicts() -> None:
    query_profile = build_room_profile(normalized=normalize_room_name("Queen Bed Room with Balcony"))
    good_candidate = build_room_profile(
        normalized=normalize_room_name("Superior Room, Queen Bed, Balcony, Non Smoking")
    )
    bad_candidate = build_room_profile(
        normalized=normalize_room_name("Standard Room, 1 Queen Bed, Smoking")
    )

    good_score = adjust_prediction_score(0.95, query_profile, good_candidate)
    bad_score = adjust_prediction_score(0.95, query_profile, bad_candidate)

    assert good_score > 0.8
    assert bad_score < 0.2


def test_live_match_compatibility_blocks_explicit_conflicts() -> None:
    query_profile = build_room_profile(normalized=normalize_room_name("STDSD QN RM MEERBLICK NSMK"))

    assert is_candidate_compatible_for_live_match(
        query_profile,
        build_room_profile(normalized=normalize_room_name("Standard Queen Room with Sea View Non Smoking")),
    )
    assert not is_candidate_compatible_for_live_match(
        query_profile,
        build_room_profile(normalized=normalize_room_name("STD KG RM MEERBLICK NSMK")),
    )
    assert not is_candidate_compatible_for_live_match(
        query_profile,
        build_room_profile(normalized=normalize_room_name("STD QN RM POOLVW NSMK")),
    )
    assert not is_candidate_compatible_for_live_match(
        query_profile,
        build_room_profile(normalized=normalize_room_name("appt 2 qn rm bayview nsmk")),
    )
    assert not is_candidate_compatible_for_live_match(
        build_room_profile(normalized=normalize_room_name("Standard Queen Room")),
        build_room_profile(normalized=normalize_room_name("Superior Queen Room")),
    )
