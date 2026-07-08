from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from room_matcher.cleaning import normalize_room_name


GENERIC_TOKENS = {
    "room",
    "bed",
    "with",
    "and",
    "the",
    "for",
    "non",
    "smoking",
    "nonsmoking",
    "smoke",
    "accessible",
    "standard",
}
BED_TYPE_TOKENS = {"king", "queen", "double", "twin", "single", "sofa", "bunk"}
ATTRIBUTE_TOKENS = {
    "accessible",
    "nonsmoking",
    "smoking",
    "suite",
    "studio",
    "deluxe",
    "superior",
    "premium",
    "family",
    "junior",
    "executive",
}


@dataclass(slots=True, frozen=True)
class RoomPair:
    room_name: str
    candidate_room: str
    room_name_normalized: str
    candidate_room_normalized: str
    pair_count: int


@dataclass(slots=True, frozen=True)
class CandidateRecord:
    candidate_room: str
    candidate_room_normalized: str


@dataclass(slots=True)
class CandidateScenario:
    room_name: str
    room_name_normalized: str
    actual_matches: list[str]
    candidates: list[CandidateRecord]


@dataclass(slots=True)
class ScoredScenario:
    room_name: str
    actual_matches: list[str]
    scored_candidates: list[dict[str, float | str]]


class RoomMatcherModel:
    def __init__(self, *, threshold: float = 0.5) -> None:
        self.vectorizer = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            n_features=2**18,
            alternate_sign=False,
            norm="l2",
        )
        self.classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            random_state=42,
        )
        self.threshold = threshold
        self._is_fitted = False

    def fit_batch(
        self,
        room_names: list[str],
        candidate_rooms: list[str],
        labels: list[int],
        sample_weights: list[float],
    ) -> None:
        matrix = self._build_feature_matrix(room_names, candidate_rooms)
        label_array = np.asarray(labels, dtype=np.int64)
        weight_array = np.asarray(sample_weights, dtype=np.float64)
        if not self._is_fitted:
            self.classifier.partial_fit(
                matrix,
                label_array,
                classes=np.asarray([0, 1], dtype=np.int64),
                sample_weight=weight_array,
            )
            self._is_fitted = True
            return

        self.classifier.partial_fit(matrix, label_array, sample_weight=weight_array)

    def predict_scores(
        self,
        room_name: str,
        candidate_rooms: list[str],
    ) -> list[dict[str, float | str]]:
        if not self._is_fitted:
            raise RuntimeError("Model must be trained before prediction.")

        room_names = [room_name] * len(candidate_rooms)
        matrix = self._build_feature_matrix(room_names, candidate_rooms)
        probabilities = self.classifier.predict_proba(matrix)[:, 1]
        results = [
            {
                "candidate_room": candidate_room,
                "score": float(score),
            }
            for candidate_room, score in zip(candidate_rooms, probabilities, strict=True)
        ]
        return sorted(results, key=lambda row: row["score"], reverse=True)

    def predict_matches(
        self,
        room_name: str,
        candidate_rooms: list[str],
        *,
        threshold: float | None = None,
    ) -> dict[str, object]:
        active_threshold = self.threshold if threshold is None else threshold
        scored_candidates = self.predict_scores(room_name, candidate_rooms)
        matched_rooms = [
            item["candidate_room"]
            for item in scored_candidates
            if item["score"] >= active_threshold
        ]
        return {
            "room_name": room_name,
            "threshold": active_threshold,
            "matched_rooms": matched_rooms,
            "scored_candidates": scored_candidates,
        }

    def save(self, artifact_path: str | Path, metadata: dict[str, object]) -> None:
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self,
                "metadata": metadata,
            },
            artifact_path,
        )

    @staticmethod
    def load(artifact_path: str | Path) -> tuple["RoomMatcherModel", dict[str, object]]:
        payload = joblib.load(artifact_path)
        return payload["model"], payload["metadata"]

    def _build_feature_matrix(
        self,
        room_names: list[str],
        candidate_rooms: list[str],
    ):
        pair_texts = [
            self._build_pair_text(room_name, candidate_room)
            for room_name, candidate_room in zip(room_names, candidate_rooms, strict=True)
        ]
        text_matrix = self.vectorizer.transform(pair_texts)
        numeric_features = np.asarray(
            [
                self._numeric_features(room_name, candidate_room)
                for room_name, candidate_room in zip(room_names, candidate_rooms, strict=True)
            ],
            dtype=np.float64,
        )
        numeric_matrix = csr_matrix(numeric_features)
        return hstack([text_matrix, numeric_matrix], format="csr")

    @staticmethod
    def _build_pair_text(room_name: str, candidate_room: str) -> str:
        room_name_normalized = normalize_room_name(room_name)
        candidate_room_normalized = normalize_room_name(candidate_room)
        return (
            f"query {room_name_normalized} "
            f"candidate {candidate_room_normalized} "
            f"cross {room_name_normalized} {candidate_room_normalized}"
        )

    @staticmethod
    def _numeric_features(room_name: str, candidate_room: str) -> list[float]:
        room_name_normalized = normalize_room_name(room_name)
        candidate_room_normalized = normalize_room_name(candidate_room)
        left_tokens = room_name_normalized.split()
        right_tokens = candidate_room_normalized.split()
        left_set = set(left_tokens)
        right_set = set(right_tokens)
        shared_tokens = left_set & right_set
        union_tokens = left_set | right_set
        left_numbers = extract_numbers(left_tokens)
        right_numbers = extract_numbers(right_tokens)
        left_beds = left_set & BED_TYPE_TOKENS
        right_beds = right_set & BED_TYPE_TOKENS

        jaccard = len(shared_tokens) / len(union_tokens) if union_tokens else 0.0
        overlap_left = len(shared_tokens) / len(left_set) if left_set else 0.0
        overlap_right = len(shared_tokens) / len(right_set) if right_set else 0.0
        shared_numbers = left_numbers & right_numbers
        number_overlap = len(shared_numbers) / len(left_numbers | right_numbers) if (left_numbers or right_numbers) else 0.0
        bed_overlap = len(left_beds & right_beds) / len(left_beds | right_beds) if (left_beds or right_beds) else 0.0
        attribute_match = sum(
            1.0
            for attribute in ATTRIBUTE_TOKENS
            if (attribute in left_set) == (attribute in right_set)
        ) / len(ATTRIBUTE_TOKENS)

        return [
            jaccard,
            overlap_left,
            overlap_right,
            number_overlap,
            bed_overlap,
            attribute_match,
            1.0 if room_name_normalized in candidate_room_normalized else 0.0,
            1.0 if candidate_room_normalized in room_name_normalized else 0.0,
            abs(len(room_name_normalized) - len(candidate_room_normalized))
            / max(len(room_name_normalized), len(candidate_room_normalized), 1),
        ]


def tokenize_room(value: str) -> list[str]:
    return normalize_room_name(value).split()


def extract_numbers(tokens: list[str]) -> set[str]:
    return {token for token in tokens if token.isdigit()}


def informative_tokens(value: str) -> list[str]:
    return [
        token
        for token in tokenize_room(value)
        if token not in GENERIC_TOKENS and not token.isdigit()
    ]


def pair_weight(pair_count: int) -> float:
    return 1.0 + math.log1p(pair_count)


def stable_bucket(value: str, bucket_count: int = 100) -> int:
    digest = hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16) % bucket_count


def stable_fraction(value: str) -> float:
    digest = hashlib.md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def load_positive_pairs(
    cleaned_csv_path: str | Path,
    *,
    unique_pair_count: int | None = None,
    max_positive_pairs: int | None = 250_000,
    drop_ambiguous: bool = True,
) -> list[RoomPair]:
    sampling_rate = 1.0
    if unique_pair_count and max_positive_pairs and max_positive_pairs < unique_pair_count:
        sampling_rate = max_positive_pairs / unique_pair_count

    pairs: list[RoomPair] = []
    with Path(cleaned_csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if drop_ambiguous and int(row["provider_room_ambiguity"]) > 1:
                continue

            key = f"{row['room_name_normalized']}|||{row['candidate_room_normalized']}"
            if sampling_rate < 1.0 and stable_fraction(key) > sampling_rate:
                continue

            pairs.append(
                RoomPair(
                    room_name=row["room_name"],
                    candidate_room=row["candidate_room"],
                    room_name_normalized=row["room_name_normalized"],
                    candidate_room_normalized=row["candidate_room_normalized"],
                    pair_count=int(row["pair_count"]),
                )
            )
    return pairs


def split_pairs(
    pairs: list[RoomPair],
    *,
    train_cutoff: int = 80,
    validation_cutoff: int = 90,
) -> dict[str, list[RoomPair]]:
    splits = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for pair in pairs:
        key = f"{pair.room_name_normalized}|||{pair.candidate_room_normalized}"
        bucket = stable_bucket(key)
        if bucket < train_cutoff:
            splits["train"].append(pair)
        elif bucket < validation_cutoff:
            splits["validation"].append(pair)
        else:
            splits["test"].append(pair)
    return splits


def build_provider_pool(pairs: list[RoomPair]) -> list[CandidateRecord]:
    seen: dict[str, CandidateRecord] = {}
    for pair in pairs:
        seen.setdefault(
            pair.candidate_room,
            CandidateRecord(
                candidate_room=pair.candidate_room,
                candidate_room_normalized=pair.candidate_room_normalized,
            ),
        )
    return list(seen.values())


def build_positive_lookup(
    pairs: list[RoomPair],
) -> dict[str, set[str]]:
    positives: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        positives[pair.room_name].add(pair.candidate_room)
    return positives


def build_token_index(
    provider_pool: list[CandidateRecord],
) -> dict[str, list[CandidateRecord]]:
    token_index: dict[str, list[CandidateRecord]] = defaultdict(list)
    for record in provider_pool:
        for token in informative_tokens(record.candidate_room_normalized):
            token_index[token].append(record)
    return token_index


def sample_negative_candidates(
    room_name: str,
    positive_candidates: set[str],
    provider_pool: list[CandidateRecord],
    token_index: dict[str, list[CandidateRecord]],
    rng: random.Random,
    count: int,
) -> list[CandidateRecord]:
    if count <= 0:
        return []

    selected: list[CandidateRecord] = []
    seen_candidates: set[str] = set()

    hard_pool: list[CandidateRecord] = []
    for token in informative_tokens(room_name):
        hard_pool.extend(token_index.get(token, []))

    rng.shuffle(hard_pool)
    for record in hard_pool:
        if record.candidate_room in positive_candidates or record.candidate_room in seen_candidates:
            continue
        seen_candidates.add(record.candidate_room)
        selected.append(record)
        if len(selected) >= count:
            return selected

    if not provider_pool:
        return selected

    max_attempts = max(len(provider_pool) * 3, count * 10)
    attempts = 0
    while len(selected) < count:
        attempts += 1
        if attempts > max_attempts:
            break
        record = provider_pool[rng.randrange(len(provider_pool))]
        if record.candidate_room in positive_candidates or record.candidate_room in seen_candidates:
            continue
        seen_candidates.add(record.candidate_room)
        selected.append(record)

    return selected


def train_pairwise_model(
    model: RoomMatcherModel,
    train_pairs: list[RoomPair],
    *,
    provider_pool: list[CandidateRecord],
    positive_lookup: dict[str, set[str]],
    token_index: dict[str, list[CandidateRecord]],
    negatives_per_positive: int = 2,
    epochs: int = 3,
    batch_size: int = 2_048,
    random_seed: int = 42,
) -> dict[str, int]:
    total_positive_samples = 0
    total_negative_samples = 0

    for epoch in range(epochs):
        rng = random.Random(random_seed + epoch)
        shuffled_pairs = list(train_pairs)
        rng.shuffle(shuffled_pairs)

        room_names: list[str] = []
        candidate_rooms: list[str] = []
        labels: list[int] = []
        weights: list[float] = []

        for pair in shuffled_pairs:
            weight = pair_weight(pair.pair_count)

            room_names.append(pair.room_name)
            candidate_rooms.append(pair.candidate_room)
            labels.append(1)
            weights.append(weight)
            total_positive_samples += 1

            negatives = sample_negative_candidates(
                pair.room_name_normalized,
                positive_lookup[pair.room_name],
                provider_pool,
                token_index,
                rng,
                negatives_per_positive,
            )
            for negative in negatives:
                room_names.append(pair.room_name)
                candidate_rooms.append(negative.candidate_room)
                labels.append(0)
                weights.append(weight)
                total_negative_samples += 1

            if len(labels) >= batch_size:
                model.fit_batch(room_names, candidate_rooms, labels, weights)
                room_names.clear()
                candidate_rooms.clear()
                labels.clear()
                weights.clear()

        if labels:
            model.fit_batch(room_names, candidate_rooms, labels, weights)

    return {
        "positive_samples": total_positive_samples,
        "negative_samples": total_negative_samples,
    }


def build_candidate_scenarios(
    pairs: list[RoomPair],
    *,
    provider_pool: list[CandidateRecord],
    positive_lookup: dict[str, set[str]],
    token_index: dict[str, list[CandidateRecord]],
    min_candidates: int = 5,
    max_candidates: int = 20,
    max_true_candidates: int = 3,
    max_scenarios: int = 1_500,
    random_seed: int = 42,
) -> list[CandidateScenario]:
    grouped: dict[str, list[RoomPair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.room_name].append(pair)

    rng = random.Random(random_seed)
    room_names = list(grouped)
    rng.shuffle(room_names)

    scenarios: list[CandidateScenario] = []
    for room_name in room_names:
        positives = grouped[room_name]
        if not positives:
            continue

        positive_candidates = list({pair.candidate_room: pair for pair in positives}.values())
        rng.shuffle(positive_candidates)
        true_candidate_count = min(
            len(positive_candidates),
            max_true_candidates,
            rng.randint(1, max_true_candidates),
        )
        selected_truth = positive_candidates[:true_candidate_count]
        scenario_size = rng.randint(min_candidates, max_candidates)
        negative_count = max(0, scenario_size - len(selected_truth))
        negatives = sample_negative_candidates(
            positives[0].room_name_normalized,
            positive_lookup[room_name],
            provider_pool,
            token_index,
            rng,
            negative_count,
        )

        candidates = [
            CandidateRecord(
                candidate_room=pair.candidate_room,
                candidate_room_normalized=pair.candidate_room_normalized,
            )
            for pair in selected_truth
        ]
        candidates.extend(negatives)
        rng.shuffle(candidates)

        scenarios.append(
            CandidateScenario(
                room_name=room_name,
                room_name_normalized=positives[0].room_name_normalized,
                actual_matches=[pair.candidate_room for pair in selected_truth],
                candidates=candidates,
            )
        )
        if len(scenarios) >= max_scenarios:
            break

    return scenarios


def score_scenarios(
    model: RoomMatcherModel,
    scenarios: list[CandidateScenario],
) -> list[ScoredScenario]:
    scored: list[ScoredScenario] = []
    for scenario in scenarios:
        candidate_rooms = [candidate.candidate_room for candidate in scenario.candidates]
        scored_candidates = model.predict_scores(scenario.room_name, candidate_rooms)
        scored.append(
            ScoredScenario(
                room_name=scenario.room_name,
                actual_matches=scenario.actual_matches,
                scored_candidates=scored_candidates,
            )
        )
    return scored


def evaluate_scored_scenarios(
    scored_scenarios: list[ScoredScenario],
    *,
    threshold: float,
) -> dict[str, float | int]:
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    exact_match_count = 0
    true_positive_total = 0
    false_positive_total = 0
    false_negative_total = 0

    for scenario in scored_scenarios:
        predicted = {
            item["candidate_room"]
            for item in scenario.scored_candidates
            if item["score"] >= threshold
        }
        actual = set(scenario.actual_matches)
        true_positive = len(predicted & actual)
        false_positive = len(predicted - actual)
        false_negative = len(actual - predicted)

        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(actual) if actual else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        exact_match_count += int(predicted == actual)
        true_positive_total += true_positive
        false_positive_total += false_positive
        false_negative_total += false_negative

    scenario_count = len(scored_scenarios)
    return {
        "threshold": threshold,
        "scenario_count": scenario_count,
        "precision_mean": float(np.mean(precision_values)) if precision_values else 0.0,
        "recall_mean": float(np.mean(recall_values)) if recall_values else 0.0,
        "f1_mean": float(np.mean(f1_values)) if f1_values else 0.0,
        "exact_match_rate": exact_match_count / scenario_count if scenario_count else 0.0,
        "true_positive_total": true_positive_total,
        "false_positive_total": false_positive_total,
        "false_negative_total": false_negative_total,
    }


def tune_threshold(
    scored_scenarios: list[ScoredScenario],
    *,
    candidate_thresholds: list[float] | None = None,
) -> tuple[float, list[dict[str, float | int]]]:
    thresholds = candidate_thresholds or [
        round(float(step), 2) for step in np.arange(0.2, 0.86, 0.05)
    ]
    results = [
        evaluate_scored_scenarios(scored_scenarios, threshold=threshold)
        for threshold in thresholds
    ]
    best_result = max(
        results,
        key=lambda result: (
            result["f1_mean"],
            result["precision_mean"],
            result["recall_mean"],
        ),
    )
    return float(best_result["threshold"]), results


def sample_prediction_rows(
    scored_scenarios: list[ScoredScenario],
    *,
    threshold: float,
    limit: int = 50,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scored_scenarios[:limit]:
        rows.append(
            {
                "room_name": scenario.room_name,
                "actual_matches": scenario.actual_matches,
                "predicted_matches": [
                    item["candidate_room"]
                    for item in scenario.scored_candidates
                    if item["score"] >= threshold
                ],
                "scored_candidates": scenario.scored_candidates,
            }
        )
    return rows


def write_json_report(path: str | Path, payload: dict[str, object] | list[object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
