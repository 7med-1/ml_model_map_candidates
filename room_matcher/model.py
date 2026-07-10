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
from room_matcher.progress import print_status


GENERIC_TOKENS = {
    "room",
    "bed",
    "view",
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
    "size",
}
BED_TYPE_TOKENS = {"king", "queen", "double", "twin", "single", "sofa", "bunk"}
VIEW_TOKENS = {
    "water_view",
    "pool_view",
    "garden_view",
    "city_view",
    "mountain_view",
    "runway_view",
    "bridge_view",
}
ROOM_KIND_TOKENS = {"suite", "studio", "apartment", "bungalow", "villa", "family"}
ROOM_CLASS_TOKENS = {
    "standard",
    "superior",
    "deluxe",
    "premium",
    "executive",
    "classic",
    "luxury",
    "moderate",
}
STRICT_QUERY_ATTRIBUTE_TOKENS = {"balcony", "terrace", "accessible"}
REQUIRED_QUERY_TOKENS = BED_TYPE_TOKENS | VIEW_TOKENS | ROOM_KIND_TOKENS | {
    "accessible",
    "nonsmoking",
    "smoking",
    "balcony",
    "terrace",
}
ATTRIBUTE_TOKENS = {
    "accessible",
    "nonsmoking",
    "smoking",
    "balcony",
    "terrace",
    "suite",
    "studio",
    "deluxe",
    "superior",
    "premium",
    "family",
    "junior",
    "executive",
    "water_view",
    "pool_view",
    "garden_view",
    "city_view",
    "mountain_view",
    "runway_view",
    "bridge_view",
    "room_only",
    "breakfast_included",
    "half_board",
    "full_board",
    "all_inclusive",
    "nonrefundable",
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
class RoomProfile:
    normalized: str
    tokens: list[str]
    token_set: set[str]
    numbers: set[str]
    bed_types: set[str]
    view_types: set[str]
    room_kinds: set[str]
    room_classes: set[str]
    smoking_state: str | None
    bed_counts: dict[str, int]
    total_bed_count: int | None


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
        query_profile = build_room_profile(room_name)
        results = [
            {
                "candidate_room": candidate_room,
                "score": adjust_prediction_score(
                    float(score),
                    query_profile,
                    build_room_profile(candidate_room),
                ),
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
        query_profile = build_room_profile(room_name)
        matched_rooms = [
            item["candidate_room"]
            for item in scored_candidates
            if item["score"] >= active_threshold
            and is_candidate_compatible_for_live_match(
                query_profile,
                build_room_profile(item["candidate_room"]),
            )
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
        expected_numeric_feature_count = self._expected_numeric_feature_count()
        if expected_numeric_feature_count is not None:
            if numeric_features.shape[1] > expected_numeric_feature_count:
                numeric_features = numeric_features[:, :expected_numeric_feature_count]
            elif numeric_features.shape[1] < expected_numeric_feature_count:
                padding = np.zeros(
                    (
                        numeric_features.shape[0],
                        expected_numeric_feature_count - numeric_features.shape[1],
                    ),
                    dtype=np.float64,
                )
                numeric_features = np.hstack([numeric_features, padding])
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

    def _expected_numeric_feature_count(self) -> int | None:
        if not self._is_fitted or not hasattr(self.classifier, "n_features_in_"):
            return None
        expected_count = int(self.classifier.n_features_in_) - int(self.vectorizer.n_features)
        return max(expected_count, 0)

    @staticmethod
    def _numeric_features(room_name: str, candidate_room: str) -> list[float]:
        room_name_normalized = normalize_room_name(room_name)
        candidate_room_normalized = normalize_room_name(candidate_room)
        left_profile = build_room_profile(normalized=room_name_normalized)
        right_profile = build_room_profile(normalized=candidate_room_normalized)
        pair_features = summarize_pair_features(left_profile, right_profile)

        return [
            pair_features["jaccard"],
            pair_features["overlap_left"],
            pair_features["overlap_right"],
            pair_features["number_overlap"],
            pair_features["bed_overlap"],
            pair_features["attribute_match"],
            pair_features["query_specific_coverage"],
            pair_features["required_token_miss_rate"],
            pair_features["bed_count_gap"],
            pair_features["smoking_conflict"],
            pair_features["view_conflict"],
            pair_features["room_kind_conflict"],
            pair_features["room_class_conflict"],
            pair_features["bed_type_conflict"],
            pair_features["bed_count_conflict"],
            1.0 if room_name_normalized in candidate_room_normalized else 0.0,
            1.0 if candidate_room_normalized in room_name_normalized else 0.0,
            abs(len(room_name_normalized) - len(candidate_room_normalized))
            / max(len(room_name_normalized), len(candidate_room_normalized), 1),
        ]


def tokenize_room(value: str) -> list[str]:
    return normalize_room_name(value).split()


def build_room_profile(
    value: str | None = None,
    *,
    normalized: str | None = None,
) -> RoomProfile:
    resolved_normalized = normalized if normalized is not None else normalize_room_name(value or "")
    tokens = resolved_normalized.split()
    token_set = set(tokens)
    bed_counts = extract_bed_counts(tokens)
    return RoomProfile(
        normalized=resolved_normalized,
        tokens=tokens,
        token_set=token_set,
        numbers=extract_numbers(tokens),
        bed_types=token_set & BED_TYPE_TOKENS,
        view_types=token_set & VIEW_TOKENS,
        room_kinds=token_set & ROOM_KIND_TOKENS,
        room_classes=token_set & ROOM_CLASS_TOKENS,
        smoking_state=extract_smoking_state(token_set),
        bed_counts=bed_counts,
        total_bed_count=sum(bed_counts.values()) if bed_counts else None,
    )


def extract_numbers(tokens: list[str]) -> set[str]:
    return {token for token in tokens if token.isdigit()}


def extract_smoking_state(tokens: set[str]) -> str | None:
    if "nonsmoking" in tokens:
        return "nonsmoking"
    if "smoking" in tokens:
        return "smoking"
    return None


def extract_bed_counts(tokens: list[str]) -> dict[str, int]:
    bed_counts: dict[str, int] = {}
    for index, token in enumerate(tokens):
        if token not in BED_TYPE_TOKENS:
            continue

        count = 1
        if index > 0 and tokens[index - 1].isdigit():
            count = int(tokens[index - 1])
        elif index + 1 < len(tokens) and tokens[index + 1].isdigit():
            count = int(tokens[index + 1])
        elif index > 1 and tokens[index - 1] == "bed" and tokens[index - 2].isdigit():
            count = int(tokens[index - 2])
        elif index + 2 < len(tokens) and tokens[index + 1] == "bed" and tokens[index + 2].isdigit():
            count = int(tokens[index + 2])

        bed_counts[token] = max(bed_counts.get(token, 0), count)
    return bed_counts


def summarize_pair_features(
    left_profile: RoomProfile,
    right_profile: RoomProfile,
) -> dict[str, float]:
    shared_tokens = left_profile.token_set & right_profile.token_set
    union_tokens = left_profile.token_set | right_profile.token_set
    shared_numbers = left_profile.numbers & right_profile.numbers
    shared_beds = left_profile.bed_types & right_profile.bed_types
    shared_views = left_profile.view_types & right_profile.view_types
    shared_room_kinds = left_profile.room_kinds & right_profile.room_kinds
    shared_room_classes = left_profile.room_classes & right_profile.room_classes
    query_specific_tokens = left_profile.token_set & REQUIRED_QUERY_TOKENS

    bed_overlap = (
        len(shared_beds) / len(left_profile.bed_types | right_profile.bed_types)
        if (left_profile.bed_types or right_profile.bed_types)
        else 0.0
    )
    number_overlap = (
        len(shared_numbers) / len(left_profile.numbers | right_profile.numbers)
        if (left_profile.numbers or right_profile.numbers)
        else 0.0
    )
    attribute_match = sum(
        1.0
        for attribute in ATTRIBUTE_TOKENS
        if (attribute in left_profile.token_set) == (attribute in right_profile.token_set)
    ) / len(ATTRIBUTE_TOKENS)

    query_specific_coverage = (
        len(query_specific_tokens & right_profile.token_set) / len(query_specific_tokens)
        if query_specific_tokens
        else 0.0
    )
    required_token_miss_rate = (
        len(query_specific_tokens - right_profile.token_set) / len(query_specific_tokens)
        if query_specific_tokens
        else 0.0
    )
    bed_count_gap = 0.0
    bed_count_conflict = 0.0
    if left_profile.total_bed_count and right_profile.total_bed_count:
        bed_count_gap = abs(left_profile.total_bed_count - right_profile.total_bed_count) / max(
            left_profile.total_bed_count,
            right_profile.total_bed_count,
            1,
        )
        bed_count_conflict = float(left_profile.total_bed_count != right_profile.total_bed_count)

    return {
        "jaccard": len(shared_tokens) / len(union_tokens) if union_tokens else 0.0,
        "overlap_left": len(shared_tokens) / len(left_profile.token_set) if left_profile.token_set else 0.0,
        "overlap_right": len(shared_tokens) / len(right_profile.token_set) if right_profile.token_set else 0.0,
        "number_overlap": number_overlap,
        "bed_overlap": bed_overlap,
        "attribute_match": attribute_match,
        "query_specific_coverage": query_specific_coverage,
        "required_token_miss_rate": required_token_miss_rate,
        "shared_view_ratio": (
            len(shared_views) / len(left_profile.view_types | right_profile.view_types)
            if (left_profile.view_types or right_profile.view_types)
            else 0.0
        ),
        "shared_room_kind_ratio": (
            len(shared_room_kinds) / len(left_profile.room_kinds | right_profile.room_kinds)
            if (left_profile.room_kinds or right_profile.room_kinds)
            else 0.0
        ),
        "shared_room_class_ratio": (
            len(shared_room_classes) / len(left_profile.room_classes | right_profile.room_classes)
            if (left_profile.room_classes or right_profile.room_classes)
            else 0.0
        ),
        "smoking_conflict": float(
            left_profile.smoking_state is not None
            and right_profile.smoking_state is not None
            and left_profile.smoking_state != right_profile.smoking_state
        ),
        "view_conflict": float(
            bool(left_profile.view_types)
            and bool(right_profile.view_types)
            and not shared_views
        ),
        "room_kind_conflict": float(
            bool(left_profile.room_kinds)
            and bool(right_profile.room_kinds)
            and not shared_room_kinds
        ),
        "room_class_conflict": float(
            bool(left_profile.room_classes)
            and bool(right_profile.room_classes)
            and not shared_room_classes
        ),
        "bed_type_conflict": float(
            bool(left_profile.bed_types)
            and bool(right_profile.bed_types)
            and not shared_beds
        ),
        "bed_count_gap": bed_count_gap,
        "bed_count_conflict": bed_count_conflict,
    }


def informative_tokens(value: str) -> list[str]:
    return [
        token
        for token in tokenize_room(value)
        if token not in GENERIC_TOKENS and not token.isdigit()
    ]


def hard_negative_priority(
    query_profile: RoomProfile,
    candidate_profile: RoomProfile,
) -> tuple[float, int]:
    pair_features = summarize_pair_features(query_profile, candidate_profile)
    conflict_count = int(pair_features["smoking_conflict"]) + int(pair_features["view_conflict"]) + int(
        pair_features["room_kind_conflict"]
    ) + int(pair_features["room_class_conflict"]) + int(pair_features["bed_type_conflict"]) + int(
        pair_features["bed_count_conflict"]
    )
    similarity = (
        0.35 * pair_features["overlap_left"]
        + 0.2 * pair_features["overlap_right"]
        + 0.15 * pair_features["jaccard"]
        + 0.1 * pair_features["bed_overlap"]
        + 0.1 * pair_features["query_specific_coverage"]
        + 0.05 * pair_features["shared_view_ratio"]
        + 0.05 * pair_features["shared_room_kind_ratio"]
        + 0.05 * pair_features["shared_room_class_ratio"]
    )
    priority = (
        similarity
        + 0.4 * conflict_count
        + 0.25 * pair_features["required_token_miss_rate"]
        + 0.15 * pair_features["bed_count_gap"]
    )
    return priority, conflict_count


def adjust_prediction_score(
    base_score: float,
    query_profile: RoomProfile,
    candidate_profile: RoomProfile,
) -> float:
    pair_features = summarize_pair_features(query_profile, candidate_profile)
    coverage = pair_features["query_specific_coverage"]
    if coverage > 0:
        adjusted_score = base_score * (coverage**3)
    else:
        adjusted_score = base_score

    if pair_features["required_token_miss_rate"] > 0:
        adjusted_score *= max(0.05, 1.0 - pair_features["required_token_miss_rate"]) ** 2
    if pair_features["smoking_conflict"]:
        adjusted_score *= 0.05
    if pair_features["view_conflict"]:
        adjusted_score *= 0.1
    if pair_features["room_kind_conflict"]:
        adjusted_score *= 0.05
    if pair_features["room_class_conflict"]:
        adjusted_score *= 0.05
    if pair_features["bed_type_conflict"]:
        adjusted_score *= 0.02
    if pair_features["bed_count_conflict"]:
        adjusted_score *= 0.2

    return float(max(0.0, min(1.0, adjusted_score)))


def query_required_match_tokens(query_profile: RoomProfile) -> set[str]:
    required_tokens = set(query_profile.bed_types)
    required_tokens.update(query_profile.view_types)
    required_tokens.update(query_profile.room_kinds)
    required_tokens.update(query_profile.token_set & STRICT_QUERY_ATTRIBUTE_TOKENS)
    if query_profile.smoking_state:
        required_tokens.add(query_profile.smoking_state)
    return required_tokens


def is_candidate_compatible_for_live_match(
    query_profile: RoomProfile,
    candidate_profile: RoomProfile,
) -> bool:
    pair_features = summarize_pair_features(query_profile, candidate_profile)
    if pair_features["bed_type_conflict"]:
        return False
    if pair_features["view_conflict"]:
        return False
    if pair_features["room_kind_conflict"]:
        return False
    if pair_features["room_class_conflict"]:
        return False
    if pair_features["smoking_conflict"]:
        return False
    if pair_features["bed_count_conflict"]:
        return False

    required_tokens = query_required_match_tokens(query_profile)
    if "balcony" in required_tokens and "balcony" not in candidate_profile.token_set:
        return False
    if "terrace" in required_tokens and "terrace" not in candidate_profile.token_set:
        return False
    if "accessible" in required_tokens and "accessible" not in candidate_profile.token_set:
        return False
    if query_profile.bed_types and not (query_profile.bed_types & candidate_profile.bed_types):
        return False
    if query_profile.view_types and not (query_profile.view_types & candidate_profile.view_types):
        return False
    if query_profile.room_kinds and not (query_profile.room_kinds & candidate_profile.room_kinds):
        return False

    return True


def conflict_lookup_tokens(query_profile: RoomProfile) -> set[str]:
    lookup_tokens = set(query_profile.bed_types)
    lookup_tokens.update(query_profile.view_types)
    lookup_tokens.update(query_profile.room_kinds)

    if query_profile.bed_types:
        lookup_tokens.update(BED_TYPE_TOKENS - query_profile.bed_types)
    if query_profile.view_types:
        lookup_tokens.update(VIEW_TOKENS - query_profile.view_types)
    if query_profile.room_kinds:
        lookup_tokens.update(ROOM_KIND_TOKENS - query_profile.room_kinds)
    if query_profile.smoking_state == "nonsmoking":
        lookup_tokens.add("smoking")
    elif query_profile.smoking_state == "smoking":
        lookup_tokens.add("nonsmoking")

    return lookup_tokens


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


def select_token_window(
    candidates: list[CandidateRecord],
    *,
    seed_text: str,
    limit: int,
) -> list[CandidateRecord]:
    if len(candidates) <= limit:
        return candidates

    start = stable_bucket(seed_text, bucket_count=len(candidates))
    return [candidates[(start + offset) % len(candidates)] for offset in range(limit)]


def sample_negative_candidates(
    room_name: str,
    positive_candidates: set[str],
    provider_pool: list[CandidateRecord],
    token_index: dict[str, list[CandidateRecord]],
    rng: random.Random,
    count: int,
    per_token_limit: int = 80,
) -> list[CandidateRecord]:
    if count <= 0:
        return []

    selected: list[CandidateRecord] = []
    seen_candidates: set[str] = set()
    query_profile = build_room_profile(normalized=room_name)

    hard_pool: list[CandidateRecord] = []
    lookup_tokens = list(dict.fromkeys(informative_tokens(room_name) + sorted(conflict_lookup_tokens(query_profile))))
    for token in lookup_tokens:
        hard_pool.extend(
            select_token_window(
                token_index.get(token, []),
                seed_text=f"{room_name}|||{token}",
                limit=per_token_limit,
            )
        )

    prioritized_conflicts: dict[str, tuple[float, CandidateRecord]] = {}
    prioritized_similar: dict[str, tuple[float, CandidateRecord]] = {}
    for record in hard_pool:
        if record.candidate_room in positive_candidates or record.candidate_room in seen_candidates:
            continue

        candidate_profile = build_room_profile(normalized=record.candidate_room_normalized)
        priority, conflict_count = hard_negative_priority(query_profile, candidate_profile)
        if conflict_count > 0:
            current = prioritized_conflicts.get(record.candidate_room)
            if current is None or priority > current[0]:
                prioritized_conflicts[record.candidate_room] = (priority, record)
        elif priority > 0:
            current = prioritized_similar.get(record.candidate_room)
            if current is None or priority > current[0]:
                prioritized_similar[record.candidate_room] = (priority, record)

    for _priority, record in sorted(prioritized_conflicts.values(), key=lambda item: item[0], reverse=True):
        if record.candidate_room in seen_candidates:
            continue
        seen_candidates.add(record.candidate_room)
        selected.append(record)
        if len(selected) >= count:
            return selected

    for _priority, record in sorted(prioritized_similar.values(), key=lambda item: item[0], reverse=True):
        if record.candidate_room in seen_candidates:
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
    progress_every_pairs: int = 50_000,
) -> dict[str, int]:
    total_positive_samples = 0
    total_negative_samples = 0

    for epoch in range(epochs):
        print_status(
            f"Training baseline epoch {epoch + 1}/{epochs} "
            f"on {len(train_pairs):,} positive pairs"
        )
        rng = random.Random(random_seed + epoch)
        shuffled_pairs = list(train_pairs)
        rng.shuffle(shuffled_pairs)

        room_names: list[str] = []
        candidate_rooms: list[str] = []
        labels: list[int] = []
        weights: list[float] = []
        epoch_positive_samples = 0
        epoch_negative_samples = 0

        for pair_index, pair in enumerate(shuffled_pairs, start=1):
            weight = pair_weight(pair.pair_count)

            room_names.append(pair.room_name)
            candidate_rooms.append(pair.candidate_room)
            labels.append(1)
            weights.append(weight)
            total_positive_samples += 1
            epoch_positive_samples += 1

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
                epoch_negative_samples += 1

            if len(labels) >= batch_size:
                model.fit_batch(room_names, candidate_rooms, labels, weights)
                room_names.clear()
                candidate_rooms.clear()
                labels.clear()
                weights.clear()

            if progress_every_pairs and pair_index % progress_every_pairs == 0:
                print_status(
                    f"Epoch {epoch + 1}/{epochs}: processed {pair_index:,} / {len(shuffled_pairs):,} pairs"
                )

        if labels:
            model.fit_batch(room_names, candidate_rooms, labels, weights)

        print_status(
            f"Finished epoch {epoch + 1}/{epochs}: "
            f"{epoch_positive_samples:,} positive samples, "
            f"{epoch_negative_samples:,} negative samples"
        )

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
