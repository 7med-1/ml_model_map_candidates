from __future__ import annotations

import argparse
from pathlib import Path

from room_matcher.cleaning import clean_room_matching_csv, load_cleaning_stats, normalize_room_name
from room_matcher.model import (
    BED_TYPE_TOKENS,
    ATTRIBUTE_TOKENS,
    RoomMatcherModel,
    build_candidate_scenarios,
    build_positive_lookup,
    build_provider_pool,
    build_token_index,
    evaluate_scored_scenarios,
    extract_numbers,
    load_positive_pairs,
    sample_prediction_rows,
    score_scenarios,
    split_pairs,
    tune_threshold,
    write_json_report,
)
from room_matcher.paths import (
    BASELINE_ARTIFACTS_DIR,
    BASELINE_CLEANING_REPORT_PATH,
    BASELINE_CLEAN_CSV_PATH,
    BASELINE_MODEL_PATH,
    BASELINE_SQLITE_PATH,
    REPORTS_ROOT,
    sync_legacy_baseline_artifacts,
)
from room_matcher.progress import print_status


class OverlapRoomMatcher:
    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def predict_scores(
        self,
        room_name: str,
        candidate_rooms: list[str],
    ) -> list[dict[str, float | str]]:
        left_normalized = normalize_room_name(room_name)
        left_tokens = left_normalized.split()
        left_set = set(left_tokens)
        left_numbers = extract_numbers(left_tokens)
        left_beds = left_set & BED_TYPE_TOKENS

        scored_candidates: list[dict[str, float | str]] = []
        for candidate_room in candidate_rooms:
            right_normalized = normalize_room_name(candidate_room)
            right_tokens = right_normalized.split()
            right_set = set(right_tokens)
            right_numbers = extract_numbers(right_tokens)
            right_beds = right_set & BED_TYPE_TOKENS

            shared = left_set & right_set
            union = left_set | right_set
            jaccard = len(shared) / len(union) if union else 0.0
            number_overlap = (
                len(left_numbers & right_numbers) / len(left_numbers | right_numbers)
                if (left_numbers or right_numbers)
                else 0.0
            )
            bed_overlap = (
                len(left_beds & right_beds) / len(left_beds | right_beds)
                if (left_beds or right_beds)
                else 0.0
            )
            attribute_overlap = (
                sum(
                    1.0
                    for attribute in ATTRIBUTE_TOKENS
                    if (attribute in left_set) and (attribute in right_set)
                )
                / len(ATTRIBUTE_TOKENS)
            )
            containment = max(
                float(left_normalized in right_normalized),
                float(right_normalized in left_normalized),
            )
            score = (
                0.45 * jaccard
                + 0.2 * number_overlap
                + 0.2 * bed_overlap
                + 0.1 * containment
                + 0.05 * attribute_overlap
            )
            scored_candidates.append(
                {
                    "candidate_room": candidate_room,
                    "score": float(score),
                }
            )

        return sorted(scored_candidates, key=lambda item: item["score"], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the baseline heuristic or a trained room matcher model.",
    )
    parser.add_argument("--input-csv", default="room_matching.csv")
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--reports-dir", default=str(REPORTS_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-clean-rows", type=int, default=None)
    parser.add_argument("--max-positive-pairs", type=int, default=250_000)
    parser.add_argument("--min-candidates", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-true-candidates", type=int, default=3)
    parser.add_argument("--max-eval-scenarios", type=int, default=1_500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--keep-ambiguous", action="store_true")
    parser.add_argument("--rebuild-cleaned", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def _load_or_build_cleaned_data(args: argparse.Namespace) -> tuple[Path, object]:
    sync_legacy_baseline_artifacts()

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else BASELINE_ARTIFACTS_DIR
    reports_dir = Path(args.reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv_path = artifacts_dir / BASELINE_CLEAN_CSV_PATH.name
    sqlite_path = artifacts_dir / BASELINE_SQLITE_PATH.name
    cleaning_report_path = reports_dir / BASELINE_CLEANING_REPORT_PATH.name

    if (
        args.rebuild_cleaned
        or not cleaned_csv_path.exists()
        or not cleaning_report_path.exists()
    ):
        print_status("Building cleaned dataset for evaluation")
        cleaning_stats = clean_room_matching_csv(
            input_csv_path=args.input_csv,
            output_csv_path=cleaned_csv_path,
            sqlite_path=sqlite_path,
            report_path=cleaning_report_path,
            max_rows=args.max_clean_rows,
        )
    else:
        print_status(f"Reusing existing cleaned dataset: {cleaned_csv_path}")
        cleaning_stats = load_cleaning_stats(cleaning_report_path)

    return cleaned_csv_path, cleaning_stats


def evaluate_model(args: argparse.Namespace) -> dict[str, object]:
    if args.baseline_only:
        print_status("Starting heuristic baseline evaluation")
    else:
        print_status("Starting trained baseline evaluation")

    cleaned_csv_path, cleaning_stats = _load_or_build_cleaned_data(args)

    print_status("Loading positive pairs for evaluation")
    pairs = load_positive_pairs(
        cleaned_csv_path,
        unique_pair_count=cleaning_stats.unique_pairs,
        max_positive_pairs=args.max_positive_pairs,
        drop_ambiguous=not args.keep_ambiguous,
    )
    if not pairs:
        raise RuntimeError("No usable pairs were loaded from the cleaned dataset.")

    print_status(f"Loaded {len(pairs):,} positive pairs")
    provider_pool = build_provider_pool(pairs)
    positive_lookup = build_positive_lookup(pairs)
    token_index = build_token_index(provider_pool)
    split_data = split_pairs(pairs)

    if not split_data["validation"] or not split_data["test"]:
        raise RuntimeError(
            "Validation/test split is empty. Increase max_positive_pairs or keep ambiguous rows."
        )

    print_status(
        "Evaluation split ready: "
        f"validation={len(split_data['validation']):,}, "
        f"test={len(split_data['test']):,}"
    )
    print_status("Building validation scenarios")
    validation_scenarios = build_candidate_scenarios(
        split_data["validation"],
        provider_pool=provider_pool,
        positive_lookup=positive_lookup,
        token_index=token_index,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        max_true_candidates=args.max_true_candidates,
        max_scenarios=args.max_eval_scenarios,
        random_seed=args.random_seed,
    )
    print_status("Building test scenarios")
    test_scenarios = build_candidate_scenarios(
        split_data["test"],
        provider_pool=provider_pool,
        positive_lookup=positive_lookup,
        token_index=token_index,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        max_true_candidates=args.max_true_candidates,
        max_scenarios=args.max_eval_scenarios,
        random_seed=args.random_seed + 1,
    )

    if args.baseline_only:
        model = OverlapRoomMatcher()
        print_status("Scoring heuristic baseline on validation scenarios")
        validation_scores = score_scenarios(model, validation_scenarios)
        threshold, threshold_grid = tune_threshold(validation_scores)
        model.threshold = threshold
        print_status(f"Selected heuristic threshold: {threshold:.2f}")
        print_status("Scoring heuristic baseline on test scenarios")
        test_scores = score_scenarios(model, test_scenarios)
        mode = "baseline"
        metadata: dict[str, object] = {}
    else:
        model_path = Path(args.model_path) if args.model_path else BASELINE_MODEL_PATH
        print_status(f"Loading trained baseline model from: {model_path}")
        model, metadata = RoomMatcherModel.load(model_path)
        print_status("Scoring validation scenarios")
        validation_scores = score_scenarios(model, validation_scenarios)
        threshold = float(metadata.get("threshold", model.threshold))
        print_status(f"Using stored threshold: {threshold:.2f}")
        print_status("Scoring test scenarios")
        test_scores = score_scenarios(model, test_scenarios)
        threshold_grid = []
        mode = "trained_baseline_model"

    validation_metrics = evaluate_scored_scenarios(validation_scores, threshold=threshold)
    test_metrics = evaluate_scored_scenarios(test_scores, threshold=threshold)

    report = {
        "mode": mode,
        "model_path": None if args.baseline_only else str(model_path),
        "threshold": threshold,
        "cleaning_stats": cleaning_stats.to_dict(),
        "dataset_summary": {
            "total_pairs_loaded": len(pairs),
            "train_pairs": len(split_data["train"]),
            "validation_pairs": len(split_data["validation"]),
            "test_pairs": len(split_data["test"]),
            "provider_pool_size": len(provider_pool),
            "drop_ambiguous": not args.keep_ambiguous,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "threshold_grid": threshold_grid,
        "stored_metadata": metadata,
        "sample_predictions": {
            "validation_samples": sample_prediction_rows(validation_scores, threshold=threshold),
            "test_samples": sample_prediction_rows(test_scores, threshold=threshold),
        },
    }

    if args.baseline_only:
        report_name = "baseline_evaluation_before_training.json"
    else:
        report_name = "baseline_evaluation_after_training.json"
    print_status(f"Writing evaluation report to: {Path(args.reports_dir) / report_name}")
    write_json_report(Path(args.reports_dir) / report_name, report)
    print_status(
        "Evaluation finished: "
        f"validation_f1={validation_metrics['f1_mean']:.4f}, "
        f"test_f1={test_metrics['f1_mean']:.4f}"
    )
    return report


def main() -> None:
    args = parse_args()
    report = evaluate_model(args)
    print("Mode:", report["mode"])
    print("Threshold:", report["threshold"])
    print("Validation F1:", report["validation_metrics"]["f1_mean"])
    print("Test F1:", report["test_metrics"]["f1_mean"])


if __name__ == "__main__":
    main()
