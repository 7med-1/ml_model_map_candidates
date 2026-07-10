from __future__ import annotations

import argparse
from pathlib import Path

from room_matcher.cleaning import clean_room_matching_csv, load_cleaning_stats
from room_matcher.model import (
    RoomMatcherModel,
    build_candidate_scenarios,
    build_positive_lookup,
    build_provider_pool,
    build_token_index,
    evaluate_scored_scenarios,
    load_positive_pairs,
    sample_prediction_rows,
    score_scenarios,
    split_pairs,
    train_pairwise_model,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a room matching model.",
    )
    parser.add_argument("--input-csv", default="room_matching.csv")
    parser.add_argument("--artifacts-dir", default=str(BASELINE_ARTIFACTS_DIR))
    parser.add_argument("--reports-dir", default=str(REPORTS_ROOT))
    parser.add_argument("--model-name", default=BASELINE_MODEL_PATH.name)
    parser.add_argument("--max-positive-pairs", type=int, default=250_000)
    parser.add_argument("--max-clean-rows", type=int, default=None)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--min-candidates", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-true-candidates", type=int, default=3)
    parser.add_argument("--max-eval-scenarios", type=int, default=1_500)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--keep-ambiguous", action="store_true")
    parser.add_argument("--rebuild-cleaned", action="store_true")
    return parser.parse_args()


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    print_status("Starting baseline training run")
    sync_legacy_baseline_artifacts()

    artifacts_dir = Path(args.artifacts_dir)
    reports_dir = Path(args.reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv_path = artifacts_dir / BASELINE_CLEAN_CSV_PATH.name
    sqlite_path = artifacts_dir / BASELINE_SQLITE_PATH.name
    cleaning_report_path = reports_dir / BASELINE_CLEANING_REPORT_PATH.name
    model_path = artifacts_dir / args.model_name

    if (
        args.rebuild_cleaned
        or not cleaned_csv_path.exists()
        or not cleaning_report_path.exists()
    ):
        print_status("Building cleaned baseline dataset")
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

    print_status("Loading positive training pairs from cleaned dataset")
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

    if not split_data["train"] or not split_data["validation"] or not split_data["test"]:
        raise RuntimeError(
            "Train/validation/test split is empty. Increase max_positive_pairs or keep ambiguous rows."
        )

    print_status(
        "Pair split ready: "
        f"train={len(split_data['train']):,}, "
        f"validation={len(split_data['validation']):,}, "
        f"test={len(split_data['test']):,}"
    )

    model = RoomMatcherModel()
    print_status("Training baseline pairwise model")
    training_summary = train_pairwise_model(
        model,
        split_data["train"],
        provider_pool=provider_pool,
        positive_lookup=positive_lookup,
        token_index=token_index,
        negatives_per_positive=args.negatives_per_positive,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_seed=args.random_seed,
    )

    print_status("Building validation candidate scenarios")
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
    print_status("Scoring validation scenarios and tuning threshold")
    validation_scores = score_scenarios(model, validation_scenarios)
    threshold, threshold_grid = tune_threshold(validation_scores)
    model.threshold = threshold

    print_status("Building test candidate scenarios")
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
    print_status("Scoring test scenarios")
    test_scores = score_scenarios(model, test_scenarios)
    validation_metrics = evaluate_scored_scenarios(validation_scores, threshold=threshold)
    test_metrics = evaluate_scored_scenarios(test_scores, threshold=threshold)

    metadata = {
        "model_path": str(model_path),
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
        "training_summary": training_summary,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "cli_args": vars(args),
    }
    print_status(f"Saving trained baseline model to: {model_path}")
    model.save(model_path, metadata)

    print_status("Writing baseline reports")
    write_json_report(reports_dir / "baseline_training_summary.json", metadata)
    write_json_report(
        reports_dir / "baseline_threshold_grid.json",
        {
            "selected_threshold": threshold,
            "results": threshold_grid,
        },
    )
    write_json_report(
        reports_dir / "baseline_sample_predictions.json",
        {
            "threshold": threshold,
            "validation_samples": sample_prediction_rows(validation_scores, threshold=threshold),
            "test_samples": sample_prediction_rows(test_scores, threshold=threshold),
        },
    )

    print_status(
        "Baseline training run finished: "
        f"validation_f1={validation_metrics['f1_mean']:.4f}, "
        f"test_f1={test_metrics['f1_mean']:.4f}, "
        f"threshold={threshold:.2f}"
    )
    return metadata


def main() -> None:
    args = parse_args()
    summary = train_and_evaluate(args)
    print("Model artifact:", summary["model_path"])
    print("Threshold:", summary["threshold"])
    print("Validation F1:", summary["validation_metrics"]["f1_mean"])
    print("Test F1:", summary["test_metrics"]["f1_mean"])


if __name__ == "__main__":
    main()
