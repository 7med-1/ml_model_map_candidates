from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from room_matcher.cleaning import clean_room_matching_csv, load_cleaning_stats
from room_matcher.model import (
    RoomPair,
    build_candidate_scenarios,
    build_positive_lookup,
    build_provider_pool,
    build_token_index,
    evaluate_scored_scenarios,
    load_positive_pairs,
    sample_negative_candidates,
    sample_prediction_rows,
    score_scenarios,
    split_pairs,
    tune_threshold,
    write_json_report,
)
from room_matcher.paths import (
    HF_ARTIFACTS_DIR,
    HF_CLEANING_REPORT_PATH,
    HF_CLEAN_CSV_PATH,
    HF_MODEL_PATH,
    HF_SQLITE_PATH,
    REPORTS_ROOT,
)


MODEL2_METADATA_FILE = "metadata.json"
DEFAULT_HF_MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
TOKENIZER_NAME_OVERRIDES = {
    DEFAULT_HF_MODEL_NAME: "xlm-roberta-base",
}


@dataclass(slots=True, frozen=True)
class PairExample:
    room_name: str
    candidate_room: str
    label: int


def resolve_tokenizer_name(
    model_name: str,
    tokenizer_name: str | None = None,
) -> str:
    if tokenizer_name:
        return tokenizer_name
    return TOKENIZER_NAME_OVERRIDES.get(model_name, model_name)


def _require_hf_dependencies() -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Hugging Face dependencies are not installed. Run: uv sync --extra hf --group dev"
        ) from error

    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


class HFPairDataset:
    def __init__(
        self,
        examples: list[PairExample],
        tokenizer: Any,
        *,
        max_length: int,
        dataset_cls: type[Any],
    ) -> None:
        class _Dataset(dataset_cls):
            def __init__(self, data: list[PairExample], tokenizer_obj: Any, max_len: int) -> None:
                self.data = data
                self.tokenizer = tokenizer_obj
                self.max_len = max_len

            def __len__(self) -> int:
                return len(self.data)

            def __getitem__(self, index: int) -> dict[str, Any]:
                example = self.data[index]
                encoded = self.tokenizer(
                    example.room_name,
                    example.candidate_room,
                    truncation=True,
                    max_length=self.max_len,
                )
                encoded["labels"] = example.label
                return encoded

        self.dataset = _Dataset(examples, tokenizer, max_length)


class HFRoomMatcher:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        threshold: float = 0.5,
        max_length: int = 96,
        device: str | None = None,
    ) -> None:
        hf_modules = _require_hf_dependencies()
        torch = hf_modules["torch"]
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        tokenizer_name: str | None = None,
        threshold: float = 0.5,
        max_length: int = 96,
        device: str | None = None,
    ) -> "HFRoomMatcher":
        hf_modules = _require_hf_dependencies()
        resolved_tokenizer_name = resolve_tokenizer_name(model_name, tokenizer_name)
        tokenizer = hf_modules["AutoTokenizer"].from_pretrained(resolved_tokenizer_name)
        model = hf_modules["AutoModelForSequenceClassification"].from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: "NO_MATCH", 1: "MATCH"},
            label2id={"NO_MATCH": 0, "MATCH": 1},
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            threshold=threshold,
            max_length=max_length,
            device=device,
        )

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        device: str | None = None,
    ) -> tuple["HFRoomMatcher", dict[str, Any]]:
        hf_modules = _require_hf_dependencies()
        model_path = Path(model_dir)
        metadata_path = model_path / MODEL2_METADATA_FILE
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        tokenizer = hf_modules["AutoTokenizer"].from_pretrained(model_path)
        model = hf_modules["AutoModelForSequenceClassification"].from_pretrained(model_path)
        predictor = cls(
            model=model,
            tokenizer=tokenizer,
            threshold=float(metadata.get("threshold", 0.5)),
            max_length=int(metadata.get("max_length", 96)),
            device=device,
        )
        return predictor, metadata

    def save(self, model_dir: str | Path, metadata: dict[str, Any]) -> None:
        target_dir = Path(model_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(target_dir)
        self.tokenizer.save_pretrained(target_dir)
        payload = dict(metadata)
        payload["threshold"] = self.threshold
        payload["max_length"] = self.max_length
        (target_dir / MODEL2_METADATA_FILE).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def predict_scores(
        self,
        room_name: str,
        candidate_rooms: list[str],
        *,
        batch_size: int = 16,
    ) -> list[dict[str, float | str]]:
        hf_modules = _require_hf_dependencies()
        torch = hf_modules["torch"]
        results: list[dict[str, float | str]] = []

        self.model.eval()
        for start in range(0, len(candidate_rooms), batch_size):
            batch_candidates = candidate_rooms[start : start + batch_size]
            encoded = self.tokenizer(
                [room_name] * len(batch_candidates),
                batch_candidates,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()

            for candidate_room, score in zip(batch_candidates, probabilities, strict=True):
                results.append(
                    {
                        "candidate_room": candidate_room,
                        "score": float(score),
                    }
                )

        return sorted(results, key=lambda item: item["score"], reverse=True)

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


def build_pair_examples(
    pairs: list[RoomPair],
    *,
    provider_pool: list[Any],
    positive_lookup: dict[str, set[str]],
    token_index: dict[str, list[Any]],
    negatives_per_positive: int,
    random_seed: int,
) -> list[PairExample]:
    rng = random.Random(random_seed)
    examples: list[PairExample] = []
    shuffled_pairs = list(pairs)
    rng.shuffle(shuffled_pairs)

    for pair in shuffled_pairs:
        examples.append(
            PairExample(
                room_name=pair.room_name,
                candidate_room=pair.candidate_room,
                label=1,
            )
        )
        negatives = sample_negative_candidates(
            pair.room_name_normalized,
            positive_lookup[pair.room_name],
            provider_pool,
            token_index,
            rng,
            negatives_per_positive,
        )
        for negative in negatives:
            examples.append(
                PairExample(
                    room_name=pair.room_name,
                    candidate_room=negative.candidate_room,
                    label=0,
                )
            )

    return examples


def _compute_metrics(eval_pred: tuple[Any, Any]) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predicted_labels = np.argmax(predictions, axis=1)
    precision, recall, f1, _support = precision_recall_fscore_support(
        labels,
        predicted_labels,
        average="binary",
        zero_division=0,
    )
    accuracy = accuracy_score(labels, predicted_labels)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Hugging Face room matching model.",
    )
    parser.add_argument("--input-csv", default="room_matching.csv")
    parser.add_argument("--artifacts-dir", default=str(HF_ARTIFACTS_DIR))
    parser.add_argument("--reports-dir", default=str(REPORTS_ROOT))
    parser.add_argument("--model-dir-name", default=HF_MODEL_PATH.name)
    parser.add_argument("--hf-model-name", default=DEFAULT_HF_MODEL_NAME)
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument("--max-positive-pairs", type=int, default=100_000)
    parser.add_argument("--max-clean-rows", type=int, default=None)
    parser.add_argument("--negatives-per-positive", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--min-candidates", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--max-true-candidates", type=int, default=3)
    parser.add_argument("--max-eval-scenarios", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-ambiguous", action="store_true")
    parser.add_argument("--rebuild-cleaned", action="store_true")
    return parser.parse_args()


def train_and_evaluate_hf(args: argparse.Namespace) -> dict[str, Any]:
    hf_modules = _require_hf_dependencies()
    artifacts_dir = Path(args.artifacts_dir)
    reports_dir = Path(args.reports_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cleaned_csv_path = artifacts_dir / HF_CLEAN_CSV_PATH.name
    sqlite_path = artifacts_dir / HF_SQLITE_PATH.name
    cleaning_report_path = reports_dir / HF_CLEANING_REPORT_PATH.name
    model_dir = artifacts_dir / args.model_dir_name
    trainer_output_dir = artifacts_dir / f"{args.model_dir_name}_trainer"

    if (
        args.rebuild_cleaned
        or not cleaned_csv_path.exists()
        or not cleaning_report_path.exists()
    ):
        cleaning_stats = clean_room_matching_csv(
            input_csv_path=args.input_csv,
            output_csv_path=cleaned_csv_path,
            sqlite_path=sqlite_path,
            report_path=cleaning_report_path,
            max_rows=args.max_clean_rows,
        )
    else:
        cleaning_stats = load_cleaning_stats(cleaning_report_path)

    pairs = load_positive_pairs(
        cleaned_csv_path,
        unique_pair_count=cleaning_stats.unique_pairs,
        max_positive_pairs=args.max_positive_pairs,
        drop_ambiguous=not args.keep_ambiguous,
    )
    if not pairs:
        raise RuntimeError("No usable pairs were loaded from the cleaned dataset.")

    provider_pool = build_provider_pool(pairs)
    positive_lookup = build_positive_lookup(pairs)
    token_index = build_token_index(provider_pool)
    split_data = split_pairs(pairs)
    if not split_data["train"] or not split_data["validation"] or not split_data["test"]:
        raise RuntimeError(
            "Train/validation/test split is empty. Increase max_positive_pairs or keep ambiguous rows."
        )

    train_examples = build_pair_examples(
        split_data["train"],
        provider_pool=provider_pool,
        positive_lookup=positive_lookup,
        token_index=token_index,
        negatives_per_positive=args.negatives_per_positive,
        random_seed=args.random_seed,
    )
    validation_examples = build_pair_examples(
        split_data["validation"],
        provider_pool=provider_pool,
        positive_lookup=positive_lookup,
        token_index=token_index,
        negatives_per_positive=args.negatives_per_positive,
        random_seed=args.random_seed + 1,
    )
    if not train_examples or not validation_examples:
        raise RuntimeError("HF training examples are empty.")

    resolved_tokenizer_name = resolve_tokenizer_name(
        args.hf_model_name,
        args.tokenizer_name,
    )
    tokenizer = hf_modules["AutoTokenizer"].from_pretrained(resolved_tokenizer_name)
    model = hf_modules["AutoModelForSequenceClassification"].from_pretrained(
        args.hf_model_name,
        num_labels=2,
        id2label={0: "NO_MATCH", 1: "MATCH"},
        label2id={"NO_MATCH": 0, "MATCH": 1},
    )

    train_dataset = HFPairDataset(
        train_examples,
        tokenizer,
        max_length=args.max_length,
        dataset_cls=hf_modules["Dataset"],
    ).dataset
    validation_dataset = HFPairDataset(
        validation_examples,
        tokenizer,
        max_length=args.max_length,
        dataset_cls=hf_modules["Dataset"],
    ).dataset

    training_args = hf_modules["TrainingArguments"](
        output_dir=str(trainer_output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
        seed=args.random_seed,
    )
    trainer = hf_modules["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=hf_modules["DataCollatorWithPadding"](tokenizer=tokenizer),
        compute_metrics=_compute_metrics,
    )
    trainer.train()

    predictor = HFRoomMatcher(
        model=trainer.model,
        tokenizer=tokenizer,
        threshold=0.5,
        max_length=args.max_length,
        device=args.device,
    )

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
    validation_scores = score_scenarios(predictor, validation_scenarios)
    threshold, threshold_grid = tune_threshold(validation_scores)
    predictor.threshold = threshold

    test_scenarios = build_candidate_scenarios(
        split_data["test"],
        provider_pool=provider_pool,
        positive_lookup=positive_lookup,
        token_index=token_index,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        max_true_candidates=args.max_true_candidates,
        max_scenarios=args.max_eval_scenarios,
        random_seed=args.random_seed + 2,
    )
    test_scores = score_scenarios(predictor, test_scenarios)
    validation_metrics = evaluate_scored_scenarios(validation_scores, threshold=threshold)
    test_metrics = evaluate_scored_scenarios(test_scores, threshold=threshold)

    trainer_metrics = {
        key: float(value) if hasattr(value, "__float__") else value
        for key, value in trainer.evaluate().items()
    }
    metadata = {
        "model_dir": str(model_dir),
        "threshold": threshold,
        "max_length": args.max_length,
        "base_hf_model": args.hf_model_name,
        "tokenizer_name": resolved_tokenizer_name,
        "cleaning_stats": cleaning_stats.to_dict(),
        "dataset_summary": {
            "total_pairs_loaded": len(pairs),
            "train_pairs": len(split_data["train"]),
            "validation_pairs": len(split_data["validation"]),
            "test_pairs": len(split_data["test"]),
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "drop_ambiguous": not args.keep_ambiguous,
        },
        "trainer_metrics": trainer_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "cli_args": vars(args),
    }
    predictor.save(model_dir, metadata)

    write_json_report(reports_dir / "hf_training_summary.json", metadata)
    write_json_report(
        reports_dir / "hf_threshold_grid.json",
        {
            "selected_threshold": threshold,
            "results": threshold_grid,
        },
    )
    write_json_report(
        reports_dir / "hf_sample_predictions.json",
        {
            "threshold": threshold,
            "validation_samples": sample_prediction_rows(validation_scores, threshold=threshold),
            "test_samples": sample_prediction_rows(test_scores, threshold=threshold),
        },
    )

    return metadata


def main() -> None:
    args = parse_args()
    summary = train_and_evaluate_hf(args)
    print("HF model dir:", summary["model_dir"])
    print("Base HF model:", summary["base_hf_model"])
    print("Tokenizer:", summary["tokenizer_name"])
    print("Threshold:", summary["threshold"])
    print("Validation F1:", summary["validation_metrics"]["f1_mean"])
    print("Test F1:", summary["test_metrics"]["f1_mean"])


if __name__ == "__main__":
    main()
