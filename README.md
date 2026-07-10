# Room Matcher

This project solves a hotel room-name matching problem.

Input:
- one source room name
- a list of 5 to 20 candidate provider room names

Output:
- the candidate rooms that match the source room
- a score for every candidate

Example task:

```text
source: STDSD QN RM NSMoKing
candidates:
- Standard Queen Room, Non Smoking
- Standard Room, 1 Queen Bed, Smoking
- Superior King Room with Balcony

expected match:
- Standard Queen Room, Non Smoking
```

## Solution

The repo uses one baseline ML model in `room_matcher/model.py`.

The pipeline is:

1. Clean and normalize raw room names from `room_matching.csv`.
2. Build positive pairs from real mapped rooms.
3. Generate hard negative pairs from other provider rooms.
4. Train a pairwise binary classifier.
5. Tune a decision threshold.
6. Serve the model with FastAPI.

The model is intentionally simple and editable:

- text features: `HashingVectorizer`
- numeric/conflict features: bed type, room type, numbers, room attributes
- classifier: `SGDClassifier`
- artifact path: `artifacts/baseline/room_matcher.joblib`

## Requirements

- Python `>=3.14`
- `uv`
- the raw dataset file: `room_matching.csv`

Install dependencies:

```bash
uv sync --group dev
```

## Test The Repo

Run all automated tests:

```bash
uv run pytest
```

## Clean Data

Clean a small sample:

```bash
uv run python -m room_matcher.clean --input-csv room_matching.csv --max-clean-rows 50000
```

Clean the full dataset:

```bash
uv run python -m room_matcher.clean --input-csv room_matching.csv
```

Outputs:

- `artifacts/baseline/room_matching_clean.csv`
- `artifacts/baseline/room_matching_clean.sqlite3`
- `reports/baseline_cleaning_summary.json`

## Evaluate Before Training

This evaluates the heuristic baseline before training the ML model.

```bash
uv run python -m room_matcher.evaluate --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --baseline-only --rebuild-cleaned
```

Output:

- `reports/baseline_evaluation_before_training.json`

## Train The Model

Train on a small sample:

```bash
uv run python -m room_matcher.train --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --rebuild-cleaned
```

Train on the full dataset:

```bash
uv run python -m room_matcher.train --input-csv room_matching.csv --rebuild-cleaned
```

Outputs:

- `artifacts/baseline/room_matcher.joblib`
- `reports/baseline_training_summary.json`
- `reports/baseline_threshold_grid.json`
- `reports/baseline_sample_predictions.json`

## Evaluate After Training

Evaluate the trained model:

```bash
uv run python -m room_matcher.evaluate --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --rebuild-cleaned
```

Output:

- `reports/baseline_evaluation_after_training.json`

## Run The API

Start FastAPI:

```bash
ROOM_MATCHER_MODEL_TYPE=baseline uv run uvicorn room_matcher.api:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Live prediction:

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "room_name": "STDSD QN RM NSMoKing",
    "candidate_rooms": [
      "Standard Queen Room, Non Smoking",
      "Standard Room, 1 Queen Bed, Smoking",
      "STD QN RM POOLVW NSMK",
      "Standard Queen Room Garden View",
      "1 QN RM NSMK",
      "Standard Queen Room with Pool View Non Smoking",
      "Chambre standard avec lit queen vue mer non fumeur",
      "Habitacion estandar con cama queen vista al mar para no fumadores",
      "Standard Queen Room with Sea View Non Smoking",
      "Superior King Room with Balcony and Breakfast"
    ],
    "threshold": 0.4
  }'
```

The API returns:

- `matched_rooms`: candidates with score above the threshold
- `scored_candidates`: every candidate sorted by score
- `threshold`: the threshold used for this request

## Docker

Build the API image:

```bash
docker build -t room-matcher-baseline .
```

Run the API container:

```bash
docker run --rm -p 8000:8000 room-matcher-baseline
```

## Project Layout

```text
room_matcher/
  api.py        FastAPI app
  clean.py      data cleaning CLI
  cleaning.py   normalization and cleaned dataset builder
  evaluate.py   evaluation CLI
  model.py      baseline ML model and metrics
  paths.py      artifact/report paths
  train.py      training CLI

artifacts/baseline/
  room_matcher.joblib
  room_matching_clean.csv
  room_matching_clean.sqlite3

reports/
  baseline_*.json
```

## Notes

Baseline commands automatically copy a newer legacy model from `artifacts/` into `artifacts/baseline/` when needed.

The repo is baseline-only. There is no Hugging Face or transformer implementation in the active code path.
