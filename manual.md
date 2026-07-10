# Manual

`uv sync --group dev`
Creates the local virtual environment and installs the baseline app dependencies plus the test tools.

`uv run pytest`
Runs the automated test suite.

`uv run python -m room_matcher.clean --input-csv room_matching.csv --max-clean-rows 50000`
Cleans only the first 50000 raw rows and writes the cleaned sample dataset to `artifacts/baseline/` plus the baseline cleaning report.

`uv run python -m room_matcher.evaluate --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --baseline-only --rebuild-cleaned`
Evaluates the pre-training heuristic baseline on a small sample and writes `reports/baseline_evaluation_before_training.json`.

`uv run python -m room_matcher.train --input-csv room_matching.csv --rebuild-cleaned`
Cleans the raw CSV, trains the baseline room matcher, evaluates it, and writes model artifacts into `artifacts/baseline/` plus baseline JSON reports.

`uv run python -m room_matcher.train --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --rebuild-cleaned`
Cleans a raw sample, trains on a sample of unique pairs from that cleaned data, and writes the trained artifact into `artifacts/baseline/` plus training/evaluation reports.

Baseline commands automatically copy a newer legacy artifact from `artifacts/` into `artifacts/baseline/` so the API and evaluation use the baseline path.

`uv run python -m room_matcher.evaluate --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --rebuild-cleaned`
Loads the trained baseline artifact from `artifacts/baseline/room_matcher.joblib` and evaluates it on the same sampled dataset, then writes `reports/baseline_evaluation_after_training.json`.

`ROOM_MATCHER_MODEL_TYPE=baseline uv run uvicorn room_matcher.api:app --host 0.0.0.0 --port 8000 --reload`
Starts the FastAPI app with the baseline model from `artifacts/baseline/room_matcher.joblib`.

`curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"room_name":"Standard King Room","candidate_rooms":["Room, 1 King Bed, Non Smoking","Suite, 2 Queen Beds, Non Smoking","Room, 2 Double Beds","Deluxe Ocean View","Accessible Studio"]}'`
Calls the API with one room name and five candidate rooms.

`docker build -t room-matcher-baseline .`
Builds the baseline API image.

`docker run --rm -p 8000:8000 room-matcher-baseline`
Runs the baseline-only API container.
