# Model

This project has one model.

`room_matcher/model.py`
This is the baseline model. It is a pairwise binary classifier built on cleaned room-name pairs from `room_matching.csv`. The pipeline normalizes room text, removes empty rows, merges duplicate pairs, counts how often each pair appears, and marks provider names that map to more than one target room. Training uses positive pairs from the CSV and generated negative pairs sampled from other provider room names. The classifier is `HashingVectorizer + SGDClassifier`, and prediction works by scoring every candidate room against the input room name and returning the candidates above a learned threshold.

How the model is made:
The raw CSV is first cleaned by `room_matcher/cleaning.py`.
The baseline model is trained by `room_matcher/train.py`.

How to use it:
Train the baseline model with `uv run python -m room_matcher.train --input-csv room_matching.csv --rebuild-cleaned`.
Start the baseline API with `ROOM_MATCHER_MODEL_TYPE=baseline uv run uvicorn room_matcher.api:app --reload`.

Confirmed facts:
The baseline API in `room_matcher/api.py` loads the baseline artifact from `artifacts/baseline/room_matcher.joblib`.

Assumptions and limits:
The baseline model is cheap to train and run, and its behavior is easier to inspect and modify than a transformer model.
The Docker setup serves this baseline model only.
