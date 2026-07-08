# Model

This project has two models.

`room_matcher/model.py`
This is the baseline model. It is a pairwise binary classifier built on cleaned room-name pairs from `room_matching.csv`. The pipeline normalizes room text, removes empty rows, merges duplicate pairs, counts how often each pair appears, and marks provider names that map to more than one target room. Training uses positive pairs from the CSV and generated negative pairs sampled from other provider room names. The classifier is `HashingVectorizer + SGDClassifier`, and prediction works by scoring every candidate room against the input room name and returning the candidates above a learned threshold.

`room_matcher/model2.py`
This is the optional Hugging Face model. It uses `microsoft/Multilingual-MiniLM-L12-H384` as the default starting checkpoint and fine-tunes it as a text-pair classifier on the same room-matching task. Each training example is `(room_name, candidate_room) -> match / no_match`. After fine-tuning, it uses the same candidate-list evaluation flow as the baseline model, so both models can be compared with the same metrics and reports.

How the models are made:
The raw CSV is first cleaned by `room_matcher/cleaning.py`.
The baseline model is trained by `room_matcher/train.py`.
The Hugging Face model is trained by `python -m room_matcher.model2`.

How to use them:
Train the baseline model with `uv run python -m room_matcher.train --input-csv room_matching.csv --rebuild-cleaned`.
Start the baseline API with `ROOM_MATCHER_MODEL_TYPE=baseline uv run uvicorn room_matcher.api:app --reload`.
Train the Hugging Face model with `uv sync --group dev --extra hf` and then `uv run python -m room_matcher.model2 --input-csv room_matching.csv --max-clean-rows 50000 --max-positive-pairs 5000 --rebuild-cleaned`.

Confirmed facts:
The baseline API in `room_matcher/api.py` loads the baseline artifact from `artifacts/baseline/room_matcher.joblib`.
The Hugging Face path saves a directory model under `artifacts/hf/room_matcher`.
The default tokenizer override for `microsoft/Multilingual-MiniLM-L12-H384` is `xlm-roberta-base`.

Assumptions and limits:
The baseline model is the simpler and cheaper option to run.
The Hugging Face model should capture harder text variations better, but it needs more memory, more time, and optional dependencies.
The Hugging Face path depends on PyTorch and model downloads being available in your environment.
The Docker setup is baseline-only and does not install the optional Hugging Face stack.
