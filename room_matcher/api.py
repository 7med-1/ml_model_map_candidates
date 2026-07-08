from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from room_matcher.model import RoomMatcherModel
from room_matcher.model2 import HFRoomMatcher
from room_matcher.paths import BASELINE_MODEL_PATH, HF_MODEL_PATH


DEFAULT_MODEL_TYPE = "baseline"
DEFAULT_BASELINE_MODEL_PATH = BASELINE_MODEL_PATH
DEFAULT_HF_MODEL_PATH = HF_MODEL_PATH


class PredictionRequest(BaseModel):
    room_name: str = Field(min_length=1)
    candidate_rooms: list[str] = Field(min_length=5, max_length=20)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("candidate_rooms")
    @classmethod
    def validate_candidate_rooms(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if len(cleaned) != len(values):
            raise ValueError("candidate_rooms cannot contain empty values.")
        return cleaned


class ScoredCandidate(BaseModel):
    candidate_room: str
    score: float


class PredictionResponse(BaseModel):
    room_name: str
    threshold: float
    matched_rooms: list[str]
    scored_candidates: list[ScoredCandidate]


app = FastAPI(title="Room Matcher API", version="0.1.0")


@lru_cache
def get_model_bundle() -> tuple[Any, dict[str, object]]:
    model_type = get_model_type()
    model_path = get_model_path(model_type)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_type} model artifact not found at {model_path}. Train the model first."
        )
    if model_type == "baseline":
        return RoomMatcherModel.load(model_path)
    if model_type == "hf":
        return HFRoomMatcher.load(model_path)
    raise ValueError(
        f"Unsupported ROOM_MATCHER_MODEL_TYPE={model_type!r}. Use 'baseline' or 'hf'."
    )


def get_model_type() -> str:
    return os.environ.get("ROOM_MATCHER_MODEL_TYPE", DEFAULT_MODEL_TYPE).strip().lower()


def get_model_path(model_type: str | None = None) -> Path:
    resolved_model_type = model_type or get_model_type()
    default_path = (
        DEFAULT_BASELINE_MODEL_PATH
        if resolved_model_type == "baseline"
        else DEFAULT_HF_MODEL_PATH
    )
    return Path(os.environ.get("ROOM_MATCHER_MODEL_PATH", default_path))


@app.get("/health")
def health() -> dict[str, object]:
    model_type = get_model_type()
    model_path = get_model_path(model_type)
    status = "ok" if model_path.exists() else "missing_model"
    if model_type not in {"baseline", "hf"}:
        status = "invalid_model_type"
    return {
        "status": status,
        "model_type": model_type,
        "model_path": str(model_path),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        model, _metadata = get_model_bundle()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    prediction = model.predict_matches(
        request.room_name,
        request.candidate_rooms,
        threshold=request.threshold,
    )
    return PredictionResponse(**prediction)
