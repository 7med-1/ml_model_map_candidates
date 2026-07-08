FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY room_matcher ./room_matcher

RUN uv sync --frozen --no-dev

COPY manual.md model.md ARCHITECTURE.md ./
COPY artifacts/baseline ./artifacts/baseline

ENV PATH="/app/.venv/bin:$PATH"
ENV ROOM_MATCHER_MODEL_TYPE=baseline
ENV ROOM_MATCHER_MODEL_PATH=/app/artifacts/baseline/room_matcher.joblib

EXPOSE 8000

CMD ["uvicorn", "room_matcher.api:app", "--host", "0.0.0.0", "--port", "8000"]
