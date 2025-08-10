FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Layer-cache dependencies separately from source
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY ui ./ui
COPY README.md ./
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HUB_CACHE=/models \
    SPECSAGE_API_HOST=0.0.0.0

EXPOSE 8000

CMD ["uvicorn", "specsage.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
