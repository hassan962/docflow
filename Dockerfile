# One image for the API, the workers and migrations; the command decides which runs.
FROM python:3.12-slim AS app

# Same uv version as the lockfile was written with.
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Tesseract OCR engine (English) for the OCR worker.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, in their own layer, so code edits don't reinstall them.
COPY pyproject.toml uv.lock .python-version ./
COPY packages/docflow_core/pyproject.toml packages/docflow_core/
COPY apps/api/pyproject.toml apps/api/
COPY apps/worker/pyproject.toml apps/worker/
RUN uv sync --frozen --no-dev --no-install-workspace

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

RUN useradd --system --uid 10001 docflow
USER docflow

EXPOSE 8000
CMD ["uvicorn", "docflow_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
