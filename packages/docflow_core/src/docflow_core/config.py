"""All configuration in one place.

Every setting is an environment variable, loaded through this one Pydantic settings class.
In development the values come from a local ``.env`` file; in AWS they are injected from
SSM Parameter Store. Nothing secret is ever committed to the repository.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Stage = Literal["ocr", "extract", "validate", "index"]
STAGES: tuple[Stage, ...] = ("ocr", "extract", "validate", "index")

DEFAULT_REVIEW_THRESHOLD = 0.90


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    env: Literal["local", "test", "lean", "full"] = "local"

    # Database. The API and workers connect as a role that is subject to row-level security;
    # migrations run as the table owner.
    database_url: str = "postgresql+psycopg://docflow_app:app_dev_pw@localhost:5432/docflow"
    migration_database_url: str = (
        "postgresql+psycopg://docflow_owner:owner_dev_pw@localhost:5432/docflow"
    )

    # Object storage and queues. The endpoint overrides point at the local emulators
    # (S3Mock and ElasticMQ); on AWS they are unset and boto3 uses the real services.
    # The names match boto3's own service-specific variables (AWS_ENDPOINT_URL_S3/_SQS).
    s3_bucket: str = "docflow-local"
    aws_endpoint_url_s3: str | None = None
    aws_endpoint_url_sqs: str | None = None
    # Host that browsers use for presigned URLs. Locally the API reaches S3 as
    # http://s3mock:9090 inside Docker, but your browser needs http://localhost:9090.
    s3_public_endpoint_url: str | None = None
    aws_region: str = "ap-south-1"
    queue_url_ocr: str | None = None
    queue_url_extract: str | None = None
    queue_url_validate: str | None = None
    queue_url_index: str | None = None

    # Workers.
    max_receives: int = 3  # must match the queues' redrive maxReceiveCount
    retry_base_seconds: int = 10  # backoff: 10s, 20s, 40s ...
    receive_wait_seconds: int = 20  # SQS long polling

    # OCR.
    ocr_engine: Literal["tesseract", "textract"] = "tesseract"
    ocr_dpi: int = Field(default=250, ge=150, le=400)
    max_pages: int = Field(default=20, ge=1)

    # LLM. Any OpenAI-compatible endpoint: Groq, Gemini's compatibility endpoint, Ollama
    # (http://localhost:11434/v1), or a paid provider. These values only bootstrap the first
    # extractor version; after that the active row in extractor_versions decides the model.
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    # Second opinion for the per-field agreement signal (the providers tried return no token
    # log-probabilities). A different model gives a more independent check. Empty = off.
    llm_agreement_model: str | None = "openai/gpt-oss-20b"
    # json_schema = the provider enforces the schema; json_object = JSON mode only, for
    # providers without strict structured output (e.g. older Ollama models).
    llm_response_format: Literal["json_schema", "json_object"] = "json_schema"
    llm_reasoning_effort: Literal["low", "medium", "high"] | None = "low"
    llm_timeout_seconds: float = 90
    # USD per million tokens (input, output). Check groq.com/pricing; on the free tier these
    # are notional, used for cost-per-page reporting and the promotion gate's cost check.
    llm_prices: dict[str, tuple[float, float]] = {
        "openai/gpt-oss-120b": (0.15, 0.60),
        "openai/gpt-oss-20b": (0.075, 0.30),
    }

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Guard rails.
    tenant_daily_budget_usd: float = Field(default=1.0, ge=0)
    validation_max_age_days: int = 3 * 365  # older invoice dates fail valid_date
    # Model for the tool-calling reference checks. Empty = skip them.
    llm_tools_model: str | None = "openai/gpt-oss-20b"
    max_upload_bytes: int = 15 * 1024 * 1024

    def queue_url(self, stage: Stage) -> str | None:
        url: str | None = getattr(self, f"queue_url_{stage}")
        return url


def review_threshold(doc_type: str, default: float | None = None) -> float:
    """Confidence threshold for auto-approval, per document type.

    ``REVIEW_THRESHOLD_INVOICE=0.93`` overrides it. The variable is read on every call rather
    than cached, so it can be changed in the running environment without rebuilding anything.
    Without it, ``default`` (the threshold calibrated with the extractor version) applies.
    """
    raw = os.environ.get(f"REVIEW_THRESHOLD_{doc_type.upper()}")
    if raw is None:
        return default if default is not None else DEFAULT_REVIEW_THRESHOLD
    value = float(raw)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"REVIEW_THRESHOLD_{doc_type.upper()} must be between 0 and 1")
    return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
