"""Response models. The API is the only contract the review UI depends on."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    doc_type: str
    status: str
    sha256: str
    page_count: int | None
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime


class UploadOut(DocumentOut):
    duplicate: bool = False


class PageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    page_no: int
    width: int | None
    height: int | None
    ocr_engine: str | None
    mean_ocr_conf: float | None


class StageRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stage: str
    status: str
    attempts: int
    error: str | None
    started_at: datetime
    finished_at: datetime | None


class DocumentDetail(DocumentOut):
    pages: list[PageOut]
    stages: list[StageRunOut]
    latest_extraction: dict[str, Any] | None = None


class DocumentList(BaseModel):
    items: list[DocumentOut]
    limit: int
    offset: int


class PresignedUrl(BaseModel):
    url: str
    expires_in: int
