"""SQLAlchemy models for every DocFlow table.

The schema itself is created by the hand-written SQL migration in ``db/migrations`` (it also
holds the row-level security policies, triggers and grants that models cannot express).
``tests/test_db.py`` checks that these models and the migrated database agree.

Conventions
- Every tenant-owned table carries ``tenant_id`` and is protected by a row-level security
  policy. Global tables (schema and extractor versions, eval runs) have no tenant.
- Status columns are text with CHECK constraints rather than Postgres enums: adding a status
  is then a one-line migration instead of an ``ALTER TYPE``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, ClassVar

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 384

DOCUMENT_STATUSES = (
    "uploaded", "ocr_done", "extracted", "validated", "needs_review", "approved", "indexed",
    "failed",
)  # fmt: skip
EXTRACTOR_STATUSES = ("candidate", "active", "rejected", "retired")
EXTRACTION_STATUSES = ("ok", "failed_parse", "needs_review", "approved")


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSONB,
        uuid.UUID: UUID(as_uuid=True),
    }


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(primary_key=True, server_default=text("gen_random_uuid()"))


def _tenant() -> Mapped[uuid.UUID]:
    return mapped_column(ForeignKey("tenants.id"), index=False)


def _created() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now())


# --- Tenancy and access ---------------------------------------------------------------------


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(Text)
    daily_budget_usd: Mapped[Decimal] = mapped_column(Numeric(10, 4), server_default="1.0")
    created_at: Mapped[datetime] = _created()


class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    email: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text)  # admin | reviewer | viewer
    created_at: Mapped[datetime] = _created()


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    prefix: Mapped[str] = mapped_column(String(12))  # shown in the UI to identify a key
    key_hash: Mapped[str] = mapped_column(Text)  # SHA-256 of the secret; never the secret
    created_at: Mapped[datetime] = _created()
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --- Documents and pipeline state -----------------------------------------------------------


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    s3_key: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    filename: Mapped[str] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str] = mapped_column(Text, server_default="invoice")
    status: Mapped[str] = mapped_column(Text, server_default="uploaded")
    page_count: Mapped[int | None] = mapped_column(Integer)
    failure_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _created()


class Page(Base):
    __tablename__ = "pages"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    page_no: Mapped[int] = mapped_column(Integer)
    image_key: Mapped[str] = mapped_column(Text)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    ocr_engine: Mapped[str | None] = mapped_column(Text)
    ocr_json_key: Mapped[str | None] = mapped_column(Text)
    mean_ocr_conf: Mapped[float | None]


class StageRun(Base):
    """One row per (document, stage): the idempotency key for every worker."""

    __tablename__ = "stage_runs"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    stage: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, server_default="running")  # running|done|failed
    attempts: Mapped[int] = mapped_column(Integer, server_default="1")
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = _created()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --- Schemas, extractors and extractions ----------------------------------------------------


class SchemaVersion(Base):
    __tablename__ = "schema_versions"
    id: Mapped[uuid.UUID] = _pk()
    doc_type: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer)
    json_schema: Mapped[dict[str, Any]]
    fingerprint: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = _created()


class ExtractorVersion(Base):
    __tablename__ = "extractor_versions"
    id: Mapped[uuid.UUID] = _pk()
    doc_type: Mapped[str] = mapped_column(Text)
    schema_version: Mapped[int] = mapped_column(Integer)
    model: Mapped[str] = mapped_column(Text)
    prompt_hash: Mapped[str] = mapped_column(String(64))
    fewshot_policy: Mapped[str] = mapped_column(Text, server_default="none")
    config: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    status: Mapped[str] = mapped_column(Text, server_default="candidate")
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("extractor_versions.id"))
    created_at: Mapped[datetime] = _created()
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Extraction(Base):
    __tablename__ = "extractions"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    extractor_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("extractor_versions.id"))
    fields: Mapped[dict[str, Any]] = mapped_column(server_default=text("'{}'::jsonb"))
    status: Mapped[str] = mapped_column(Text, server_default="ok")
    # Bumped on every reviewer change; the corrections table holds the full history.
    revision: Mapped[int] = mapped_column(Integer, server_default="1")
    created_at: Mapped[datetime] = _created()
    updated_at: Mapped[datetime] = _created()


class Correction(Base):
    """Append-only: a trigger rejects UPDATE and DELETE. Audit trail and training data."""

    __tablename__ = "corrections"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    extraction_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("extractions.id"))
    field: Mapped[str] = mapped_column(Text)
    action: Mapped[str] = mapped_column(Text)  # accept | edit
    old_value: Mapped[dict[str, Any] | None]
    new_value: Mapped[dict[str, Any] | None]
    reviewer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    seconds_spent: Mapped[float | None]
    created_at: Mapped[datetime] = _created()


# --- Evaluation and metering ----------------------------------------------------------------


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[uuid.UUID] = _pk()
    extractor_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("extractor_versions.id"))
    dataset: Mapped[str] = mapped_column(Text)  # golden | dev | holdout-<name>
    n_docs: Mapped[int] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any]]
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    p95_latency_ms: Mapped[int] = mapped_column(Integer)
    report_key: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created()


class UsageEvent(Base):
    __tablename__ = "usage_events"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[uuid.UUID] = _tenant()
    document_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("documents.id"))
    stage: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(Text)
    tokens_in: Mapped[int] = mapped_column(Integer, server_default="0")
    tokens_out: Mapped[int] = mapped_column(Integer, server_default="0")
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), server_default="0")
    latency_ms: Mapped[int] = mapped_column(Integer, server_default="0")
    created_at: Mapped[datetime] = _created()


# --- Retrieval ------------------------------------------------------------------------------


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    page_no: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(Text, server_default="text")  # text | field
    text: Mapped[str] = mapped_column(Text)
    bbox: Mapped[list[int] | None] = mapped_column(JSONB)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )


# --- Reference data for the tool-calling checks ---------------------------------------------


class Vendor(Base):
    __tablename__ = "vendors"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    name: Mapped[str] = mapped_column(Text)
    normalised_name: Mapped[str] = mapped_column(Text)
    tax_id: Mapped[str | None] = mapped_column(Text)


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"
    id: Mapped[uuid.UUID] = _pk()
    tenant_id: Mapped[uuid.UUID] = _tenant()
    po_number: Mapped[str] = mapped_column(Text)
    vendor_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("vendors.id"))
    total: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    currency: Mapped[str] = mapped_column(String(3))
    status: Mapped[str] = mapped_column(Text, server_default="open")


TENANT_TABLES = (
    "users", "api_keys", "documents", "pages", "stage_runs", "extractions", "corrections",
    "usage_events", "chunks", "vendors", "purchase_orders",
)  # fmt: skip
