"""Review endpoints: the queue, an extraction's fields, corrections and approval.

Reviewers and admins may change data; viewers may only read. Every route runs inside the
caller's tenant session, so an extraction id from another tenant is simply not found.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from docflow_api.deps import AppSettings, CurrentPrincipal, Principal, Queue
from docflow_core.db.models import Document, Extraction
from docflow_core.db.session import tenant_session
from docflow_core.review import ReviewError, approve_document, correct_field

router = APIRouter(prefix="/v1", tags=["review"])


class ExtractionOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    extractor_version_id: uuid.UUID
    status: str
    revision: int
    fields: dict[str, dict[str, Any]]
    issues: list[dict[str, Any]]
    counts: dict[str, int]  # fields by status: auto_approved, needs_review, reviewed

    @classmethod
    def of(cls, ex: Extraction) -> ExtractionOut:
        counts = {"auto_approved": 0, "needs_review": 0, "reviewed": 0}
        for f in ex.fields.values():
            if f.get("status") in counts:
                counts[f["status"]] += 1
        return cls(
            id=ex.id, document_id=ex.document_id, extractor_version_id=ex.extractor_version_id,
            status=ex.status, revision=ex.revision, fields=ex.fields, issues=ex.issues or [],
            counts=counts,
        )  # fmt: skip


class QueueItem(BaseModel):
    document_id: uuid.UUID
    filename: str
    created_at: datetime
    extraction_id: uuid.UUID | None
    fields_to_review: int


class CorrectionIn(BaseModel):
    field: str
    action: Literal["accept", "edit"]
    value: Any = None
    seconds_spent: float | None = Field(default=None, ge=0, le=3600)


def _can_edit(principal: Principal) -> None:
    if principal.role not in ("admin", "reviewer"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "viewers cannot change extractions")


def _load(session: Session, extraction_id: uuid.UUID) -> tuple[Extraction, Document]:
    ex = session.get(Extraction, extraction_id, with_for_update=True)
    if ex is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "extraction not found")
    doc = session.get(Document, ex.document_id, with_for_update=True)
    assert doc is not None
    return ex, doc


def _raise(err: ReviewError) -> HTTPException:
    detail = {"message": err.message, **(err.detail or {})} if err.detail else err.message
    return HTTPException(err.status, detail)


@router.get("/review/queue", response_model=list[QueueItem])
def review_queue(
    principal: CurrentPrincipal, limit: Annotated[int, Query(ge=1, le=200)] = 50
) -> list[QueueItem]:
    """Documents waiting for a person, oldest first, with how many fields need a look."""
    with tenant_session(principal.tenant_id) as s:
        docs = s.scalars(
            select(Document)
            .where(Document.status == "needs_review")
            .order_by(Document.created_at)
            .limit(limit)
        ).all()
        items = []
        for doc in docs:
            ex = s.scalar(
                select(Extraction)
                .where(Extraction.document_id == doc.id)
                .order_by(Extraction.updated_at.desc())
                .limit(1)
            )
            pending = (
                sum(f.get("status") == "needs_review" for f in ex.fields.values()) if ex else 0
            )
            if ex is not None and ex.status == "failed_parse":
                pending = -1  # the model could not read it: the whole document needs entry
            items.append(
                QueueItem(
                    document_id=doc.id,
                    filename=doc.filename,
                    created_at=doc.created_at,
                    extraction_id=ex.id if ex else None,
                    fields_to_review=pending,
                )
            )
        return items


@router.get("/extractions/{extraction_id}", response_model=ExtractionOut)
def get_extraction(principal: CurrentPrincipal, extraction_id: uuid.UUID) -> ExtractionOut:
    with tenant_session(principal.tenant_id) as s:
        ex = s.get(Extraction, extraction_id)
        if ex is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "extraction not found")
        return ExtractionOut.of(ex)


@router.post("/extractions/{extraction_id}/corrections", response_model=ExtractionOut)
def add_correction(
    principal: CurrentPrincipal,
    queue: Queue,
    settings: AppSettings,
    extraction_id: uuid.UUID,
    body: CorrectionIn,
) -> ExtractionOut:
    """Accept (no change) or edit one field. Appends a correction; re-runs the rules."""
    _can_edit(principal)
    try:
        with tenant_session(principal.tenant_id) as s:
            ex, doc = _load(s, extraction_id)
            correct_field(
                s, ex, doc, key=body.field, action=body.action, value=body.value,
                reviewer_id=principal.user_id, seconds_spent=body.seconds_spent,
                queue=queue, settings=settings,
            )  # fmt: skip
            return ExtractionOut.of(ex)
    except ReviewError as err:
        raise _raise(err) from err


@router.post("/documents/{document_id}/approve", response_model=ExtractionOut)
def approve(
    principal: CurrentPrincipal,
    queue: Queue,
    settings: AppSettings,
    document_id: uuid.UUID,
    acknowledge_issues: bool = False,
) -> ExtractionOut:
    """Approve once no field needs review (409 lists what is left)."""
    _can_edit(principal)
    try:
        with tenant_session(principal.tenant_id) as s:
            doc = s.get(Document, document_id, with_for_update=True)
            if doc is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
            ex = s.scalar(
                select(Extraction)
                .where(Extraction.document_id == document_id)
                .order_by(Extraction.updated_at.desc())
                .limit(1)
            )
            if ex is None or ex.status == "failed_parse":
                raise HTTPException(status.HTTP_409_CONFLICT, "nothing extracted to approve")
            approve_document(
                s, ex, doc, queue, settings,
                reviewer_id=principal.user_id, acknowledge_issues=acknowledge_issues,
            )  # fmt: skip
            return ExtractionOut.of(ex)
    except ReviewError as err:
        raise _raise(err) from err
