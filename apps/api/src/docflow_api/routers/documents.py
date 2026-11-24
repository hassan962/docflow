"""Document upload and status.

Upload flow (POST /v1/documents):

1. Read at most MAX_UPLOAD_BYTES + 1 bytes, so an oversized file is rejected without
   reading all of it into memory.
2. Decide the type from the file's magic bytes (PDF, PNG, JPEG), not from its name.
3. Hash it. If this tenant already uploaded the same bytes, return that document: nothing
   is reprocessed, so a re-upload costs nothing.
4. Otherwise store the original in S3, insert the row, enqueue the OCR message and commit.

The enqueue happens inside the database transaction, before the commit. If the commit then
fails, the OCR worker receives a message for a document that does not exist and simply
drops it. The reverse order (commit, then enqueue) could leave a document stuck in
"uploaded" forever if the enqueue failed. This is a lightweight alternative to a
transactional outbox, made safe by the workers' idempotency.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from docflow_api.deps import AppSettings, CurrentPrincipal, Queue, Store
from docflow_api.schemas import (
    DocumentDetail,
    DocumentList,
    DocumentOut,
    PageOut,
    PresignedUrl,
    StageRunOut,
    UploadOut,
)
from docflow_core.db.models import DOCUMENT_STATUSES, Document, Extraction, Page, StageRun
from docflow_core.db.session import tenant_session
from docflow_core.queue import StageMessage
from docflow_core.schemas import registry
from docflow_core.storage import raw_key, sniff_content_type

router = APIRouter(prefix="/v1/documents", tags=["documents"])

PRESIGN_SECONDS = 300


def _by_hash(session: Session, sha256: str) -> Document | None:
    return session.scalar(select(Document).where(Document.sha256 == sha256))


def _duplicate(doc: Document) -> JSONResponse:
    body = UploadOut.model_validate(doc).model_copy(update={"duplicate": True})
    return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump(mode="json"))


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=UploadOut,
    responses={200: {"model": UploadOut, "description": "Same file already uploaded"}},
)
def upload_document(
    principal: CurrentPrincipal,
    store: Store,
    queue: Queue,
    settings: AppSettings,
    file: Annotated[UploadFile, File(description="PDF, PNG or JPEG")],
    doc_type: Annotated[str, Form()] = "invoice",
) -> UploadOut | JSONResponse:
    if doc_type not in registry.doc_types():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"unknown doc_type {doc_type!r}")

    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"file is larger than {settings.max_upload_bytes // (1024 * 1024)} MB",
        )
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "file is empty")
    content_type = sniff_content_type(data)
    if content_type is None:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "only PDF, PNG and JPEG files are accepted"
        )
    sha256 = hashlib.sha256(data).hexdigest()

    try:
        with tenant_session(principal.tenant_id) as session:
            existing = _by_hash(session, sha256)
            if existing is not None:
                return _duplicate(existing)

            document_id = uuid.uuid4()
            key = raw_key(principal.tenant_id, document_id, content_type)
            store.put(key, data, content_type)
            doc = Document(
                id=document_id,
                tenant_id=principal.tenant_id,
                s3_key=key,
                sha256=sha256,
                filename=(file.filename or "upload")[:255],
                content_type=content_type,
                doc_type=doc_type,
            )
            session.add(doc)
            session.flush()
            session.refresh(doc)
            queue.send(
                StageMessage(tenant_id=principal.tenant_id, document_id=document_id, stage="ocr")
            )
            return UploadOut.model_validate(doc)
    except IntegrityError:
        # Two identical uploads raced; the other one won. Return its document.
        with tenant_session(principal.tenant_id) as session:
            winner = _by_hash(session, sha256)
            if winner is None:
                raise
            return _duplicate(winner)


@router.get("", response_model=DocumentList)
def list_documents(
    principal: CurrentPrincipal,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    doc_type: str | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    if status_filter is not None and status_filter not in DOCUMENT_STATUSES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"unknown status {status_filter!r}"
        )
    query = select(Document).order_by(Document.created_at)  # oldest first, like a work queue
    if status_filter:
        query = query.where(Document.status == status_filter)
    if doc_type:
        query = query.where(Document.doc_type == doc_type)
    with tenant_session(principal.tenant_id) as session:
        docs = session.scalars(query.limit(limit).offset(offset)).all()
        return DocumentList(
            items=[DocumentOut.model_validate(d) for d in docs], limit=limit, offset=offset
        )


def _get_or_404(session: Session, document_id: uuid.UUID) -> Document:
    # Another tenant's document is invisible under RLS, so it is a 404 too: the API never
    # confirms that a document id exists elsewhere.
    doc = session.get(Document, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return doc


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(principal: CurrentPrincipal, document_id: uuid.UUID) -> DocumentDetail:
    with tenant_session(principal.tenant_id) as session:
        doc = _get_or_404(session, document_id)
        pages = session.scalars(
            select(Page).where(Page.document_id == document_id).order_by(Page.page_no)
        ).all()
        stages = session.scalars(
            select(StageRun)
            .where(StageRun.document_id == document_id)
            .order_by(StageRun.started_at)
        ).all()
        extraction = session.scalar(
            select(Extraction)
            .where(Extraction.document_id == document_id)
            .order_by(Extraction.created_at.desc())
            .limit(1)
        )
        return DocumentDetail(
            **DocumentOut.model_validate(doc).model_dump(),
            pages=[PageOut.model_validate(p) for p in pages],
            stages=[StageRunOut.model_validate(s) for s in stages],
            latest_extraction=(
                {"id": str(extraction.id), "status": extraction.status, "fields": extraction.fields}
                if extraction
                else None
            ),
        )


@router.get("/{document_id}/pages/{page_no}/image", response_model=PresignedUrl)
def page_image(
    principal: CurrentPrincipal, store: Store, document_id: uuid.UUID, page_no: int
) -> PresignedUrl:
    with tenant_session(principal.tenant_id) as session:
        _get_or_404(session, document_id)
        page = session.scalar(
            select(Page).where(Page.document_id == document_id, Page.page_no == page_no)
        )
        if page is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "page not found")
        image_key = page.image_key
    return PresignedUrl(
        url=store.presign_get(image_key, PRESIGN_SECONDS), expires_in=PRESIGN_SECONDS
    )
