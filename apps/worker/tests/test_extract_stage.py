"""Extract stage end to end: OCR JSON in S3 -> extraction row, usage events, routing."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from sqlalchemy import select

from conftest import new_tenant
from docflow_core.config import Settings
from docflow_core.db.models import Document, Extraction, ExtractorVersion, Page, UsageEvent
from docflow_core.db.session import owner_session, tenant_session
from docflow_core.extract.llm import LlmClient, ScriptedClient
from docflow_core.extract.versions import active_extractor
from docflow_core.ocr import OcrResult, OcrWord
from docflow_core.queue import InMemoryQueue, StageMessage
from docflow_core.storage import InMemoryStore
from docflow_worker.runner import Deps, Outcome, handle
from docflow_worker.stages.extract import ExtractStage

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("migrated_db")]

LINES = ["Acme Traders", "Invoice No.: INV-7", "Date: 2026-03-05", "Item 1 100.00 100.00",
         "Total Rs. 100.00"]  # fmt: skip
LABEL: dict[str, Any] = {
    "vendor_name": "Acme Traders", "invoice_number": "INV-7", "invoice_date": "2026-03-05",
    "currency": "INR", "subtotal": None, "tax": None, "total": "100.00",
    "line_items": [
        {"description": "Item", "quantity": "1", "unit_price": "100.00", "amount": "100.00"}
    ],
    "vendor_tax_id": None, "po_number": None,
}  # fmt: skip


def _ocr() -> OcrResult:
    words = [
        OcrWord(t, (10 * i, 30 * n, 10 * i + 8, 30 * n + 20), 0.9, n)
        for n, line in enumerate(LINES)
        for i, t in enumerate(line.split())
    ]
    return OcrResult(words, 1000, 1400, "test")


def _setup(
    replies: list[str | None],
) -> tuple[uuid.UUID, uuid.UUID, Deps, ExtractStage, InMemoryQueue]:
    tenant_id, _, _ = new_tenant()
    store, queue = InMemoryStore(), InMemoryQueue()
    with owner_session() as s:  # whichever version is active, give it a second-opinion model
        active_extractor(s, "invoice", Settings())
        row = s.scalar(
            select(ExtractorVersion).where(
                ExtractorVersion.doc_type == "invoice", ExtractorVersion.status == "active"
            )
        )
        assert row is not None
        row.config = {**(row.config or {}), "agreement_model": "openai/gpt-oss-20b"}
    doc_id = uuid.uuid4()
    key = f"{tenant_id}/ocr/{doc_id}/1.test.json"
    store.put(key, _ocr().to_json(), "application/json")
    with tenant_session(tenant_id) as s:
        s.add(Document(id=doc_id, tenant_id=tenant_id, s3_key="raw", sha256=uuid.uuid4().hex * 2,
                       filename="x", content_type="image/png", status="ocr_done"))  # fmt: skip
        s.flush()
        s.add(Page(tenant_id=tenant_id, document_id=doc_id, page_no=1, image_key="p.png",
                   ocr_json_key=key))  # fmt: skip
    queue.send(StageMessage(tenant_id=tenant_id, document_id=doc_id, stage="extract"))
    primary = ScriptedClient(model="primary", replies=list(replies))
    second = ScriptedClient(model="second", replies=[json.dumps(LABEL)])

    def client_for(model: str) -> LlmClient:
        return second if model == "openai/gpt-oss-20b" else primary

    deps = Deps(settings=Settings(), store=store, queue=queue)
    return tenant_id, doc_id, deps, ExtractStage(client_for), queue


def test_extraction_is_saved_with_usage_and_handed_to_validation() -> None:
    tenant_id, doc_id, deps, stage, queue = _setup([json.dumps(LABEL)])
    [received] = queue.receive("extract")
    assert handle(received, stage, deps) is Outcome.DONE

    with tenant_session(tenant_id) as s:
        doc = s.get(Document, doc_id)
        [ex] = s.scalars(select(Extraction).where(Extraction.document_id == doc_id)).all()
        usage = s.scalars(select(UsageEvent).where(UsageEvent.document_id == doc_id)).all()
    assert doc is not None and doc.status == "extracted"
    assert ex.status == "ok" and ex.fields["total"]["value"] == "100.00"
    assert ex.fields["total"]["bbox"] is not None
    assert sorted(u.stage for u in usage) == ["extract:agreement", "extract:primary"]
    assert [m.stage for m in queue.pending("validate")] == ["validate"]


def test_unusable_replies_send_the_document_straight_to_review() -> None:
    tenant_id, doc_id, deps, stage, queue = _setup(["not json", "still not json"])
    [received] = queue.receive("extract")
    assert handle(received, stage, deps) is Outcome.DONE

    with tenant_session(tenant_id) as s:
        doc = s.get(Document, doc_id)
        ex = s.scalar(select(Extraction).where(Extraction.document_id == doc_id))
    assert doc is not None and doc.status == "needs_review"
    assert "unusable after retry" in (doc.failure_reason or "")
    assert ex is not None and ex.status == "failed_parse"
    assert queue.pending("validate") == []
