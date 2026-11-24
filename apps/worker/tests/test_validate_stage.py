"""Validate stage end to end: extraction -> rules + reference checks -> routed document."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from sqlalchemy import select
from test_reference import seed_reference

from conftest import new_tenant
from docflow_core.config import Settings
from docflow_core.db.models import Document, Extraction, UsageEvent
from docflow_core.db.session import owner_session, tenant_session
from docflow_core.extract.llm import LlmClient, ScriptedClient, ToolCall
from docflow_core.extract.versions import active_extractor
from docflow_core.queue import InMemoryQueue, StageMessage
from docflow_core.storage import InMemoryStore
from docflow_worker.runner import Deps, Outcome, handle
from docflow_worker.stages.validate import ValidateStage

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("migrated_db")]


def _field(value: Any, match: str = "exact") -> dict[str, Any]:
    signals = {"match": match, "ocr_conf": 0.96, "similarity": 1.0, "agreement": True}
    return {"value": value, "confidence": 0.99, "signals": signals, "checks": [], "source": "llm"}


GOOD = {
    "vendor_name": _field("Acme Traders Pvt Ltd"),
    "invoice_number": _field("INV-1"),
    "invoice_date": _field("2026-03-05"),
    "currency": _field("INR"),
    "line_items[0].description": _field("Widget"),
    "line_items[0].quantity": _field("2"),
    "line_items[0].unit_price": _field("500.00"),
    "line_items[0].amount": _field("1000.00"),
    "subtotal": _field("1000.00"),
    "tax": _field("180.00"),
    "total": _field("1180.00"),
    "vendor_tax_id": _field("27AAPFU0939F1ZV"),
    "po_number": _field("PO-1"),
}


def _run(
    fields: dict[str, Any], replies: list[Any]
) -> tuple[uuid.UUID, uuid.UUID, Outcome, InMemoryQueue]:
    tenant_id, _, _ = new_tenant()
    seed_reference(tenant_id)
    with owner_session() as s:
        version = active_extractor(s, "invoice", Settings())
    doc_id = uuid.uuid4()
    with tenant_session(tenant_id) as s:
        s.add(Document(id=doc_id, tenant_id=tenant_id, s3_key="k", sha256=uuid.uuid4().hex * 2,
                       filename="x", content_type="image/png", status="extracted"))  # fmt: skip
        s.flush()
        s.add(Extraction(tenant_id=tenant_id, document_id=doc_id, extractor_version_id=version.id,
                         fields=fields, status="ok"))  # fmt: skip
    queue = InMemoryQueue()
    queue.send(StageMessage(tenant_id=tenant_id, document_id=doc_id, stage="validate"))
    model = ScriptedClient(replies=list(replies))

    def client_for(name: str) -> LlmClient:
        return model

    deps = Deps(settings=Settings(), store=InMemoryStore(), queue=queue)
    [received] = queue.receive("validate")
    outcome = handle(received, ValidateStage(client_for, "tools-model"), deps)
    return tenant_id, doc_id, outcome, queue


LOOKUPS = [
    ToolCall("a", "lookup_vendor", '{"name": "Acme Traders Pvt Ltd"}'),
    ToolCall("b", "check_purchase_order", '{"po_number": "PO-1", "total": 1180}'),
]


def test_a_clean_invoice_is_approved_and_sent_to_indexing() -> None:
    tenant_id, doc_id, outcome, queue = _run(GOOD, [LOOKUPS, '{"findings": []}'])
    assert outcome is Outcome.DONE
    with tenant_session(tenant_id) as s:
        doc = s.get(Document, doc_id)
        ex = s.scalar(select(Extraction).where(Extraction.document_id == doc_id))
        usage = s.scalars(select(UsageEvent).where(UsageEvent.document_id == doc_id)).all()
    assert doc is not None and doc.status == "approved"
    assert ex is not None and ex.status == "approved"
    assert {f["status"] for f in ex.fields.values()} == {"auto_approved"}
    assert {c["rule"] for c in ex.fields["total"]["checks"]} >= {
        "total_consistency",
        "reference_checks",
    }
    assert len(usage) == 2 and {u.stage for u in usage} == {"validate:reference"}
    assert [m.stage for m in queue.pending("index")] == ["index"]


def test_a_bad_tax_id_and_a_po_mismatch_go_to_review_with_a_suggestion() -> None:
    fields = {**GOOD, "vendor_tax_id": _field("2TAAPFU0939F1ZV")}
    final = {"findings": [{"field": "total", "severity": "error", "message": "PO total differs"}]}
    tenant_id, doc_id, outcome, queue = _run(fields, [LOOKUPS, json.dumps(final)])
    assert outcome is Outcome.DONE
    with tenant_session(tenant_id) as s:
        doc = s.get(Document, doc_id)
        ex = s.scalar(select(Extraction).where(Extraction.document_id == doc_id))
    assert doc is not None and doc.status == "needs_review"
    assert ex is not None and ex.status == "needs_review"
    review = sorted(k for k, f in ex.fields.items() if f["status"] == "needs_review")
    assert review == ["total", "vendor_tax_id"]
    [tax_check] = [c for c in ex.fields["vendor_tax_id"]["checks"] if c["rule"] == "tax_id_format"]
    assert tax_check["suggestion"] == "27AAPFU0939F1ZV"
    assert queue.pending("index") == []


def test_a_provider_outage_sends_the_document_to_review_instead_of_failing_it() -> None:
    import httpx
    import openai

    def quota_exhausted(request: object) -> str:
        response = httpx.Response(429, request=httpx.Request("POST", "https://llm.test"))
        raise openai.RateLimitError("tokens per day exhausted", response=response, body=None)

    tenant_id, _, _ = new_tenant()
    seed_reference(tenant_id)
    with owner_session() as s:
        version = active_extractor(s, "invoice", Settings())
    doc_id = uuid.uuid4()
    with tenant_session(tenant_id) as s:
        s.add(Document(id=doc_id, tenant_id=tenant_id, s3_key="k", sha256=uuid.uuid4().hex * 2,
                       filename="x", content_type="image/png", status="extracted"))  # fmt: skip
        s.flush()
        s.add(Extraction(tenant_id=tenant_id, document_id=doc_id, extractor_version_id=version.id,
                         fields=GOOD, status="ok"))  # fmt: skip
    queue = InMemoryQueue()
    queue.send(StageMessage(tenant_id=tenant_id, document_id=doc_id, stage="validate"))
    stage = ValidateStage(lambda name: ScriptedClient(responder=quota_exhausted), "tools-model")
    [received] = queue.receive("validate")
    assert handle(received, stage, Deps(Settings(), InMemoryStore(), queue)) is Outcome.DONE

    with tenant_session(tenant_id) as s:
        doc = s.get(Document, doc_id)
        ex = s.scalar(select(Extraction).where(Extraction.document_id == doc_id))
    assert doc is not None and doc.status == "needs_review"  # not "failed"
    assert ex is not None and ex.issues[0]["rule"] == "reference_checks"
    assert "RateLimitError" in ex.issues[0]["message"]
    assert {f["status"] for f in ex.fields.values()} == {"auto_approved"}  # the fields were fine
