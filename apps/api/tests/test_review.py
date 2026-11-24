"""Review API: queue, corrections (accept/edit), re-validation, approval, permissions."""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from conftest import new_tenant
from docflow_api.deps import get_queue, get_store
from docflow_api.main import create_app
from docflow_core.config import Settings
from docflow_core.db.models import Correction, Document, Extraction, User
from docflow_core.db.session import owner_session, tenant_session
from docflow_core.extract.confidence import IsotonicCalibrator
from docflow_core.extract.versions import active_extractor
from docflow_core.queue import InMemoryQueue
from docflow_core.schemas import registry
from docflow_core.security import generate_api_key
from docflow_core.storage import InMemoryStore
from docflow_core.validate.apply import route
from docflow_core.validate.rules import RuleContext, run_rules

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("migrated_db")]


def _field(value: Any) -> dict[str, Any]:
    signals = {"match": "exact", "ocr_conf": 0.96, "similarity": 1.0, "agreement": True}
    return {"value": value, "confidence": 0.99, "signals": signals, "checks": [], "source": "llm",
            "page": 1, "bbox": [10, 10, 50, 20]}  # fmt: skip


def _fields(**overrides: Any) -> dict[str, dict[str, Any]]:
    values = {
        "vendor_name": "Acme Traders", "invoice_number": "INV-1", "invoice_date": "2026-03-05",
        "currency": "INR", "line_items[0].description": "Widget", "line_items[0].quantity": "2",
        "line_items[0].unit_price": "500.00", "line_items[0].amount": "1000.00",
        "subtotal": "1000.00", "tax": "180.00", "total": "1180.00",
        "vendor_tax_id": "27AAPFU0939F1ZV", "po_number": None,
    }  # fmt: skip
    values.update(overrides)
    return {k: _field(v) for k, v in values.items()}


class Ctx:
    def __init__(self, fields: dict[str, dict[str, Any]]) -> None:
        self.queue = InMemoryQueue()
        app = create_app()
        app.dependency_overrides[get_store] = lambda: InMemoryStore()
        app.dependency_overrides[get_queue] = lambda: self.queue
        self.client = TestClient(app)
        self.tenant_id, _, self.key = new_tenant(role="reviewer")
        with owner_session() as s:
            version = active_extractor(s, "invoice", Settings())
        spec = registry.get("invoice", version.schema_version)
        values = {k: f["value"] for k, f in fields.items()}
        results = run_rules(values, RuleContext(spec, date.today()))
        routed = route(fields, results, IsotonicCalibrator.identity(), threshold=0.9)
        self.doc_id = uuid.uuid4()
        with tenant_session(self.tenant_id) as s:
            s.add(Document(id=self.doc_id, tenant_id=self.tenant_id, s3_key="k", sha256=uuid.uuid4().hex * 2,
                           filename="inv.pdf", content_type="application/pdf", status="needs_review"))  # fmt: skip
            s.flush()
            ex = Extraction(tenant_id=self.tenant_id, document_id=self.doc_id, extractor_version_id=version.id,
                            fields=routed.fields, issues=routed.issues, status="needs_review")  # fmt: skip
            s.add(ex)
            s.flush()
            self.extraction_id = ex.id

    def h(self, key: str | None = None) -> dict[str, str]:
        return {"X-API-Key": key or self.key}

    def correct(
        self, field: str, action: str = "accept", value: Any = None, key: str | None = None
    ) -> Any:
        return self.client.post(
            f"/v1/extractions/{self.extraction_id}/corrections",
            json={"field": field, "action": action, "value": value, "seconds_spent": 3.5},
            headers=self.h(key),
        )

    def statuses(self) -> dict[str, str]:
        body = self.client.get(f"/v1/extractions/{self.extraction_id}", headers=self.h()).json()
        return {k: f["status"] for k, f in body["fields"].items()}


def test_queue_and_extraction_show_what_needs_a_person() -> None:
    ctx = Ctx(_fields(vendor_tax_id="2TAAPFU0939F1ZV"))  # bad check character
    queue = ctx.client.get("/v1/review/queue", headers=ctx.h()).json()
    [item] = [q for q in queue if q["document_id"] == str(ctx.doc_id)]
    assert item["fields_to_review"] == 1
    body = ctx.client.get(f"/v1/extractions/{ctx.extraction_id}", headers=ctx.h()).json()
    assert body["counts"]["needs_review"] == 1
    [check] = body["fields"]["vendor_tax_id"]["checks"]
    assert check["suggestion"] == "27AAPFU0939F1ZV"


def test_accepting_the_suggestion_approves_and_queues_indexing() -> None:
    ctx = Ctx(_fields(vendor_tax_id="2TAAPFU0939F1ZV"))
    r = ctx.correct("vendor_tax_id", "edit", "27AAPFU0939F1ZV")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "approved" and body["revision"] == 2
    assert body["fields"]["vendor_tax_id"]["status"] == "reviewed"
    assert body["fields"]["vendor_tax_id"]["source"] == "review"
    assert [m.stage for m in ctx.queue.pending("index")] == ["index"]
    with tenant_session(ctx.tenant_id) as s:
        doc = s.get(Document, ctx.doc_id)
        [c] = s.scalars(
            select(Correction).where(Correction.extraction_id == ctx.extraction_id)
        ).all()
    assert doc is not None and doc.status == "approved"
    assert (c.action, c.old_value, c.new_value, c.seconds_spent) == (
        "edit", {"value": "2TAAPFU0939F1ZV"}, {"value": "27AAPFU0939F1ZV"}, 3.5,
    )  # fmt: skip


def test_fixing_one_number_clears_the_flags_it_caused() -> None:
    ctx = Ctx(_fields(total="1189.00"))  # subtotal + tax != total
    assert {k for k, v in ctx.statuses().items() if v == "needs_review"} == {
        "subtotal",
        "tax",
        "total",
    }
    body = ctx.correct("total", "edit", "Rs. 1,180.00").json()
    assert body["fields"]["total"]["value"] == "1180.00"  # normalised like a model reply
    assert body["fields"]["subtotal"]["status"] == "auto_approved"
    assert body["fields"]["tax"]["status"] == "auto_approved"
    assert body["status"] == "approved"


def test_an_accept_is_stored_as_a_positive_label() -> None:
    ctx = Ctx(_fields(total="1189.00"))
    body = ctx.correct("tax").json()
    assert body["fields"]["tax"]["status"] == "reviewed" and body["status"] == "needs_review"
    with tenant_session(ctx.tenant_id) as s:
        [c] = s.scalars(
            select(Correction).where(Correction.extraction_id == ctx.extraction_id)
        ).all()
    assert c.action == "accept" and c.old_value == c.new_value == {"value": "180.00"}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("total", "twelve", "not an amount"), ("invoice_date", "31/02/2026", "not a date"),
     ("currency", "XYZ", "not one of"), ("total", "", "required"), ("nope", "1", "no field")],
)  # fmt: skip
def test_bad_edits_are_rejected(field: str, value: str, message: str) -> None:
    ctx = Ctx(_fields(total="1189.00"))
    r = ctx.correct(field, "edit", value)
    assert r.status_code in (404, 422) and message in str(r.json()["detail"])


def test_approve_is_refused_while_fields_need_review() -> None:
    ctx = Ctx(_fields(total="1189.00"))
    r = ctx.client.post(f"/v1/documents/{ctx.doc_id}/approve", headers=ctx.h())
    assert r.status_code == 409
    assert r.json()["detail"]["fields"] == ["subtotal", "tax", "total"]
    for f in ("subtotal", "tax", "total"):
        ctx.correct(f)
    assert (
        ctx.client.post(f"/v1/documents/{ctx.doc_id}/approve", headers=ctx.h()).status_code == 200
    )


def test_viewers_read_but_cannot_change_and_other_tenants_see_nothing() -> None:
    ctx = Ctx(_fields(total="1189.00"))
    key = generate_api_key()
    with owner_session() as s:
        viewer = User(
            tenant_id=ctx.tenant_id, email=f"v-{uuid.uuid4().hex[:6]}@t.test", role="viewer"
        )
        s.add(viewer)
        s.flush()
        from docflow_core.db.models import ApiKey

        s.add(
            ApiKey(
                tenant_id=ctx.tenant_id, user_id=viewer.id, prefix=key.prefix, key_hash=key.key_hash
            )
        )
    assert (
        ctx.client.get(
            f"/v1/extractions/{ctx.extraction_id}", headers=ctx.h(key.secret)
        ).status_code
        == 200
    )
    assert ctx.correct("tax", key=key.secret).status_code == 403
    _, _, other = new_tenant()
    assert (
        ctx.client.get(f"/v1/extractions/{ctx.extraction_id}", headers=ctx.h(other)).status_code
        == 404
    )
    assert ctx.correct("tax", key=other).status_code == 404
