"""Reviewer actions: accept or edit a field, approve a document.

Every action appends a row to ``corrections`` (append-only: the audit trail and, in phase 5,
the training data). An accepted field with no change is stored too, because "this value was
right" is a positive label. The extraction itself is updated in place with a new revision.

After each correction the deterministic rules run again on the new values, so fixing one
number can clear the flags it caused elsewhere (correct the subtotal and ``total_consistency``
passes again). Reference-check results are kept as they were: re-running them would call the
model on every keystroke.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Literal

from sqlalchemy.orm import Session

from docflow_core.config import Settings
from docflow_core.db.models import Correction, Document, Extraction
from docflow_core.extract.fields import allowed_codes, base_name, field_kinds, required_keys
from docflow_core.extract.versions import version_by_id
from docflow_core.normalise import normalise_amount, normalise_date
from docflow_core.queue import QueueClient, StageMessage
from docflow_core.schemas import registry
from docflow_core.schemas.registry import SchemaSpec
from docflow_core.validate.apply import Routed, route, stored_reference_results, threshold_for
from docflow_core.validate.rules import RuleContext, run_rules


class ReviewError(Exception):
    def __init__(self, status: int, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status, self.message, self.detail = status, message, detail


def parse_value(spec: SchemaSpec, key: str, value: Any) -> Any:
    """A reviewer's edit, checked and normalised exactly like a model reply."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if base_name(key) in required_keys(spec.model):
            raise ReviewError(422, f"{key} is required")
        return None
    kind = field_kinds(spec.model)[base_name(key)]
    if kind == "amount":
        amount = normalise_amount(value)
        if amount is None:
            raise ReviewError(422, f"{value!r} is not an amount")
        return str(amount)
    if kind == "date":
        parsed = normalise_date(value)
        if parsed is None:
            raise ReviewError(422, f"{value!r} is not a date")
        return parsed.isoformat()
    if kind == "code":
        code = str(value).strip().upper()
        allowed = allowed_codes(spec.model, key)
        if allowed and code not in allowed:
            raise ReviewError(422, f"{code} is not one of {', '.join(map(str, allowed))}")
        return code
    return str(value).strip()


def _reroute(
    session: Session, extraction: Extraction, settings: Settings
) -> tuple[SchemaSpec, Routed]:
    version = version_by_id(session, extraction.extractor_version_id)
    spec = registry.get(version.doc_type, version.schema_version)
    fields = extraction.fields
    values = {k: f["value"] for k, f in fields.items()}
    grounded = {k: f.get("signals", {}).get("match", "none") for k, f in fields.items()}
    ctx = RuleContext(spec, date.today(), settings.validation_max_age_days, grounded=grounded)
    results = run_rules(values, ctx) + stored_reference_results(fields)
    threshold = threshold_for(version.doc_type, version.review_threshold)
    return spec, route(fields, results, version.calibrator, threshold)


def _finish_if_clear(
    extraction: Extraction, doc: Document, routed: Routed, queue: QueueClient
) -> None:
    extraction.fields, extraction.issues = routed.fields, routed.issues
    if routed.approved:
        extraction.status = "approved"
        if doc.status != "approved":
            doc.status = "approved"
            queue.send(StageMessage(tenant_id=doc.tenant_id, document_id=doc.id, stage="index"))
    else:
        extraction.status = "needs_review"


def correct_field(
    session: Session,
    extraction: Extraction,
    doc: Document,
    *,
    key: str,
    action: Literal["accept", "edit"],
    value: Any,
    reviewer_id: uuid.UUID,
    seconds_spent: float | None,
    queue: QueueClient,
    settings: Settings,
) -> Extraction:
    if key not in extraction.fields:
        raise ReviewError(404, f"no field {key!r} in this extraction")
    version = version_by_id(session, extraction.extractor_version_id)
    spec = registry.get(version.doc_type, version.schema_version)
    current = extraction.fields[key]
    old = current["value"]
    new = old if action == "accept" else parse_value(spec, key, value)

    session.add(
        Correction(
            tenant_id=extraction.tenant_id,
            extraction_id=extraction.id,
            field=key,
            action=action,
            old_value={"value": old},
            new_value={"value": new},
            reviewer_id=reviewer_id,
            seconds_spent=seconds_spent,
        )
    )
    fields = dict(extraction.fields)
    fields[key] = {
        **current,
        "value": new,
        "source": "review",
        "status": "reviewed",
        "confidence": 1.0,
    }
    extraction.fields = fields  # reassign so SQLAlchemy writes the JSONB column
    extraction.revision += 1
    _, routed = _reroute(session, extraction, settings)
    _finish_if_clear(extraction, doc, routed, queue)
    return extraction


def approve_document(
    session: Session, extraction: Extraction, doc: Document, queue: QueueClient, settings: Settings
) -> Extraction:
    _, routed = _reroute(session, extraction, settings)
    if not routed.approved:
        pending = sorted(k for k, f in routed.fields.items() if f["status"] == "needs_review")
        raise ReviewError(
            409, "fields still need review", {"fields": pending, "issues": routed.issues}
        )
    _finish_if_clear(extraction, doc, routed, queue)
    return extraction
