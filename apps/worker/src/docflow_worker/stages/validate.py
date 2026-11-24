"""Validate stage: rules + reference checks -> re-scored fields -> approved or needs review.

Runs the deterministic rules, then (when the tenant has reference data) the tool-calling
reference checks, folds every result into the fields' confidence with the calibration of the
extractor version that produced them, and routes the document:

- approved: every field auto-approved -> next stage ``index`` (phase 6);
- needs_review: at least one field for a person -> the review queue, pipeline pauses.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import openai
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from docflow_core.config import Stage
from docflow_core.db.models import Document, Extraction, UsageEvent, Vendor
from docflow_core.db.session import tenant_session, unscoped_session
from docflow_core.errors import PermanentError
from docflow_core.extract.llm import ChatResult, LlmClient
from docflow_core.extract.versions import version_by_id
from docflow_core.schemas import registry
from docflow_core.validate.apply import Routed, route, threshold_for
from docflow_core.validate.reference import (
    execute_tool,
    findings_to_results,
    run_reference_checks,
    unavailable,
)
from docflow_core.validate.rules import RuleContext, run_rules
from docflow_worker.runner import Deps, DocumentSnapshot


@dataclass
class ValidateResult:
    extraction_id: uuid.UUID
    routed: Routed
    calls: list[ChatResult] = field(default_factory=list)
    reference_error: str | None = None


class ValidateStage:
    name: Stage = "validate"

    def __init__(self, client_for: Callable[[str], LlmClient], tools_model: str | None) -> None:
        self.client_for = client_for
        self.tools_model = tools_model

    def run(self, doc: DocumentSnapshot, deps: Deps) -> ValidateResult:
        with tenant_session(doc.tenant_id) as s:
            extraction = s.scalar(
                select(Extraction)
                .where(Extraction.document_id == doc.id, Extraction.status != "failed_parse")
                .order_by(Extraction.updated_at.desc())
                .limit(1)
            )
            if extraction is None:
                raise PermanentError("no usable extraction to validate")
            fields: dict[str, dict[str, Any]] = extraction.fields
            extraction_id, version_id = extraction.id, extraction.extractor_version_id
            has_reference = bool(s.scalar(select(exists().where(Vendor.id.isnot(None)))))
        with unscoped_session() as s:
            version = version_by_id(s, version_id)
        spec = registry.get(version.doc_type, version.schema_version)

        values = {k: f["value"] for k, f in fields.items()}
        grounded = {k: f.get("signals", {}).get("match", "none") for k, f in fields.items()}
        ctx = RuleContext(
            spec, date.today(), deps.settings.validation_max_age_days, grounded=grounded
        )
        results = run_rules(values, ctx)

        calls: list[ChatResult] = []
        reference_error = None
        wants_reference = values.get("vendor_name") or values.get("po_number")
        if has_reference and self.tools_model and wants_reference:

            def run_tool(name: str, arguments: str) -> dict[str, Any]:
                with tenant_session(doc.tenant_id) as session:  # RLS scopes every lookup
                    return execute_tool(session, name, arguments)

            try:
                outcome = run_reference_checks(self.client_for(self.tools_model), values, run_tool)
            except openai.APIError as exc:
                # Rate limits, quota, outages: the rules still ran, so degrade to "needs a
                # person" instead of failing the document.
                reference_error = f"{type(exc).__name__}: {exc}"
                results.append(unavailable(reference_error))
            else:
                results += findings_to_results(outcome, values)
                calls, reference_error = outcome.calls, outcome.error

        threshold = threshold_for(doc.doc_type, version.review_threshold)
        routed = route(fields, results, version.calibrator, threshold)
        return ValidateResult(extraction_id, routed, calls, reference_error)

    def save(self, session: Session, doc: Document, result: ValidateResult) -> None:
        extraction = session.get(Extraction, result.extraction_id, with_for_update=True)
        if extraction is None:
            raise PermanentError("extraction disappeared during validation")
        extraction.fields = result.routed.fields
        extraction.issues = result.routed.issues
        extraction.status = "approved" if result.routed.approved else "needs_review"
        for call in result.calls:
            session.add(
                UsageEvent(
                    tenant_id=doc.tenant_id,
                    document_id=doc.id,
                    stage="validate:reference",
                    model=call.model,
                    tokens_in=call.tokens_in,
                    tokens_out=call.tokens_out,
                    cost_usd=call.cost_usd,
                    latency_ms=call.latency_ms,
                )
            )

    def route(self, result: ValidateResult) -> tuple[str, Stage | None]:
        if result.routed.approved:
            return "approved", "index"
        return "needs_review", None
