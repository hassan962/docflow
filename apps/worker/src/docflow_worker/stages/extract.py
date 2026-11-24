"""Extract stage: OCR JSON -> typed fields with value, confidence, page and box.

The active extractor version (model, prompt template, calibration) is read from the database
on every document, so promoting a new version changes the next document's extraction with no
redeploy. Every LLM call is recorded in ``usage_events`` with tokens, cost and latency.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from docflow_core.config import Stage
from docflow_core.db.models import Document, Extraction, UsageEvent
from docflow_core.db.session import tenant_session, unscoped_session
from docflow_core.errors import PermanentError
from docflow_core.extract.extractor import ExtractionOutcome, extract
from docflow_core.extract.llm import LlmClient
from docflow_core.extract.versions import active_extractor
from docflow_core.learn.fewshot import PoolCache, select_examples
from docflow_core.ocr import OcrResult
from docflow_core.schemas import registry
from docflow_worker.runner import Deps, DocumentSnapshot

ClientFactory = Callable[[str], LlmClient]
log = structlog.get_logger()


@dataclass(frozen=True)
class ExtractResult:
    extractor_version_id: uuid.UUID
    outcome: ExtractionOutcome


class ExtractStage:
    name: Stage = "extract"

    def __init__(self, client_for: ClientFactory) -> None:
        self.client_for = client_for
        self.pools: PoolCache | None = None

    def run(self, doc: DocumentSnapshot, deps: Deps) -> ExtractResult:
        if not doc.ocr_keys:
            raise PermanentError("no OCR output for this document")
        with unscoped_session() as session:  # extractor_versions is global, not tenant data
            version = active_extractor(session, doc.doc_type, deps.settings)
        spec = registry.get(version.doc_type, version.schema_version)
        pages = [OcrResult.from_json(deps.store.get(key)) for key in doc.ocr_keys]
        examples = []
        if version.fewshot_k:
            self.pools = self.pools or PoolCache(deps.store)
            with tenant_session(doc.tenant_id) as session:  # this tenant's corrections only
                pool = self.pools.get(session, doc.tenant_id, doc.doc_type)
            examples = select_examples(pool, pages, version.fewshot_k)
        outcome = extract(
            spec,
            pages,
            self.client_for(version.model),
            agreement_client=self.client_for(version.agreement_model)
            if version.agreement_model
            else None,
            scorer=version.scorer,
            examples=examples,
        )
        if outcome.template_hash != version.prompt_hash:
            # The code's prompt changed but no new version was registered, so results are
            # being attributed to a version that does not describe them.
            log.warning(
                "prompt_drift",
                active_version=str(version.id),
                version_hash=version.prompt_hash[:12],
                code_hash=outcome.template_hash[:12],
                fix="python scripts/admin.py extractors register --promote",
            )
        return ExtractResult(extractor_version_id=version.id, outcome=outcome)

    def save(self, session: Session, doc: Document, result: ExtractResult) -> None:
        outcome = result.outcome
        values = {
            "fields": {k: f.model_dump(mode="json") for k, f in outcome.fields.items()},
            "status": outcome.status,
            "revision": 1,
        }
        stmt = insert(Extraction).values(
            tenant_id=doc.tenant_id,
            document_id=doc.id,
            extractor_version_id=result.extractor_version_id,
            **values,
        )
        # A rerun of the same version on the same document replaces its result.
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=["document_id", "extractor_version_id"], set_=values
            )
        )
        for call in outcome.calls:
            session.add(
                UsageEvent(
                    tenant_id=doc.tenant_id,
                    document_id=doc.id,
                    stage=f"extract:{call.role}",
                    model=call.result.model,
                    tokens_in=call.result.tokens_in,
                    tokens_out=call.result.tokens_out,
                    cost_usd=call.result.cost_usd,
                    latency_ms=call.result.latency_ms,
                )
            )
        if outcome.status == "failed_parse":
            reason = f"extract: model reply unusable after retry: {outcome.error}"
            doc.failure_reason = reason[:2000]

    def route(self, result: ExtractResult) -> tuple[str, Stage | None]:
        if result.outcome.status == "failed_parse":
            return "needs_review", None  # a person reads it; nothing to validate
        return "extracted", "validate"
