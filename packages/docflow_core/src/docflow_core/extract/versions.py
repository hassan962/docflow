"""Extractor versions.

A version is the tuple (doc_type, schema_version, model, prompt template hash, few-shot policy,
config), and the active row in ``extractor_versions`` is what the workers run. Settings only
bootstrap the very first version: after that, changing the model or the calibration means
registering a new version and promoting it, which phase 5's evaluation gate controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from docflow_core.config import Settings
from docflow_core.db.models import ExtractorVersion
from docflow_core.db.publish import publish_schemas
from docflow_core.extract.confidence import Scorer, scorer_from_config
from docflow_core.extract.prompt import PROMPT_VERSION, build_prompt
from docflow_core.schemas import registry


@dataclass(frozen=True)
class ActiveExtractor:
    id: Any
    doc_type: str
    schema_version: int
    model: str
    agreement_model: str | None
    prompt_hash: str
    scorer: Scorer
    fewshot_k: int  # corrected examples per prompt; 0 = zero-shot
    review_threshold: float | None


def _snapshot(row: ExtractorVersion) -> ActiveExtractor:
    cfg = row.config or {}
    return ActiveExtractor(
        id=row.id,
        doc_type=row.doc_type,
        schema_version=row.schema_version,
        model=row.model,
        agreement_model=cfg.get("agreement_model"),
        prompt_hash=row.prompt_hash,
        scorer=scorer_from_config(cfg),
        fewshot_k=int((cfg.get("fewshot") or {}).get("k", 0)),
        review_threshold=cfg.get("review_threshold"),
    )


def version_by_id(session: Session, version_id: Any) -> ActiveExtractor:
    """Any version, active or not: validation uses the one that made the extraction."""
    row = session.get(ExtractorVersion, version_id)
    if row is None:
        raise ValueError(f"no extractor version {version_id}")
    return _snapshot(row)


def _active(session: Session, doc_type: str) -> ExtractorVersion | None:
    return session.scalar(
        select(ExtractorVersion).where(
            ExtractorVersion.doc_type == doc_type, ExtractorVersion.status == "active"
        )
    )


def bootstrap_config(settings: Settings) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "agreement_model": settings.llm_agreement_model or None,
        "response_format": settings.llm_response_format,
        "reasoning_effort": settings.llm_reasoning_effort,
        "calibration": None,
        "review_threshold": None,
    }


def active_extractor(session: Session, doc_type: str, settings: Settings) -> ActiveExtractor:
    row = _active(session, doc_type)
    if row is not None:
        return _snapshot(row)

    # First run for this document type: register version 1 from settings and activate it.
    publish_schemas(session)
    spec = registry.latest(doc_type)
    candidate = ExtractorVersion(
        doc_type=doc_type,
        schema_version=spec.version,
        model=settings.llm_model,
        prompt_hash=build_prompt(spec, []).template_hash,
        fewshot_policy="none",
        config=bootstrap_config(settings),
        status="active",
    )
    try:
        with session.begin_nested():  # savepoint: losing a bootstrap race is not an error
            session.add(candidate)
            session.flush()
    except IntegrityError:
        pass  # another worker activated one first; the unique index guarantees just one
    row = _active(session, doc_type)
    assert row is not None
    return _snapshot(row)


def current_template_hash(doc_type: str, schema_version: int) -> str:
    """The prompt template hash the code would send today (no few-shot examples)."""
    return build_prompt(registry.get(doc_type, schema_version), []).template_hash


def register_candidate(
    session: Session,
    doc_type: str,
    settings: Settings,
    *,
    config_overrides: dict[str, Any] | None = None,
) -> ExtractorVersion:
    """A new candidate from the current code and settings, child of the active version.

    Candidates do nothing until promoted. Phase 5's gate promotes only on a measured win.
    """
    publish_schemas(session)
    parent = _active(session, doc_type)
    spec = registry.latest(doc_type)
    config = {**bootstrap_config(settings), **(config_overrides or {})}
    row = ExtractorVersion(
        doc_type=doc_type,
        schema_version=spec.version,
        model=settings.llm_model,
        prompt_hash=current_template_hash(doc_type, spec.version),
        fewshot_policy=f"layout-knn-k{config['fewshot']['k']}" if config.get("fewshot") else "none",
        config=config,
        status="candidate",
        parent_id=parent.id if parent else None,
    )
    session.add(row)
    session.flush()
    return row


def promote(session: Session, version_id: Any) -> ExtractorVersion:
    """Make a version active and retire the previous one, in one transaction.

    Versions are never deleted, so rolling back is promoting the retired version again.
    """
    row = session.get(ExtractorVersion, version_id, with_for_update=True)
    if row is None:
        raise ValueError(f"no extractor version {version_id}")
    current = _active(session, row.doc_type)
    if current is not None and current.id == row.id:
        return row
    if current is not None:
        current.status = "retired"
        session.flush()  # free the one-active slot before taking it
    row.status = "active"
    row.promoted_at = datetime.now(UTC)
    session.flush()
    return row
