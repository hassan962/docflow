"""The worker contract, shared by every stage.

For each message:

1. **Claim.** In a short transaction, load the document and its ``stage_runs`` row. If the
   stage is already ``done``, this is a redelivery: acknowledge and stop. Otherwise mark it
   ``running`` and count the attempt.
2. **Run.** Do the slow work (OCR, an LLM call) with no database transaction open, so no
   locks are held for seconds or minutes. Outputs are written to S3 under deterministic
   keys, so a rerun overwrites them with the same content.
3. **Save.** In a second transaction, write the stage's rows, mark the stage ``done``,
   advance the document status and enqueue the next stage's message, then commit.
4. **Acknowledge.** Delete the incoming message only after that commit.

Output first, then enqueue, then delete: a crash at any point leaves either nothing done
(the message reappears) or work done that step 1 will recognise next time.

Failures:

- A ``PermanentError`` (unreadable file, too many pages) marks the document failed at once;
  retrying cannot help.
- Any other error is retried with exponential backoff (10s, 20s, 40s, ...) by shortening the
  message's visibility timeout. On the last allowed receive the document is marked failed
  with the reason, and SQS moves the message to the dead-letter queue on its next receive.
- A message that cannot even be parsed (a poison message) is never acknowledged, so it ends
  up in the dead-letter queue without touching any document.
"""

from __future__ import annotations

import enum
import signal
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from types import FrameType
from typing import Any, Protocol, TypeVar

import structlog
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from docflow_core.config import Settings, Stage
from docflow_core.db.models import Document, Page, StageRun
from docflow_core.db.session import tenant_session
from docflow_core.errors import PermanentError
from docflow_core.queue import QueueClient, Received, StageMessage
from docflow_core.storage import ObjectStore

log = structlog.get_logger()

R = TypeVar("R")
MAX_ERROR_CHARS = 2000


@dataclass(frozen=True)
class DocumentSnapshot:
    """What a stage may read while no transaction is open."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    s3_key: str
    content_type: str
    doc_type: str
    ocr_keys: tuple[str, ...] = ()  # page OCR JSON keys in page order (after the OCR stage)


@dataclass
class Deps:
    settings: Settings
    store: ObjectStore
    queue: QueueClient


class StageHandler(Protocol[R]):
    name: Stage

    def run(self, doc: DocumentSnapshot, deps: Deps) -> R: ...
    def save(self, session: Session, doc: Document, result: R) -> None: ...
    def route(self, result: R) -> tuple[str, Stage | None]:
        """The document's new status and the next stage (None: the pipeline stops here)."""
        ...


class Outcome(enum.StrEnum):
    DONE = "done"
    ALREADY_DONE = "already_done"
    MISSING = "missing_document"
    RETRY = "retry_scheduled"
    FAILED = "failed"
    POISON = "poison_message"


def _now() -> datetime:
    return datetime.now(UTC)


def _claim(session: Session, msg: StageMessage) -> tuple[Outcome | None, DocumentSnapshot | None]:
    doc = session.get(Document, msg.document_id)
    if doc is None:
        return Outcome.MISSING, None
    run = session.scalar(
        select(StageRun)
        .where(StageRun.document_id == msg.document_id, StageRun.stage == msg.stage)
        .with_for_update()
    )
    if run is not None and run.status == "done":
        return Outcome.ALREADY_DONE, None
    if run is None:
        session.add(StageRun(tenant_id=msg.tenant_id, document_id=doc.id, stage=msg.stage))
    else:
        run.status, run.attempts, run.error, run.started_at = (
            "running",
            run.attempts + 1,
            None,
            _now(),
        )
    ocr_keys = session.scalars(
        select(Page.ocr_json_key).where(Page.document_id == doc.id).order_by(Page.page_no)
    ).all()
    snapshot = DocumentSnapshot(
        doc.id,
        doc.tenant_id,
        doc.s3_key,
        doc.content_type,
        doc.doc_type,
        tuple(k for k in ocr_keys if k),
    )
    return None, snapshot


def _stage_run(session: Session, msg: StageMessage) -> StageRun:
    run = session.scalar(
        select(StageRun).where(StageRun.document_id == msg.document_id, StageRun.stage == msg.stage)
    )
    assert run is not None, "claimed stage run disappeared"
    return run


def _record_failure(msg: StageMessage, error: str, *, final: bool) -> None:
    with tenant_session(msg.tenant_id) as session:
        run = _stage_run(session, msg)
        run.status, run.error, run.finished_at = "failed", error[:MAX_ERROR_CHARS], _now()
        if final:
            doc = session.get(Document, msg.document_id)
            if doc is not None:
                doc.status = "failed"
                doc.failure_reason = f"{msg.stage}: {error}"[:MAX_ERROR_CHARS]


def handle(received: Received, handler: StageHandler[Any], deps: Deps) -> Outcome:
    stage = handler.name
    try:
        msg = StageMessage.model_validate_json(received.body)
        if msg.stage != stage:
            raise ValueError(f"message for stage {msg.stage!r} on the {stage!r} queue")
    except (ValidationError, ValueError) as exc:
        log.error("poison_message", stage=stage, error=str(exc), receives=received.receive_count)
        return Outcome.POISON  # not deleted: the queue's redrive policy dead-letters it

    bound = log.bind(tenant_id=str(msg.tenant_id), document_id=str(msg.document_id), stage=stage)

    with tenant_session(msg.tenant_id) as session:
        early, snapshot = _claim(session, msg)
    if early is not None:
        deps.queue.delete(stage, received.receipt)
        bound.info("skipped", outcome=early.value)
        return early
    assert snapshot is not None

    try:
        result = handler.run(snapshot, deps)
        with tenant_session(msg.tenant_id) as session:
            doc = session.get(Document, msg.document_id, with_for_update=True)
            if doc is None:
                raise PermanentError("document was deleted while being processed")
            handler.save(session, doc, result)
            doc.status, next_stage = handler.route(result)
            run = _stage_run(session, msg)
            run.status, run.finished_at = "done", _now()
            if next_stage is not None:
                deps.queue.send(
                    StageMessage(
                        tenant_id=msg.tenant_id, document_id=msg.document_id, stage=next_stage
                    )
                )
    except PermanentError as exc:
        _record_failure(msg, str(exc), final=True)
        deps.queue.delete(stage, received.receipt)
        bound.warning("failed_permanently", error=str(exc))
        return Outcome.FAILED
    except Exception as exc:
        final = received.receive_count >= deps.settings.max_receives
        _record_failure(msg, f"{type(exc).__name__}: {exc}", final=final)
        if final:
            bound.error("failed_after_retries", error=str(exc), receives=received.receive_count)
            return Outcome.FAILED
        delay = deps.settings.retry_base_seconds * 2 ** (received.receive_count - 1)
        deps.queue.retry_later(stage, received.receipt, delay)
        bound.warning("retry_scheduled", error=str(exc), delay_s=delay)
        return Outcome.RETRY

    deps.queue.delete(stage, received.receipt)
    bound.info("stage_done")
    return Outcome.DONE


class _Stop:
    requested = False

    def __call__(self, signum: int, frame: FrameType | None) -> None:
        log.info("shutdown_requested", signal=signum)
        self.requested = True


def run_forever(handler: StageHandler[Any], deps: Deps) -> None:
    """Poll the stage's queue until SIGTERM/SIGINT, finishing the message in hand first."""
    stop = _Stop()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("worker_started", stage=handler.name)
    while not stop.requested:
        try:
            for received in deps.queue.receive(handler.name):
                handle(received, handler, deps)
        except Exception:
            # e.g. the database is briefly unreachable. The message was not deleted, so it
            # reappears after its visibility timeout; the worker itself keeps running.
            log.exception("worker_loop_error", stage=handler.name)
    log.info("worker_stopped", stage=handler.name)
