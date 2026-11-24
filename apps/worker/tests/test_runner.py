"""The worker contract, tested on the OCR stage with in-memory S3/SQS and a real Postgres."""

from __future__ import annotations

import uuid
from io import BytesIO

import pytest
from PIL import Image, ImageDraw
from sqlalchemy import select

from conftest import new_tenant
from docflow_core.config import Settings
from docflow_core.db.models import Document, Page, StageRun
from docflow_core.db.session import tenant_session
from docflow_core.ocr import OcrResult, OcrWord
from docflow_core.queue import InMemoryQueue, StageMessage
from docflow_core.storage import InMemoryStore, raw_key
from docflow_worker.runner import Deps, Outcome, handle
from docflow_worker.stages.ocr import OcrStage

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("migrated_db")]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeEngine:
    name = "fake"

    def __init__(self, fail_times: int = 0) -> None:
        self.calls = 0
        self.fail_times = fail_times

    def run(self, image: bytes) -> OcrResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("engine crashed")
        with Image.open(BytesIO(image)) as img:
            w, h = img.size
        words = [
            OcrWord("Total", (10, 10, 60, 25), 0.9, 0),
            OcrWord("1180.00", (70, 10, 140, 25), 0.8, 0),
        ]
        return OcrResult(words=words, page_width=w, page_height=h, engine=self.name)


def _png() -> bytes:
    img = Image.new("RGB", (400, 300), "white")
    ImageDraw.Draw(img).rectangle([40, 40, 300, 60], fill="black")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class World:
    """One tenant, one uploaded document, and the worker's dependencies."""

    def __init__(self, data: bytes | None = None, content_type: str = "image/png") -> None:
        self.tenant_id, _, _ = new_tenant()
        self.clock = Clock()
        self.queue = InMemoryQueue(visibility_timeout=300, max_receives=3, clock=self.clock)
        self.store = InMemoryStore()
        self.settings = Settings(max_receives=3, retry_base_seconds=10, ocr_dpi=150, max_pages=2)
        self.deps = Deps(settings=self.settings, store=self.store, queue=self.queue)
        self.document_id = uuid.uuid4()
        key = raw_key(self.tenant_id, self.document_id, content_type)
        self.store.put(key, data if data is not None else _png(), content_type)
        with tenant_session(self.tenant_id) as s:
            s.add(
                Document(
                    id=self.document_id,
                    tenant_id=self.tenant_id,
                    s3_key=key,
                    sha256=uuid.uuid4().hex * 2,
                    filename="x",
                    content_type=content_type,
                )
            )
        self.queue.send(self.message())

    def message(self, stage: str = "ocr") -> StageMessage:
        return StageMessage(tenant_id=self.tenant_id, document_id=self.document_id, stage=stage)  # type: ignore[arg-type]

    def step(self, stage: OcrStage) -> Outcome | None:
        received = self.queue.receive("ocr")
        if not received:
            return None
        return handle(received[0], stage, self.deps)

    def document(self) -> Document:
        with tenant_session(self.tenant_id) as s:
            doc = s.get(Document, self.document_id)
            assert doc is not None
            return doc

    def stage_run(self) -> StageRun:
        with tenant_session(self.tenant_id) as s:
            run = s.scalar(select(StageRun).where(StageRun.document_id == self.document_id))
            assert run is not None
            return run


def test_happy_path_writes_pages_and_hands_off_to_extract() -> None:
    w = World()
    assert w.step(OcrStage(FakeEngine())) is Outcome.DONE

    doc = w.document()
    assert (doc.status, doc.page_count) == ("ocr_done", 1)
    with tenant_session(w.tenant_id) as s:
        [page] = s.scalars(select(Page).where(Page.document_id == w.document_id)).all()
        assert (page.ocr_engine, page.mean_ocr_conf) == ("fake", pytest.approx(0.85))
    assert w.store.exists(page.image_key) and w.store.exists(page.ocr_json_key or "")
    assert OcrResult.from_json(w.store.get(page.ocr_json_key or "")).lines() == ["Total 1180.00"]
    assert (w.stage_run().status, w.stage_run().attempts) == ("done", 1)
    assert w.queue.pending("ocr") == []  # acknowledged
    assert w.queue.pending("extract") == [w.message("extract")]  # next stage queued


def test_redelivered_message_is_acknowledged_without_redoing_work() -> None:
    w, engine = World(), FakeEngine()
    stage = OcrStage(engine)
    assert w.step(stage) is Outcome.DONE
    w.queue.send(w.message())  # e.g. SQS at-least-once delivery hands it out again
    assert w.step(stage) is Outcome.ALREADY_DONE
    assert engine.calls == 1
    assert w.queue.pending("ocr") == []
    assert len(w.queue.pending("extract")) == 1


def test_transient_failure_is_retried_with_backoff_then_succeeds() -> None:
    w = World()
    stage = OcrStage(FakeEngine(fail_times=1))
    assert w.step(stage) is Outcome.RETRY
    assert w.stage_run().status == "failed" and "engine crashed" in (w.stage_run().error or "")
    assert w.step(stage) is None  # hidden for the 10 s backoff...
    w.clock.now += 10
    assert w.step(stage) is Outcome.DONE  # ...then retried
    assert (w.stage_run().status, w.stage_run().attempts) == ("done", 2)
    assert w.document().status == "ocr_done"


def test_repeated_failure_marks_the_document_failed_and_dead_letters_the_message() -> None:
    w = World()
    stage = OcrStage(FakeEngine(fail_times=99))
    assert w.step(stage) is Outcome.RETRY  # receive 1, retry in 10 s
    w.clock.now += 10
    assert w.step(stage) is Outcome.RETRY  # receive 2, retry in 20 s
    w.clock.now += 19
    assert w.step(stage) is None  # backoff doubled: not yet
    w.clock.now += 1
    assert w.step(stage) is Outcome.FAILED  # receive 3: the last allowed

    doc = w.document()
    assert doc.status == "failed"
    assert (doc.failure_reason or "").startswith("ocr: RuntimeError: engine crashed")
    w.clock.now += 300  # visibility timeout passes; the queue redrives on the next receive
    assert w.step(stage) is None
    assert len(w.queue.dead_letters["ocr"]) == 1
    assert w.queue.pending("extract") == []


def test_permanent_error_fails_at_once_without_retries() -> None:
    w = World(data=b"%PDF-1.4 definitely not a pdf", content_type="application/pdf")
    engine = FakeEngine()
    assert w.step(OcrStage(engine)) is Outcome.FAILED
    doc = w.document()
    assert doc.status == "failed" and "unreadable PDF" in (doc.failure_reason or "")
    assert engine.calls == 0
    assert w.queue.pending("ocr") == [] and "ocr" not in w.queue.dead_letters


def test_poison_message_is_never_acknowledged_and_ends_in_the_dlq() -> None:
    w = World()
    w.queue.queues["ocr"].clear()
    w.queue.send_raw("ocr", '{"this is": "not a stage message"}')
    stage = OcrStage(FakeEngine())
    for _ in range(3):
        assert w.step(stage) is Outcome.POISON
        w.clock.now += 300
    assert w.step(stage) is None
    assert w.queue.dead_letters["ocr"] == ['{"this is": "not a stage message"}']
    assert w.document().status == "uploaded"  # no document was touched


def test_message_for_a_missing_document_is_dropped() -> None:
    w = World()
    w.queue.queues["ocr"].clear()
    w.queue.send(StageMessage(tenant_id=w.tenant_id, document_id=uuid.uuid4(), stage="ocr"))
    assert w.step(OcrStage(FakeEngine())) is Outcome.MISSING
    assert w.queue.pending("ocr") == []
