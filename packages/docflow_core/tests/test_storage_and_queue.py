import uuid

import pytest

from docflow_core.config import Settings
from docflow_core.queue import InMemoryQueue, SqsQueue, StageMessage
from docflow_core.storage import (
    InMemoryStore,
    S3Store,
    ocr_json_key,
    page_image_key,
    raw_key,
    sniff_content_type,
)

T, D = uuid.UUID(int=1), uuid.UUID(int=2)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"%PDF-1.7\n...", "application/pdf"),
        (b"\x89PNG\r\n\x1a\n....", "image/png"),
        (b"\xff\xd8\xff\xe0....", "image/jpeg"),
        (b"GIF89a....", None),
        (b"hello", None),
        (b"", None),
    ],
)
def test_content_type_comes_from_magic_bytes(data: bytes, expected: str | None) -> None:
    assert sniff_content_type(data) == expected


def test_key_layout_is_per_tenant() -> None:
    assert raw_key(T, D, "application/pdf") == f"{T}/raw/{D}/original.pdf"
    assert page_image_key(T, D, 2) == f"{T}/pages/{D}/2.png"
    assert ocr_json_key(T, D, 2, "tesseract") == f"{T}/ocr/{D}/2.tesseract.json"


def test_in_memory_store() -> None:
    store = InMemoryStore()
    store.put("k", b"v", "text/plain")
    assert store.get("k") == b"v" and store.exists("k") and not store.exists("nope")
    with pytest.raises(KeyError):
        store.get("nope")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_in_memory_queue_follows_sqs_rules() -> None:
    clock = Clock()
    q = InMemoryQueue(visibility_timeout=30, max_receives=3, clock=clock)
    q.send(StageMessage(tenant_id=T, document_id=D, stage="ocr"))

    first = q.receive("ocr")
    assert [m.receive_count for m in first] == [1]
    assert q.receive("ocr") == []  # hidden while "in flight"

    clock.now = 31
    second = q.receive("ocr")
    assert [m.receive_count for m in second] == [2]

    q.retry_later("ocr", second[0].receipt, 5)  # backoff shorter than the timeout
    clock.now = 37
    third = q.receive("ocr")
    assert [m.receive_count for m in third] == [3]

    clock.now = 100
    assert q.receive("ocr") == []  # 4th receive attempt: moved to the DLQ instead
    assert len(q.dead_letters["ocr"]) == 1


def test_deleted_messages_do_not_come_back() -> None:
    clock = Clock()
    q = InMemoryQueue(clock=clock)
    q.send(StageMessage(tenant_id=T, document_id=D, stage="extract"))
    [m] = q.receive("extract")
    q.delete("extract", m.receipt)
    clock.now = 1000
    assert q.receive("extract") == [] and q.pending("extract") == []


# --- the real adapters against the local emulators -------------------------------------------


def _local_settings() -> Settings:
    return Settings(
        aws_endpoint_url_s3="http://localhost:9090",
        aws_endpoint_url_sqs="http://localhost:9324",
        s3_bucket="docflow-local",
        queue_url_validate="http://localhost:9324/000000000000/docflow-validate",
        receive_wait_seconds=1,
    )


@pytest.fixture
def aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "local")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "local")


@pytest.mark.emulators
@pytest.mark.usefixtures("aws_env")
def test_s3_store_against_s3mock() -> None:
    store = S3Store(_local_settings())
    key = f"tests/{uuid.uuid4()}.txt"
    assert not store.exists(key)
    store.put(key, b"hello", "text/plain")
    assert store.exists(key) and store.get(key) == b"hello"
    assert store.presign_get(key).startswith("http://localhost:9090/docflow-local/tests/")


@pytest.mark.emulators
@pytest.mark.usefixtures("aws_env")
def test_sqs_queue_against_elasticmq() -> None:
    """Uses the validate queue, which has no worker yet, so nothing else consumes it."""
    q = SqsQueue(_local_settings())
    while q.receive("validate", max_messages=10):  # drain leftovers from earlier runs
        pass
    msg = StageMessage(tenant_id=uuid.uuid4(), document_id=uuid.uuid4(), stage="validate")
    q.send(msg)
    [received] = q.receive("validate")
    assert StageMessage.model_validate_json(received.body) == msg
    assert received.receive_count == 1

    q.retry_later("validate", received.receipt, 0)  # make it visible again now
    [again] = q.receive("validate")
    assert again.receive_count == 2  # the worker's retry logic depends on this attribute
    q.delete("validate", again.receipt)
