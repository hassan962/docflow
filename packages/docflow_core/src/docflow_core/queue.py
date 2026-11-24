"""Per-stage queues: the message format, an SQS adapter and an in-memory adapter for tests.

Every stage has its own queue (and its own dead-letter queue), so a slow OCR run never holds
up extraction. The semantics the workers rely on are SQS's:

- a received message is hidden for the queue's visibility timeout, then reappears unless it
  was deleted (that is the retry);
- every receive increments ``ApproximateReceiveCount``;
- after ``maxReceiveCount`` receives the queue moves the message to its dead-letter queue.

The in-memory queue reproduces exactly those three rules so the worker can be tested
without Docker.
"""

from __future__ import annotations

import itertools
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import boto3
from pydantic import BaseModel, ConfigDict

from docflow_core.config import STAGES, Settings, Stage


class StageMessage(BaseModel):
    """What a stage needs to find its work. Deliberately tiny: the data lives in S3/Postgres."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: uuid.UUID
    document_id: uuid.UUID
    stage: Stage


@dataclass(frozen=True)
class Received:
    body: str  # parsed by the worker, so a malformed body can be handled as a poison message
    receipt: str
    receive_count: int


class QueueClient(Protocol):
    def send(self, message: StageMessage) -> None: ...
    def receive(self, stage: Stage, max_messages: int = 1) -> list[Received]: ...
    def delete(self, stage: Stage, receipt: str) -> None: ...
    def retry_later(self, stage: Stage, receipt: str, delay_seconds: int) -> None: ...


class SqsQueue:
    def __init__(self, settings: Settings) -> None:
        self._client: Any = boto3.client(
            "sqs", region_name=settings.aws_region, endpoint_url=settings.aws_endpoint_url_sqs
        )
        self._wait = settings.receive_wait_seconds
        self._urls: dict[Stage, str] = {}
        for stage in STAGES:
            url = settings.queue_url(stage)
            if url:
                self._urls[stage] = url

    def _url(self, stage: Stage) -> str:
        try:
            return self._urls[stage]
        except KeyError:
            raise RuntimeError(f"QUEUE_URL_{stage.upper()} is not configured") from None

    def send(self, message: StageMessage) -> None:
        self._client.send_message(
            QueueUrl=self._url(message.stage), MessageBody=message.model_dump_json()
        )

    def receive(self, stage: Stage, max_messages: int = 1) -> list[Received]:
        response = self._client.receive_message(
            QueueUrl=self._url(stage),
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=self._wait,  # long polling: no busy loop on an empty queue
            MessageSystemAttributeNames=["ApproximateReceiveCount"],
        )
        return [
            Received(
                body=m["Body"],
                receipt=m["ReceiptHandle"],
                receive_count=int(m.get("Attributes", {}).get("ApproximateReceiveCount", 1)),
            )
            for m in response.get("Messages", [])
        ]

    def delete(self, stage: Stage, receipt: str) -> None:
        self._client.delete_message(QueueUrl=self._url(stage), ReceiptHandle=receipt)

    def retry_later(self, stage: Stage, receipt: str, delay_seconds: int) -> None:
        # Shortening the visibility timeout makes the message reappear after the backoff
        # delay instead of after the full timeout.
        self._client.change_message_visibility(
            QueueUrl=self._url(stage), ReceiptHandle=receipt, VisibilityTimeout=delay_seconds
        )


@dataclass
class _Entry:
    body: str
    receive_count: int = 0
    visible_at: float = 0.0
    receipt: str | None = None


@dataclass
class InMemoryQueue:
    """SQS semantics in a few lines: visibility timeout, receive counts, redrive to a DLQ."""

    visibility_timeout: float = 30.0
    max_receives: int = 3
    clock: Callable[[], float] = time.monotonic
    queues: dict[Stage, list[_Entry]] = field(default_factory=dict)
    dead_letters: dict[Stage, list[str]] = field(default_factory=dict)
    _ids: itertools.count[int] = field(default_factory=itertools.count)

    def send(self, message: StageMessage) -> None:
        self.send_raw(message.stage, message.model_dump_json())

    def send_raw(self, stage: Stage, body: str) -> None:
        self.queues.setdefault(stage, []).append(_Entry(body=body))

    def receive(self, stage: Stage, max_messages: int = 1) -> list[Received]:
        now = self.clock()
        out: list[Received] = []
        entries = self.queues.setdefault(stage, [])
        for entry in list(entries):
            if len(out) >= max_messages or entry.visible_at > now:
                continue
            if entry.receive_count >= self.max_receives:  # redrive, as SQS does
                entries.remove(entry)
                self.dead_letters.setdefault(stage, []).append(entry.body)
                continue
            entry.receive_count += 1
            entry.visible_at = now + self.visibility_timeout
            entry.receipt = f"r{next(self._ids)}"
            out.append(Received(entry.body, entry.receipt, entry.receive_count))
        return out

    def _find(self, stage: Stage, receipt: str) -> _Entry:
        for entry in self.queues.get(stage, []):
            if entry.receipt == receipt:
                return entry
        raise KeyError(f"unknown receipt {receipt}")

    def delete(self, stage: Stage, receipt: str) -> None:
        self.queues[stage].remove(self._find(stage, receipt))

    def retry_later(self, stage: Stage, receipt: str, delay_seconds: int) -> None:
        self._find(stage, receipt).visible_at = self.clock() + delay_seconds

    def pending(self, stage: Stage) -> list[StageMessage]:
        return [StageMessage.model_validate_json(e.body) for e in self.queues.get(stage, [])]
