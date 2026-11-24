"""Object storage: the S3 key layout, an S3 adapter and an in-memory adapter for tests.

Key layout (one prefix per tenant, so a bucket policy or a lifecycle rule can target one):

    <tenant_id>/raw/<document_id>/original.<ext>
    <tenant_id>/pages/<document_id>/<page_no>.png
    <tenant_id>/ocr/<document_id>/<page_no>.<engine>.json

Originals are written once and never modified ("stored immutably"). Page images and OCR
JSON are derived data: a retried OCR run overwrites them with identical output.
"""

from __future__ import annotations

import uuid
from typing import Any, Protocol

import boto3
from botocore.config import Config

from docflow_core.config import Settings

CONTENT_TYPES = {"application/pdf": "pdf", "image/png": "png", "image/jpeg": "jpg"}


def sniff_content_type(data: bytes) -> str | None:
    """Content type from the file's magic bytes, never from its name or the client's header."""
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return None


def raw_key(tenant_id: uuid.UUID, document_id: uuid.UUID, content_type: str) -> str:
    return f"{tenant_id}/raw/{document_id}/original.{CONTENT_TYPES[content_type]}"


def page_image_key(tenant_id: uuid.UUID, document_id: uuid.UUID, page_no: int) -> str:
    return f"{tenant_id}/pages/{document_id}/{page_no}.png"


def ocr_json_key(tenant_id: uuid.UUID, document_id: uuid.UUID, page_no: int, engine: str) -> str:
    return f"{tenant_id}/ocr/{document_id}/{page_no}.{engine}.json"


class ObjectStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def presign_get(self, key: str, expires_seconds: int = 300) -> str: ...


def _s3_client(settings: Settings, endpoint: str | None) -> Any:
    return boto3.client(
        "s3",
        region_name=settings.aws_region,
        endpoint_url=endpoint,
        # Path-style URLs (http://host/bucket/key) work with every S3 emulator.
        config=Config(
            s3={"addressing_style": "path"} if endpoint else {}, signature_version="s3v4"
        ),
    )


class S3Store:
    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket
        self._client = _s3_client(settings, settings.aws_endpoint_url_s3)
        # Presigned URLs embed the host in their signature, so they must be signed for the
        # host the browser will use, not the one this container uses.
        public = settings.s3_public_endpoint_url or settings.aws_endpoint_url_s3
        self._presigner = _s3_client(settings, public)

    def put(self, key: str, data: bytes, content_type: str) -> None:
        # Encryption at rest is the bucket's default (SSE-S3, set in Terraform), so every
        # object is encrypted without each call having to remember it.
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        body: bytes = self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return body

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except self._client.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise
        return True

    def presign_get(self, key: str, expires_seconds: int = 300) -> str:
        url: str = self._presigner.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_seconds
        )
        return url


class InMemoryStore:
    """Dict-backed store for tests."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = (data, content_type)

    def get(self, key: str) -> bytes:
        try:
            return self.objects[key][0]
        except KeyError:
            raise KeyError(f"no such object: {key}") from None

    def exists(self, key: str) -> bool:
        return key in self.objects

    def presign_get(self, key: str, expires_seconds: int = 300) -> str:
        return f"memory://{key}?expires={expires_seconds}"
