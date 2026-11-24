"""Upload and document endpoints against a real Postgres, with in-memory S3 and SQS."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import new_tenant
from docflow_api.deps import get_queue, get_store
from docflow_api.main import create_app
from docflow_core.config import Settings, get_settings
from docflow_core.db.models import Page
from docflow_core.db.session import tenant_session
from docflow_core.queue import InMemoryQueue
from docflow_core.storage import InMemoryStore

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("migrated_db")]


class Ctx:
    def __init__(self) -> None:
        self.store = InMemoryStore()
        self.queue = InMemoryQueue()
        self.app = create_app()
        self.app.dependency_overrides[get_store] = lambda: self.store
        self.app.dependency_overrides[get_queue] = lambda: self.queue
        self.client = TestClient(self.app)
        self.tenant_id, _, self.key = new_tenant()

    def headers(self, key: str | None = None) -> dict[str, str]:
        return {"X-API-Key": key or self.key}

    def upload(self, data: bytes, name: str = "inv.png", key: str | None = None) -> object:
        return self.client.post(
            "/v1/documents", files={"file": (name, data, "application/octet-stream")},
            headers=self.headers(key),
        )  # fmt: skip


@pytest.fixture
def ctx() -> Iterator[Ctx]:
    yield Ctx()


def _png(seed: int = 0) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (50, 50), (seed % 256, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def test_upload_requires_a_valid_api_key(ctx: Ctx) -> None:
    assert ctx.client.post("/v1/documents", files={"file": ("a", b"x")}).status_code == 401
    assert ctx.upload(_png(), key="dfk_00000000_wrong").status_code == 401  # type: ignore[attr-defined]


def test_upload_stores_the_file_and_queues_ocr(ctx: Ctx) -> None:
    response = ctx.upload(_png(1))
    assert response.status_code == 202  # type: ignore[attr-defined]
    body = response.json()  # type: ignore[attr-defined]
    assert (body["status"], body["content_type"], body["duplicate"]) == (
        "uploaded",
        "image/png",
        False,
    )
    doc_id = uuid.UUID(body["id"])
    assert ctx.store.exists(f"{ctx.tenant_id}/raw/{doc_id}/original.png")
    [msg] = ctx.queue.pending("ocr")
    assert (msg.document_id, msg.tenant_id, msg.stage) == (doc_id, ctx.tenant_id, "ocr")


def test_reupload_returns_the_existing_document_and_costs_nothing(ctx: Ctx) -> None:
    first = ctx.upload(_png(2)).json()  # type: ignore[attr-defined]
    again = ctx.upload(_png(2), name="renamed.png")
    assert again.status_code == 200  # type: ignore[attr-defined]
    assert again.json()["id"] == first["id"] and again.json()["duplicate"] is True  # type: ignore[attr-defined]
    assert len(ctx.queue.pending("ocr")) == 1  # nothing reprocessed


def test_same_file_from_another_tenant_is_a_separate_document(ctx: Ctx) -> None:
    _, _, other_key = new_tenant()
    a = ctx.upload(_png(3)).json()  # type: ignore[attr-defined]
    b = ctx.upload(_png(3), key=other_key)
    assert b.status_code == 202 and b.json()["id"] != a["id"]  # type: ignore[attr-defined]


def test_rejects_unsupported_and_empty_files(ctx: Ctx) -> None:
    assert ctx.upload(b"GIF89a fake gif", name="x.pdf").status_code == 415  # type: ignore[attr-defined]
    assert ctx.upload(b"").status_code == 422  # type: ignore[attr-defined]
    response = ctx.client.post(
        "/v1/documents", files={"file": ("a.png", _png())}, data={"doc_type": "passport"},
        headers=ctx.headers(),
    )  # fmt: skip
    assert response.status_code == 422


def test_rejects_files_over_the_size_limit(ctx: Ctx) -> None:
    ctx.app.dependency_overrides[get_settings] = lambda: Settings(max_upload_bytes=100)
    assert ctx.upload(_png() + b"\0" * 200).status_code == 413  # type: ignore[attr-defined]


def test_document_detail_and_list(ctx: Ctx) -> None:
    doc_id = ctx.upload(_png(4)).json()["id"]  # type: ignore[attr-defined]
    detail = ctx.client.get(f"/v1/documents/{doc_id}", headers=ctx.headers()).json()
    assert (detail["status"], detail["pages"], detail["stages"]) == ("uploaded", [], [])
    assert detail["latest_extraction"] is None

    listed = ctx.client.get("/v1/documents?status=uploaded", headers=ctx.headers()).json()
    assert doc_id in [d["id"] for d in listed["items"]]
    assert (
        ctx.client.get("/v1/documents?status=approved", headers=ctx.headers()).json()["items"] == []
    )
    assert ctx.client.get("/v1/documents?status=bogus", headers=ctx.headers()).status_code == 422


def test_other_tenants_get_404_not_403(ctx: Ctx) -> None:
    doc_id = ctx.upload(_png(5)).json()["id"]  # type: ignore[attr-defined]
    _, _, other_key = new_tenant()
    assert (
        ctx.client.get(f"/v1/documents/{doc_id}", headers=ctx.headers(other_key)).status_code == 404
    )
    assert ctx.client.get("/v1/documents", headers=ctx.headers(other_key)).json()["items"] == []


def test_page_image_returns_a_short_lived_url(ctx: Ctx) -> None:
    doc_id = uuid.UUID(ctx.upload(_png(6)).json()["id"])  # type: ignore[attr-defined]
    url = f"/v1/documents/{doc_id}/pages/1/image"
    assert ctx.client.get(url, headers=ctx.headers()).status_code == 404  # not OCR'd yet
    with tenant_session(ctx.tenant_id) as s:
        s.add(Page(tenant_id=ctx.tenant_id, document_id=doc_id, page_no=1, image_key="k/1.png"))
    body = ctx.client.get(url, headers=ctx.headers()).json()
    assert body == {"url": "memory://k/1.png?expires=300", "expires_in": 300}
