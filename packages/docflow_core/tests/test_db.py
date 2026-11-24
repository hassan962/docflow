"""Database guarantees: tenant isolation, append-only corrections, one active extractor.

These are the promises the design makes "in the database, not only in API code", so they
are tested against a real Postgres, not mocks.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from docflow_core.db.models import (
    TENANT_TABLES,
    ApiKey,
    Base,
    Correction,
    Document,
    Extraction,
    ExtractorVersion,
    Tenant,
    User,
)
from docflow_core.db.publish import publish_schemas
from docflow_core.db.session import owner_session, tenant_session, unscoped_session
from docflow_core.security import generate_api_key

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("migrated_db")]


def _make_tenant(name: str) -> tuple[uuid.UUID, uuid.UUID]:
    """Create a tenant and one reviewer as the owner; return (tenant_id, user_id)."""
    with owner_session() as s:
        t = Tenant(name=name)
        s.add(t)
        s.flush()
        u = User(tenant_id=t.id, email=f"reviewer@{name}.test", role="reviewer")
        s.add(u)
        s.flush()
        return t.id, u.id


def _add_document(tenant_id: uuid.UUID, sha: str) -> uuid.UUID:
    with tenant_session(tenant_id) as s:
        d = Document(
            tenant_id=tenant_id, s3_key=f"{tenant_id}/raw/x.pdf", sha256=sha,
            filename="x.pdf", content_type="application/pdf",
        )  # fmt: skip
        s.add(d)
        s.flush()
        return d.id


def test_models_match_the_migrated_schema() -> None:
    with owner_session() as s:
        insp = inspect(s.connection())
        for table in Base.metadata.sorted_tables:
            db_cols = {c["name"] for c in insp.get_columns(table.name)}
            model_cols = {c.name for c in table.columns}
            assert db_cols == model_cols, f"{table.name}: models and migration disagree"


def test_every_tenant_table_has_rls_enabled() -> None:
    with owner_session() as s:
        rows = s.execute(
            text("SELECT relname FROM pg_class WHERE relrowsecurity AND relkind = 'r'")
        ).scalars()
        protected = set(rows)
    assert set(TENANT_TABLES) | {"tenants"} <= protected


def test_a_tenant_sees_zero_rows_of_another_tenant() -> None:
    tenant_a, _ = _make_tenant(f"a-{uuid.uuid4().hex[:6]}")
    tenant_b, _ = _make_tenant(f"b-{uuid.uuid4().hex[:6]}")
    doc_a = _add_document(tenant_a, uuid.uuid4().hex * 2)

    with tenant_session(tenant_b) as s:
        assert s.get(Document, doc_a) is None
        assert s.scalars(select(Document).where(Document.tenant_id == tenant_a)).all() == []
        assert s.scalars(select(Tenant)).all() == [s.get(Tenant, tenant_b)]

    with tenant_session(tenant_a) as s:
        assert s.get(Document, doc_a) is not None


def test_no_tenant_set_means_no_rows() -> None:
    tenant, _ = _make_tenant(f"c-{uuid.uuid4().hex[:6]}")
    _add_document(tenant, uuid.uuid4().hex * 2)
    with unscoped_session() as s:
        assert s.scalars(select(Document)).all() == []


def test_a_tenant_cannot_write_rows_for_another_tenant() -> None:
    tenant_a, _ = _make_tenant(f"d-{uuid.uuid4().hex[:6]}")
    tenant_b, _ = _make_tenant(f"e-{uuid.uuid4().hex[:6]}")
    with pytest.raises(DBAPIError, match="row-level security"), tenant_session(tenant_a) as s:
        s.add(
            Document(
                tenant_id=tenant_b,
                s3_key="k",
                sha256="f" * 64,
                filename="x.pdf",
                content_type="application/pdf",
            )
        )
        s.flush()


def test_duplicate_upload_is_rejected_per_tenant() -> None:
    tenant, _ = _make_tenant(f"f-{uuid.uuid4().hex[:6]}")
    sha = uuid.uuid4().hex * 2
    _add_document(tenant, sha)
    with pytest.raises(IntegrityError):
        _add_document(tenant, sha)


def test_api_key_lookup_works_before_the_tenant_is_known() -> None:
    tenant, user = _make_tenant(f"g-{uuid.uuid4().hex[:6]}")
    key = generate_api_key()
    with owner_session() as s:
        s.add(ApiKey(tenant_id=tenant, user_id=user, prefix=key.prefix, key_hash=key.key_hash))
    with unscoped_session() as s:
        row = s.execute(
            text("SELECT tenant_id, user_id, role FROM auth_api_key(:h)"), {"h": key.key_hash}
        ).one()
        assert (row.tenant_id, row.user_id, row.role) == (tenant, user, "reviewer")
        assert s.execute(text("SELECT * FROM auth_api_key('nope')")).first() is None


def _active_extractor() -> uuid.UUID:
    with owner_session() as s:
        publish_schemas(s)
        existing = s.scalar(
            select(ExtractorVersion).where(
                ExtractorVersion.doc_type == "invoice", ExtractorVersion.status == "active"
            )
        )
        if existing:
            return existing.id
        ev = ExtractorVersion(
            doc_type="invoice", schema_version=2, model="test-model", prompt_hash="0" * 64,
            status="active",
        )  # fmt: skip
        s.add(ev)
        s.flush()
        return ev.id


def test_only_one_active_extractor_per_doc_type() -> None:
    _active_extractor()
    with pytest.raises(IntegrityError), owner_session() as s:
        s.add(
            ExtractorVersion(
                doc_type="invoice",
                schema_version=2,
                model="other",
                prompt_hash="1" * 64,
                status="active",
            )
        )
        s.flush()


def test_corrections_are_append_only() -> None:
    tenant, reviewer = _make_tenant(f"h-{uuid.uuid4().hex[:6]}")
    doc = _add_document(tenant, uuid.uuid4().hex * 2)
    extractor = _active_extractor()

    with tenant_session(tenant) as s:
        ex = Extraction(tenant_id=tenant, document_id=doc, extractor_version_id=extractor)
        s.add(ex)
        s.flush()
        c = Correction(
            tenant_id=tenant, extraction_id=ex.id, field="total", action="edit",
            old_value={"value": "1180.00"}, new_value={"value": "1108.00"}, reviewer_id=reviewer,
        )  # fmt: skip
        s.add(c)
        s.flush()
        correction_id = c.id

    # The app role has no UPDATE/DELETE privilege on corrections at all...
    with pytest.raises(DBAPIError, match="permission denied"), tenant_session(tenant) as s:
        s.execute(text("DELETE FROM corrections WHERE id = :id"), {"id": correction_id})

    # ...and even the table owner is stopped by the trigger.
    with pytest.raises(DBAPIError, match="append-only"), owner_session() as s:
        s.execute(
            text("UPDATE corrections SET field = 'tax' WHERE id = :id"), {"id": correction_id}
        )
