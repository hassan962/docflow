"""Publish the registered schemas into ``schema_versions``.

Safe to run on every deploy: an already-published version is left alone if its fingerprint
matches, and the run fails loudly if someone edited a published schema in code.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from docflow_core.db.models import SchemaVersion
from docflow_core.schemas import registry


class SchemaDriftError(RuntimeError):
    pass


def publish_schemas(session: Session) -> list[str]:
    published: list[str] = []
    for spec in registry.all():
        row = session.scalar(
            select(SchemaVersion).where(
                SchemaVersion.doc_type == spec.doc_type, SchemaVersion.version == spec.version
            )
        )
        if row is None:
            session.add(
                SchemaVersion(
                    doc_type=spec.doc_type,
                    version=spec.version,
                    json_schema=spec.json_schema(),
                    fingerprint=spec.fingerprint(),
                )
            )
            published.append(spec.key)
        elif row.fingerprint != spec.fingerprint():
            raise SchemaDriftError(
                f"{spec.key} in code differs from the published version; "
                "register a new version instead of editing a published one"
            )
    return published
