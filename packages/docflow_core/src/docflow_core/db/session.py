"""Engines and sessions.

``tenant_session`` is how every request and every worker talks to the database. It opens a
transaction and sets ``app.tenant_id`` with ``set_config(..., is_local => true)``, which is the
same as ``SET LOCAL``: the setting lives only until the transaction ends. A pooled connection
handed to the next request therefore cannot carry the previous tenant's id, and the RLS
policies see exactly one tenant per transaction.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from docflow_core.config import get_settings


@lru_cache
def app_engine() -> Engine:
    """Engine for the RLS-restricted application role."""
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def owner_engine() -> Engine:
    """Engine for the table owner. Only migrations, seeding and admin scripts use it."""
    return create_engine(get_settings().migration_database_url, pool_pre_ping=True)


@lru_cache
def _app_sessions() -> sessionmaker[Session]:
    return sessionmaker(app_engine(), expire_on_commit=False)


@contextmanager
def tenant_session(tenant_id: uuid.UUID | str) -> Iterator[Session]:
    """A transaction scoped to one tenant. Commits on success, rolls back on error."""
    tenant = uuid.UUID(str(tenant_id))  # rejects anything that is not a UUID
    with _app_sessions()() as session, session.begin():
        session.execute(
            text("SELECT set_config('app.tenant_id', :tenant, true)"), {"tenant": str(tenant)}
        )
        yield session


@contextmanager
def unscoped_session() -> Iterator[Session]:
    """App-role transaction with no tenant: RLS hides every tenant-owned row.

    Used for global tables (schema and extractor versions, eval runs) and for API-key
    lookup through the ``auth_api_key`` function.
    """
    with _app_sessions()() as session, session.begin():
        yield session


@contextmanager
def owner_session() -> Iterator[Session]:
    with sessionmaker(owner_engine(), expire_on_commit=False)() as session, session.begin():
        yield session
