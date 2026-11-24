"""Request dependencies: who is calling, and the storage/queue clients.

Storage and queue are dependencies (not module globals) so tests can swap in the in-memory
versions with ``app.dependency_overrides``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import text

from docflow_core.config import Settings, get_settings
from docflow_core.db.session import unscoped_session
from docflow_core.queue import QueueClient, SqsQueue
from docflow_core.security import hash_api_key
from docflow_core.storage import ObjectStore, S3Store


@dataclass(frozen=True)
class Principal:
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    role: str


_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="missing or invalid API key",
    headers={"WWW-Authenticate": "ApiKey"},
)


def get_principal(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    if not x_api_key:
        raise _UNAUTHORIZED
    with unscoped_session() as session:
        row = session.execute(
            text("SELECT tenant_id, user_id, role FROM auth_api_key(:h)"),
            {"h": hash_api_key(x_api_key)},
        ).first()
    if row is None:
        raise _UNAUTHORIZED
    return Principal(tenant_id=row.tenant_id, user_id=row.user_id, role=row.role)


@lru_cache
def _store() -> ObjectStore:
    return S3Store(get_settings())


@lru_cache
def _queue() -> QueueClient:
    return SqsQueue(get_settings())


def get_store() -> ObjectStore:
    return _store()


def get_queue() -> QueueClient:
    return _queue()


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
Store = Annotated[ObjectStore, Depends(get_store)]
Queue = Annotated[QueueClient, Depends(get_queue)]
AppSettings = Annotated[Settings, Depends(get_settings)]
