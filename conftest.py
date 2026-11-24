"""Shared test configuration.

Database tests run against ``docflow_test`` (created by the Postgres init script), never the
dev database. Emulator tests talk to S3Mock and ElasticMQ from docker compose. Either group
is skipped when its service is not reachable, so ``pytest`` always works on a laptop.
"""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
OWNER_URL = os.environ.get(
    "TEST_MIGRATION_DATABASE_URL",
    "postgresql+psycopg://docflow_owner:owner_dev_pw@localhost:5432/docflow_test",
)
APP_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://docflow_app:app_dev_pw@localhost:5432/docflow_test",
)


def _postgres_up() -> bool:
    try:
        import psycopg

        dsn = OWNER_URL.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(dsn, connect_timeout=2):
            return True
    except Exception:
        return False


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("localhost", port)) == 0


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    checks = {
        "db": (_postgres_up, "Postgres not reachable; run `docker compose up -d db`"),
        "emulators": (
            lambda: _port_open(9090) and _port_open(9324),
            "S3Mock/ElasticMQ not reachable; run `docker compose up -d s3mock elasticmq`",
        ),
    }
    for marker, (check, reason) in checks.items():
        marked = [i for i in items if i.get_closest_marker(marker)]
        if marked and not check():
            for item in marked:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def migrated_db() -> Iterator[None]:
    """Point the app at the test database and migrate it from scratch."""
    from alembic import command
    from alembic.config import Config

    from docflow_core.config import get_settings
    from docflow_core.db import session as db_session

    os.environ["DATABASE_URL"] = APP_URL
    os.environ["MIGRATION_DATABASE_URL"] = OWNER_URL
    get_settings.cache_clear()
    for cached in (db_session.app_engine, db_session.owner_engine, db_session._app_sessions):
        cached.cache_clear()

    cfg = Config(str(ROOT / "alembic.ini"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    yield


def new_tenant(name: str | None = None, role: str = "admin") -> tuple[uuid.UUID, uuid.UUID, str]:
    """Create a tenant, one user and an API key. Returns (tenant_id, user_id, api_key)."""
    from docflow_core.db.models import ApiKey, Tenant, User
    from docflow_core.db.session import owner_session
    from docflow_core.security import generate_api_key

    name = name or f"t-{uuid.uuid4().hex[:8]}"
    key = generate_api_key()
    with owner_session() as s:
        tenant = Tenant(name=name)
        s.add(tenant)
        s.flush()
        user = User(tenant_id=tenant.id, email=f"{role}@{name}.test", role=role)
        s.add(user)
        s.flush()
        s.add(
            ApiKey(tenant_id=tenant.id, user_id=user.id, prefix=key.prefix, key_hash=key.key_hash)
        )
        return tenant.id, user.id, key.secret
