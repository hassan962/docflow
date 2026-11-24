"""Seed a local database: publish schemas, create a demo tenant, an admin and an API key.

    uv run python scripts/seed.py

Prints the API key once. Runs as the table owner, because creating the first tenant is an
admin action that no tenant-scoped session is allowed to do.
"""

from __future__ import annotations

from sqlalchemy import select

from docflow_core.db.models import ApiKey, Tenant, User
from docflow_core.db.publish import publish_schemas
from docflow_core.db.session import owner_session
from docflow_core.security import generate_api_key

DEMO_TENANT = "Demo Co"
DEMO_ADMIN = "admin@demo.local"


def main() -> None:
    with owner_session() as session:
        published = publish_schemas(session)
        print(f"schemas published: {', '.join(published) or 'none new'}")

        tenant = session.scalar(select(Tenant).where(Tenant.name == DEMO_TENANT))
        if tenant is not None:
            print(f"tenant {DEMO_TENANT!r} already exists ({tenant.id}); nothing else to do")
            return

        tenant = Tenant(name=DEMO_TENANT)
        session.add(tenant)
        session.flush()
        admin = User(tenant_id=tenant.id, email=DEMO_ADMIN, role="admin")
        session.add(admin)
        session.flush()
        key = generate_api_key()
        session.add(
            ApiKey(tenant_id=tenant.id, user_id=admin.id, prefix=key.prefix, key_hash=key.key_hash)
        )

    print(f"tenant: {tenant.id}")
    print(f"admin:  {DEMO_ADMIN}")
    print(f"API key (shown once, store it in .env as DOCFLOW_API_KEY): {key.secret}")


if __name__ == "__main__":
    main()
