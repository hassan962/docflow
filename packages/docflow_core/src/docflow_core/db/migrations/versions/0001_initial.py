"""Initial schema: all tables, indexes, row-level security, append-only corrections.

Revision ID: 0001
Revises:
Create Date: 2026-11-24
"""

from __future__ import annotations

from pathlib import Path

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def _run(name: str) -> None:
    # Send the file straight to psycopg with no parameters, so nothing parses "::uuid", "$$"
    # or the "%" in format('%I', ...) as placeholders. (SQLAlchemy's exec_driver_sql passes an
    # empty parameter list, which makes psycopg treat "%" as a placeholder.) It runs on the
    # same connection, inside Alembic's transaction.
    sql = (SQL_DIR / name).read_text(encoding="utf-8")
    driver_connection = op.get_bind().connection.driver_connection
    assert driver_connection is not None
    driver_connection.execute(sql)


def upgrade() -> None:
    _run("0001_initial.sql")


def downgrade() -> None:
    _run("0001_initial_down.sql")
