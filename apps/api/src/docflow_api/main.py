"""FastAPI application.

Health checks plus one router per area under /v1. Every /v1 route requires an API key
(X-API-Key header) and runs inside a tenant-scoped database session.
"""

from __future__ import annotations

from fastapi import FastAPI, Response, status
from sqlalchemy import text

from docflow_api.routers import documents
from docflow_core import __version__
from docflow_core.db.session import unscoped_session
from docflow_core.logs import configure_logging
from docflow_core.schemas import registry


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="DocFlow API", version=__version__)
    app.include_router(documents.router)

    @app.get("/healthz", tags=["ops"])
    def healthz() -> dict[str, str]:
        """Liveness: the process is up. Never touches dependencies."""
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    def readyz(response: Response) -> dict[str, object]:
        """Readiness: the database answers and the schemas are loaded."""
        try:
            with unscoped_session() as session:
                session.execute(text("SELECT 1"))
            database = "ok"
        except Exception as exc:  # any failure means "not ready"
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            database = f"error: {type(exc).__name__}"
        return {"database": database, "schemas": [s.key for s in registry.all()]}

    return app


app = create_app()
