"""Operator commands. Run inside the stack so they see its database and queues:

    docker compose exec api python scripts/admin.py extractors list
    docker compose exec api python scripts/admin.py extractors register --promote
    docker compose exec api python scripts/admin.py extractors promote <version-id>
    docker compose exec api python scripts/admin.py reprocess --stage extract --status extracted

Promotion is manual until phase 5's evaluation gate, which calls the same ``promote``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlalchemy import delete, select

from docflow_core.config import STAGES, get_settings
from docflow_core.db.models import Document, ExtractorVersion, StageRun
from docflow_core.db.session import owner_session
from docflow_core.extract.versions import current_template_hash, promote, register_candidate
from docflow_core.queue import SqsQueue, StageMessage


def cmd_list(args: argparse.Namespace) -> None:
    with owner_session() as s:
        rows = s.scalars(select(ExtractorVersion).order_by(ExtractorVersion.created_at)).all()
        for r in rows:
            cfg = r.config or {}
            drift = ""
            if r.status == "active" and r.prompt_hash != current_template_hash(
                r.doc_type, r.schema_version
            ):
                drift = "  <- prompt in code differs: register a new version"
            calibrated = "yes" if cfg.get("calibration") else "no"
            print(
                f"{r.id}  {r.doc_type}/v{r.schema_version}  {r.status:<9}  {r.model:<22}"
                f"  prompt={r.prompt_hash[:10]}  calibrated={calibrated}"
                f"  threshold={cfg.get('review_threshold')}{drift}"
            )


def cmd_register(args: argparse.Namespace) -> None:
    overrides = json.loads(Path(args.calibration).read_text()) if args.calibration else {}
    overrides = {k: overrides[k] for k in ("calibration", "review_threshold") if k in overrides}
    with owner_session() as s:
        row = register_candidate(s, args.doc_type, get_settings(), config_overrides=overrides)
        print(f"registered candidate {row.id} (prompt {row.prompt_hash[:10]}, model {row.model})")
        if args.promote:
            promote(s, row.id)
            print(f"promoted {row.id}; the previous active version is retired")


def cmd_promote(args: argparse.Namespace) -> None:
    with owner_session() as s:
        row = promote(s, args.version_id)
        print(f"{row.id} is now active for {row.doc_type}")


def cmd_reprocess(args: argparse.Namespace) -> None:
    """Run a stage again for documents in a given status, with the active extractor.

    Clears the idempotency records of that stage and every later one, so the workers treat
    it as new work all the way down. Extractions are unique per (document, extractor version):
    a rerun with the same version replaces its row, other versions' rows are kept.
    """
    queue = SqsQueue(get_settings())
    # New output from a stage invalidates everything built on it, so later stages rerun too.
    later = list(STAGES[STAGES.index(args.stage) :])
    with owner_session() as s:
        query = select(Document).where(Document.status == args.status).limit(args.limit)
        docs = s.scalars(query).all()
        for doc in docs:
            s.execute(
                delete(StageRun).where(StageRun.document_id == doc.id, StageRun.stage.in_(later))
            )
        s.flush()
        pairs = [(d.tenant_id, d.id) for d in docs]
    for tenant_id, doc_id in pairs:  # enqueue after the commit, so workers see the reset
        queue.send(StageMessage(tenant_id=tenant_id, document_id=doc_id, stage=args.stage))
    print(f"queued {len(pairs)} document(s) for {args.stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="group", required=True)

    ext = sub.add_parser("extractors").add_subparsers(dest="cmd", required=True)
    ext.add_parser("list").set_defaults(func=cmd_list)
    reg = ext.add_parser("register", help="new candidate from the current code and settings")
    reg.add_argument("--doc-type", default="invoice")
    reg.add_argument("--calibration", help="JSON file written by scripts/calibrate.py")
    reg.add_argument("--promote", action="store_true")
    reg.set_defaults(func=cmd_register)
    pro = ext.add_parser("promote")
    pro.add_argument("version_id")
    pro.set_defaults(func=cmd_promote)

    rep = sub.add_parser("reprocess")
    rep.add_argument("--stage", choices=STAGES, required=True)
    rep.add_argument("--status", required=True, help="document status to select, e.g. extracted")
    rep.add_argument("--limit", type=int, default=1000)
    rep.set_defaults(func=cmd_reprocess)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
