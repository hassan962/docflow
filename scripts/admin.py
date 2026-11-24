"""Operator commands. Run inside the stack so they see its database and queues:

    docker compose exec api python scripts/admin.py extractors list
    docker compose exec api python scripts/admin.py extractors register --promote
    docker compose exec api python scripts/admin.py extractors promote <version-id>
    docker compose exec api python scripts/admin.py reprocess --stage extract --status extracted
    docker compose exec api python scripts/admin.py dlq --stage validate [--redrive | --purge]

Promotion is manual until phase 5's evaluation gate, which calls the same ``promote``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import boto3
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
    if args.calibration is None:  # keep the active version's calibration and threshold
        with owner_session() as s:
            active = s.scalar(
                select(ExtractorVersion).where(
                    ExtractorVersion.doc_type == args.doc_type, ExtractorVersion.status == "active"
                )
            )
            if active is not None:
                cfg = active.config or {}
                overrides = {
                    k: cfg[k] for k in ("calibration", "review_threshold", "scorer") if cfg.get(k)
                }
    if args.fewshot_k:
        overrides["fewshot"] = {"k": args.fewshot_k, "eval_tenant": args.eval_tenant}
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
        if args.reason:  # e.g. --reason RateLimitError: only documents that failed for that
            query = query.where(Document.failure_reason.ilike(f"%{args.reason}%"))
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


def cmd_dlq(args: argparse.Namespace) -> None:
    """Dead-letter queues are named after their stage queue with a ``-dlq`` suffix.

    Without a flag: count the messages and show a sample. ``--redrive`` sends each one back
    to its stage queue (the worker's idempotency makes duplicates harmless); ``--purge``
    deletes them. Always look before you redrive: a poison message just comes back.
    """
    settings = get_settings()
    source = settings.queue_url(args.stage)
    if not source:
        raise SystemExit(f"QUEUE_URL_{args.stage.upper()} is not configured")
    client = boto3.client(
        "sqs", region_name=settings.aws_region, endpoint_url=settings.aws_endpoint_url_sqs
    )
    dlq = f"{source}-dlq"
    moved = 0
    sample: list[str] = []
    while True:
        batch = client.receive_message(QueueUrl=dlq, MaxNumberOfMessages=10, WaitTimeSeconds=1)
        messages = batch.get("Messages", [])
        if not messages:
            break
        for m in messages:
            if args.redrive:
                client.send_message(QueueUrl=source, MessageBody=m["Body"])
            if args.redrive or args.purge:
                client.delete_message(QueueUrl=dlq, ReceiptHandle=m["ReceiptHandle"])
            elif len(sample) < 5:
                sample.append(m["Body"][:160])
            moved += 1
        if not (args.redrive or args.purge):
            break  # peeking: one batch is enough, and the messages reappear on their own
    verb = "redrove" if args.redrive else "purged" if args.purge else "found (sample)"
    print(f"{verb} {moved} message(s) in {dlq.rsplit('/', 1)[-1]}")
    for body in sample:
        print(f"  {body}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="group", required=True)

    ext = sub.add_parser("extractors").add_subparsers(dest="cmd", required=True)
    ext.add_parser("list").set_defaults(func=cmd_list)
    reg = ext.add_parser("register", help="new candidate from the current code and settings")
    reg.add_argument("--doc-type", default="invoice")
    reg.add_argument("--calibration", help="JSON file written by scripts/calibrate.py")
    reg.add_argument("--fewshot-k", type=int, default=0, help="corrected examples per prompt")
    reg.add_argument("--eval-tenant", default="Demo Co", help="tenant whose pool feeds evaluation")
    reg.add_argument("--promote", action="store_true")
    reg.set_defaults(func=cmd_register)
    pro = ext.add_parser("promote")
    pro.add_argument("version_id")
    pro.set_defaults(func=cmd_promote)

    rep = sub.add_parser("reprocess")
    rep.add_argument("--stage", choices=STAGES, required=True)
    rep.add_argument("--status", required=True, help="document status to select, e.g. extracted")
    rep.add_argument("--limit", type=int, default=1000)
    rep.add_argument("--reason", help="only documents whose failure reason contains this text")
    rep.set_defaults(func=cmd_reprocess)

    dlq = sub.add_parser("dlq", help="inspect, redrive or purge a stage's dead-letter queue")
    dlq.add_argument("--stage", choices=STAGES, required=True)
    action = dlq.add_mutually_exclusive_group()
    action.add_argument("--redrive", action="store_true", help="move messages back to the stage")
    action.add_argument("--purge", action="store_true", help="delete them (after fixing the cause)")
    dlq.set_defaults(func=cmd_dlq)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
