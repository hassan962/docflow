"""Evaluation CLI. Runs inside the stack (it needs Tesseract, the database and the LLM key):

    docker compose run --rm eval run --version active --dataset golden
    docker compose run --rm eval run --version <candidate-id> --dataset golden
    docker compose run --rm eval compare eval/runs/<active>.json eval/runs/<candidate>.json --decide

``compare`` applies the promotion gate. With ``--decide`` a passing candidate is promoted (the
old version retired in the same transaction) and a failing one is marked ``rejected`` with
the gate report attached. Without it, nothing changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import select

from docflow_core.config import get_settings
from docflow_core.db.models import ExtractorVersion
from docflow_core.db.session import owner_session
from docflow_core.eval.cache import CachedClient, OcrCache
from docflow_core.eval.datasets import GOLDEN_MANIFEST, assert_disjoint, load
from docflow_core.eval.gate import GateConfig, decide
from docflow_core.eval.runner import evaluate, load_run, record_run, save_run
from docflow_core.extract.llm import OpenAICompatibleClient
from docflow_core.extract.versions import ActiveExtractor, promote, version_by_id
from docflow_core.ocr import get_engine

CACHE = Path("data/cache")


def _version(session: object, ref: str, doc_type: str) -> ActiveExtractor:
    if ref == "active":
        row = session.scalar(  # type: ignore[attr-defined]
            select(ExtractorVersion).where(
                ExtractorVersion.doc_type == doc_type, ExtractorVersion.status == "active"
            )
        )
        if row is None:
            sys.exit("no active extractor version")
        ref = str(row.id)
    return version_by_id(session, ref)  # type: ignore[arg-type]


def cmd_run(args: argparse.Namespace) -> None:
    settings = get_settings()
    if args.dataset == "golden":
        dataset = load(GOLDEN_MANIFEST, Path("data/golden"), "golden")
    else:
        dataset = load(Path(args.dataset))
        assert_disjoint(dataset.hashes, f"dataset {args.dataset}")  # dev data must stay dev
    missing = [i.id for i in dataset.items if not i.path.exists()]
    if missing:
        sys.exit(f"{len(missing)} document(s) missing; for golden run scripts/golden.py build")

    with owner_session() as s:
        version = _version(s, args.version, args.doc_type)
        examples_for = None
        if version.fewshot_k:
            from docflow_core.learn.fewshot import examples_source

            examples_for = examples_source(s, version, settings)

    clients: dict[str, CachedClient] = {}

    def client_for(model: str) -> CachedClient:
        if model not in clients:
            live = None if args.offline else OpenAICompatibleClient(settings, model)
            clients[model] = CachedClient(live, model, CACHE / "llm")
        return clients[model]

    ocr = OcrCache(
        get_engine(settings), CACHE / "ocr", dpi=settings.ocr_dpi, max_pages=settings.max_pages
    )
    n = len(dataset.items)
    print(f"evaluating {version.id} ({version.model}) on {dataset.name}: {n} documents")
    docs = evaluate(
        version, dataset, client_for=client_for, ocr=ocr, examples_for=examples_for, progress=print
    )
    path = save_run(version, dataset, docs, note=args.note)
    with owner_session() as s:
        record_run(s, version, dataset, docs, path)
    hits = sum(c.hits for c in clients.values())
    misses = sum(c.misses for c in clients.values())
    body, _ = load_run(path)
    print(json.dumps({k: v for k, v in body["summary"].items() if k != "fields"}, indent=1))
    print(f"LLM replies: {hits} from cache, {misses} live  |  report: {path}")


def cmd_compare(args: argparse.Namespace) -> None:
    head_a, docs_a = load_run(Path(args.active))
    head_c, docs_c = load_run(Path(args.candidate))
    if (
        head_a["dataset"] != head_c["dataset"]
        or head_a["dataset_hashes"] != head_c["dataset_hashes"]
    ):
        sys.exit("the two runs are not on the same dataset")
    report = decide(docs_a, docs_c, GateConfig(target_precision=args.target_precision))
    print(f"active    {head_a['extractor_version_id']}  {head_a['summary']['macro_f1']} macro F1")
    print(f"candidate {head_c['extractor_version_id']}  {head_c['summary']['macro_f1']} macro F1\n")
    for check in report.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    print(f"\ngate: {'PASSED' if report.passed else 'FAILED'}")
    if not args.decide:
        return
    with owner_session() as s:
        row = s.get(ExtractorVersion, head_c["extractor_version_id"], with_for_update=True)
        assert row is not None
        gate_report = {
            "against": head_a["extractor_version_id"],
            "run": args.candidate,
            **report.as_dict(),
        }
        row.config = {**(row.config or {}), "gate_report": gate_report}
        if report.passed:
            promote(s, row.id)
            print(f"promoted {row.id}; the previous version is retired (roll back by promoting it)")
        else:
            row.status = "rejected"
            print(f"{row.id} marked rejected; the active version is unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--version", default="active", help="'active' or an extractor version id")
    run.add_argument("--dataset", default="golden", help="'golden' or a manifest.jsonl path")
    run.add_argument("--doc-type", default="invoice")
    run.add_argument("--offline", action="store_true", help="cached replies only, no LLM calls")
    run.add_argument("--note", default="")
    run.set_defaults(func=cmd_run)
    cmp = sub.add_parser("compare")
    cmp.add_argument("active")
    cmp.add_argument("candidate")
    cmp.add_argument("--target-precision", type=float, default=0.995)
    cmp.add_argument("--decide", action="store_true", help="promote on pass, reject on fail")
    cmp.set_defaults(func=cmd_compare)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
