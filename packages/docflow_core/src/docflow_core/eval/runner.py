"""Run one extractor version on a dataset, exactly the way production would.

For every document: OCR (cached) -> extraction with the version's model, prompt, few-shot
examples and scorer -> the deterministic rules -> routing with the version's threshold ->
score against the label. Reference checks are left out: golden documents belong to no tenant,
so there is no vendor list to check against.

The result is a list of per-document results (what the bootstrap resamples) plus a summary.
Both are written to ``eval/runs/`` and a row is added to ``eval_runs``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from docflow_core.eval.cache import OcrCache
from docflow_core.eval.datasets import Dataset
from docflow_core.eval.metrics import Counts, DocResult, score_document, summary
from docflow_core.extract.extractor import extract
from docflow_core.extract.llm import LlmClient
from docflow_core.extract.prompt import FewShotExample
from docflow_core.extract.versions import ActiveExtractor
from docflow_core.ocr import OcrResult
from docflow_core.schemas import registry
from docflow_core.validate.apply import route, threshold_for
from docflow_core.validate.rules import RuleContext, run_rules

RUNS = Path("eval/runs")
ExampleSource = Callable[[list[OcrResult]], list[FewShotExample]]


def evaluate(
    version: ActiveExtractor,
    dataset: Dataset,
    *,
    client_for: Callable[[str], LlmClient],
    ocr: OcrCache,
    examples_for: ExampleSource | None = None,
    today: date | None = None,
    progress: Callable[[str], None] = lambda _: None,
) -> list[DocResult]:
    spec = registry.get(version.doc_type, version.schema_version)
    threshold = threshold_for(version.doc_type, version.review_threshold)
    agreement = client_for(version.agreement_model) if version.agreement_model else None
    results = []
    for n, item in enumerate(dataset.items, 1):
        pages = ocr.pages(item.path, item.sha256)
        examples = examples_for(pages) if examples_for else []
        outcome = extract(
            spec, pages, client_for(version.model),
            agreement_client=agreement, scorer=version.scorer, examples=examples,
        )  # fmt: skip
        cost = sum(c.result.cost_usd for c in outcome.calls)
        latency = sum(c.result.latency_ms for c in outcome.calls)
        if outcome.status == "failed_parse":
            results.append(
                score_document(spec, item.id, None, item.label, cost_usd=cost, latency_ms=latency)
            )
            progress(f"[{n}/{len(dataset.items)}] {item.id}: reply unusable")
            continue
        fields = {k: f.model_dump(mode="json") for k, f in outcome.fields.items()}
        values = {k: f["value"] for k, f in fields.items()}
        grounded = {k: f["signals"].get("match", "none") for k, f in fields.items()}
        rules = run_rules(values, RuleContext(spec, today or date.today(), grounded=grounded))
        routed = route(fields, rules, version.scorer, threshold)
        result = score_document(
            spec, item.id, values, item.label,
            confidence={k: f["confidence"] for k, f in routed.fields.items()},
            status={k: f["status"] for k, f in routed.fields.items()},
            cost_usd=cost, latency_ms=latency,
        )  # fmt: skip
        results.append(result)
        progress(
            f"[{n}/{len(dataset.items)}] {item.id} ({item.layout}/{item.noise}): "
            f"{'straight-through' if result.straight_through else 'has errors'}"
        )
    return results


def _doc_to_dict(d: DocResult) -> dict[str, Any]:
    out = asdict(d)
    out["fields"] = {k: asdict(c) for k, c in d.fields.items()}
    return out


def _doc_from_dict(raw: dict[str, Any]) -> DocResult:
    raw = dict(raw)
    raw["fields"] = {k: Counts(**c) for k, c in raw["fields"].items()}
    return DocResult(**raw)


def save_run(
    version: ActiveExtractor, dataset: Dataset, docs: list[DocResult], *, note: str = ""
) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    path = RUNS / f"{stamp}-{str(version.id)[:8]}-{dataset.name}.json"
    body = {
        "extractor_version_id": str(version.id),
        "model": version.model,
        "prompt_hash": version.prompt_hash,
        "dataset": dataset.name,
        "dataset_hashes": sorted(dataset.hashes),
        "created_at": stamp,
        "note": note,
        "summary": summary(docs),
        "documents": [_doc_to_dict(d) for d in docs],
    }
    path.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return path


def load_run(path: Path) -> tuple[dict[str, Any], list[DocResult]]:
    body = json.loads(path.read_text(encoding="utf-8"))
    return body, [_doc_from_dict(d) for d in body["documents"]]


def record_run(
    session: Any, version: ActiveExtractor, dataset: Dataset, docs: list[DocResult], path: Path
) -> None:
    from docflow_core.db.models import EvalRun

    s = summary(docs)
    session.add(
        EvalRun(
            extractor_version_id=version.id,
            dataset=dataset.name,
            n_docs=len(docs),
            metrics=s,
            cost_usd=Decimal(str(s["cost_per_page_usd"])) * len(docs),
            p95_latency_ms=s["p95_latency_ms"],
            report_key=str(path),
        )
    )
