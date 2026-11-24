"""Few-shot selection: show the model corrected documents that look like the one in hand.

The correction pool is every document a reviewer has approved after touching at least one
field: its OCR text, and its final values as the answer the model should have given. For a
new document, the pool entries with the most similar *layout* go into the prompt as worked
examples.

Layout signature: the OCR lines with every digit replaced by 9 and letters lower-cased, cut
into character 3-grams. Two invoices from the same template share their labels, column headers
and number shapes ("99/99/9999", "9,99,999.99") even when every value differs, so the cosine
similarity of these profiles finds same-template documents without any model.

Two rules keep the pool safe:

- **Tenant isolation.** Examples come only from the same tenant's documents. One company's
  invoices must never appear in a prompt for another company's document.
- **No leakage.** The pool is checked against the golden set's hashes and refuses to build if
  any golden document is in it.
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from docflow_core.config import Settings
from docflow_core.db.models import Correction, Document, Extraction, Page, Tenant
from docflow_core.eval.datasets import assert_disjoint
from docflow_core.extract.fields import unflatten
from docflow_core.extract.prompt import FewShotExample, ocr_text
from docflow_core.extract.versions import ActiveExtractor
from docflow_core.ocr import OcrResult
from docflow_core.schemas import registry
from docflow_core.storage import ObjectStore, S3Store

MIN_SIMILARITY = 0.3


def signature(pages: Sequence[OcrResult]) -> Counter[str]:
    text = "\n".join(line for p in pages for line in p.lines()[:60])
    shape = re.sub(r"\d", "9", text.lower())
    return Counter(shape[i : i + 3] for i in range(len(shape) - 2))


def similarity(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(v * b[k] for k, v in a.items() if k in b)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


@dataclass(frozen=True)
class PoolEntry:
    example: FewShotExample
    signature: Counter[str]


def build_pool(
    session: Session, tenant_id: uuid.UUID, doc_type: str, store: ObjectStore, limit: int = 200
) -> list[PoolEntry]:
    """Reviewed and approved documents of one tenant, newest first."""
    reviewed = exists().where(Correction.extraction_id == Extraction.id)
    rows = session.execute(
        select(Document, Extraction)
        .join(Extraction, Extraction.document_id == Document.id)
        .where(
            Document.tenant_id == tenant_id,
            Document.doc_type == doc_type,
            Document.status.in_(("approved", "indexed")),
            Extraction.status == "approved",
            reviewed,
        )
        .order_by(Extraction.updated_at.desc())
        .limit(limit)
    ).all()
    assert_disjoint((d.sha256 for d, _ in rows), "the correction pool")

    pool = []
    for doc, ex in rows:
        keys = session.scalars(
            select(Page.ocr_json_key).where(Page.document_id == doc.id).order_by(Page.page_no)
        ).all()
        pages = [OcrResult.from_json(store.get(k)) for k in keys if k]
        if not pages:
            continue
        spec = registry.latest(doc_type)
        answer = unflatten(spec.model, {k: f["value"] for k, f in ex.fields.items()})
        example = FewShotExample(
            example_id=doc.sha256, ocr_text=ocr_text(pages), answer_json=json.dumps(answer)
        )
        pool.append(PoolEntry(example, signature(pages)))
    return pool


def select_examples(
    pool: Sequence[PoolEntry], pages: Sequence[OcrResult], k: int
) -> list[FewShotExample]:
    target = signature(pages)
    ranked = sorted(pool, key=lambda e: similarity(target, e.signature), reverse=True)
    return [e.example for e in ranked[:k] if similarity(target, e.signature) >= MIN_SIMILARITY]


def fewshot_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    cfg = (config or {}).get("fewshot")
    return cfg if cfg and cfg.get("k", 0) > 0 else None


class PoolCache:
    """Per-tenant pools for the extract worker, rebuilt every ``ttl`` seconds."""

    def __init__(self, store: ObjectStore, ttl: float = 600) -> None:
        self.store, self.ttl = store, ttl
        self._pools: dict[tuple[uuid.UUID, str], tuple[float, list[PoolEntry]]] = {}

    def get(self, session: Session, tenant_id: uuid.UUID, doc_type: str) -> list[PoolEntry]:
        key = (tenant_id, doc_type)
        hit = self._pools.get(key)
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1]
        pool = build_pool(session, tenant_id, doc_type, self.store)
        self._pools[key] = (time.monotonic(), pool)
        return pool


def examples_source(
    session: Session, version: ActiveExtractor, settings: Settings
) -> Callable[[list[OcrResult]], list[FewShotExample]]:
    """For offline evaluation: examples from the tenant named in the version's config.

    Golden documents belong to no tenant, so the evaluation borrows one tenant's pool (the
    dev tenant), the same way a new customer's first documents would be served.
    """
    from docflow_core.db.models import ExtractorVersion

    row = session.get(ExtractorVersion, version.id)
    cfg = fewshot_config(row.config if row else None) or {}
    tenant = session.scalar(select(Tenant).where(Tenant.name == cfg.get("eval_tenant", "Demo Co")))
    if tenant is None:
        raise ValueError("the evaluation tenant for few-shot examples does not exist")
    pool = build_pool(session, tenant.id, version.doc_type, S3Store(settings))
    k = int(cfg["k"])
    return lambda pages: select_examples(pool, pages, k)
