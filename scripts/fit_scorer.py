"""Fit a learned confidence scorer on the dev set and register it as a candidate version.

    uv run python scripts/fit_scorer.py data/dev data/synthetic data/batch2 [--register]

Training rows are the active version's extracted fields on the dev documents: the stored
evidence (grounding, OCR confidence, agreement, rule results) and whether the value the
extractor *originally* produced was right. Reviewed fields now hold the corrected value, so
the original comes from the first correction's ``old_value``: the append-only audit trail is
what makes re-learning possible.

The scorer is a logistic regression (``LogisticScorer``); the threshold is chosen on the dev
set for the target precision. These are in-sample numbers. Whether the candidate really is
better is decided by the promotion gate on the golden set, which it has never seen.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from docflow_core.config import get_settings
from docflow_core.db.models import Correction, Document, Extraction, ExtractorVersion
from docflow_core.db.session import owner_session
from docflow_core.eval.datasets import assert_disjoint, load
from docflow_core.extract.confidence import (
    FEATURES,
    LogisticScorer,
    Signals,
    choose_threshold,
    scorer_from_config,
)
from docflow_core.extract.fields import base_name, field_kinds, flatten, same_value
from docflow_core.extract.versions import promote, register_candidate
from docflow_core.schemas import registry


def signals_of(stored: dict[str, Any]) -> Signals:
    return Signals(
        match=stored.get("match", "none"),
        ocr_conf=stored.get("ocr_conf"),
        similarity=stored.get("similarity", 0.0),
        agreement=stored.get("agreement"),
        validators_passed=stored.get("validators_passed"),
        required_missing=stored.get("required_missing", False),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--target", type=float, default=0.995)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--register", action="store_true", help="register as a candidate version")
    parser.add_argument("--promote", action="store_true", help="skip the gate (do not use)")
    args = parser.parse_args()

    labels: dict[str, dict[str, Any]] = {}
    for folder in args.folders:
        ds = load(folder / "manifest.jsonl")
        assert_disjoint(ds.hashes, f"{folder} (scorer training)")
        labels.update({i.sha256: i.label for i in ds.items})

    rows: list[tuple[Signals, bool]] = []
    with owner_session() as s:
        active = s.scalar(select(ExtractorVersion).where(ExtractorVersion.status == "active"))
        assert active is not None
        spec = registry.get(active.doc_type, active.schema_version)
        kinds = field_kinds(spec.model)
        current = scorer_from_config(active.config)
        pairs = s.execute(
            select(Document.sha256, Extraction)
            .join(Extraction, Extraction.document_id == Document.id)
            .where(Extraction.extractor_version_id == active.id, Document.sha256.in_(list(labels)))
        ).all()
        for sha, ex in pairs:
            truth = flatten(spec.model, labels[sha])
            originals: dict[str, Any] = {}
            for c in s.scalars(
                select(Correction)
                .where(Correction.extraction_id == ex.id)
                .order_by(Correction.created_at)
            ):
                originals.setdefault(c.field, (c.old_value or {}).get("value"))
            for key, f in ex.fields.items():
                value = originals.get(key, f["value"])
                ok = same_value(kinds[base_name(key)], value, truth.get(key))
                rows.append((signals_of(f.get("signals", {})), ok))

    signals = [r[0] for r in rows]
    correct = [r[1] for r in rows]
    learned = LogisticScorer.fit(signals, correct, l2=args.l2)
    wrong = sum(not ok for ok in correct)
    print(f"trained on {len(rows)} fields from {len(pairs)} dev documents ({wrong} wrong)\n")
    print("weights (log-odds; positive raises confidence):")
    for name, w in learned.explain().items():
        print(f"  {name:<18} {w:+.3f}")

    header = f"{'scorer':<14}{'threshold':>10}{'auto-approved':>15}{'precision':>11}"
    print(f"\n{header}  (dev set, in-sample)")
    for name, scorer in (("current", current), ("learned", learned)):
        pick = choose_threshold([scorer(x) for x in signals], correct, args.target)
        print(
            f"{name:<14}{pick.threshold:>10.4f}{pick.auto_approve_rate:>15.1%}{pick.precision:>11.2%}"
        )
    choice = choose_threshold([learned(x) for x in signals], correct, args.target)

    if not args.register:
        return
    config = {
        "scorer": {
            "type": "logistic",
            "weights": dict(zip(FEATURES, learned.weights, strict=True)),
            "bias": learned.bias,
            "fitted_on": [str(f) for f in args.folders],
            "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "training_fields": len(rows),
        },
        "review_threshold": choice.threshold,
    }
    with owner_session() as s:
        row = register_candidate(s, active.doc_type, get_settings(), config_overrides=config)
        print(f"\nregistered candidate {row.id} (parent {active.id})")
        print(json.dumps({"review_threshold": choice.threshold}))
        if args.promote:
            promote(s, row.id)


if __name__ == "__main__":
    main()
