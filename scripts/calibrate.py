"""Fit the confidence calibration on a labelled dev set and pick the review threshold.

    uv run python scripts/calibrate.py data/dev data/synthetic data/batch2 --target 0.98

For every dev document it takes the active extractor's stored extraction (no model calls:
each field already carries its raw score in ``signals``), compares each field with the label,
and then:

1. fits an isotonic calibration from raw score to the observed rate of correct fields;
2. picks the lowest threshold whose auto-approved fields reach the target precision;
3. prints a reliability table and per-field accuracy;
4. writes the result to eval/calibration/, ready for ``admin.py extractors register``.

Never run this on the golden set: the golden set must not influence thresholds.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from docflow_core.db.models import Document, Extraction, ExtractorVersion
from docflow_core.db.session import owner_session
from docflow_core.eval.match import field_correctness
from docflow_core.extract.confidence import IsotonicCalibrator, choose_threshold
from docflow_core.extract.fields import base_name
from docflow_core.schemas import registry

BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01]


def load_labels(folders: list[Path]) -> dict[str, dict[str, Any]]:
    """sha256 -> label, from each folder's manifest.jsonl."""
    labels: dict[str, dict[str, Any]] = {}
    for folder in folders:
        for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            labels[row["sha256"]] = json.loads((folder / row["label"]).read_text(encoding="utf-8"))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", type=Path, help="dev-set folders with manifest.jsonl")
    parser.add_argument("--doc-type", default="invoice")
    parser.add_argument(
        "--target", type=float, default=0.98, help="precision of auto-approved fields"
    )
    parser.add_argument("--out", type=Path, default=Path("eval/calibration"))
    args = parser.parse_args()

    labels = load_labels(args.folders)
    scores: list[float] = []
    correct: list[bool] = []
    per_field: dict[str, list[bool]] = defaultdict(list)
    docs_used = docs_perfect = parse_failures = 0
    with owner_session() as s:
        version = s.scalar(
            select(ExtractorVersion).where(
                ExtractorVersion.doc_type == args.doc_type, ExtractorVersion.status == "active"
            )
        )
        if version is None:
            raise SystemExit("no active extractor version")
        spec = registry.get(version.doc_type, version.schema_version)
        rows = s.execute(
            select(Document.sha256, Extraction)
            .join(Extraction, Extraction.document_id == Document.id)
            .where(
                Extraction.extractor_version_id == version.id,
                Document.sha256.in_(list(labels)),
            )
        ).all()
        for sha, extraction in rows:
            if extraction.status == "failed_parse":
                parse_failures += 1
                continue
            fields = extraction.fields
            values = {k: f["value"] for k, f in fields.items()}
            result = field_correctness(spec, values, labels[sha])
            docs_used += 1
            docs_perfect += all(result.values())
            for key, ok in result.items():
                per_field[base_name(key)].append(ok)
                if key in fields:  # a row the model missed has no score to calibrate
                    scores.append(float(fields[key]["signals"]["raw_score"]))
                    correct.append(ok)

    if not scores:
        raise SystemExit(f"no extractions by the active version {version.id} for these documents")

    calibrator = IsotonicCalibrator.fit(scores, correct)
    calibrated = [calibrator(x) for x in scores]
    choice = choose_threshold(calibrated, correct, args.target)

    print(f"extractor {version.id} ({version.model}), {docs_used} documents, {len(scores)} fields")
    if parse_failures:
        print(f"{parse_failures} document(s) skipped: failed_parse")
    print(
        f"field accuracy {sum(correct) / len(correct):.1%}  |  straight-through "
        f"(every field right) {docs_perfect}/{docs_used}\n"
    )
    print("raw score      fields  correct  calibrated")
    for lo, hi in itertools.pairwise(BINS):
        idx = [i for i, x in enumerate(scores) if lo <= x < hi]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            cal = sum(calibrated[i] for i in idx) / len(idx)
            print(f"{lo:4.2f}-{min(hi, 1):4.2f}  {len(idx):>9}  {acc:>7.1%}  {cal:>10.3f}")
    print("\nfield accuracy (worst first)")
    for name, oks in sorted(per_field.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {name:<26} {sum(oks):>4}/{len(oks):<4} {sum(oks) / len(oks):6.1%}")
    # The target is a business decision: show what each choice costs in review work.
    print("\ntarget precision  threshold  auto-approved  precision  wrong fields let through")
    for target in (0.98, 0.99, 0.995, 0.999):
        c = choose_threshold(calibrated, correct, target)
        through = [ok for conf, ok in zip(calibrated, correct, strict=True) if conf >= c.threshold]
        print(
            f"{target:>15.1%}  {c.threshold:>9.4f}  {c.auto_approve_rate:>13.1%}"
            f"  {c.precision:>9.1%}  {len(through) - sum(through):>24}"
        )
    print(
        f"\nthreshold for {args.target:.1%} precision: {choice.threshold:.4f}  ->  "
        f"{choice.auto_approve_rate:.1%} of fields auto-approved "
        f"at {choice.precision:.1%} precision"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.doc_type}-{str(version.id)[:8]}.json"
    out.write_text(
        json.dumps(
            {
                "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "extractor_version_id": str(version.id),
                "dev_sets": [str(f) for f in args.folders],
                "documents": docs_used,
                "fields": len(scores),
                "field_accuracy": round(sum(correct) / len(correct), 4),
                "target_precision": args.target,
                "review_threshold": choice.threshold,
                "auto_approve_rate": choice.auto_approve_rate,
                "precision_at_threshold": choice.precision,
                "calibration": json.loads(calibrator.to_json()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
