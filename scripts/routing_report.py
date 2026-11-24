"""How well does routing protect the data? Scored against labels on the dev set.

    uv run python scripts/routing_report.py data/dev data/synthetic data/batch2

For the active extractor's validated extractions:

- catch rate: of the fields the extractor got wrong, the share sent to a person;
- auto-approve precision: of the fields that skipped review, the share that are right;
- review load: the share of fields a person has to look at;
- reference checks: flagged fields per document against data/reference_expectations.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sqlalchemy import select

from docflow_core.db.models import Document, Extraction, ExtractorVersion
from docflow_core.db.session import owner_session
from docflow_core.eval.match import field_correctness
from docflow_core.schemas import registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument(
        "--expectations", type=Path, default=Path("data/reference_expectations.json")
    )
    args = parser.parse_args()

    labels: dict[str, dict[str, Any]] = {}
    for folder in args.folders:
        for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            labels[row["sha256"]] = json.loads((folder / row["label"]).read_text(encoding="utf-8"))
    expected = json.loads(args.expectations.read_text()) if args.expectations.exists() else {}

    counts: Counter[str] = Counter()
    missed: list[str] = []
    rules_on_wrong: Counter[str] = Counter()
    ref: Counter[str] = Counter()
    with owner_session() as s:
        version = s.scalar(select(ExtractorVersion).where(ExtractorVersion.status == "active"))
        assert version is not None
        spec = registry.get(version.doc_type, version.schema_version)
        rows = s.execute(
            select(Document.sha256, Document.filename, Extraction)
            .join(Extraction, Extraction.document_id == Document.id)
            .where(Extraction.extractor_version_id == version.id, Document.sha256.in_(list(labels)))
        ).all()
        for sha, filename, ex in rows:
            if ex.status not in ("approved", "needs_review"):
                continue
            counts["documents"] += 1
            counts["documents_auto_approved"] += ex.status == "approved"
            values = {k: f["value"] for k, f in ex.fields.items()}
            correct = field_correctness(spec, values, labels[sha])
            for key, f in ex.fields.items():
                ok, review = correct.get(key, False), f["status"] == "needs_review"
                counts["fields"] += 1
                counts["review"] += review
                counts["wrong"] += not ok
                counts["wrong_caught"] += (not ok) and review
                counts["auto"] += not review
                counts["auto_right"] += (not review) and ok
                if not ok:
                    for c in f.get("checks", []):
                        if not c["passed"]:
                            rules_on_wrong[c["rule"]] += 1
                    if not review:
                        missed.append(f"{filename} {key}: {f['value']!r}")
            if sha in expected:
                want = {fld for fld, _ in expected[sha]}
                got = {
                    k
                    for k, f in ex.fields.items()
                    for c in f.get("checks", [])
                    if c["rule"] == "reference_checks" and not c["passed"]
                }
                ref["documents"] += 1
                ref["exact"] += got == want
                ref["expected_flags"] += len(want)
                ref["found_flags"] += len(want & got)
                ref["false_flags"] += len(got - want)

    c = counts
    print(f"extractor {version.id}: {c['documents']} documents, {c['fields']} fields")
    print(
        f"documents approved with no human touch: {c['documents_auto_approved']}/{c['documents']}"
    )
    print(f"review load: {c['review']}/{c['fields']} fields ({c['review'] / c['fields']:.1%})")
    print(
        f"catch rate: {c['wrong_caught']}/{c['wrong']} wrong fields sent to review "
        f"({c['wrong_caught'] / max(c['wrong'], 1):.1%})"
    )
    precision = c["auto_right"] / max(c["auto"], 1)
    print(f"auto-approve precision: {c['auto_right']}/{c['auto']} ({precision:.2%})")
    print(f"rules that fired on wrong fields: {dict(rules_on_wrong)}")
    for m in missed:
        print(f"  wrong but auto-approved: {m}")
    if ref["documents"]:
        print(
            f"\nreference checks: {ref['exact']}/{ref['documents']} documents exactly right, "
            f"{ref['found_flags']}/{ref['expected_flags']} expected flags found, "
            f"{ref['false_flags']} false flags"
        )


if __name__ == "__main__":
    main()
