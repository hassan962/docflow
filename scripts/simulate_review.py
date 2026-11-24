"""Stand-in for human reviewers: clears the review queue through the real API, using labels.

    uv run python scripts/simulate_review.py data/dev data/synthetic data/batch2

For each document waiting for review whose file is in a labelled dev set: every flagged field
whose value already matches the label is accepted (a positive label), every other flagged field
is edited to the label's value, document-level issues are acknowledged, and the document is
approved. Everything goes through the same endpoints the review UI uses, so the corrections
table fills exactly as it would with people, and phase 5's learning loop has a correction pool.

Golden documents are refused: they must never become corrections or few-shot examples.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from docflow_core.eval.datasets import assert_disjoint, load
from docflow_core.extract.fields import base_name, field_kinds, flatten, same_value
from docflow_core.schemas import registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()
    load_dotenv(".env")
    key = os.environ.get("DOCFLOW_API_KEY") or sys.exit("DOCFLOW_API_KEY is not set")

    labels: dict[str, dict[str, Any]] = {}
    for folder in args.folders:
        dataset = load(folder / "manifest.jsonl")
        assert_disjoint(dataset.hashes, f"{folder} (simulated review)")
        labels.update({i.sha256: i.label for i in dataset.items})

    spec = registry.get("invoice", 2)
    kinds = field_kinds(spec.model)
    stats = {"documents": 0, "accepted": 0, "edited": 0, "acknowledged": 0, "skipped": 0}
    with httpx.Client(base_url=args.api, headers={"X-API-Key": key}, timeout=60) as api:
        for item in api.get("/v1/review/queue", params={"limit": 200}).json():
            doc = api.get(f"/v1/documents/{item['document_id']}").json()
            label = labels.get(doc["sha256"])
            if label is None or item["extraction_id"] is None or item["fields_to_review"] < 0:
                stats["skipped"] += 1
                continue
            truth = flatten(spec.model, label)
            ex = api.get(f"/v1/extractions/{item['extraction_id']}").json()
            for key_, f in ex["fields"].items():
                if f["status"] != "needs_review":
                    continue
                want = truth.get(key_)
                if same_value(kinds[base_name(key_)], f["value"], want):
                    body = {"field": key_, "action": "accept"}
                    stats["accepted"] += 1
                else:
                    body = {"field": key_, "action": "edit", "value": want}
                    stats["edited"] += 1
                r = api.post(f"/v1/extractions/{ex['id']}/corrections", json=body)
                r.raise_for_status()
                ex = r.json()
            if ex["status"] != "approved":
                stats["acknowledged"] += bool(ex["issues"])
                r = api.post(
                    f"/v1/documents/{doc['id']}/approve",
                    params={"acknowledge_issues": bool(ex["issues"])},
                )
                r.raise_for_status()
            stats["documents"] += 1
    print(", ".join(f"{k}: {v}" for k, v in stats.items()))


if __name__ == "__main__":
    main()
