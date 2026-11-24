"""Evaluation datasets and the leakage guard.

Three datasets, strictly separated:

- golden: the headline result and the promotion gate. Never used for prompts, few-shot
  examples, thresholds or training.
- dev: tuning OCR, calibration, scorers and thresholds. Never reported as the result.
- correction pool: few-shot examples and training data. Never used for evaluation.

The repository holds the golden *manifest* (ids, hashes, labels), not the documents: they
are regenerated from the generator seed and checked against the hashes. ``assert_disjoint``
is called wherever data could leak into the golden set's territory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GOLDEN_DIR = Path("eval/golden")
GOLDEN_MANIFEST = GOLDEN_DIR / "manifest.jsonl"


class LeakageError(RuntimeError):
    """A golden document showed up where it must never be."""


@dataclass(frozen=True)
class Item:
    id: str
    path: Path  # the document file
    sha256: str
    label: dict[str, Any]
    layout: str = ""
    noise: str = ""


@dataclass(frozen=True)
class Dataset:
    name: str
    items: list[Item]

    @property
    def hashes(self) -> set[str]:
        return {i.sha256 for i in self.items}


def load(manifest: Path, docs_dir: Path | None = None, name: str | None = None) -> Dataset:
    """Read a manifest.jsonl. Labels are inline (golden) or in a file next to it (dev sets)."""
    docs_dir = docs_dir or manifest.parent
    items = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = row["label"]
        if isinstance(label, str):
            label = json.loads((manifest.parent / label).read_text(encoding="utf-8"))
        items.append(
            Item(
                id=row["id"],
                path=docs_dir / row["file"],
                sha256=row["sha256"],
                label=label,
                layout=row.get("layout", ""),
                noise=row.get("noise", ""),
            )
        )
    return Dataset(name or manifest.parent.name, items)


def golden_hashes(root: Path = Path(".")) -> set[str]:
    path = root / GOLDEN_MANIFEST
    if not path.exists():
        return set()
    return {json.loads(line)["sha256"] for line in path.read_text().splitlines() if line.strip()}


def assert_disjoint(hashes: Iterable[str], where: str, root: Path = Path(".")) -> None:
    leaked = golden_hashes(root) & set(hashes)
    if leaked:
        raise LeakageError(
            f"{len(leaked)} golden document(s) found in {where}; refusing to continue"
        )
