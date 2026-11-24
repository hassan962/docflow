"""Freeze and rebuild the golden evaluation set.

    uv run python scripts/golden.py freeze    # once: generate, check for leakage, write manifest
    uv run python scripts/golden.py build     # any machine: regenerate documents, verify hashes

The manifest (ids, hashes, layout, noise, labels) is committed under eval/golden/. The
documents are not: they are regenerated deterministically from the seed into data/golden/,
and every file is checked against its committed SHA-256 before it may be used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from docflow_core.eval.datasets import GOLDEN_DIR, GOLDEN_MANIFEST, load
from generate_invoices import generate

SEED = 2027  # never used by any dev set
N = 27  # 3 layouts x 3 noise levels x 3
DOCS = Path("data/golden")
META = GOLDEN_DIR / "dataset.json"


def _dev_hashes() -> set[str]:
    hashes: set[str] = set()
    for manifest in Path("data").glob("*/manifest.jsonl"):
        if manifest.parent != DOCS:
            hashes |= load(manifest).hashes
    return hashes


def freeze(force: bool) -> None:
    if GOLDEN_MANIFEST.exists() and not force:
        sys.exit(f"{GOLDEN_MANIFEST} exists: the golden set is frozen. (--force replaces it)")
    if DOCS.exists():
        shutil.rmtree(DOCS)
    rows = generate(N, DOCS, SEED, stratify=True)
    overlap = {r["sha256"] for r in rows} & _dev_hashes()
    if overlap:
        sys.exit(f"{len(overlap)} golden document(s) also appear in a dev set; pick another seed")
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with GOLDEN_MANIFEST.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            label = json.loads((DOCS / r["label"]).read_text(encoding="utf-8"))
            entry = {k: r[k] for k in ("id", "file", "sha256", "layout", "noise", "currency")}
            fh.write(json.dumps({**entry, "label": label}) + "\n")
    META.write_text(
        json.dumps(
            {
                "name": "golden-v1",
                "seed": SEED,
                "documents": N,
                "generator": "scripts/generate_invoices.py (stratified)",
                "frozen_at": datetime.now(UTC).date().isoformat(),
                "rules": "never used for prompts, few-shot examples, thresholds or training",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    mix: dict[str, int] = {}
    for r in rows:
        mix[f"{r['layout']}/{r['noise']}"] = mix.get(f"{r['layout']}/{r['noise']}", 0) + 1
    print(f"froze {N} documents (seed {SEED}) into {GOLDEN_MANIFEST}")
    print("mix: " + ", ".join(f"{k} {v}" for k, v in sorted(mix.items())))


def build() -> None:
    meta = json.loads(META.read_text(encoding="utf-8"))
    generate(meta["documents"], DOCS, meta["seed"], stratify=True)
    dataset = load(GOLDEN_MANIFEST, DOCS, "golden")
    bad = [
        i.id for i in dataset.items if hashlib.sha256(i.path.read_bytes()).hexdigest() != i.sha256
    ]
    if bad:
        sys.exit(f"{len(bad)} document(s) differ from the frozen hashes: {bad[:5]}")
    print(f"rebuilt {len(dataset.items)} golden documents in {DOCS}; all hashes match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--force", action="store_true")
    sub.add_parser("build")
    args = parser.parse_args()
    freeze(args.force) if args.cmd == "freeze" else build()


if __name__ == "__main__":
    main()
