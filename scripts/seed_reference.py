"""Seed a tenant's vendors and purchase orders from labelled invoices, with known mistakes.

    uv run python scripts/seed_reference.py data/dev data/synthetic data/batch2

Per vendor: 80% on file exactly, 10% on file with a spelling difference, 10% missing.
Per purchase order (of vendors on file): 80% matching, 10% with a different total, 10%
belonging to another vendor. Those are the five reference-check cases. The finding each
document should get is written to ``data/reference_expectations.json``, so the model's
reference checks can be scored instead of assumed.
"""

from __future__ import annotations

import argparse
import json
import random
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select

from docflow_core.db.models import PurchaseOrder, Tenant, Vendor
from docflow_core.db.session import owner_session
from docflow_core.normalise import normalise_name


def misspell(name: str, rng: random.Random) -> str:
    words = name.split()
    i = max(range(len(words)), key=lambda j: len(words[j]))
    w = words[i]
    if len(w) > 4:
        k = rng.randrange(1, len(w) - 1)
        words[i] = w[:k] + w[k + 1 :]  # drop one letter, the commonest typo
    return " ".join(words).replace(" and ", " & ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--tenant", default="Demo Co")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("data/reference_expectations.json"))
    args = parser.parse_args()
    rng = random.Random(args.seed)

    labelled: list[tuple[str, dict[str, Any]]] = []
    for folder in args.folders:
        for line in (folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            labelled.append(
                (row["sha256"], json.loads((folder / row["label"]).read_text(encoding="utf-8")))
            )

    expectations: dict[str, list[list[str]]] = {}
    with owner_session() as s:
        tenant = s.scalar(select(Tenant).where(Tenant.name == args.tenant))
        if tenant is None:
            raise SystemExit(f"no tenant {args.tenant!r}; run scripts/seed.py first")
        s.execute(delete(PurchaseOrder).where(PurchaseOrder.tenant_id == tenant.id))
        s.execute(delete(Vendor).where(Vendor.tenant_id == tenant.id))
        vendors: dict[str, Vendor | None] = {}
        decoy = Vendor(tenant_id=tenant.id, name="Northwind Office Supplies",
                       normalised_name="northwind office supplies", tax_id=None)  # fmt: skip
        s.add(decoy)
        for sha, label in labelled:
            name = label["vendor_name"]
            expected: set[tuple[str, str]] = set()
            if name not in vendors:
                roll = rng.random()
                if roll < 0.8:
                    on_file = name
                elif roll < 0.9:
                    on_file = misspell(name, rng)
                else:
                    on_file = None
                vendors[name] = (
                    Vendor(
                        tenant_id=tenant.id,
                        name=on_file,
                        normalised_name=normalise_name(on_file) or "",
                        tax_id=label["vendor_tax_id"],
                    )
                    if on_file
                    else None
                )
                if vendors[name] is not None:
                    s.add(vendors[name])
            vendor = vendors[name]
            if vendor is None:
                expected.add(("vendor_name", "error"))
            elif vendor.name != name:
                expected.add(("vendor_name", "warning"))
            if label["po_number"]:
                if vendor is None:
                    expected.add(("po_number", "warning"))  # cannot be on file
                else:
                    s.flush()
                    roll, total = rng.random(), Decimal(label["total"])
                    owner = vendor
                    if roll < 0.8:
                        po_total = total
                    elif roll < 0.9:
                        po_total = (total * Decimal("1.03")).quantize(Decimal("0.01"))
                        expected.add(("total", "error"))
                    else:
                        po_total, owner = total, decoy
                        expected |= {("vendor_name", "error"), ("po_number", "error")}
                    s.add(
                        PurchaseOrder(
                            tenant_id=tenant.id,
                            po_number=label["po_number"],
                            vendor_id=owner.id,
                            total=po_total,
                            currency=label["currency"],
                        )
                    )
            expectations[sha] = sorted(list(e) for e in expected)

    args.out.write_text(json.dumps(expectations, indent=1), encoding="utf-8")
    cases: dict[str, int] = {}
    for found in expectations.values():
        key = ", ".join(f"{f}:{sev}" for f, sev in found) or "all match"
        cases[key] = cases.get(key, 0) + 1
    print(f"seeded {sum(v is not None for v in vendors.values())} vendors for {args.tenant!r}")
    for key, n in sorted(cases.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>3}  {key}")
    print(f"expectations written to {args.out}")


if __name__ == "__main__":
    main()
