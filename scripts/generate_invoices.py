"""Synthetic invoice generator with exact ground truth.

    uv run python scripts/generate_invoices.py --n 60 --out data/synthetic --seed 7

For each invoice it writes:

- the document: a clean PDF, or a degraded scan-like JPEG (rotation, blur, speckle and
  compression at controlled levels);
- ``<id>.json``: the label, an ``InvoiceV2`` object validated against the real schema;
- one line in ``manifest.jsonl``: id, file, sha256, layout, noise level, label path.

Because the generator writes both the document and its label, the ground truth is exact,
which real scans can never guarantee. Three layouts vary field positions, labels, date
formats and number formatting, so a model cannot pass by memorising one template. All
content comes from Faker: no real company or person data is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pypdfium2 as pdfium
from faker import Faker
from PIL import Image, ImageFilter
from reportlab.lib.colors import Color, black, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas

from docflow_core.schemas.invoice import Currency, InvoiceV2, LineItem
from docflow_core.taxid import gstin_check_char

LAYOUTS = ("classic", "modern", "compact")
CENT = Decimal("0.01")
GREY = Color(0.92, 0.92, 0.92)

LOCALES = {"INR": "en_IN", "USD": "en_US", "GBP": "en_GB", "EUR": "en_IE"}
SYMBOLS = {"INR": "Rs.", "USD": "$", "GBP": "£", "EUR": "€"}  # WinAnsi has £ and €, not ₹
ITEMS = [
    ("A4 copier paper, 500 sheets", 240, 420),
    ("Laser toner cartridge", 1800, 5200),
    ("USB-C cable 1 m", 150, 650),
    ("Wireless mouse", 450, 1500),
    ("Office chair", 4500, 14000),
    ("Monthly cloud hosting", 2500, 9000),
    ("Consulting, per hour", 1500, 4000),
    ("Annual maintenance contract", 8000, 25000),
    ("Printer servicing", 900, 3000),
    ("Whiteboard markers, box of 10", 180, 420),
    ("LED desk lamp", 700, 2200),
    ("Network switch, 8 port", 1600, 4800),
]


@dataclass(frozen=True)
class NoiseSpec:
    max_rotation: float  # degrees, sign chosen at random
    blur: float  # Gaussian radius in pixels at 200 DPI
    speckle: float  # fraction of pixels flipped to black or white
    jpeg_quality: int


NOISE: dict[str, NoiseSpec | None] = {
    "clean": None,
    "light": NoiseSpec(max_rotation=1.0, blur=0.5, speckle=0.001, jpeg_quality=80),
    "heavy": NoiseSpec(max_rotation=3.0, blur=1.1, speckle=0.004, jpeg_quality=40),
}


# --------------------------------------------------------------------------- content


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def format_amount(value: Decimal, currency: str) -> str:
    """1234567.5 -> '12,34,567.50' for INR (Indian grouping), '1,234,567.50' otherwise."""
    whole, frac = f"{value:.2f}".split(".")
    sign = "-" if whole.startswith("-") else ""
    whole = whole.lstrip("-")
    if currency == "INR" and len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups: list[str] = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join([*groups, tail])
    else:
        whole = f"{int(whole):,}"
    return f"{sign}{whole}.{frac}"


def fake_gstin(rng: random.Random) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pan = (
        "".join(rng.choice(letters) for _ in range(3))
        + rng.choice("CPHFT")  # 4th PAN letter encodes holder type: company, person, ...
        + rng.choice(letters)
        + f"{rng.randint(0, 9999):04d}"
        + rng.choice(letters)
    )
    first14 = f"{rng.randint(1, 37):02d}{pan}{rng.choice('123456789')}Z"
    return first14 + gstin_check_char(first14)


def make_invoice(rng: random.Random, fake: Faker, currency: str) -> dict[str, Any]:
    """Return the label (InvoiceV2 JSON) plus render-only extras under '_render'."""
    fx = Decimal({"INR": "1", "USD": "0.012", "GBP": "0.0095", "EUR": "0.011"}[currency])
    items: list[LineItem] = []
    for desc, lo, hi in rng.sample(ITEMS, k=rng.randint(1, 6)):
        qty = Decimal(rng.randint(1, 12))
        unit = money(Decimal(rng.randint(lo, hi)) * fx)
        items.append(
            LineItem(description=desc, quantity=qty, unit_price=unit, amount=money(qty * unit))
        )
    subtotal = money(sum((i.amount for i in items), Decimal(0)))
    tax_rate = Decimal(rng.choice(["0.05", "0.12", "0.18", "0.18", "0.28"]))
    has_tax = rng.random() > 0.1
    tax = money(subtotal * tax_rate) if has_tax else None
    total = subtotal + (tax or Decimal(0))
    invoice_date = date(2026, 1, 1) + timedelta(days=rng.randint(0, 300))
    number = rng.choice(
        [
            f"INV-2026-{rng.randint(1, 9999):04d}",
            f"BILL/26/{rng.randint(100, 9999)}",
            f"A-{rng.randint(1, 99999):05d}",
        ]
    )
    label = InvoiceV2(
        vendor_name=fake.company(),
        invoice_number=number,
        invoice_date=invoice_date,
        currency=cast(Currency, currency),
        line_items=items,
        subtotal=subtotal if has_tax else None,
        tax=tax,
        total=total,
        vendor_tax_id=fake_gstin(rng) if currency == "INR" else None,
        po_number=f"PO-{rng.randint(2025, 2026)}-{rng.randint(1, 9999):04d}"
        if rng.random() < 0.6
        else None,
    )
    data = label.model_dump(mode="json")
    data["_render"] = {
        "vendor_address": fake.address().replace("\n", ", "),
        "buyer": fake.company(),
        "buyer_address": fake.address().replace("\n", ", "),
        "tax_rate_pct": f"{tax_rate * 100:.0f}",
    }
    return data


# --------------------------------------------------------------------------- rendering


def _amounts(inv: dict[str, Any]) -> dict[str, str]:
    cur = inv["currency"]
    out = {k: format_amount(Decimal(inv[k]), cur) for k in ("subtotal", "tax", "total") if inv[k]}
    return out


def _date(inv: dict[str, Any], layout: str) -> str:
    d = date.fromisoformat(inv["invoice_date"])
    return {
        "classic": d.strftime("%d/%m/%Y"),
        "modern": d.strftime("%d %b %Y"),
        "compact": d.isoformat(),
    }[layout]


def _render_classic(c: Canvas, inv: dict[str, Any]) -> None:
    r, cur, amt = inv["_render"], inv["currency"], _amounts(inv)
    w, h = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h - 60, inv["vendor_name"])
    c.setFont("Helvetica", 9)
    c.drawString(40, h - 76, r["vendor_address"][:95])
    if inv["vendor_tax_id"]:
        c.drawString(40, h - 90, f"GSTIN: {inv['vendor_tax_id']}")
    c.setFont("Helvetica-Bold", 14)
    c.drawRightString(w - 40, h - 60, "TAX INVOICE")
    c.setFont("Helvetica", 10)
    c.drawRightString(w - 40, h - 80, f"Invoice No.: {inv['invoice_number']}")
    c.drawRightString(w - 40, h - 95, f"Date: {_date(inv, 'classic')}")
    if inv["po_number"]:
        c.drawRightString(w - 40, h - 110, f"PO No.: {inv['po_number']}")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, h - 140, "Bill To:")
    c.setFont("Helvetica", 10)
    c.drawString(40, h - 155, r["buyer"])
    c.drawString(40, h - 169, r["buyer_address"][:95])

    y = h - 210
    cols = (40, 330, 400, w - 40)
    c.setFont("Helvetica-Bold", 10)
    for x, text in zip(cols[:3], ("Description", "Qty", "Rate"), strict=True):
        c.drawString(x, y, text)
    c.drawRightString(cols[3], y, f"Amount ({cur})")
    c.line(40, y - 5, w - 40, y - 5)
    c.setFont("Helvetica", 10)
    for item in inv["line_items"]:
        y -= 20
        c.drawString(cols[0], y, item["description"])
        c.drawString(cols[1], y, str(int(Decimal(item["quantity"]))))
        c.drawString(cols[2], y, format_amount(Decimal(item["unit_price"]), cur))
        c.drawRightString(cols[3], y, format_amount(Decimal(item["amount"]), cur))
    c.line(40, y - 8, w - 40, y - 8)
    y -= 30
    if "subtotal" in amt:
        c.drawRightString(440, y, "Subtotal")
        c.drawRightString(w - 40, y, amt["subtotal"])
        y -= 16
    if "tax" in amt:
        c.drawRightString(440, y, f"GST @ {r['tax_rate_pct']}%")
        c.drawRightString(w - 40, y, amt["tax"])
        y -= 16
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(440, y, "Total")
    c.drawRightString(w - 40, y, f"{SYMBOLS[cur]} {amt['total']}")


def _render_modern(c: Canvas, inv: dict[str, Any]) -> None:
    r, cur, amt = inv["_render"], inv["currency"], _amounts(inv)
    w, h = A4
    c.setFont("Helvetica-Bold", 28)
    c.drawString(40, h - 70, "INVOICE")
    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(w - 40, h - 50, inv["vendor_name"])
    c.setFont("Helvetica", 9)
    c.drawRightString(w - 40, h - 64, r["vendor_address"][:80])
    if inv["vendor_tax_id"]:
        c.drawRightString(w - 40, h - 78, f"Tax ID {inv['vendor_tax_id']}")

    c.setFillColor(GREY)
    c.rect(40, h - 160, w - 80, 55, stroke=0, fill=1)
    c.setFillColor(black)
    c.setFont("Helvetica", 9)
    meta = [("Bill #", inv["invoice_number"]), ("Issued", _date(inv, "modern"))]
    if inv["po_number"]:
        meta.append(("Purchase order", inv["po_number"]))
    for i, (k, v) in enumerate(meta):
        x = 55 + i * 170
        c.drawString(x, h - 125, k.upper())
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, h - 142, v)
        c.setFont("Helvetica", 9)
    c.drawString(40, h - 185, f"Billed to: {r['buyer']}")

    y = h - 225
    c.setFillColor(black)
    c.rect(40, y - 6, w - 80, 20, stroke=0, fill=1)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(48, y, "ITEM")
    c.drawRightString(360, y, "QTY")
    c.drawRightString(450, y, "UNIT PRICE")
    c.drawRightString(w - 48, y, "LINE TOTAL")
    c.setFillColor(black)
    c.setFont("Helvetica", 10)
    for item in inv["line_items"]:
        y -= 22
        c.drawString(48, y, item["description"])
        c.drawRightString(360, y, str(int(Decimal(item["quantity"]))))
        c.drawRightString(450, y, format_amount(Decimal(item["unit_price"]), cur))
        c.drawRightString(w - 48, y, format_amount(Decimal(item["amount"]), cur))
    y -= 40
    box_top = y + 14
    lines = []
    if "subtotal" in amt:
        lines.append(("Net amount", amt["subtotal"]))
    if "tax" in amt:
        lines.append((f"Tax ({r['tax_rate_pct']}%)", amt["tax"]))
    for k, v in lines:
        c.drawString(330, y, k)
        c.drawRightString(w - 48, y, v)
        y -= 18
    c.setFont("Helvetica-Bold", 13)
    c.drawString(330, y - 4, f"Grand Total ({cur})")
    c.drawRightString(w - 48, y - 4, amt["total"])
    c.rect(320, y - 14, w - 360, box_top - y + 14, stroke=1, fill=0)


def _render_compact(c: Canvas, inv: dict[str, Any]) -> None:
    cur, amt = inv["currency"], _amounts(inv)
    w, h = A4
    left, right, mid = 150, w - 150, w / 2
    c.setFont("Courier-Bold", 13)
    c.drawCentredString(mid, h - 60, inv["vendor_name"].upper()[:38])
    c.setFont("Courier", 9)
    if inv["vendor_tax_id"]:
        c.drawCentredString(mid, h - 74, f"GSTIN {inv['vendor_tax_id']}")
    c.drawString(left, h - 100, f"Receipt: {inv['invoice_number']}")
    c.drawString(left, h - 113, f"Date:    {_date(inv, 'compact')}")
    if inv["po_number"]:
        c.drawString(left, h - 126, f"PO:      {inv['po_number']}")
    y = h - 150
    c.drawString(left, y, "-" * 50)
    for item in inv["line_items"]:
        y -= 14
        c.drawString(left, y, item["description"][:30])
        y -= 12
        qty = int(Decimal(item["quantity"]))
        c.drawString(left + 10, y, f"{qty} x {format_amount(Decimal(item['unit_price']), cur)}")
        c.drawRightString(right, y, format_amount(Decimal(item["amount"]), cur))
    y -= 10
    c.drawString(left, y, "-" * 50)
    if "subtotal" in amt:
        y -= 14
        c.drawString(left, y, "Sub-total")
        c.drawRightString(right, y, amt["subtotal"])
    if "tax" in amt:
        y -= 14
        c.drawString(left, y, "Tax")
        c.drawRightString(right, y, amt["tax"])
    y -= 18
    c.setFont("Courier-Bold", 11)
    c.drawString(left, y, f"TOTAL {cur}")
    c.drawRightString(right, y, amt["total"])


RENDERERS = {"classic": _render_classic, "modern": _render_modern, "compact": _render_compact}


def render_pdf(inv: dict[str, Any], layout: str) -> bytes:
    buf = BytesIO()
    # invariant=1 drops timestamps and random ids, so the same seed gives identical bytes.
    canvas = Canvas(buf, pagesize=A4, invariant=1)
    canvas.setTitle(f"Invoice {inv['invoice_number']}")
    RENDERERS[layout](canvas, inv)
    canvas.showPage()
    canvas.save()
    return buf.getvalue()


def degrade(pdf: bytes, spec: NoiseSpec, rng: random.Random, dpi: int = 200) -> bytes:
    """Make a clean PDF look like a phone scan: rotate, blur, speckle, compress."""
    doc = pdfium.PdfDocument(pdf)
    try:
        img = doc[0].render(scale=dpi / 72).to_pil().convert("L")
    finally:
        doc.close()
    angle = rng.uniform(0.3, spec.max_rotation) * rng.choice((-1, 1))
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(spec.blur))
    arr = np.asarray(img).copy()
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    mask = np_rng.random(arr.shape) < spec.speckle
    arr[mask] = np_rng.choice(np.array([0, 255], dtype=np.uint8), size=int(np.count_nonzero(mask)))
    out = BytesIO()
    Image.fromarray(arr).save(out, format="JPEG", quality=spec.jpeg_quality)
    return out.getvalue()


# --------------------------------------------------------------------------- driver


def generate(n: int, out_dir: Path, seed: int) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fakers = {cur: Faker(loc) for cur, loc in LOCALES.items()}
    for i, f in enumerate(fakers.values()):
        f.seed_instance(seed + i)

    rows: list[dict[str, Any]] = []
    for idx in range(n):
        currency = rng.choices(list(LOCALES), weights=[70, 15, 8, 7])[0]
        layout = LAYOUTS[idx % len(LAYOUTS)]
        noise = rng.choices(list(NOISE), weights=[40, 40, 20])[0]
        inv = make_invoice(rng, fakers[currency], currency)
        pdf = render_pdf(inv, layout)
        spec = NOISE[noise]
        doc_id = f"syn-{seed}-{idx:04d}"
        if spec is None:
            data, ext = pdf, "pdf"
        else:
            data, ext = degrade(pdf, spec, rng), "jpg"
        file = out_dir / f"{doc_id}.{ext}"
        file.write_bytes(data)
        label = {k: v for k, v in inv.items() if not k.startswith("_")}
        label_path = out_dir / f"{doc_id}.json"
        label_path.write_text(json.dumps(label, indent=2), encoding="utf-8")
        rows.append(
            {
                "id": doc_id,
                "file": file.name,
                "sha256": hashlib.sha256(data).hexdigest(),
                "layout": layout,
                "noise": noise,
                "currency": currency,
                "label": label_path.name,
            }
        )
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rows = generate(args.n, args.out, args.seed)
    by_noise = {k: sum(r["noise"] == k for r in rows) for k in NOISE}
    print(f"wrote {len(rows)} invoices to {args.out} ({by_noise})")


if __name__ == "__main__":
    main()
