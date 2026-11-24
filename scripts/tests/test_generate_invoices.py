import json
import random
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from faker import Faker
from PIL import Image

from docflow_core.schemas.invoice import InvoiceV2
from docflow_core.taxid import is_valid_gstin
from generate_invoices import (
    LAYOUTS,
    NOISE,
    degrade,
    fake_gstin,
    format_amount,
    generate,
    make_invoice,
    render_pdf,
)


@pytest.mark.parametrize(
    ("value", "currency", "expected"),
    [
        ("1234567.5", "INR", "12,34,567.50"),
        ("103095", "INR", "1,03,095.00"),
        ("999", "INR", "999.00"),
        ("1234567.5", "USD", "1,234,567.50"),
        ("12.3", "EUR", "12.30"),
    ],
)
def test_amount_formatting(value: str, currency: str, expected: str) -> None:
    assert format_amount(Decimal(value), currency) == expected


def test_generated_gstins_are_valid() -> None:
    rng = random.Random(1)
    assert all(is_valid_gstin(fake_gstin(rng)) for _ in range(200))


@pytest.mark.parametrize("currency", ["INR", "USD", "GBP", "EUR"])
def test_labels_are_valid_and_arithmetic_holds(currency: str) -> None:
    rng, fake = random.Random(3), Faker()
    fake.seed_instance(3)
    for _ in range(25):
        inv = make_invoice(rng, fake, currency)
        label = InvoiceV2.model_validate({k: v for k, v in inv.items() if k != "_render"})
        items_sum = sum((i.amount for i in label.line_items), Decimal(0))
        for item in label.line_items:
            assert item.quantity is not None and item.unit_price is not None
            assert item.quantity * item.unit_price == item.amount
        if label.tax is None:
            assert label.total == items_sum
        else:
            assert label.subtotal == items_sum and label.subtotal + label.tax == label.total


@pytest.mark.parametrize("layout", LAYOUTS)
def test_pdfs_are_deterministic_and_contain_the_fields(layout: str) -> None:
    def build() -> tuple[dict[str, object], bytes]:
        fake = Faker("en_IN")
        fake.seed_instance(9)
        inv = make_invoice(random.Random(9), fake, "INR")
        return inv, render_pdf(inv, layout)

    inv, pdf = build()
    assert build()[1] == pdf
    doc = pdfium.PdfDocument(pdf)
    text = doc[0].get_textpage().get_text_range()
    doc.close()
    assert str(inv["invoice_number"]) in text
    assert format_amount(Decimal(str(inv["total"])), "INR") in text


def test_degrade_produces_a_rotated_jpeg() -> None:
    fake = Faker()
    fake.seed_instance(4)
    pdf = render_pdf(make_invoice(random.Random(4), fake, "USD"), "modern")
    spec = NOISE["heavy"]
    assert spec is not None
    img = Image.open(BytesIO(degrade(pdf, spec, random.Random(4))))
    assert img.format == "JPEG"
    assert img.size != (1654, 2339)  # an unrotated A4 page at 200 DPI; rotation expands it


def test_generate_writes_documents_labels_and_manifest(tmp_path: Path) -> None:
    rows = generate(6, tmp_path, seed=11)
    manifest = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert manifest == rows and len(rows) == 6
    assert {r["layout"] for r in rows} == set(LAYOUTS)
    for row in rows:
        assert (tmp_path / row["file"]).exists()
        InvoiceV2.model_validate_json((tmp_path / row["label"]).read_text())
