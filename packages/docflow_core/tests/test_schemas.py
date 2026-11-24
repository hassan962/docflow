from datetime import date
from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from docflow_core.schemas import ExtractedField, SchemaSpec, registry
from docflow_core.schemas.invoice import InvoiceV1, InvoiceV2
from docflow_core.schemas.registry import SchemaRegistry

VALID_INVOICE = {
    "vendor_name": "Acme Traders",
    "invoice_number": "INV-0042",
    "invoice_date": "2026-03-12",
    "currency": "INR",
    "line_items": [
        {"description": "Widget", "quantity": "2", "unit_price": "500", "amount": "1000"}
    ],
    "subtotal": "1000.00",
    "tax": "180.00",
    "total": "1180.00",
}


def test_invoice_v1_parses_a_valid_reply() -> None:
    inv = InvoiceV1.model_validate(VALID_INVOICE)
    assert inv.total == Decimal("1180.00")
    assert inv.invoice_date == date(2026, 3, 12)


def test_invoice_rejects_unknown_fields_and_currencies() -> None:
    with pytest.raises(ValidationError):
        InvoiceV1.model_validate({**VALID_INVOICE, "made_up_field": 1})
    with pytest.raises(ValidationError):
        InvoiceV1.model_validate({**VALID_INVOICE, "currency": "XYZ"})


def test_v2_adds_fields_without_breaking_v1_data() -> None:
    v1_fields = set(InvoiceV1.model_fields)
    v2_fields = set(InvoiceV2.model_fields)
    assert v2_fields - v1_fields == {"vendor_tax_id", "po_number"}
    # A v1 reply is still a valid v2 reply because the new fields are optional.
    InvoiceV2.model_validate(VALID_INVOICE)


def test_registry_lookup() -> None:
    assert registry.latest("invoice").version == 2
    assert registry.get("invoice", 1).model is InvoiceV1
    assert "invoice" in registry.doc_types()
    with pytest.raises(KeyError):
        registry.get("invoice", 99)


def test_required_and_critical_fields() -> None:
    spec = registry.get("invoice", 1)
    assert set(spec.required_fields) == {
        "vendor_name", "invoice_number", "invoice_date", "currency", "line_items", "total",
    }  # fmt: skip
    assert "total" in spec.critical_fields


def test_published_versions_are_immutable() -> None:
    class A(BaseModel):
        x: int

    class B(BaseModel):
        x: str

    reg = SchemaRegistry()
    reg.register(SchemaSpec(doc_type="t", version=1, model=A))
    reg.register(SchemaSpec(doc_type="t", version=1, model=A))  # same schema: fine
    with pytest.raises(ValueError, match="immutable"):
        reg.register(SchemaSpec(doc_type="t", version=1, model=B))


def test_fingerprint_is_stable() -> None:
    spec = registry.get("invoice", 1)
    assert spec.fingerprint() == spec.fingerprint()
    assert spec.fingerprint() != registry.get("invoice", 2).fingerprint()


def test_extracted_field_shape() -> None:
    f = ExtractedField.model_validate(
        {
            "value": "1180.00",
            "raw": "Rs. 1,180.00",
            "confidence": 0.93,
            "page": 1,
            "bbox": [412, 880, 530, 902],
            "checks": [{"rule": "line_items_sum", "passed": True}],
        }
    )
    assert f.bbox == (412, 880, 530, 902)
    assert f.failed_rules == []
    with pytest.raises(ValidationError):
        ExtractedField(value="x", bbox=(10, 10, 5, 20))  # x1 < x0
    with pytest.raises(ValidationError):
        ExtractedField(value="x", confidence=1.5)
