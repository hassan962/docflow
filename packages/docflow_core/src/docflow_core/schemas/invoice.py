"""Invoice schemas.

The Pydantic model is the single contract for a document type: it generates the JSON schema
sent to the LLM, validates the model's reply, and defines the fields the review UI shows.

v1 is the design doc's original schema. v2 adds ``vendor_tax_id`` and ``po_number``, which the
tax-ID validator and the purchase-order tool need. Adding them created a new version rather
than editing v1, so every v1 extraction already stored stays valid.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from docflow_core.schemas.registry import SchemaSpec, registry

Currency = Literal["INR", "USD", "EUR", "GBP"]


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal


class InvoiceV1(BaseModel):
    model_config = ConfigDict(extra="forbid", title="Invoice v1")

    vendor_name: str = Field(description="Legal name of the seller as printed.")
    invoice_number: str
    invoice_date: date
    currency: Currency
    line_items: list[LineItem]
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal


class InvoiceV2(InvoiceV1):
    model_config = ConfigDict(extra="forbid", title="Invoice v2")

    vendor_tax_id: str | None = Field(
        default=None, description="Seller's tax registration, e.g. an Indian GSTIN."
    )
    po_number: str | None = Field(default=None, description="Buyer's purchase-order number.")


CRITICAL = ("total", "invoice_date", "vendor_name", "currency")

INVOICE_V1 = registry.register(
    SchemaSpec(doc_type="invoice", version=1, model=InvoiceV1, critical_fields=CRITICAL)
)
INVOICE_V2 = registry.register(
    SchemaSpec(doc_type="invoice", version=2, model=InvoiceV2, critical_fields=CRITICAL)
)
