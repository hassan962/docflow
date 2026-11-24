"""Validation rules, GSTIN repair and routing (pure functions, no database)."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from docflow_core.extract.confidence import IsotonicCalibrator
from docflow_core.extract.fields import flatten
from docflow_core.schemas import registry
from docflow_core.schemas.invoice import InvoiceV2
from docflow_core.taxid import is_valid_gstin, repair_gstin
from docflow_core.validate.apply import route, stored_reference_results
from docflow_core.validate.rules import RuleContext, RuleResult, run_rules

SPEC = registry.get("invoice", 2)
TODAY = date(2026, 11, 24)
GOOD: dict[str, Any] = {
    "vendor_name": "Acme Traders",
    "invoice_number": "INV-1",
    "invoice_date": "2026-03-05",
    "currency": "INR",
    "line_items": [
        {"description": "Widget", "quantity": "2", "unit_price": "500.00", "amount": "1000.00"},
        {"description": "Bolt", "quantity": "4", "unit_price": "25.00", "amount": "100.00"},
    ],
    "subtotal": "1100.00",
    "tax": "198.00",
    "total": "1298.00",
    "vendor_tax_id": "27AAPFU0939F1ZV",
    "po_number": None,
}


def failures(data: dict[str, Any], **ctx: Any) -> dict[str, set[str]]:
    results = run_rules(flatten(InvoiceV2, data), RuleContext(SPEC, TODAY, **ctx))
    out: dict[str, set[str]] = {}
    for r in results:
        if not r.passed:
            out.setdefault(r.rule, set()).update(r.fields)
    return out


def test_a_consistent_invoice_passes_every_rule() -> None:
    assert failures(GOOD) == {}


def test_arithmetic_rules_point_at_the_right_fields() -> None:
    assert failures({**GOOD, "total": "1289.00"}) == {
        "total_consistency": {"subtotal", "tax", "total"}
    }
    broken_row = {
        **GOOD,
        "line_items": [{**GOOD["line_items"][0], "amount": "1060.00"}, GOOD["line_items"][1]],
    }
    found = failures(broken_row)
    assert "line_items[0].amount" in found["row_arithmetic"]
    assert "line_items[0].amount" in found["line_items_sum"]


def test_receipts_without_subtotal_sum_to_the_total() -> None:
    receipt = {**GOOD, "subtotal": None, "tax": None, "total": "1100.00"}
    assert failures(receipt) == {}


@pytest.mark.parametrize(
    ("value", "rule_fails"),
    [("2026-03-05", False), ("2027-01-01", True), ("2019-01-01", True), ("not a date", True)],
)
def test_valid_date(value: str, rule_fails: bool) -> None:
    assert ("valid_date" in failures({**GOOD, "invoice_date": value})) is rule_fails


def test_required_fields_and_currency_printed() -> None:
    found = failures({**GOOD, "total": None, "invoice_number": None})
    assert found["required_fields"] == {"total", "invoice_number"}
    assert "known_currency" in failures(GOOD, grounded={"currency": "none"})


# --- GSTIN ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("misread", "truth"),
    [
        ("OSWOPTQ6379M421", "05WOPTQ6379M4Z1"),  # O for 0, 2 for Z
        ("30PVRFY514606ZR", "30PVRFY5146O6ZR"),  # 0 for O in a letter position
        ("16FULHL9785J22Z1", "16FULHL9785J2Z1"),  # doubled glyph: 16 characters
        ("11AHPPD13960Q9ZQ", None),  # ambiguous: left for a person
    ],
)
def test_gstin_repair_undoes_ocr_look_alikes(misread: str, truth: str | None) -> None:
    assert not is_valid_gstin(misread)
    repaired = repair_gstin(misread)
    if truth is None:
        assert repaired is None or is_valid_gstin(repaired)
    else:
        assert repaired == truth


def test_tax_id_rule_attaches_the_repair_as_a_suggestion() -> None:
    results = run_rules(
        flatten(InvoiceV2, {**GOOD, "vendor_tax_id": "2TAAPFU0939F1ZV"}), RuleContext(SPEC, TODAY)
    )
    [tax] = [r for r in results if r.rule == "tax_id_format"]
    assert not tax.passed and tax.suggestion == {"vendor_tax_id": "27AAPFU0939F1ZV"}
    # Non-Indian invoices are not held to the GSTIN format.
    assert "tax_id_format" not in failures({**GOOD, "currency": "USD", "vendor_tax_id": "X1"})


# --- routing ---------------------------------------------------------------------------------


def field(value: Any, match: str = "exact", conf: float = 0.95, **extra: Any) -> dict[str, Any]:
    signals = {"match": match, "ocr_conf": conf, "similarity": 1.0, "agreement": True}
    return {"value": value, "confidence": 0.0, "signals": signals, "checks": [], **extra}


def test_route_sends_flagged_and_low_confidence_fields_to_review() -> None:
    fields = {
        "total": field("1298.00"),
        "tax": field("198.00"),
        "po_number": field("PO-9", match="none"),  # not on the page
        "vendor_tax_id": field("2TAAPFU0939F1ZV"),
    }
    results = [
        RuleResult("total_consistency", True, ("total", "tax")),
        RuleResult(
            "tax_id_format", False, ("vendor_tax_id",), "bad", {"vendor_tax_id": "27AAPFU0939F1ZV"}
        ),
        RuleResult("required_fields", False, ("line_items",), "no line items found"),
    ]
    routed = route(fields, results, IsotonicCalibrator.identity(), threshold=0.9)
    status = {k: f["status"] for k, f in routed.fields.items()}
    assert status == {
        "total": "auto_approved",
        "tax": "auto_approved",
        "po_number": "needs_review",
        "vendor_tax_id": "needs_review",
    }
    assert routed.fields["vendor_tax_id"]["checks"][0]["suggestion"] == "27AAPFU0939F1ZV"
    assert routed.fields["vendor_tax_id"]["signals"]["validators_passed"] is False
    assert routed.issues == [
        {
            "rule": "required_fields",
            "passed": False,
            "message": "no line items found",
            "field": "line_items",
        }
    ]
    assert not routed.approved and routed.needs_review == 2


def test_reviewed_fields_keep_their_status_and_reference_results_survive() -> None:
    reviewed = {**field("1298.00"), "source": "review"}
    check = {"rule": "reference_checks", "passed": False, "message": "m", "suggestion": "Acme"}
    fields = {"total": reviewed, "vendor_name": {**field("Acme Tradrs"), "checks": [check]}}
    kept = stored_reference_results(fields)
    assert kept == [
        RuleResult("reference_checks", False, ("vendor_name",), "m", {"vendor_name": "Acme"})
    ]
    routed = route(fields, kept, IsotonicCalibrator.identity(), threshold=0.5)
    assert routed.fields["total"]["status"] == "reviewed"
    assert routed.fields["vendor_name"]["status"] == "needs_review"
