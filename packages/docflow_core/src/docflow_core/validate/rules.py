"""Validation rules: pure functions from an extraction's values to rule results.

Each rule names the fields it covers. A failure lowers the confidence of exactly those fields
(through the ``validators_passed`` signal) and is shown to the reviewer with its message.
Rules never change a value. When a rule can say what the value probably should be (a
GSTIN with OCR look-alikes undone), it attaches a *suggestion* the reviewer can accept with
one key.

Rules are registered per document type, so a new document family brings its own list.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from docflow_core.extract.fields import required_keys
from docflow_core.normalise import normalise_amount, normalise_date
from docflow_core.schemas.registry import SchemaSpec
from docflow_core.taxid import is_valid_gstin, repair_gstin

TOLERANCE = Decimal("0.01")


@dataclass(frozen=True)
class RuleResult:
    rule: str
    passed: bool
    fields: tuple[str, ...]  # flat keys the rule covers
    message: str | None = None
    suggestion: dict[str, Any] = field(default_factory=dict)  # {flat key: suggested value}


@dataclass(frozen=True)
class RuleContext:
    spec: SchemaSpec
    today: date
    max_age_days: int = 3 * 365
    grounded: dict[str, str] = field(default_factory=dict)  # flat key -> grounding match


Rule = Callable[[dict[str, Any], RuleContext], list[RuleResult]]


def _rows(values: dict[str, Any]) -> list[int]:
    rows = {int(k.split("[")[1].split("]")[0]) for k in values if k.startswith("line_items[")}
    return sorted(rows)


def _amount(values: dict[str, Any], key: str) -> Decimal | None:
    return normalise_amount(values.get(key))


def line_items_sum(values: dict[str, Any], ctx: RuleContext) -> list[RuleResult]:
    rows = _rows(values)
    amounts = [_amount(values, f"line_items[{i}].amount") for i in rows]
    if not rows or any(a is None for a in amounts):
        return []
    total_of_rows = sum((a for a in amounts if a is not None), Decimal(0))
    # Against the subtotal when printed; receipts without tax print only a total.
    target_key = "subtotal" if _amount(values, "subtotal") is not None else "total"
    target = _amount(values, target_key)
    if target is None:
        return []
    covered = (*(f"line_items[{i}].amount" for i in rows), target_key)
    ok = abs(total_of_rows - target) <= TOLERANCE
    msg = None if ok else f"line amounts add up to {total_of_rows}, {target_key} is {target}"
    return [RuleResult("line_items_sum", ok, covered, msg)]


def total_consistency(values: dict[str, Any], ctx: RuleContext) -> list[RuleResult]:
    subtotal, tax, total = (_amount(values, k) for k in ("subtotal", "tax", "total"))
    if subtotal is None or total is None:
        return []
    expected = subtotal + (tax or Decimal(0))
    ok = abs(expected - total) <= TOLERANCE
    msg = None if ok else f"subtotal + tax = {expected}, but total is {total}"
    return [RuleResult("total_consistency", ok, ("subtotal", "tax", "total"), msg)]


def row_arithmetic(values: dict[str, Any], ctx: RuleContext) -> list[RuleResult]:
    """quantity x unit price = amount, per row. Catches a misread digit in any of the three."""
    out = []
    for i in _rows(values):
        keys = tuple(f"line_items[{i}].{c}" for c in ("quantity", "unit_price", "amount"))
        qty, price, amount = (_amount(values, k) for k in keys)
        if qty is None or price is None or amount is None:
            continue
        ok = abs(qty * price - amount) <= TOLERANCE
        msg = None if ok else f"row {i + 1}: {qty} x {price} = {qty * price}, amount is {amount}"
        out.append(RuleResult("row_arithmetic", ok, keys, msg))
    return out


def valid_date(values: dict[str, Any], ctx: RuleContext) -> list[RuleResult]:
    raw = values.get("invoice_date")
    if raw is None:
        return []
    parsed = normalise_date(raw)
    if parsed is None:
        return [RuleResult("valid_date", False, ("invoice_date",), f"{raw!r} is not a date")]
    if parsed > ctx.today:
        return [RuleResult("valid_date", False, ("invoice_date",), f"{parsed} is in the future")]
    if parsed < ctx.today - timedelta(days=ctx.max_age_days):
        msg = f"{parsed} is more than {ctx.max_age_days // 365} years old"
        return [RuleResult("valid_date", False, ("invoice_date",), msg)]
    return [RuleResult("valid_date", True, ("invoice_date",))]


def known_currency(values: dict[str, Any], ctx: RuleContext) -> list[RuleResult]:
    currency = values.get("currency")
    if currency is None:
        return []
    allowed = ctx.spec.model.model_fields["currency"].annotation
    allowed_codes = set(getattr(allowed, "__args__", ()))
    if allowed_codes and currency not in allowed_codes:
        return [RuleResult("known_currency", False, ("currency",), f"{currency} is not accepted")]
    if ctx.grounded.get("currency") == "none":
        msg = f"no {currency} symbol or code is printed on the page"
        return [RuleResult("known_currency", False, ("currency",), msg)]
    return [RuleResult("known_currency", True, ("currency",))]


def required_fields(values: dict[str, Any], ctx: RuleContext) -> list[RuleResult]:
    required = required_keys(ctx.spec.model)
    out = []
    for key in values:
        base = key.split("[")[0] + ("." + key.split(".")[-1] if "[" in key else "")
        if base in required and values[key] in (None, ""):
            out.append(RuleResult("required_fields", False, (key,), "required but not found"))
    if not any(k.startswith("line_items[") for k in values) and "line_items.amount" in required:
        out.append(RuleResult("required_fields", False, ("line_items",), "no line items found"))
    return out


def tax_id_format(values: dict[str, Any], ctx: RuleContext) -> list[RuleResult]:
    tax_id = values.get("vendor_tax_id")
    if not tax_id:
        return []
    if values.get("currency") != "INR":
        return []  # only the Indian GSTIN format is implemented
    if is_valid_gstin(str(tax_id)):
        return [RuleResult("tax_id_format", True, ("vendor_tax_id",))]
    repaired = repair_gstin(str(tax_id))
    if repaired:
        msg = f"not a valid GSTIN; OCR look-alikes suggest {repaired}"
        return [
            RuleResult("tax_id_format", False, ("vendor_tax_id",), msg, {"vendor_tax_id": repaired})
        ]
    return [RuleResult("tax_id_format", False, ("vendor_tax_id",), "not a valid GSTIN")]


VALIDATORS: dict[str, list[Rule]] = {
    "invoice": [
        line_items_sum,
        total_consistency,
        row_arithmetic,
        valid_date,
        known_currency,
        required_fields,
        tax_id_format,
    ],
}


def run_rules(values: dict[str, Any], ctx: RuleContext) -> list[RuleResult]:
    results: list[RuleResult] = []
    for rule in VALIDATORS.get(ctx.spec.doc_type, []):
        results.extend(rule(values, ctx))
    return results
