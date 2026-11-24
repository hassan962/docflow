"""Extraction pieces: field keys, strict schema, prompt, grounding, confidence, orchestration."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest

from docflow_core.eval.match import field_correctness
from docflow_core.extract.confidence import (
    IsotonicCalibrator,
    Signals,
    choose_threshold,
    raw_score,
)
from docflow_core.extract.extractor import extract
from docflow_core.extract.fields import base_name, field_kinds, flatten, same_value, unflatten
from docflow_core.extract.grounding import ground
from docflow_core.extract.llm import ChatRequest, ScriptedClient
from docflow_core.extract.llm_schema import object_schema
from docflow_core.extract.prompt import FewShotExample, build_prompt, max_tokens_for, ocr_text
from docflow_core.ocr import OcrResult, OcrWord
from docflow_core.schemas import registry
from docflow_core.schemas.invoice import InvoiceV2

SPEC = registry.get("invoice", 2)


def page(*lines: str, conf: float = 0.95) -> OcrResult:
    """An OCR page from text lines, with simple left-to-right word boxes."""
    words = []
    for line_no, line in enumerate(lines):
        x = 10
        for token in line.split():
            words.append(
                OcrWord(
                    token,
                    (x, 20 + 30 * line_no, x + 10 * len(token), 40 + 30 * line_no),
                    conf,
                    line_no,
                )
            )
            x += 10 * len(token) + 8
    return OcrResult(words=words, page_width=1000, page_height=1400, engine="test")


INVOICE_PAGE = page(
    "Acme Traders Pvt Ltd",
    "GSTIN: 27AAPFU0939F1ZV",
    "TAX INVOICE",
    "Invoice No.: A-08109",
    "Date: 05/03/2026  PO No.: PO-2026-0042",
    "Widget, blue 2 500.00 1,000.00",
    "Subtotal 1,000.00",
    "GST @ 18% 180.00",
    "Total Rs. 1,180.00",
)

LABEL: dict[str, Any] = {
    "vendor_name": "Acme Traders Pvt Ltd",
    "invoice_number": "A-08109",
    "invoice_date": "2026-03-05",
    "currency": "INR",
    "line_items": [
        {
            "description": "Widget, blue",
            "quantity": "2",
            "unit_price": "500.00",
            "amount": "1000.00",
        }
    ],
    "subtotal": "1000.00",
    "tax": "180.00",
    "total": "1180.00",
    "vendor_tax_id": "27AAPFU0939F1ZV",
    "po_number": "PO-2026-0042",
}


# --- field keys and kinds -----------------------------------------------------------------


def test_kinds_come_from_the_schema() -> None:
    kinds = field_kinds(InvoiceV2)
    assert kinds["total"] == "amount" and kinds["invoice_date"] == "date"
    assert kinds["currency"] == "code" and kinds["vendor_name"] == "text"
    assert kinds["line_items.quantity"] == "amount" and kinds["line_items.description"] == "text"


def test_flatten_round_trip() -> None:
    flat = flatten(InvoiceV2, LABEL)
    assert flat["line_items[0].amount"] == "1000.00" and flat["total"] == "1180.00"
    assert base_name("line_items[0].amount") == "line_items.amount"
    assert unflatten(InvoiceV2, flat) == LABEL


@pytest.mark.parametrize(
    ("kind", "a", "b", "same"),
    [
        ("amount", "Rs. 1,180.00", "1180", True),
        ("date", "05/03/2026", "2026-03-05", True),
        ("text", "INV-0042", "inv 0042", True),
        ("text", None, None, True),
        ("amount", None, "0", False),
        ("amount", "1180.00", "1108.00", False),
    ],
)
def test_same_value(kind: Any, a: Any, b: Any, same: bool) -> None:
    assert same_value(kind, a, b) is same


# --- strict schema and prompt ---------------------------------------------------------------


def test_llm_schema_is_strict() -> None:
    schema = object_schema(InvoiceV2)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(InvoiceV2.model_fields)
    assert schema["properties"]["vendor_tax_id"]["type"] == ["string", "null"]
    # Required fields are nullable too: the model must be able to say "not on the page".
    assert schema["properties"]["total"]["type"] == ["string", "null"]
    assert schema["properties"]["currency"]["enum"] == ["INR", "USD", "EUR", "GBP", None]
    item = schema["properties"]["line_items"]["items"]
    assert item["additionalProperties"] is False and "amount" in item["required"]
    assert "$defs" not in json.dumps(schema)


def test_prompt_tags_lines_and_hash_covers_the_template_only() -> None:
    p1 = build_prompt(SPEC, [INVOICE_PAGE])
    p2 = build_prompt(SPEC, [page("something else")])
    assert "[p1 l09] Total Rs. 1,180.00" in p1.messages[-1]["content"]
    assert p1.template_hash == p2.template_hash  # same version, different documents
    ex = FewShotExample("abc", "[p1 l01] x", "{}")
    with_example = build_prompt(SPEC, [INVOICE_PAGE], [ex])
    assert with_example.template_hash != p1.template_hash
    assert [m["role"] for m in with_example.messages] == ["system", "user", "assistant", "user"]
    assert max_tokens_for(SPEC) > 1500
    assert ocr_text([page("a b"), page("c")]) == "[p1 l01] a b\n[p2 l01] c"


# --- grounding ---------------------------------------------------------------------------------


def test_grounds_amounts_dates_codes_and_text() -> None:
    pages = [INVOICE_PAGE]
    total = ground("total", "amount", "1180.00", pages)
    assert (total.match, total.page, total.line, total.raw) == ("exact", 1, 8, "1,180.00")
    assert ground("invoice_date", "date", "2026-03-05", pages).raw == "05/03/2026"
    assert ground("currency", "code", "INR", pages).match == "exact"
    assert ground("vendor_name", "text", "Acme Traders Pvt. Ltd.", pages).line == 0


def test_label_hints_break_ties() -> None:
    # 1,000.00 is printed as the line amount and as the subtotal.
    assert ground("subtotal", "amount", "1000.00", [INVOICE_PAGE]).line == 6
    row = ground("line_items[0].amount", "amount", "1000.00", [INVOICE_PAGE], prefer_line=(1, 5))
    assert row.line == 5


def test_fuzzy_and_missing_values() -> None:
    misread = page("Acme Tradcrs Pvt Ltd")
    g = ground("vendor_name", "text", "Acme Traders Pvt Ltd", [misread])
    assert g.match == "fuzzy" and 0.8 <= g.similarity < 1
    assert ground("total", "amount", "9999.00", [INVOICE_PAGE]).match == "none"
    assert ground("po_number", "text", None, [INVOICE_PAGE]).match == "absent"


def test_a_bare_number_is_not_read_as_a_date() -> None:
    assert ground("invoice_date", "date", "2026-11-02", [page("Qty 2")]).match == "none"


# --- confidence ----------------------------------------------------------------------------


def test_raw_score_orders_the_evidence() -> None:
    exact = raw_score(Signals("exact", 0.95, 1.0, agreement=True))
    fuzzy = raw_score(Signals("fuzzy", 0.95, 0.85, agreement=True))
    missing = raw_score(Signals("none"))
    disputed = raw_score(Signals("exact", 0.95, 1.0, agreement=False))
    failed_rule = raw_score(Signals("exact", 0.95, 1.0, validators_passed=False))
    assert exact > fuzzy > missing
    assert exact > disputed and exact > failed_rule


def test_isotonic_calibration_is_monotone_and_maps_to_frequencies() -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95] * 10
    correct = [s > 0.5 or (s == 0.5 and i % 2 == 0) for i, s in enumerate(scores)]
    cal = IsotonicCalibrator.fit(scores, correct)
    mapped = [cal(s) for s in sorted(set(scores))]
    assert mapped == sorted(mapped)
    assert cal(0.1) < 0.1 and cal(0.95) > 0.9  # smoothed, never exactly 0 or 1
    assert IsotonicCalibrator.from_json(cal.to_json()) == cal
    assert IsotonicCalibrator.identity()(0.42) == 0.42


def test_pav_pools_violators() -> None:
    cal = IsotonicCalibrator.fit([0.1, 0.2, 0.3], [True, False, True])
    # 0.1 (right) and 0.2 (wrong) violate monotonicity and are pooled into one step.
    assert cal.thresholds == [0.1, 0.3]


def test_threshold_choice_hits_the_target_precision() -> None:
    conf = [0.99, 0.98, 0.97, 0.9, 0.8, 0.6, 0.4]
    ok = [True, True, True, True, False, True, False]
    choice = choose_threshold(conf, ok, target_precision=0.98)
    assert choice.threshold == 0.9 and choice.precision == 1.0
    assert choice.auto_approve_rate == pytest.approx(4 / 7, abs=1e-3)
    assert choose_threshold([0.5], [False], 0.98).threshold > 1  # nothing qualifies


# --- orchestration with a scripted model -------------------------------------------------------


def reply(**overrides: Any) -> str:
    return json.dumps({**LABEL, **overrides})


def test_extract_grounds_scores_and_uses_the_second_opinion() -> None:
    primary = ScriptedClient(replies=[reply()])
    second = ScriptedClient(model="second", replies=[reply(total="1108.00")])
    out = extract(SPEC, [INVOICE_PAGE], primary, agreement_client=second)
    assert out.status == "ok" and [c.role for c in out.calls] == ["primary", "agreement"]
    total, tax = out.fields["total"], out.fields["tax"]
    assert total.signals["agreement"] is False and tax.signals["agreement"] is True
    assert total.confidence < tax.confidence
    assert total.bbox is not None and total.page == 1
    assert out.fields["line_items[0].amount"].value == "1000.00"
    assert primary.requests[0].json_schema is not None


def test_invalid_reply_is_retried_once_with_the_error() -> None:
    client = ScriptedClient(replies=['{"total": "oops"', reply()])
    out = extract(SPEC, [INVOICE_PAGE], client)
    assert out.status == "ok" and [c.role for c in out.calls] == ["primary", "retry"]
    follow_up = client.requests[1].messages[-1]["content"]
    assert follow_up.startswith("That reply was invalid")


def test_two_bad_replies_mean_failed_parse() -> None:
    client = ScriptedClient(replies=[reply(currency="XYZ"), None])
    out = extract(SPEC, [INVOICE_PAGE], client)
    assert out.status == "failed_parse" and out.fields == {}
    assert out.error and "empty reply" in out.error


def test_a_failing_second_opinion_only_removes_that_signal() -> None:
    def boom(request: ChatRequest) -> str:
        raise RuntimeError("rate limited")

    out = extract(
        SPEC, [INVOICE_PAGE], ScriptedClient(replies=[reply()]),
        agreement_client=ScriptedClient(responder=boom),
    )  # fmt: skip
    assert out.status == "ok" and out.fields["total"].signals["agreement"] is None


def test_hallucinated_value_gets_low_confidence() -> None:
    out = extract(SPEC, [INVOICE_PAGE], ScriptedClient(replies=[reply(po_number="PO-9999-1111")]))
    assert out.fields["po_number"].signals["match"] == "none"
    assert out.fields["po_number"].bbox is None
    assert out.fields["po_number"].confidence < out.fields["invoice_number"].confidence


def test_a_missing_required_value_is_kept_as_null_with_near_zero_confidence() -> None:
    out = extract(
        SPEC,
        [page("Acme Traders", "Invoice No.: A-1")],
        ScriptedClient(replies=[reply(total=None)]),
    )
    assert out.status == "ok"  # not a parse failure: "not on the page" is a valid answer
    assert out.fields["total"].value is None
    assert out.fields["total"].signals["required_missing"] is True
    assert out.fields["total"].confidence == 0.05
    assert out.fields["po_number"].signals["required_missing"] is False  # optional field


def test_a_provider_rejection_is_retried_like_an_invalid_reply() -> None:
    import httpx
    import openai

    calls = {"n": 0}

    def responder(request: ChatRequest) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            response = httpx.Response(400, request=httpx.Request("POST", "https://llm.test"))
            raise openai.BadRequestError(
                "Generated JSON does not match the schema", response=response, body=None
            )
        return reply()

    client = ScriptedClient(responder=responder)
    out = extract(SPEC, [INVOICE_PAGE], client)
    assert out.status == "ok" and [c.role for c in out.calls] == ["primary", "retry"]
    assert "provider rejected the reply" in client.requests[1].messages[-1]["content"]
    # The rejected reply had no content, so no empty assistant turn is replayed to the model.
    assert all(m["role"] != "assistant" for m in client.requests[1].messages)


def test_field_correctness() -> None:
    extracted = flatten(InvoiceV2, {**LABEL, "total": "Rs. 1,180.00", "tax": "18.00"})
    result = field_correctness(SPEC, extracted, LABEL)
    assert result["total"] is True and result["tax"] is False
    assert Decimal(LABEL["total"]) == Decimal("1180.00")
