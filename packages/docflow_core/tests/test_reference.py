"""Reference checks: the two tools against a real Postgres, and the tool-calling loop.

The five design-doc cases are set up as reference data. The tools are checked directly
(deterministic), the loop is driven by a scripted model, and ``test_llm_live.py`` runs the
same five cases with the real model.
"""

from __future__ import annotations

import json
import uuid
from decimal import Decimal
from typing import Any

import pytest

from conftest import new_tenant
from docflow_core.db.models import PurchaseOrder, Vendor
from docflow_core.db.session import owner_session, tenant_session
from docflow_core.extract.llm import ScriptedClient, ToolCall
from docflow_core.normalise import normalise_name
from docflow_core.validate.reference import (
    execute_tool,
    findings_to_results,
    run_reference_checks,
)

pytestmark = [pytest.mark.db, pytest.mark.usefixtures("migrated_db")]


def seed_reference(tenant_id: uuid.UUID) -> None:
    def vendor(name: str, tax_id: str | None) -> Vendor:
        normalised = normalise_name(name) or ""
        return Vendor(tenant_id=tenant_id, name=name, normalised_name=normalised, tax_id=tax_id)

    def po(number: str, vendor_id: uuid.UUID, total: str) -> PurchaseOrder:
        return PurchaseOrder(
            tenant_id=tenant_id, po_number=number, vendor_id=vendor_id, total=Decimal(total),
            currency="INR",
        )  # fmt: skip

    with owner_session() as s:
        acme = vendor("Acme Traders Pvt Ltd", "27AAPFU0939F1ZV")
        globex = vendor("Globex Supplies", None)
        s.add_all([acme, globex])
        s.flush()
        s.add_all([po("PO-1", acme.id, "1180.00"), po("PO-2", globex.id, "500.00")])


@pytest.fixture
def tenant() -> uuid.UUID:
    tenant_id, _, _ = new_tenant()
    seed_reference(tenant_id)
    return tenant_id


def tool(tenant_id: uuid.UUID, tool_name: str, /, **args: Any) -> dict[str, Any]:
    with tenant_session(tenant_id) as s:
        return execute_tool(s, tool_name, json.dumps(args))


# --- the five cases, tool by tool ----------------------------------------------------------


def test_case1_vendor_and_po_match(tenant: uuid.UUID) -> None:
    [best, *_] = tool(tenant, "lookup_vendor", name="Acme Traders Pvt. Ltd.")["matches"]
    assert best["name"] == "Acme Traders Pvt Ltd" and best["similarity"] == 1.0
    po = tool(tenant, "check_purchase_order", po_number="po-1", total=1180, currency="INR",
              vendor_name="Acme Traders Pvt Ltd")  # fmt: skip
    assert po["found"] and po["total_matches"] and po["vendor_matches"] and po["currency_matches"]


def test_case2_close_spelling_is_a_close_match(tenant: uuid.UUID) -> None:
    [best, *_] = tool(tenant, "lookup_vendor", name="Acme Tradrs")["matches"]
    assert best["name"] == "Acme Traders Pvt Ltd" and 0.3 < best["similarity"] < 1.0


def test_case3_unknown_vendor_has_no_match(tenant: uuid.UUID) -> None:
    assert tool(tenant, "lookup_vendor", name="Initech Solutions")["matches"] == []


def test_case4_po_total_differs(tenant: uuid.UUID) -> None:
    po = tool(tenant, "check_purchase_order", po_number="PO-1", total=1200)
    assert po["total_matches"] is False and po["total_difference"] == "20.00"


def test_case5_po_belongs_to_another_vendor(tenant: uuid.UUID) -> None:
    po = tool(tenant, "check_purchase_order", po_number="PO-2", vendor_name="Acme Traders Pvt Ltd")
    assert po["vendor_on_po"] == "Globex Supplies" and po["vendor_matches"] is False


def test_tax_id_finds_the_vendor_even_with_a_different_name(tenant: uuid.UUID) -> None:
    [best, *_] = tool(tenant, "lookup_vendor", name="Totally Different", tax_id="27aapfu0939f1zv")[
        "matches"
    ]
    assert best["tax_id_matches"] is True


def test_bad_arguments_are_reported_to_the_model_not_executed(tenant: uuid.UUID) -> None:
    assert "invalid arguments" in tool(tenant, "lookup_vendor")["error"]
    assert (
        "invalid arguments"
        in tool(tenant, "check_purchase_order", po_number="PO-1", tenant_id="x")["error"]
    )
    assert "unknown tool" in tool(tenant, "drop_table")["error"]


def test_tools_cannot_see_another_tenants_records(tenant: uuid.UUID) -> None:
    other, _, _ = new_tenant()
    assert tool(other, "lookup_vendor", name="Acme Traders Pvt Ltd")["matches"] == []
    assert tool(other, "check_purchase_order", po_number="PO-1")["found"] is False


# --- the control loop with a scripted model -------------------------------------------------


def test_loop_executes_tool_calls_and_parses_findings(tenant: uuid.UUID) -> None:
    final = {
        "findings": [
            {"field": "total", "severity": "error", "message": "PO total differs by 20.00"}
        ]
    }
    model = ScriptedClient(replies=[
        [ToolCall("c1", "lookup_vendor", '{"name": "Acme Traders Pvt Ltd"}'),
         ToolCall("c2", "check_purchase_order", '{"po_number": "PO-1", "total": 1200}')],
        "```json\n" + json.dumps(final) + "\n```",
    ])  # fmt: skip

    def run_tool(name: str, arguments: str) -> dict[str, Any]:
        with tenant_session(tenant) as s:
            return execute_tool(s, name, arguments)

    values = {"vendor_name": "Acme Traders Pvt Ltd", "po_number": "PO-1", "total": "1200.00"}
    outcome = run_reference_checks(model, values, run_tool)
    assert outcome.error is None and [f.field for f in outcome.findings] == ["total"]
    tool_messages = [m for m in model.requests[1].messages if m["role"] == "tool"]
    assert json.loads(tool_messages[1]["content"])["total_difference"] == "20.00"
    assert model.requests[1].messages[2]["tool_calls"][0]["function"]["name"] == "lookup_vendor"

    results = findings_to_results(outcome, values)
    assert [(r.passed, r.fields) for r in results] == [
        (False, ("total",)),
        (True, ("vendor_name", "po_number")),
    ]


def test_loop_gives_up_after_three_rounds() -> None:
    looping = ScriptedClient(responder=lambda r: [ToolCall("c", "lookup_vendor", '{"name": "x"}')])
    outcome = run_reference_checks(looping, {"vendor_name": "x"}, lambda n, a: {"matches": []})
    assert outcome.error == "no final answer after 3 rounds" and len(outcome.calls) == 3
    # Not verified must not mean approved: the failure becomes a document-level issue.
    [issue] = findings_to_results(outcome, {"vendor_name": "x"})
    assert issue.fields == ("_reference",) and not issue.passed
    assert issue.message and issue.message.startswith("reference checks unavailable")
