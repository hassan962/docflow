"""One real call to the configured provider. Opt-in, so test runs never spend quota:

RUN_LLM_TESTS=1 uv run pytest -m llm
"""

from __future__ import annotations

import os

import pytest
from dotenv import dotenv_values

from docflow_core.config import Settings
from docflow_core.extract.extractor import extract
from docflow_core.extract.llm import OpenAICompatibleClient
from docflow_core.ocr import OcrResult, OcrWord
from docflow_core.schemas import registry

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(os.environ.get("RUN_LLM_TESTS") != "1", reason="set RUN_LLM_TESTS=1"),
]

LINES = [
    "Acme Traders Pvt Ltd",
    "TAX INVOICE",
    "Invoice No.: INV-2026-0042   Date: 12/03/2026",
    "Description Qty Rate Amount (INR)",
    "Laser toner cartridge 2 1,953.00 3,906.00",
    "Subtotal 3,906.00",
    "GST @ 18% 703.08",
    "Total Rs. 4,609.08",
]


def test_live_extraction_of_a_simple_invoice() -> None:
    env = dotenv_values(".env")
    settings = Settings(llm_api_key=env.get("LLM_API_KEY") or os.environ.get("LLM_API_KEY"))
    if settings.llm_api_key is None:
        pytest.skip("LLM_API_KEY is not configured")
    words = [
        OcrWord(t, (12 * i, 30 * n, 12 * i + 10, 30 * n + 20), 0.95, n)
        for n, line in enumerate(LINES)
        for i, t in enumerate(line.split())
    ]
    page = OcrResult(words, 1200, 1600, "test")
    out = extract(
        registry.get("invoice", 2), [page], OpenAICompatibleClient(settings, settings.llm_model)
    )

    assert out.status == "ok", out.error
    values = {k: f.value for k, f in out.fields.items()}
    assert values["total"] == "4609.08" and values["tax"] == "703.08"
    assert values["invoice_date"] == "2026-03-12"  # read day-first
    assert values["currency"] == "INR" and values["invoice_number"] == "INV-2026-0042"
    assert values["line_items[0].amount"] == "3906.00"
    assert out.fields["total"].signals["match"] == "exact"


CASES = [
    # (invoice values, expected set of (field, severity) findings)
    ({"vendor_name": "Acme Traders Pvt Ltd", "po_number": "PO-1", "total": "1180.00", "currency": "INR"}, set()),
    ({"vendor_name": "Acme Tradrs Pvt Ltd", "po_number": None, "total": "1180.00", "currency": "INR"}, {("vendor_name", "warning")}),
    ({"vendor_name": "Initech Solutions", "po_number": None, "total": "99.00", "currency": "INR"}, {("vendor_name", "error")}),
    ({"vendor_name": "Acme Traders Pvt Ltd", "po_number": "PO-1", "total": "1200.00", "currency": "INR"}, {("total", "error")}),
    ({"vendor_name": "Acme Traders Pvt Ltd", "po_number": "PO-2", "total": "500.00", "currency": "INR"}, {("vendor_name", "error"), ("po_number", "error")}),
]  # fmt: skip


@pytest.mark.db
@pytest.mark.usefixtures("migrated_db")
@pytest.mark.parametrize(
    ("values", "expected"), CASES, ids=["match", "spelling", "unknown", "total", "other-vendor"]
)
def test_live_reference_checks_reach_the_expected_findings(values: dict, expected: set) -> None:
    from test_reference import seed_reference

    from conftest import new_tenant
    from docflow_core.db.session import tenant_session
    from docflow_core.validate.reference import execute_tool, run_reference_checks

    env = dotenv_values(".env")
    settings = Settings(llm_api_key=env.get("LLM_API_KEY") or os.environ.get("LLM_API_KEY"))
    if settings.llm_api_key is None or not settings.llm_tools_model:
        pytest.skip("LLM_API_KEY is not configured")
    tenant_id, _, _ = new_tenant()
    seed_reference(tenant_id)

    def run_tool(name: str, arguments: str) -> dict:
        with tenant_session(tenant_id) as s:
            return execute_tool(s, name, arguments)

    client = OpenAICompatibleClient(settings, settings.llm_tools_model)
    outcome = run_reference_checks(client, values, run_tool)
    assert outcome.error is None
    found = {(f.field, f.severity) for f in outcome.findings if f.severity != "info"}
    assert found == expected, [f.model_dump() for f in outcome.findings]
