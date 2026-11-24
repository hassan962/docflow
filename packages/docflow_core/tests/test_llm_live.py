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
