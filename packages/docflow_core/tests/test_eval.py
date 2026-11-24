"""Phase 5: metrics, bootstrap, gate, datasets, caches, learned scorer, few-shot, runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from docflow_core.eval.cache import CachedClient
from docflow_core.eval.datasets import Dataset, Item, LeakageError, assert_disjoint, load
from docflow_core.eval.gate import GateConfig, decide, paired_bootstrap
from docflow_core.eval.metrics import DocResult, macro_f1, routing, score_document
from docflow_core.eval.runner import evaluate
from docflow_core.extract.confidence import (
    CalibratedRawScorer,
    IsotonicCalibrator,
    LogisticScorer,
    Signals,
    scorer_from_config,
)
from docflow_core.extract.fields import flatten
from docflow_core.extract.llm import ChatRequest, ScriptedClient
from docflow_core.extract.prompt import FewShotExample
from docflow_core.extract.versions import ActiveExtractor
from docflow_core.learn.fewshot import PoolEntry, select_examples, signature, similarity
from docflow_core.ocr import OcrResult, OcrWord
from docflow_core.schemas import registry
from docflow_core.schemas.invoice import InvoiceV2

SPEC = registry.get("invoice", 2)
LABEL: dict[str, Any] = {
    "vendor_name": "Acme Traders", "invoice_number": "INV-1", "invoice_date": "2026-03-05",
    "currency": "INR", "subtotal": "1300.00", "tax": None, "total": "1300.00",
    "vendor_tax_id": None, "po_number": None,
    "line_items": [
        {"description": "Widget", "quantity": "2", "unit_price": "500.00", "amount": "1000.00"},
        {"description": "Bolt pack", "quantity": "4", "unit_price": "25.00", "amount": "100.00"},
        {"description": "Cable", "quantity": "1", "unit_price": "200.00", "amount": "200.00"},
    ],
}  # fmt: skip


def flat(**overrides: Any) -> dict[str, Any]:
    return flatten(InvoiceV2, {**deepcopy(LABEL), **overrides})


# --- metrics --------------------------------------------------------------------------------


def test_a_perfect_extraction_is_straight_through() -> None:
    d = score_document(SPEC, "a", flat(), LABEL)
    assert d.straight_through and d.rows_exact == 3 and macro_f1([d]) == 1.0


def test_a_wrong_value_is_both_a_false_positive_and_a_false_negative() -> None:
    d = score_document(SPEC, "a", flat(total="1390.00"), LABEL)
    assert (d.fields["total"].fp, d.fields["total"].fn) == (1, 1)
    assert not d.straight_through and d.correct["total"] is False


def test_rows_are_aligned_so_one_missed_row_does_not_cascade() -> None:
    rows = [deepcopy(LABEL["line_items"][0]), deepcopy(LABEL["line_items"][2])]  # middle row missed
    d = score_document(SPEC, "a", flat(line_items=rows), LABEL)
    amount = d.fields["line_items.amount"]
    assert (amount.tp, amount.fp, amount.fn) == (2, 0, 1)  # not 1 right + 2 wrong
    assert d.rows_exact == 2 and d.rows_expected == 3 and not d.straight_through


def test_unparseable_reply_scores_every_labelled_field_as_missed() -> None:
    d = score_document(SPEC, "a", None, LABEL)
    assert d.parse_failed and d.fields["total"].fn == 1 and not d.straight_through


def test_routing_metrics() -> None:
    d = score_document(
        SPEC, "a", flat(total="1390.00"), LABEL,
        status={"total": "needs_review", "vendor_name": "auto_approved", "tax": "auto_approved"},
    )  # fmt: skip
    r = routing([d])
    assert (r["catch_rate"], r["auto_approve_precision"], r["auto_approve_rate"]) == (
        1.0,
        1.0,
        2 / 3,
    )


# --- bootstrap and gate ------------------------------------------------------------------------


def run(
    n: int, *, wrong_total: Sequence[int] = (), auto: float = 0.5, leak: Sequence[int] = ()
) -> list[DocResult]:
    """n documents; some with a wrong total; a share of fields auto-approved."""
    docs = []
    for i in range(n):
        values = flat(total="9.99") if i in wrong_total else flat()
        keys = sorted(values)
        cut = int(len(keys) * auto)
        status = {k: ("auto_approved" if j < cut else "needs_review") for j, k in enumerate(keys)}
        if i in wrong_total:
            status["total"] = "auto_approved" if i in leak else "needs_review"
        docs.append(
            score_document(
                SPEC, f"d{i}", values, LABEL, status=status, cost_usd=0.001, latency_ms=1000
            )
        )
    return docs


def test_bootstrap_of_identical_runs_is_exactly_zero() -> None:
    a = run(20, wrong_total=[1, 2])
    interval = paired_bootstrap(a, a, macro_f1, resamples=200)
    assert (interval.diff, interval.low, interval.high) == (0.0, 0.0, 0.0)


def test_gate_passes_a_real_accuracy_gain() -> None:
    active, candidate = run(30, wrong_total=range(0, 30, 2)), run(30)
    report = decide(active, candidate, GateConfig(resamples=300))
    assert report.passed and report.intervals["macro_f1"].low > 0


def test_gate_passes_a_safe_routing_gain() -> None:
    active, candidate = run(30, wrong_total=[3], auto=0.4), run(30, wrong_total=[3], auto=0.8)
    report = decide(active, candidate, GateConfig(resamples=300))
    assert report.passed and "routing gain" in report.checks[0].detail


def test_gate_rejects_a_routing_gain_that_lets_an_error_through() -> None:
    active = run(30, wrong_total=[3, 9], auto=0.4)
    candidate = run(30, wrong_total=[3, 9], auto=0.8, leak=[9])
    report = decide(active, candidate, GateConfig(resamples=300, target_precision=0.9))
    safety = next(c for c in report.checks if c.name == "safety")
    assert not report.passed and not safety.passed and "catch rate 0.500 vs 1.000" in safety.detail


def test_gate_rejects_a_critical_field_regression_and_no_gain() -> None:
    active, candidate = run(30), run(30, wrong_total=[5])
    report = decide(active, candidate, GateConfig(resamples=300))
    assert not report.passed
    assert not next(c for c in report.checks if c.name == "no critical regression").passed


# --- datasets and leakage -----------------------------------------------------------------------


def test_load_reads_inline_and_file_labels(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps(LABEL))
    rows = [
        {
            "id": "a",
            "file": "a.pdf",
            "sha256": "aa",
            "label": LABEL,
            "layout": "classic",
            "noise": "clean",
        },
        {"id": "b", "file": "b.pdf", "sha256": "bb", "label": "x.json"},
    ]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    ds = load(tmp_path / "manifest.jsonl")
    assert (
        ds.hashes == {"aa", "bb"} and ds.items[1].label == LABEL and ds.items[0].layout == "classic"
    )


def test_leakage_guard(tmp_path: Path) -> None:
    golden = tmp_path / "eval" / "golden"
    golden.mkdir(parents=True)
    (golden / "manifest.jsonl").write_text(json.dumps({"sha256": "secret"}) + "\n")
    assert_disjoint({"other"}, "the pool", root=tmp_path)
    with pytest.raises(LeakageError, match="the pool"):
        assert_disjoint({"other", "secret"}, "the pool", root=tmp_path)


def test_the_real_golden_set_is_frozen_balanced_and_disjoint_from_nothing_it_must_avoid() -> None:
    root = Path(__file__).resolve().parents[3]
    ds = load(root / "eval/golden/manifest.jsonl")
    assert len(ds.items) == 27 and len(ds.hashes) == 27
    mix: dict[tuple[str, str], int] = {}
    for i in ds.items:
        mix[(i.layout, i.noise)] = mix.get((i.layout, i.noise), 0) + 1
        InvoiceV2.model_validate(i.label)
    assert set(mix.values()) == {3} and len(mix) == 9


# --- caches -------------------------------------------------------------------------------------


def test_reply_cache_replays_and_refuses_misses_offline(tmp_path: Path) -> None:
    live = ScriptedClient(model="m", replies=["one", None])
    cached = CachedClient(live, "m", tmp_path)
    request = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    first = cached.chat(request)
    again = cached.chat(request)
    assert again == first and (cached.hits, cached.misses) == (1, 1) and len(live.requests) == 1
    other = ChatRequest(messages=[{"role": "user", "content": "other"}])
    assert cached.chat(other).content is None  # empty replies are not cached...
    with pytest.raises(LookupError):
        CachedClient(None, "m", tmp_path).chat(other)  # ...so offline, that one is a miss


# --- learned scorer -----------------------------------------------------------------------------


def test_logistic_scorer_learns_that_failed_rules_mean_wrong() -> None:
    good = [Signals("exact", 0.95, 1.0, True, True)] * 60
    bad = [Signals("exact", 0.95, 1.0, True, False)] * 20
    mixed = [Signals("exact", 0.6, 1.0, None, None)] * 20
    signals = good + bad + mixed
    correct = [True] * 60 + [False] * 18 + [True] * 2 + [True] * 15 + [False] * 5
    scorer = LogisticScorer.fit(signals, correct)
    assert scorer.explain()["rules_fail"] < 0 < scorer.explain()["rules_pass"]
    assert scorer(good[0]) > 0.9 > 0.5 > scorer(bad[0])
    config = {"scorer": {"type": "logistic", "weights": dict(zip(
        ("match_fuzzy", "match_none", "match_absent", "ocr_conf", "agree_true", "agree_false",
         "rules_pass", "rules_fail", "required_missing"), scorer.weights, strict=True)), "bias": scorer.bias}}  # fmt: skip
    assert scorer_from_config(config)(good[0]) == scorer(good[0])
    assert isinstance(scorer_from_config({}), CalibratedRawScorer)


# --- few-shot selection ------------------------------------------------------------------------


def page(*lines: str) -> OcrResult:
    words = [OcrWord(t, (10 * i, 30 * n, 10 * i + 8, 30 * n + 20), 0.9, n)
             for n, line in enumerate(lines) for i, t in enumerate(line.split())]  # fmt: skip
    return OcrResult(words, 1000, 1400, "test")


def test_layout_signature_matches_templates_not_values() -> None:
    classic_a = page(
        "TAX INVOICE",
        "Invoice No.: INV-2026-0042",
        "Date: 12/03/2026",
        "Subtotal 1,180.00",
        "GST @ 18% 212.40",
    )
    classic_b = page(
        "TAX INVOICE",
        "Invoice No.: INV-2026-0917",
        "Date: 30/11/2026",
        "Subtotal 9,005.50",
        "GST @ 18% 1,620.99",
    )
    receipt = page("RECEIPT", "Receipt: BILL/26/7", "Date: 2026-11-30", "TOTAL INR", "Sub-total")
    sa, sb, sr = signature([classic_a]), signature([classic_b]), signature([receipt])
    assert similarity(sa, sb) > similarity(sa, sr)

    pool = [
        PoolEntry(FewShotExample(n, n, "{}"), signature([p]))
        for n, p in (("b", classic_b), ("r", receipt))
    ]
    assert [e.example_id for e in select_examples(pool, [classic_a], k=1)] == ["b"]


# --- runner, end to end with a scripted model ---------------------------------------------------


class FakeOcr:
    def pages(self, path: Path, sha256: str) -> list[OcrResult]:
        return [page("Acme Traders", "Invoice No.: INV-1", "Date: 2026-03-05", "Widget 2 500.00 1,000.00",
                     "Bolt pack 4 25.00 100.00", "Cable 1 200.00 200.00", "Total Rs. 1,300.00")]  # fmt: skip


def test_evaluate_runs_extraction_rules_routing_and_scoring() -> None:
    items = [Item(id="g1", path=Path("unused"), sha256="s1", label=LABEL)]
    version = ActiveExtractor(
        id="v", doc_type="invoice", schema_version=2, model="m", agreement_model=None,
        prompt_hash="h", scorer=CalibratedRawScorer(IsotonicCalibrator.identity()),
        review_threshold=0.5, fewshot_k=0,
    )  # fmt: skip
    reply = json.dumps(LABEL)
    docs = evaluate(version, Dataset("t", items), client_for=lambda m: ScriptedClient(replies=[reply]),
                    ocr=FakeOcr())  # type: ignore[arg-type]  # fmt: skip
    [d] = docs
    assert d.straight_through and d.status["total"] == "auto_approved" and d.cost_usd > 0
