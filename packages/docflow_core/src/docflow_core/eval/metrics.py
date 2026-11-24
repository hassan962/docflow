"""Evaluation metrics, kept per document so that versions can be compared by bootstrap.

Field F1 (after normalisation), per base field, summed over documents:

- TP: a value was extracted and it equals the label;
- FP: a value was extracted but the label differs or says "not printed";
- FN: the label has a value the extractor missed or got wrong.

A wrong value is both an FP and an FN, which is the usual convention for extraction. Both
"not printed" and "returned null" is a true negative: correct, but outside F1.

Line items are **aligned before scoring**: rows are paired by description similarity (plus
a bonus when amounts agree), so one missed row does not make every later row count as wrong.
Unpaired predicted rows are FPs for every column, unpaired label rows FNs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from docflow_core.extract.fields import (
    base_name,
    field_kinds,
    flatten,
    item_model,
    normalised,
    required_keys,
)
from docflow_core.schemas.registry import SchemaSpec

CRITICAL = ("total", "invoice_date", "vendor_name", "currency")


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, other: Counts) -> None:
        self.tp, self.fp, self.fn, self.tn = (
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )  # fmt: skip

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return 1.0 if denom == 0 else 2 * self.tp / denom


@dataclass
class DocResult:
    """Everything one document contributes to the metrics."""

    doc_id: str
    fields: dict[str, Counts]  # base field name -> counts on this document
    correct: dict[str, bool]  # flat key -> extracted value right (aligned rows)
    straight_through: bool
    rows_expected: int
    rows_exact: int
    rows_predicted: int
    confidence: dict[str, float] = field(default_factory=dict)  # flat key -> confidence
    status: dict[str, str] = field(default_factory=dict)  # flat key -> routing status
    cost_usd: float = 0.0
    latency_ms: int = 0
    parse_failed: bool = False


def _compare(kind: Any, pred: Any, truth: Any) -> Counts:
    p, t = normalised(kind, pred), normalised(kind, truth)
    if p is None and t is None:
        return Counts(tn=1)
    if p is not None and p == t:
        return Counts(tp=1)
    return Counts(fp=int(p is not None), fn=int(t is not None))


def _align(
    pred_rows: list[dict[str, Any]], true_rows: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """Greedy pairing by description similarity, with a bonus for equal amounts."""
    scored = []
    for i, p in enumerate(pred_rows):
        for j, t in enumerate(true_rows):
            a = normalised("text", p.get("description")) or ""
            b = normalised("text", t.get("description")) or ""
            sim = SequenceMatcher(None, a, b).ratio() if a and b else 0.0
            same_amount = normalised("amount", p.get("amount")) == normalised(
                "amount", t.get("amount")
            )
            score = sim + (0.5 if same_amount and p.get("amount") is not None else 0.0)
            if score >= 0.5:
                scored.append((score, i, j))
    pairs, used_p, used_t = [], set(), set()
    for _, i, j in sorted(scored, reverse=True):
        if i not in used_p and j not in used_t:
            pairs.append((i, j))
            used_p.add(i)
            used_t.add(j)
    return pairs


def score_document(
    spec: SchemaSpec,
    doc_id: str,
    extracted: dict[str, Any] | None,
    label: dict[str, Any],
    *,
    confidence: dict[str, float] | None = None,
    status: dict[str, str] | None = None,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
) -> DocResult:
    """``extracted`` is the flat dict of values, or None when the reply could not be parsed."""
    kinds = field_kinds(spec.model)
    truth = flatten(spec.model, label)
    pred = extracted or {}
    counts: dict[str, Counts] = defaultdict(Counts)
    correct: dict[str, bool] = {}

    table = next((n for n in spec.model.model_fields if item_model(spec.model, n)), None)
    for key in truth:
        if "[" not in key:
            c = _compare(kinds[key], pred.get(key), truth[key])
            counts[key].add(c)
            correct[key] = c.fp == 0 and c.fn == 0
    for key in pred:
        if "[" not in key and key not in truth:
            counts[key].add(_compare(kinds[key], pred[key], None))

    rows_exact = rows_expected = rows_predicted = 0
    if table is not None:
        row_model = item_model(spec.model, table)
        assert row_model is not None
        columns = list(row_model.model_fields)

        def rows(flat: dict[str, Any]) -> list[dict[str, Any]]:
            out: dict[int, dict[str, Any]] = {}
            for k, v in flat.items():
                if k.startswith(f"{table}["):
                    i = int(k[len(table) + 1 : k.index("]")])
                    out.setdefault(i, {})[k.rsplit(".", 1)[1]] = v
            return [out[i] for i in sorted(out)]

        p_rows, t_rows = rows(pred), rows(truth)
        rows_expected, rows_predicted = len(t_rows), len(p_rows)
        pairs = _align(p_rows, t_rows)
        for i, j in pairs:
            exact = True
            for col in columns:
                c = _compare(kinds[f"{table}.{col}"], p_rows[i].get(col), t_rows[j].get(col))
                counts[f"{table}.{col}"].add(c)
                ok = c.fp == 0 and c.fn == 0
                correct[f"{table}[{i}].{col}"] = ok
                exact &= ok
            rows_exact += exact
        paired_p = {i for i, _ in pairs}
        paired_t = {j for _, j in pairs}
        for i, row in enumerate(p_rows):
            if i not in paired_p:
                for col in columns:
                    if row.get(col) is not None:
                        counts[f"{table}.{col}"].add(Counts(fp=1))
                    correct[f"{table}[{i}].{col}"] = False
        for j, row in enumerate(t_rows):
            if j not in paired_t:
                for col in columns:
                    if row.get(col) is not None:
                        counts[f"{table}.{col}"].add(Counts(fn=1))

    required = required_keys(spec.model)
    header_ok = all(correct.get(k, False) for k in required if "." not in k and k in truth)
    straight = extracted is not None and header_ok and rows_exact == rows_expected == rows_predicted
    return DocResult(
        doc_id=doc_id, fields=dict(counts), correct=correct, straight_through=straight,
        rows_expected=rows_expected, rows_exact=rows_exact, rows_predicted=rows_predicted,
        confidence=confidence or {}, status=status or {}, cost_usd=cost_usd,
        latency_ms=latency_ms, parse_failed=extracted is None,
    )  # fmt: skip


# --- aggregates over any subset of documents (the bootstrap resamples them) ------------------


def field_counts(docs: Iterable[DocResult]) -> dict[str, Counts]:
    total: dict[str, Counts] = defaultdict(Counts)
    for d in docs:
        for name, c in d.fields.items():
            total[name].add(c)
    return dict(total)


def macro_f1(docs: Sequence[DocResult]) -> float:
    counts = field_counts(docs)
    scored = [c.f1 for c in counts.values() if c.tp + c.fp + c.fn > 0]
    return sum(scored) / len(scored) if scored else 1.0


def straight_through_rate(docs: Sequence[DocResult]) -> float:
    return sum(d.straight_through for d in docs) / max(len(docs), 1)


def routing(docs: Sequence[DocResult]) -> dict[str, float]:
    """Auto-approve rate and precision, and catch rate (wrong fields sent to review)."""
    auto = auto_right = wrong = wrong_caught = total = 0
    for d in docs:
        for key, status in d.status.items():
            ok = d.correct.get(key, True)  # a field the label lacks entirely counts as right
            total += 1
            is_auto = status == "auto_approved"
            auto += is_auto
            auto_right += is_auto and ok
            wrong += not ok
            wrong_caught += (not ok) and not is_auto
    return {
        "auto_approve_rate": auto / max(total, 1),
        "auto_approve_precision": auto_right / auto if auto else 1.0,
        "catch_rate": wrong_caught / wrong if wrong else 1.0,
        "wrong_fields": float(wrong),
        "fields": float(total),
    }


def calibration_error(docs: Sequence[DocResult], bins: int = 10) -> float:
    """Expected calibration error: does 0.9 confidence mean right 90% of the time?"""
    pairs = [(d.confidence[k], d.correct.get(k, True)) for d in docs for k in d.confidence]
    if not pairs:
        return 0.0
    total, ece = len(pairs), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        chunk = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if chunk:
            conf = sum(c for c, _ in chunk) / len(chunk)
            acc = sum(ok for _, ok in chunk) / len(chunk)
            ece += len(chunk) / total * abs(conf - acc)
    return ece


def p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def summary(docs: Sequence[DocResult]) -> dict[str, Any]:
    counts = field_counts(docs)
    return {
        "documents": len(docs),
        "macro_f1": round(macro_f1(docs), 4),
        "straight_through_rate": round(straight_through_rate(docs), 4),
        "parse_failures": sum(d.parse_failed for d in docs),
        "line_item_rows": {
            "expected": sum(d.rows_expected for d in docs),
            "exact": sum(d.rows_exact for d in docs),
            "predicted": sum(d.rows_predicted for d in docs),
        },
        "fields": {
            name: {"f1": round(c.f1, 4), "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn}
            for name, c in sorted(counts.items())
        },
        "routing": {k: round(v, 4) for k, v in routing(docs).items()},
        "calibration_error": round(calibration_error(docs), 4),
        "cost_per_page_usd": round(sum(d.cost_usd for d in docs) / max(len(docs), 1), 6),
        "p95_latency_ms": int(p95([d.latency_ms for d in docs])),
    }


def critical_f1(docs: Sequence[DocResult]) -> dict[str, float]:
    counts = field_counts(docs)
    return {name: counts[name].f1 if name in counts else 1.0 for name in CRITICAL}


__all__ = [
    "CRITICAL", "Counts", "DocResult", "base_name", "calibration_error", "critical_f1",
    "field_counts", "macro_f1", "routing", "score_document", "straight_through_rate", "summary",
]  # fmt: skip
