"""Fold rule results into the fields, re-score them and decide what a person must look at.

Routing policy: a field is ``auto_approved`` when its calibrated confidence is at or above the
review threshold and none of its rules failed; otherwise it ``needs_review``. A document is
approved when no field needs review. Fields a reviewer already handled keep ``reviewed``.

Threshold precedence: the ``REVIEW_THRESHOLD_<DOC_TYPE>`` environment variable when set (an
operator override that needs no deploy), else the threshold calibrated with the extractor
version, else a conservative default.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from docflow_core.config import review_threshold
from docflow_core.extract.confidence import IsotonicCalibrator, Signals, raw_score
from docflow_core.validate.rules import RuleResult

DOCUMENT_LEVEL = "_document"


@dataclass(frozen=True)
class Routed:
    fields: dict[str, dict[str, Any]]
    issues: list[dict[str, Any]]  # rule failures that belong to no single field
    needs_review: int
    auto_approved: int
    reviewed: int

    @property
    def approved(self) -> bool:
        return self.needs_review == 0 and not self.issues


def threshold_for(doc_type: str, calibrated: float | None) -> float:
    return review_threshold(doc_type, default=calibrated)


def _signals(stored: dict[str, Any], validators_passed: bool | None) -> Signals:
    return Signals(
        match=stored.get("match", "none"),
        ocr_conf=stored.get("ocr_conf"),
        similarity=stored.get("similarity", 0.0),
        agreement=stored.get("agreement"),
        validators_passed=validators_passed,
        required_missing=stored.get("required_missing", False),
    )


def stored_reference_results(fields: dict[str, dict[str, Any]]) -> list[RuleResult]:
    """Reference-check results already on the fields, so re-validating after a correction
    keeps them without calling the model again."""
    out = []
    for key, f in fields.items():
        for c in f.get("checks", []):
            if c["rule"] == "reference_checks":
                out.append(
                    RuleResult(
                        "reference_checks",
                        c["passed"],
                        (key,),
                        c.get("message"),
                        # checks store the suggested value itself; results map field -> value
                        {key: c["suggestion"]} if c.get("suggestion") is not None else {},
                    )
                )
    return out


def route(
    fields: dict[str, dict[str, Any]],
    results: Iterable[RuleResult],
    calibrator: IsotonicCalibrator,
    threshold: float,
) -> Routed:
    by_field: dict[str, list[dict[str, Any]]] = {k: [] for k in fields}
    issues: list[dict[str, Any]] = []
    for r in results:
        check = {"rule": r.rule, "passed": r.passed, "message": r.message}
        for key in r.fields:
            if key in by_field:
                by_field[key].append({**check, "suggestion": r.suggestion.get(key)})
            elif not r.passed:
                issues.append({**check, "field": key})

    out: dict[str, dict[str, Any]] = {}
    counts = {"needs_review": 0, "auto_approved": 0, "reviewed": 0}
    for key, f in fields.items():
        checks = by_field[key]
        passed = None if not checks else all(c["passed"] for c in checks)
        signals = {k: v for k, v in f.get("signals", {}).items() if k != "raw_score"}
        updated = {**f, "checks": checks}
        if f.get("source") == "review":
            status = "reviewed"  # a person decided; rules are shown, not re-applied
        else:
            raw = raw_score(_signals(signals, passed))
            updated["confidence"] = calibrator(raw)
            updated["signals"] = {**signals, "validators_passed": passed, "raw_score": raw}
            ok = updated["confidence"] >= threshold and passed is not False
            status = "auto_approved" if ok else "needs_review"
        updated["status"] = status
        counts[status] += 1
        out[key] = updated
    return Routed(out, issues, **counts)
