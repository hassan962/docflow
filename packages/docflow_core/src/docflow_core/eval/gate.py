"""The promotion gate: a candidate replaces the active extractor only on a measured win.

Both versions are evaluated on the same frozen golden set, and every difference is judged by
a **paired bootstrap over documents**: resample the documents with replacement (the same
resample for both versions), recompute the metric for each, and take the 95% interval of the
difference. If the interval sits above zero, the gain is unlikely to be luck.

All conditions must hold:

- Accuracy: macro field F1 gains (interval above 0); **or**, for a change that only touches
  routing, the auto-approve rate gains (interval above 0) while macro F1 does not fall
  (interval above -tolerance).
- Safety: auto-approved fields stay at or above the target precision, and the catch rate
  does not fall.
- No critical regression: total, invoice_date, vendor_name and currency each score no lower
  than the active version.
- Cost: cost per page within the budget.
- Latency: p95 latency no worse than the active version plus a margin.

The "or" in the first row extends the design doc's gate. A calibration or scorer change leaves
every extracted value as it was, so it can never win on F1. It is judged on what it changes,
the share of fields that skip review, with the safety row making sure that is not bought with
wrong approvals.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from docflow_core.eval.metrics import DocResult, critical_f1, macro_f1, p95, routing

Metric = Callable[[Sequence[DocResult]], float]


@dataclass(frozen=True)
class GateConfig:
    target_precision: float = 0.995
    f1_tolerance: float = 0.005  # how much F1 may move for a routing-only change
    cost_budget_per_page: float = 0.01
    latency_margin_ms: int = 3000
    resamples: int = 1000
    seed: int = 7


@dataclass
class Interval:
    diff: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.diff:+.4f} [{self.low:+.4f}, {self.high:+.4f}]"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class GateReport:
    passed: bool
    checks: list[Check] = field(default_factory=list)
    intervals: dict[str, Interval] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.__dict__ for c in self.checks],
            "intervals": {k: v.__dict__ for k, v in self.intervals.items()},
        }


def paired_bootstrap(
    active: Sequence[DocResult],
    candidate: Sequence[DocResult],
    metric: Metric,
    *,
    resamples: int = 1000,
    seed: int = 7,
) -> Interval:
    """95% interval of metric(candidate) - metric(active), resampling documents in pairs."""
    by_id = {d.doc_id: d for d in active}
    pairs = [(by_id[c.doc_id], c) for c in candidate if c.doc_id in by_id]
    if not pairs:
        raise ValueError("no documents in common")
    rng = random.Random(seed)
    diffs = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        diffs.append(metric([c for _, c in sample]) - metric([a for a, _ in sample]))
    diffs.sort()
    point = metric([c for _, c in pairs]) - metric([a for a, _ in pairs])
    return Interval(point, diffs[int(0.025 * resamples)], diffs[int(0.975 * resamples) - 1])


def _auto_rate(docs: Sequence[DocResult]) -> float:
    return routing(docs)["auto_approve_rate"]


def decide(
    active: Sequence[DocResult],
    candidate: Sequence[DocResult],
    cfg: GateConfig = GateConfig(),
) -> GateReport:
    boot = {"resamples": cfg.resamples, "seed": cfg.seed}
    f1 = paired_bootstrap(active, candidate, macro_f1, **boot)
    auto = paired_bootstrap(active, candidate, _auto_rate, **boot)
    report = GateReport(passed=False, intervals={"macro_f1": f1, "auto_approve_rate": auto})

    f1_gain = f1.low > 0
    routing_gain = auto.low > 0 and f1.low > -cfg.f1_tolerance
    report.checks.append(
        Check(
            "accuracy",
            f1_gain or routing_gain,
            f"macro F1 {f1}; auto-approve rate {auto}"
            + (" -> F1 gain" if f1_gain else " -> routing gain" if routing_gain else ""),
        )
    )

    r_a, r_c = routing(active), routing(candidate)
    safe = (
        r_c["auto_approve_precision"] >= cfg.target_precision
        and r_c["catch_rate"] >= r_a["catch_rate"]
    )
    report.checks.append(
        Check(
            "safety",
            safe,
            f"auto-approve precision {r_c['auto_approve_precision']:.4f}"
            f" (target {cfg.target_precision}),"
            f" catch rate {r_c['catch_rate']:.3f} vs {r_a['catch_rate']:.3f}",
        )
    )

    crit_a, crit_c = critical_f1(active), critical_f1(candidate)
    worse = {k: (crit_a[k], crit_c[k]) for k in crit_a if crit_c[k] < crit_a[k] - 1e-9}
    report.checks.append(
        Check(
            "no critical regression",
            not worse,
            "all critical fields hold" if not worse else
            ", ".join(f"{k} {a:.3f}->{c:.3f}" for k, (a, c) in worse.items()),
        )
    )  # fmt: skip

    cost = sum(d.cost_usd for d in candidate) / max(len(candidate), 1)
    report.checks.append(
        Check(
            "cost",
            cost <= cfg.cost_budget_per_page,
            f"${cost:.5f}/page (budget ${cfg.cost_budget_per_page})",
        )
    )

    lat_a = p95([d.latency_ms for d in active])
    lat_c = p95([d.latency_ms for d in candidate])
    report.checks.append(
        Check(
            "latency",
            lat_c <= lat_a + cfg.latency_margin_ms,
            f"p95 {lat_c:.0f} ms vs {lat_a:.0f} ms (+{cfg.latency_margin_ms} ms allowed)",
        )
    )
    report.passed = all(c.passed for c in report.checks)
    return report
