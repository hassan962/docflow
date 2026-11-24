"""Field confidence: combine signals into a raw score, then calibrate it.

A model's own "I am 95% sure" is not reliable, so confidence is built from evidence:

| Signal          | Source                                                        |
|-----------------|---------------------------------------------------------------|
| grounding       | was the value found on the page: exact, fuzzy, or not at all  |
| OCR confidence  | mean confidence of the matched OCR words                      |
| agreement       | did an independent second model extract the same value        |
| validators      | did the rules covering the field pass (added in phase 4)      |

``raw_score`` multiplies these into a number that *ranks* fields from risky to safe. A rank
is not a probability, so an isotonic regression fitted on a labelled dev set maps it to one:
after calibration, fields scored 0.9 are right about 90% of the time. The review threshold
is then read off the calibration curve so that auto-approved fields meet a target precision.

Isotonic regression is implemented here with the pool-adjacent-violators algorithm (about 30
lines), which avoids pulling in scikit-learn for one function.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from docflow_core.extract.grounding import Match


@dataclass(frozen=True)
class Signals:
    match: Match
    ocr_conf: float | None = None
    similarity: float = 0.0
    agreement: bool | None = None  # None: no second opinion was available
    validators_passed: bool | None = None  # None: no rule covers this field (yet)
    required_missing: bool = False  # the schema requires it, the model found nothing

    def as_dict(self) -> dict[str, object]:
        return {
            "match": self.match,
            "ocr_conf": None if self.ocr_conf is None else round(self.ocr_conf, 3),
            "similarity": self.similarity,
            "agreement": self.agreement,
            "validators_passed": self.validators_passed,
            "required_missing": self.required_missing,
        }


def raw_score(s: Signals) -> float:
    if s.required_missing:
        return 0.05  # a required value is missing: always worth a human look
    if s.match == "absent":
        base = 0.8  # "not printed": only the agreement signal can support or doubt it
    elif s.match == "exact":
        base = 0.6 + 0.4 * (s.ocr_conf if s.ocr_conf is not None else 0.5)
    elif s.match == "fuzzy":
        base = 0.5 * s.similarity * (s.ocr_conf if s.ocr_conf is not None else 0.5)
    else:  # value not found on the page: likely misread or hallucinated
        base = 0.15
    if s.agreement is False:
        base *= 0.4
    elif s.agreement is True:
        base = min(1.0, base * 1.05)
    if s.validators_passed is False:
        base *= 0.5
    return round(base, 4)


@dataclass
class IsotonicCalibrator:
    """Monotone step function from raw score to probability of being correct."""

    thresholds: list[float]  # lower edge of each step, ascending
    values: list[float]  # calibrated probability for scores in that step

    @classmethod
    def identity(cls) -> IsotonicCalibrator:
        return cls(thresholds=[], values=[])

    @classmethod
    def fit(cls, scores: Sequence[float], correct: Sequence[bool]) -> IsotonicCalibrator:
        if len(scores) != len(correct) or not scores:
            raise ValueError("need equally many scores and labels, and at least one")
        pairs = sorted(zip(scores, correct, strict=True))
        # Each block: [min score, sum of labels, count]. Merge while a block's mean is not
        # above the previous block's mean (a "violator"), so means end up non-decreasing.
        blocks: list[list[float]] = []
        for score, ok in pairs:
            blocks.append([score, float(ok), 1.0])
            while (
                len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] >= blocks[-1][1] / blocks[-1][2]
            ):
                _, s, n = blocks.pop()
                blocks[-1][1] += s
                blocks[-1][2] += n
        # Laplace smoothing: a block of 3 correct fields is evidence, not certainty.
        values = [(s + 1) / (n + 2) for _, s, n in blocks]
        for i in range(1, len(values)):  # smoothing can reorder slightly; keep it monotone
            values[i] = max(values[i], values[i - 1])
        return cls(thresholds=[b[0] for b in blocks], values=[round(v, 4) for v in values])

    def __call__(self, score: float) -> float:
        if not self.thresholds:
            return score
        i = bisect.bisect_right(self.thresholds, score) - 1
        return self.values[max(i, 0)]

    def to_json(self) -> str:
        return json.dumps({"thresholds": self.thresholds, "values": self.values})

    @classmethod
    def from_json(cls, data: str | dict[str, list[float]] | None) -> IsotonicCalibrator:
        if not data:
            return cls.identity()
        raw = json.loads(data) if isinstance(data, str) else data
        return cls(thresholds=list(raw["thresholds"]), values=list(raw["values"]))


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    precision: float  # of the fields at or above the threshold, on the dev set
    auto_approve_rate: float  # share of fields at or above it


def choose_threshold(
    confidences: Sequence[float], correct: Sequence[bool], target_precision: float
) -> ThresholdChoice:
    """Lowest threshold whose auto-approved fields still reach ``target_precision``.

    Lower thresholds approve more fields, so this maximises the share of fields that skip
    review while holding precision at the target.
    """
    pairs = sorted(zip(confidences, correct, strict=True), key=lambda p: -p[0])
    best = ThresholdChoice(threshold=1.01, precision=1.0, auto_approve_rate=0.0)
    hits = 0
    for i, (conf, ok) in enumerate(pairs, start=1):
        hits += ok
        next_conf = pairs[i][0] if i < len(pairs) else None
        if next_conf == conf:
            continue  # a threshold cannot split equal confidences
        if hits / i >= target_precision:
            best = ThresholdChoice(conf, round(hits / i, 4), round(i / len(pairs), 4))
    return best


# --- scorers: Signals -> probability of being right ---------------------------------------------


class Scorer(Protocol):
    def __call__(self, s: Signals) -> float: ...


@dataclass(frozen=True)
class CalibratedRawScorer:
    """The phase 3 method: hand-made raw score, then isotonic calibration."""

    calibrator: IsotonicCalibrator

    def __call__(self, s: Signals) -> float:
        return self.calibrator(raw_score(s))


FEATURES = (
    "match_fuzzy", "match_none", "match_absent", "ocr_conf",
    "agree_true", "agree_false", "rules_pass", "rules_fail", "required_missing",
)  # fmt: skip


def features(s: Signals) -> list[float]:
    """One row of the learned scorer's design matrix. Baselines: exact match, no second
    opinion, no rule covering the field (each absent column means "the baseline")."""
    return [
        s.similarity if s.match == "fuzzy" else 0.0,
        float(s.match == "none"),
        float(s.match == "absent"),
        s.ocr_conf if s.ocr_conf is not None else 0.0,
        float(s.agreement is True),
        float(s.agreement is False),
        float(s.validators_passed is True),
        float(s.validators_passed is False),
        float(s.required_missing),
    ]


@dataclass(frozen=True)
class LogisticScorer:
    """P(field is right) = sigmoid(bias + weights . features(signals)).

    Learned from labelled fields instead of hand-tuned, so the data decides how much an
    agreed absence or a low OCR confidence should count. Weights are readable: a positive
    weight raises confidence, and its size says by how much (in log-odds).
    """

    weights: tuple[float, ...]
    bias: float

    def __call__(self, s: Signals) -> float:
        z = self.bias + sum(w * x for w, x in zip(self.weights, features(s), strict=True))
        return round(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))), 4)

    @classmethod
    def fit(
        cls, signals: Sequence[Signals], correct: Sequence[bool], l2: float = 1.0, steps: int = 50
    ) -> LogisticScorer:
        """Newton's method on L2-regularised log loss (the bias is not penalised)."""
        import numpy as np

        x = np.array([[1.0, *features(s)] for s in signals])
        y = np.array([float(c) for c in correct])
        w = np.zeros(x.shape[1])
        penalty = np.eye(x.shape[1]) * l2
        penalty[0, 0] = 0.0
        for _ in range(steps):
            p = 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))
            grad = x.T @ (p - y) + penalty @ w
            hess = (x * (p * (1 - p))[:, None]).T @ x + penalty
            step = np.linalg.solve(hess, grad)
            w -= step
            if float(np.abs(step).max()) < 1e-8:
                break
        return cls(weights=tuple(round(float(v), 4) for v in w[1:]), bias=round(float(w[0]), 4))

    def explain(self) -> dict[str, float]:
        return {"bias": self.bias, **dict(zip(FEATURES, self.weights, strict=True))}


def scorer_from_config(config: dict[str, Any] | None) -> Scorer:
    """An extractor version's scorer: a learned one if configured, else calibrated raw."""
    cfg = config or {}
    learned = cfg.get("scorer")
    if learned and learned.get("type") == "logistic":
        weights = tuple(float(learned["weights"][name]) for name in FEATURES)
        return LogisticScorer(weights=weights, bias=float(learned["bias"]))
    return CalibratedRawScorer(IsotonicCalibrator.from_json(cfg.get("calibration")))
