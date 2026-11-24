"""The shape of one extracted field.

Every entry in ``extractions.fields`` is one of these objects, so the review UI, the
validators and the evals all read exactly the same structure:

    {"total": {"value": "1180.00", "raw": "Rs. 1,180.00", "confidence": 0.93, "page": 1,
               "bbox": [412, 880, 530, 902],
               "checks": [{"rule": "line_items_sum", "passed": true}], "source": "llm"}}
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

BBox = tuple[int, int, int, int]  # x0, y0, x1, y1 in page-image pixels


class RuleCheck(BaseModel):
    rule: str
    passed: bool
    message: str | None = None
    suggestion: Any = None  # what the rule thinks the value should be, for one-key accept


class ExtractedField(BaseModel):
    value: Any = Field(description="Normalised value, JSON-serialisable (amounts as strings).")
    raw: str | None = Field(default=None, description="Text as it appears on the page.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    page: int | None = Field(default=None, ge=1)
    bbox: BBox | None = None
    checks: list[RuleCheck] = Field(default_factory=list)
    source: Literal["llm", "review", "rule"] = "llm"
    status: Literal["auto_approved", "needs_review", "reviewed"] | None = None
    # The evidence behind ``confidence`` (grounding match, OCR confidence, agreement, raw
    # score). Shown to reviewers as "why is this flagged", and lets calibration be refitted
    # from stored extractions without calling the model again.
    signals: dict[str, Any] = Field(default_factory=dict)

    @field_validator("bbox")
    @classmethod
    def _bbox_is_a_box(cls, v: BBox | None) -> BBox | None:
        if v is not None and (v[2] < v[0] or v[3] < v[1]):
            raise ValueError("bbox must be (x0, y0, x1, y1) with x1 >= x0 and y1 >= y0")
        return v

    @property
    def failed_rules(self) -> list[str]:
        return [c.rule for c in self.checks if not c.passed]
