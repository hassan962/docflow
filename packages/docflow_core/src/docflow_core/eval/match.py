"""Is each extracted field right? Compared after normalisation, field by field.

The same normalisation the validators use decides equality, so "Rs. 1,180.00" and "1180.00"
match and "12/03/2026" equals "2026-03-12". A field the label says is absent is right only
if the extractor also returned null.

Line items are compared row by row in order here. Phase 5 adds row alignment (pairing rows
before scoring), so one missed row does not make every later row count as wrong.
"""

from __future__ import annotations

from typing import Any

from docflow_core.extract.fields import base_name, field_kinds, flatten, same_value
from docflow_core.schemas.registry import SchemaSpec


def field_correctness(
    spec: SchemaSpec, extracted_flat: dict[str, Any], label: dict[str, Any]
) -> dict[str, bool]:
    """For every key in the extraction or the label: True if the values agree."""
    kinds = field_kinds(spec.model)
    truth = flatten(spec.model, label)
    keys = sorted(set(extracted_flat) | set(truth))
    return {
        key: same_value(kinds[base_name(key)], extracted_flat.get(key), truth.get(key))
        for key in keys
    }
