"""Turn a Pydantic document schema into the JSON schema the LLM must answer with.

Strict structured output (the provider guarantees the reply matches the schema) only accepts
a subset of JSON Schema: every property listed in ``required``, ``additionalProperties:
false``, and optional values written as a ``null`` alternative. Pydantic's own schema uses
``$defs`` references, ``anyOf`` and formats that some providers reject, so this builds the
strict form directly from the field annotations.

Amounts are strings ("1180.00"), not JSON numbers: a float cannot hold 0.1 exactly, and the
reply is parsed straight into ``Decimal``.

Every field may be null, required ones included. The model must always be able to say "this
is not on the page": forcing a value would force a guess. "Required" is a business rule,
enforced downstream as near-zero confidence (and in phase 4 a failed rule), so a person sees
the gap instead of a plausible invention.
"""

from __future__ import annotations

import typing
from datetime import date
from decimal import Decimal
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel

from docflow_core.extract.fields import _unwrap_optional


def _type_schema(annotation: Any) -> dict[str, Any]:
    ann = _unwrap_optional(annotation)
    if ann is Decimal:
        schema: dict[str, Any] = {
            "type": "string",
            "description": "Plain decimal number, no currency symbol or thousands separator.",
        }
    elif ann is date:
        schema = {"type": "string", "description": "Date as YYYY-MM-DD."}
    elif ann is int:
        schema = {"type": "integer"}
    elif get_origin(ann) is Literal:
        schema = {"type": "string", "enum": list(get_args(ann))}
    elif get_origin(ann) is list:
        (inner,) = get_args(ann)
        schema = {"type": "array", "items": object_schema(inner)}
    elif isinstance(ann, type) and issubclass(ann, BaseModel):
        schema = object_schema(ann)
    else:
        schema = {"type": "string"}
    schema = {**schema, "type": [schema["type"], "null"]}
    if "enum" in schema:
        schema["enum"] = [*schema["enum"], None]
    return schema


def object_schema(model: type[BaseModel]) -> dict[str, Any]:
    hints = typing.get_type_hints(model)
    properties: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        prop = _type_schema(hints[name])
        if field.description:
            prop["description"] = (prop.get("description", "") + " " + field.description).strip()
        properties[name] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
