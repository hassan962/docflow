"""Field keys, kinds and value comparison, derived from the schema rather than hard-coded.

An extraction is stored *flat*: every reviewable value has its own key, and line items are
indexed, e.g. ``total``, ``line_items[2].amount``. Flat keys let routing, corrections and
per-field correction rates treat a table cell exactly like a header field. The base name
(``line_items.amount``) groups cells across rows for metrics.

Each field has a *kind* taken from its type annotation (Decimal -> amount, date -> date,
Literal -> code, str -> text), so grounding and comparison work for any registered schema,
not just invoices.
"""

from __future__ import annotations

import re
import types
import typing
from datetime import date
from decimal import Decimal
from functools import cache
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, create_model

from docflow_core.normalise import normalise_amount, normalise_date, normalise_text

Kind = Literal["amount", "date", "code", "text", "list"]
_INDEX = re.compile(r"\[\d+\]")


def _unwrap_optional(annotation: Any) -> Any:
    if get_origin(annotation) in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _kind(annotation: Any) -> Kind:
    ann = _unwrap_optional(annotation)
    if ann is Decimal or ann is int or ann is float:
        return "amount"
    if ann is date:
        return "date"
    if get_origin(ann) is Literal:
        return "code"
    if get_origin(ann) is list:
        return "list"
    return "text"


def item_model(model: type[BaseModel], field: str) -> type[BaseModel] | None:
    """For a list-of-models field (line_items), the row model (LineItem)."""
    ann = _unwrap_optional(model.model_fields[field].annotation)
    if get_origin(ann) is list:
        (inner,) = get_args(ann)
        if isinstance(inner, type) and issubclass(inner, BaseModel):
            return inner
    return None


@cache
def lenient_model(model: type[BaseModel]) -> type[BaseModel]:
    """The same schema with every field optional: how model replies are parsed.

    Types are still enforced (a bad date or an unknown currency is still an error), but a
    null for a required field is accepted and becomes a low-confidence field to review.
    """
    hints = typing.get_type_hints(model)
    fields: dict[str, Any] = {}
    for name in model.model_fields:
        ann = _unwrap_optional(hints[name])
        row = item_model(model, name)
        if row is not None:
            ann = list[lenient_model(row)]  # type: ignore[misc]
        fields[name] = (ann | None, None)
    lenient: type[BaseModel] = create_model(
        f"Lenient{model.__name__}", __config__=ConfigDict(extra="forbid"), **fields
    )
    return lenient


def required_keys(model: type[BaseModel]) -> set[str]:
    """Base names of required fields, including required columns of table rows."""
    out: set[str] = set()
    for name, f in model.model_fields.items():
        row = item_model(model, name)
        if row is not None:
            out |= {f"{name}.{sub}" for sub, rf in row.model_fields.items() if rf.is_required()}
        elif f.is_required():
            out.add(name)
    return out


def field_kinds(model: type[BaseModel]) -> dict[str, Kind]:
    """Kinds by base name: {'total': 'amount', 'line_items.amount': 'amount', ...}."""
    hints = typing.get_type_hints(model)
    kinds: dict[str, Kind] = {}
    for name in model.model_fields:
        kind = _kind(hints[name])
        row = item_model(model, name)
        if kind == "list" and row is not None:
            row_hints = typing.get_type_hints(row)
            for sub in row.model_fields:
                kinds[f"{name}.{sub}"] = _kind(row_hints[sub])
        else:
            kinds[name] = kind
    return kinds


def base_name(key: str) -> str:
    """'line_items[3].amount' -> 'line_items.amount'."""
    return _INDEX.sub("", key)


def flatten(model: type[BaseModel], data: dict[str, Any]) -> dict[str, Any]:
    """Nested -> flat: {'line_items': [{'amount': '10'}]} -> {'line_items[0].amount': '10'}."""
    flat: dict[str, Any] = {}
    for name in model.model_fields:
        value = data.get(name)
        row = item_model(model, name)
        if row is not None:
            for i, item in enumerate(value or []):
                for sub in row.model_fields:
                    flat[f"{name}[{i}].{sub}"] = item.get(sub) if isinstance(item, dict) else None
        else:
            flat[name] = value
    return flat


_KEY = re.compile(r"^(?P<name>\w+)\[(?P<i>\d+)\]\.(?P<sub>\w+)$")


def unflatten(model: type[BaseModel], flat: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    for key, value in flat.items():
        m = _KEY.match(key)
        if m:
            rows.setdefault(m["name"], {}).setdefault(int(m["i"]), {})[m["sub"]] = value
        else:
            out[key] = value
    for name in model.model_fields:
        if item_model(model, name) is not None:
            out[name] = [rows.get(name, {})[i] for i in sorted(rows.get(name, {}))]
    return out


def normalised(kind: Kind, value: Any) -> Any:
    """The comparable form of a value: Decimal, date, or compact lower-case text."""
    if value is None:
        return None
    if kind == "amount":
        return normalise_amount(value)
    if kind == "date":
        return normalise_date(value)
    text = normalise_text(value)
    return re.sub(r"[^0-9a-z]", "", text) if text else None


def same_value(kind: Kind, a: Any, b: Any) -> bool:
    """Equal after normalisation. Two absent values are equal; absent vs present is not."""
    equal: bool = normalised(kind, a) == normalised(kind, b)
    return equal
