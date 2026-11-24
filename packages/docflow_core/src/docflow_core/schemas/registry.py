"""Schema registry.

A document type is added by registering a schema here (and, in later phases, its validators
and prompt template). Pipeline code looks types up by (doc_type, version) and never imports a
specific schema, which is what lets a second document family plug in without code changes.

Published versions are immutable: changing a schema means registering a new version.
Old extractions stay valid against the version that produced them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class SchemaSpec:
    doc_type: str
    version: int
    model: type[BaseModel]
    # Fields the reviewer and the gate care about most; a regression on any of them blocks
    # promotion even if the average improves.
    critical_fields: tuple[str, ...] = ()
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.doc_type}/v{self.version}"

    @property
    def field_names(self) -> list[str]:
        return list(self.model.model_fields)

    @property
    def required_fields(self) -> list[str]:
        return [name for name, f in self.model.model_fields.items() if f.is_required()]

    def json_schema(self) -> dict[str, Any]:
        return self.model.model_json_schema()

    def fingerprint(self) -> str:
        """Stable hash of the JSON schema, stored with the published version."""
        canonical = json.dumps(self.json_schema(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class SchemaRegistry:
    def __init__(self) -> None:
        self._specs: dict[tuple[str, int], SchemaSpec] = {}

    def register(self, spec: SchemaSpec) -> SchemaSpec:
        key = (spec.doc_type, spec.version)
        existing = self._specs.get(key)
        if existing is not None and existing.fingerprint() != spec.fingerprint():
            raise ValueError(
                f"{spec.key} is already registered with a different schema; "
                "published versions are immutable, register a new version instead"
            )
        self._specs[key] = spec
        return spec

    def get(self, doc_type: str, version: int) -> SchemaSpec:
        try:
            return self._specs[(doc_type, version)]
        except KeyError:
            raise KeyError(f"unknown schema {doc_type}/v{version}") from None

    def latest(self, doc_type: str) -> SchemaSpec:
        versions = [v for (t, v) in self._specs if t == doc_type]
        if not versions:
            raise KeyError(f"no schema registered for doc_type {doc_type!r}")
        return self._specs[(doc_type, max(versions))]

    def doc_types(self) -> list[str]:
        return sorted({t for (t, _) in self._specs})

    def all(self) -> list[SchemaSpec]:
        return sorted(self._specs.values(), key=lambda s: (s.doc_type, s.version))


registry = SchemaRegistry()
