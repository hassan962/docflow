"""Versioned document schemas and the registry that maps (doc_type, version) to them."""

from docflow_core.schemas.field import ExtractedField, RuleCheck
from docflow_core.schemas.registry import SchemaSpec, registry

# Importing the schema modules registers them.
from docflow_core.schemas import invoice as _invoice  # noqa: F401  # isort: skip

__all__ = ["ExtractedField", "RuleCheck", "SchemaSpec", "registry"]
