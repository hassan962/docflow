"""Prompt construction, in a fixed order:

1. system prompt: the task, the output rules and the schema;
2. few-shot examples: corrected documents chosen by the learning loop (none at first);
3. the OCR text, every line tagged with its page and line number;
4. the instruction to answer only with JSON that matches the schema.

``template_hash`` covers parts 1 and 4: the instructions and the schema. It is the
``prompt_hash`` stored on an extractor version. Few-shot examples (part 2) are chosen per
document by the version's few-shot policy and recorded as ``example_ids``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from docflow_core.extract.llm_schema import object_schema
from docflow_core.ocr import OcrResult
from docflow_core.schemas.registry import SchemaSpec

PROMPT_VERSION = "extract-v1"

SYSTEM = """You extract data from the OCR text of one business document ({doc_type}).

Rules:
- Answer with one JSON object that matches the schema below. No other text.
- Use null when a field is not printed on the document. Never guess, and never calculate a \
value that is not printed (for example, do not add up line items to produce a missing total).
- Amounts: plain decimal strings exactly as printed, without currency symbols or thousands \
separators. "Rs. 1,03,095.00" becomes "103095.00"; "1.234,56" (European) becomes "1234.56".
- Dates: YYYY-MM-DD. Read dd/mm/yyyy unless the document clearly uses another order.
- currency: the ISO code. "Rs.", "₹" and "INR" are INR; "$" is USD; "€" is EUR; "£" is GBP.
- line_items: one entry per printed row of the item table, in order, nothing else.
- OCR may break words or misread characters; use the surrounding layout to decide.

Schema:
{schema}"""

INSTRUCTION = "Return only the JSON object for the document above."


@dataclass(frozen=True)
class FewShotExample:
    example_id: str  # document hash, so the leakage check can compare against the golden set
    ocr_text: str
    answer_json: str


@dataclass(frozen=True)
class Prompt:
    messages: list[dict[str, str]]
    template_hash: str  # the base template: instructions and schema, never the examples
    example_ids: tuple[str, ...] = ()  # which corrected documents were shown as examples


def ocr_text(pages: Sequence[OcrResult]) -> str:
    """'[p1 l03] Invoice No.: A-08109' — line tags let the model (and a reviewer) point back."""
    out = []
    for page_no, page in enumerate(pages, start=1):
        for line_no, line in enumerate(page.lines(), start=1):
            out.append(f"[p{page_no} l{line_no:02d}] {line}")
    return "\n".join(out)


def build_prompt(
    spec: SchemaSpec, pages: Sequence[OcrResult], examples: Sequence[FewShotExample] = ()
) -> Prompt:
    schema = json.dumps(object_schema(spec.model), separators=(",", ":"))
    system = SYSTEM.format(doc_type=f"{spec.doc_type} v{spec.version}", schema=schema)
    messages = [{"role": "system", "content": system}]
    for ex in examples:
        messages.append({"role": "user", "content": f"{ex.ocr_text}\n\n{INSTRUCTION}"})
        messages.append({"role": "assistant", "content": ex.answer_json})
    messages.append({"role": "user", "content": f"{ocr_text(pages)}\n\n{INSTRUCTION}"})

    # Examples differ per document, so they are not part of the version's prompt hash (that
    # would make every document look like prompt drift). The version's fewshot_policy says
    # how they are chosen; example_ids record which were used.
    template = json.dumps(
        {"version": PROMPT_VERSION, "system": system, "instruction": INSTRUCTION, "examples": []},
        sort_keys=True,
    )
    return Prompt(
        messages=messages,
        template_hash=hashlib.sha256(template.encode()).hexdigest(),
        example_ids=tuple(e.example_id for e in examples),
    )


def max_tokens_for(spec: SchemaSpec) -> int:
    """Reply budget from the schema size, plus headroom for a reasoning model's thinking."""
    schema_chars = len(json.dumps(object_schema(spec.model)))
    return 1500 + schema_chars
