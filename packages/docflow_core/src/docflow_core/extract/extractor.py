"""Run one extractor on one document: prompt -> reply -> schema check -> grounding -> confidence.

Failure handling: if the reply is not valid JSON or fails the schema, the model is asked once
more with the validation error appended. A second failure returns ``failed_parse``, and the
whole document goes to human review rather than guessing.

The second-opinion model gets the identical prompt. Its answer is used only for the
per-field agreement signal, never as a value, and if it fails the signal is simply missing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import openai
from pydantic import ValidationError

from docflow_core.extract.confidence import (
    CalibratedRawScorer,
    IsotonicCalibrator,
    Scorer,
    Signals,
    raw_score,
)
from docflow_core.extract.fields import (
    base_name,
    field_kinds,
    flatten,
    item_model,
    lenient_model,
    required_keys,
    same_value,
)
from docflow_core.extract.grounding import Grounding, ground
from docflow_core.extract.llm import ChatRequest, ChatResult, LlmClient
from docflow_core.extract.llm_schema import object_schema
from docflow_core.extract.prompt import FewShotExample, build_prompt, max_tokens_for
from docflow_core.ocr import OcrResult
from docflow_core.schemas.field import ExtractedField
from docflow_core.schemas.registry import SchemaSpec

MAX_ERROR_CHARS = 1500


@dataclass(frozen=True)
class Call:
    role: Literal["primary", "retry", "agreement"]
    result: ChatResult


@dataclass
class ExtractionOutcome:
    status: Literal["ok", "failed_parse"]
    fields: dict[str, ExtractedField]
    calls: list[Call] = field(default_factory=list)
    error: str | None = None
    template_hash: str = ""


def _parse(spec: SchemaSpec, content: str | None, finish_reason: str | None) -> dict[str, Any]:
    if content is None or not content.strip():
        raise ValueError(f"empty reply (finish_reason={finish_reason})")
    data = json.loads(content)
    return lenient_model(spec.model).model_validate(data).model_dump(mode="json")


def _call(client: LlmClient, request: ChatRequest) -> tuple[ChatResult, str | None]:
    """One chat call. A provider-side rejection (HTTP 400, e.g. a strict-schema mismatch)
    is reported as an invalid reply, so it gets the same one retry, not a stage failure."""
    try:
        return client.chat(request), None
    except openai.BadRequestError as exc:
        rejected = ChatResult(None, client.model, 0, 0, 0, 0.0, "rejected")
        return rejected, f"provider rejected the reply: {exc}"[:MAX_ERROR_CHARS]


def _ask(
    client: LlmClient, request: ChatRequest, spec: SchemaSpec, *, retry: bool
) -> tuple[dict[str, Any] | None, list[Call], str | None]:
    calls: list[Call] = []
    result, error = _call(client, request)
    calls.append(Call("primary", result))
    if error is None:
        try:
            return _parse(spec, result.content, result.finish_reason), calls, None
        except (ValueError, ValidationError) as exc:
            error = str(exc)[:MAX_ERROR_CHARS]
    if not retry:
        return None, calls, error

    previous = [{"role": "assistant", "content": result.content}] if result.content else []
    follow_up = ChatRequest(
        messages=[
            *request.messages,
            *previous,
            {
                "role": "user",
                "content": f"That reply was invalid:\n{error}\n\nReply again with only the "
                "corrected JSON object. Use null for anything not printed on the document.",
            },
        ],
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        json_schema=request.json_schema,
    )
    second, error = _call(client, follow_up)
    calls.append(Call("retry", second))
    if error is not None:
        return None, calls, error
    try:
        return _parse(spec, second.content, second.finish_reason), calls, None
    except (ValueError, ValidationError) as exc:
        return None, calls, str(exc)[:MAX_ERROR_CHARS]


def score_fields(
    spec: SchemaSpec,
    data: dict[str, Any],
    pages: Sequence[OcrResult],
    second_opinion: dict[str, Any] | None,
    scorer: Scorer,
) -> dict[str, ExtractedField]:
    kinds = field_kinds(spec.model)
    required = required_keys(spec.model)
    flat = flatten(spec.model, data)
    flat2 = flatten(spec.model, second_opinion) if second_opinion is not None else None

    # Table rows: ground each row's first text column (the description) first, so the
    # row's other cells can prefer that line when a number is printed more than once.
    anchors: dict[str, Grounding] = {}
    for name in spec.model.model_fields:
        row = item_model(spec.model, name)
        if row is None:
            continue
        anchor_col = next(c for c in row.model_fields if kinds[f"{name}.{c}"] == "text")
        for key in flat:
            if key.startswith(f"{name}[") and key.endswith(f".{anchor_col}"):
                anchors[key.rsplit(".", 1)[0]] = ground(key, "text", flat[key], pages)

    fields: dict[str, ExtractedField] = {}
    for key, value in flat.items():
        kind = kinds[base_name(key)]
        row_key = key.rsplit(".", 1)[0] if "[" in key else None
        anchor = anchors.get(row_key) if row_key else None
        prefer = (
            (anchor.page, anchor.line)
            if anchor and anchor.page and anchor.line is not None
            else None
        )
        g = ground(key, kind, value, pages, prefer_line=prefer)
        agreement = (
            None if flat2 is None else (key in flat2 and same_value(kind, value, flat2[key]))
        )
        missing = value is None and base_name(key) in required
        signals = Signals(g.match, g.ocr_conf, g.similarity, agreement, required_missing=missing)
        raw = raw_score(signals)
        fields[key] = ExtractedField(
            value=value,
            raw=g.raw,
            confidence=scorer(signals),
            page=g.page,
            bbox=g.bbox,
            source="llm",
            signals={**signals.as_dict(), "raw_score": raw},
        )
    return fields


def extract(
    spec: SchemaSpec,
    pages: Sequence[OcrResult],
    client: LlmClient,
    *,
    agreement_client: LlmClient | None = None,
    scorer: Scorer | None = None,
    examples: Sequence[FewShotExample] = (),
) -> ExtractionOutcome:
    prompt = build_prompt(spec, pages, examples)
    request = ChatRequest(
        messages=prompt.messages,
        temperature=0.0,
        max_tokens=max_tokens_for(spec),
        json_schema=object_schema(spec.model),
    )
    data, calls, error = _ask(client, request, spec, retry=True)
    if data is None:
        return ExtractionOutcome("failed_parse", {}, calls, error, prompt.template_hash)

    second: dict[str, Any] | None = None
    if agreement_client is not None:
        # The second opinion always reads the plain zero-shot prompt: it stays an
        # independent check of the extractor (including of its few-shot examples), and costs
        # no extra tokens when examples are added.
        plain = (
            request
            if not examples
            else ChatRequest(
                messages=build_prompt(spec, pages).messages,
                temperature=0.0,
                max_tokens=request.max_tokens,
                json_schema=request.json_schema,
            )
        )
        try:
            second, more, _ = _ask(agreement_client, plain, spec, retry=False)
            calls.extend(Call("agreement", c.result) for c in more)
        except Exception:  # a failing second opinion only removes one signal
            second = None

    default = CalibratedRawScorer(IsotonicCalibrator.identity())
    fields = score_fields(spec, data, pages, second, scorer or default)
    return ExtractionOutcome("ok", fields, calls, None, prompt.template_hash)
