"""Caches that make evaluation cheap, repeatable and fair.

``CachedClient`` stores every LLM reply under a hash of the exact request (model, messages,
schema, tools, temperature). Re-running an evaluation then costs nothing, and two versions that
send identical prompts get identical replies, so a change that only touches routing is compared
on exactly the same extractions. A cached reply keeps the tokens, cost and latency of the real
call, so cost and latency metrics stay honest.

``OcrCache`` stores each document's OCR output by file hash and engine. OCR is not part of an
extractor version, so every version is scored on the same text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from docflow_core.extract.llm import ChatRequest, ChatResult, LlmClient, ToolCall
from docflow_core.ocr import OcrEngine, OcrResult
from docflow_core.ocr.pipeline import read_document
from docflow_core.storage import sniff_content_type


def request_key(model: str, request: ChatRequest) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": request.messages,
            "schema": request.json_schema,
            "tools": request.tools,
            "temperature": request.temperature,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(body.encode()).hexdigest()


class CachedClient:
    def __init__(self, inner: LlmClient | None, model: str, directory: Path) -> None:
        self.inner, self.model, self.directory = inner, model, directory
        self.hits = self.misses = 0
        directory.mkdir(parents=True, exist_ok=True)

    def chat(self, request: ChatRequest) -> ChatResult:
        path = self.directory / f"{request_key(self.model, request)}.json"
        if path.exists():
            self.hits += 1
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["tool_calls"] = tuple(ToolCall(**c) for c in raw.get("tool_calls", []))
            return ChatResult(**raw)
        if self.inner is None:
            raise LookupError("reply not cached and no live client (offline evaluation)")
        self.misses += 1
        result = self.inner.chat(request)
        if result.content is not None or result.tool_calls:  # never cache an empty reply
            path.write_text(json.dumps(asdict(result)), encoding="utf-8")
        return result


class OcrCache:
    def __init__(self, engine: OcrEngine, directory: Path, *, dpi: int, max_pages: int) -> None:
        self.engine, self.directory, self.dpi, self.max_pages = engine, directory, dpi, max_pages
        directory.mkdir(parents=True, exist_ok=True)

    def pages(self, path: Path, sha256: str) -> list[OcrResult]:
        cached = self.directory / f"{sha256}.{self.engine.name}.json"
        if cached.exists():
            return [OcrResult.from_json(json.dumps(p)) for p in json.loads(cached.read_text())]
        data = path.read_bytes()
        content_type = sniff_content_type(data) or "application/octet-stream"
        pages = [
            p.ocr
            for p in read_document(
                data, content_type, self.engine, dpi=self.dpi, max_pages=self.max_pages
            )
        ]
        cached.write_text(json.dumps([json.loads(p.to_json()) for p in pages]), encoding="utf-8")
        return pages
