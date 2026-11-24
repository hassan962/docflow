"""OCR engine interface.

Every engine turns one page image into words with pixel bounding boxes and a confidence.
Engines are adapters behind this one interface, chosen by ``OCR_ENGINE``, so they can be
swapped and compared on the same pages.

``line`` is an addition to the design doc's ``OcrWord``: a per-page line index. Both
Tesseract and Textract report lines natively, and the prompt builder (line-tagged OCR text)
and the chunker (lines into chunks) both need them, so it is cheaper to keep them than to
rebuild lines from boxes later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from docflow_core.config import Settings


@dataclass(frozen=True)
class OcrWord:
    text: str
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in image pixels
    conf: float  # 0..1
    line: int = 0  # line index on the page, in reading order


@dataclass
class OcrResult:
    words: list[OcrWord]
    page_width: int
    page_height: int
    engine: str
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def mean_conf(self) -> float | None:
        if not self.words:
            return None
        return sum(w.conf for w in self.words) / len(self.words)

    def lines(self) -> list[str]:
        """Words grouped into text lines, in reading order."""
        grouped: dict[int, list[str]] = {}
        for w in self.words:
            grouped.setdefault(w.line, []).append(w.text)
        return [" ".join(grouped[k]) for k in sorted(grouped)]

    def to_json(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, data: bytes | str) -> OcrResult:
        raw = json.loads(data)
        words = [
            OcrWord(text=w["text"], bbox=tuple(w["bbox"]), conf=w["conf"], line=w.get("line", 0))
            for w in raw["words"]
        ]
        return cls(
            words=words,
            page_width=raw["page_width"],
            page_height=raw["page_height"],
            engine=raw["engine"],
            meta=raw.get("meta", {}),
        )


class OcrEngine(Protocol):
    name: str

    def run(self, image: bytes) -> OcrResult: ...


def get_engine(settings: Settings) -> OcrEngine:
    if settings.ocr_engine == "tesseract":
        from docflow_core.ocr.tesseract import TesseractEngine

        return TesseractEngine()
    if settings.ocr_engine == "textract":
        from docflow_core.ocr.textract import TextractEngine

        return TextractEngine(region=settings.aws_region)
    raise ValueError(f"unknown OCR engine {settings.ocr_engine!r}")
