"""AWS Textract adapter, the paid comparison point for the OCR bake-off.

Uses the synchronous DetectDocumentText API on one page image. Textract must run in the
project's selected Region (it is not offered in every Region, e.g. not in eu-north-1).
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import boto3
from PIL import Image

from docflow_core.ocr import OcrResult, OcrWord


class TextractEngine:
    name = "textract"

    def __init__(self, region: str, client: Any | None = None) -> None:
        self._client = client or boto3.client("textract", region_name=region)

    def run(self, image: bytes) -> OcrResult:
        with Image.open(BytesIO(image)) as img:
            width, height = img.size
        response = self._client.detect_document_text(Document={"Bytes": image})
        return parse_textract(response["Blocks"], width, height)


def parse_textract(blocks: list[dict[str, Any]], width: int, height: int) -> OcrResult:
    """Textract boxes are fractions of the page; convert them to pixels of our image."""
    by_id = {b["Id"]: b for b in blocks}
    words: list[OcrWord] = []
    lines = [b for b in blocks if b["BlockType"] == "LINE"]
    for line_no, line in enumerate(lines):
        child_ids: list[str] = []
        for rel in line.get("Relationships", []):
            if rel["Type"] == "CHILD":
                child_ids.extend(rel["Ids"])
        for wid in child_ids:
            w = by_id[wid]
            box = w["Geometry"]["BoundingBox"]
            x0 = round(box["Left"] * width)
            y0 = round(box["Top"] * height)
            x1 = round((box["Left"] + box["Width"]) * width)
            y1 = round((box["Top"] + box["Height"]) * height)
            words.append(
                OcrWord(
                    text=w["Text"],
                    bbox=(x0, y0, x1, y1),
                    conf=max(0.0, min(1.0, float(w["Confidence"]) / 100)),
                    line=line_no,
                )
            )
    return OcrResult(words=words, page_width=width, page_height=height, engine="textract")
