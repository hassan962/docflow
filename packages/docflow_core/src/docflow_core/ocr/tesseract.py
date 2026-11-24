"""Tesseract adapter (via pytesseract). Free, local, runs inside the worker container."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pytesseract
from PIL import Image

from docflow_core.ocr import OcrResult, OcrWord


class TesseractEngine:
    name = "tesseract"

    def __init__(self, lang: str = "eng", psm: int = 3) -> None:
        # psm 3 = fully automatic page segmentation, right for whole invoice pages.
        self.lang = lang
        self.config = f"--psm {psm}"

    def run(self, image: bytes) -> OcrResult:
        img = Image.open(BytesIO(image))
        data = pytesseract.image_to_data(
            img, lang=self.lang, config=self.config, output_type=pytesseract.Output.DICT
        )
        result = parse_tesseract(data, img.width, img.height)
        result.meta["version"] = str(pytesseract.get_tesseract_version())
        return result


def parse_tesseract(data: dict[str, list[Any]], width: int, height: int) -> OcrResult:
    """Convert pytesseract's column-oriented dict into words with line indices.

    Tesseract reports one row per layout element; rows with conf == -1 are blocks,
    paragraphs and lines rather than words, and empty text rows carry nothing.
    """
    words: list[OcrWord] = []
    line_ids: dict[tuple[int, int, int], int] = {}
    for i, text in enumerate(data["text"]):
        conf = float(data["conf"][i])
        if conf < 0 or not str(text).strip():
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        line = line_ids.setdefault(key, len(line_ids))
        left, top = int(data["left"][i]), int(data["top"][i])
        words.append(
            OcrWord(
                text=str(text).strip(),
                bbox=(left, top, left + int(data["width"][i]), top + int(data["height"][i])),
                conf=max(0.0, min(1.0, conf / 100)),
                line=line,
            )
        )
    return OcrResult(words=words, page_width=width, page_height=height, engine="tesseract")
