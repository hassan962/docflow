"""Rebuild text lines from word geometry: words at the same height form one visual row.

OCR engines number lines inside their own layout blocks. On invoices that breaks tables:
Tesseract reads the Description column, then the Qty column, then the Amount column as
separate blocks, and can even split "20,312.88" into "20,312." and "88" in different blocks.
The LLM then has to guess which quantity belongs to which item.

Grouping words by vertical overlap restores what a person sees, a row reading left to
right ("Office chair 9 11,455.00 1,03,095.00"), whatever engine produced the words.
"""

from __future__ import annotations

from dataclasses import replace
from statistics import median

from docflow_core.ocr import OcrResult, OcrWord


def _center(w: OcrWord) -> float:
    return (w.bbox[1] + w.bbox[3]) / 2


def visual_lines(result: OcrResult, overlap: float = 0.5) -> OcrResult:
    """Reassign ``line`` so each line is one visual row, ordered top to bottom.

    A word joins the current row when its vertical centre lies within ``overlap`` times the
    row's median word height of the row's median centre. Words in a row are sorted by x.
    """
    if not result.words:
        return result
    rows: list[list[OcrWord]] = []
    for word in sorted(result.words, key=_center):
        if rows:
            row = rows[-1]
            height = median(w.bbox[3] - w.bbox[1] for w in row) or 1
            if abs(_center(word) - median(_center(w) for w in row)) <= overlap * height:
                row.append(word)
                continue
        rows.append([word])
    words = [
        replace(w, line=line_no)
        for line_no, row in enumerate(rows)
        for w in sorted(row, key=lambda w: w.bbox[0])
    ]
    meta = {**result.meta, "lines": "visual"}
    return OcrResult(words, result.page_width, result.page_height, result.engine, meta)
