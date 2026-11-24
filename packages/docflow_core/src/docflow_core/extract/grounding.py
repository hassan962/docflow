"""Grounding: find where each extracted value is printed on the page.

Every value the LLM returns is matched back to OCR words, which gives the review UI a box to
draw and gives confidence two signals: the OCR engine's own confidence for those words, and
whether the value was found at all. A value that appears nowhere on the page is the
signature of a hallucination, so it keeps no box and its confidence is cut.

Matching is by kind, on contiguous word spans within one line:

- amounts: the span parses to the same Decimal ("Rs. 1,180.00" == "1180.00");
- dates: the span parses to the same date ("05/03/2026", "5 Mar 2026");
- currency codes: a symbol or code for that currency ("Rs.", "₹", "(INR)");
- text: exact after normalisation, else the closest fuzzy match above 0.8 similarity.

When a value is printed more than once (a unit price equal to its line amount, a total
repeated in words), labels on the same line break the tie: "Total" for ``total`` but not
"Sub-total", "GST" for ``tax``. Table cells prefer the line where their row's description was
found.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from docflow_core.extract.fields import Kind, normalised
from docflow_core.ocr import OcrResult, OcrWord
from docflow_core.schemas.field import BBox

Match = Literal["exact", "fuzzy", "none", "absent"]
FUZZY_MIN = 0.8

CURRENCY_TOKENS = {
    "INR": {"inr", "rs", "₹", "rupees"},
    "USD": {"usd", "$", "us$"},
    "EUR": {"eur", "€", "euro", "euros"},
    "GBP": {"gbp", "£"},
}

# Positive and negative label hints for tie-breaking, by base field name.
HINTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "total": (("total", "grand", "amount due", "balance due"), ("sub", "net amount")),
    "subtotal": (("subtotal", "sub-total", "sub total", "net amount", "taxable"), ()),
    "tax": (("tax", "gst", "vat", "igst", "cgst", "sgst"), ("gstin", "tax id")),
    "invoice_date": (("date", "issued", "dated"), ("due",)),
    "invoice_number": (("invoice", "bill", "receipt", "no", "#"), ()),
    "po_number": (("po", "purchase", "order"), ()),
    "vendor_tax_id": (("gstin", "tax id", "vat"), ()),
}


@dataclass(frozen=True)
class Grounding:
    match: Match
    page: int | None = None
    line: int | None = None  # (page, line) identifies a row for table-cell tie-breaking
    bbox: BBox | None = None
    ocr_conf: float | None = None
    similarity: float = 0.0
    raw: str | None = None


ABSENT = Grounding(match="absent")
NOT_FOUND = Grounding(match="none")


def _lines(pages: Sequence[OcrResult]) -> list[tuple[int, int, list[OcrWord]]]:
    out: list[tuple[int, int, list[OcrWord]]] = []
    for page_no, page in enumerate(pages, start=1):
        by_line: dict[int, list[OcrWord]] = {}
        for w in page.words:
            by_line.setdefault(w.line, []).append(w)
        out.extend((page_no, line, words) for line, words in sorted(by_line.items()))
    return out


def _union(words: Sequence[OcrWord]) -> BBox:
    return (
        min(w.bbox[0] for w in words),
        min(w.bbox[1] for w in words),
        max(w.bbox[2] for w in words),
        max(w.bbox[3] for w in words),
    )


def _strip(token: str) -> str:
    return re.sub(r"^[^\w$€£₹#]+|[^\w$€£₹#]+$", "", token.lower())


def _score(kind: Kind, field: str, target: Any, span: str) -> float:
    """1.0 for an exact match, the similarity for a fuzzy text match, else 0."""
    if kind == "amount":
        return 1.0 if normalised("amount", span) == target else 0.0
    if kind == "date":
        # dateutil reads a bare "10" as the 10th of this month, so a quantity could pass
        # for a date. Require a year or a month name in the span.
        if not re.search(r"\d{4}|[a-z]{3}", span.lower()):
            return 0.0
        return 1.0 if normalised("date", span) == target else 0.0
    if kind == "code" and field == "currency":
        tokens = {_strip(t) for t in span.split()} | {span.strip().lower()}
        return 1.0 if tokens & CURRENCY_TOKENS.get(str(target), set()) else 0.0
    candidate = normalised("text", span) or ""
    if candidate == target:
        return 1.0
    if not candidate or abs(len(candidate) - len(target)) > max(3, len(target) // 3):
        return 0.0
    return SequenceMatcher(None, candidate, target).ratio()


def _hint_bonus(field: str, line_text: str) -> float:
    base = field.split(".")[-1] if "[" not in field else ""
    positive, negative = HINTS.get(base, ((), ()))
    text = line_text.lower()
    if any(n in text for n in negative):
        return -0.05
    return 0.05 if any(p in text for p in positive) else 0.0


def ground(
    field: str,
    kind: Kind,
    value: Any,
    pages: Sequence[OcrResult],
    prefer_line: tuple[int, int] | None = None,
) -> Grounding:
    """Locate ``value`` (of the given kind) on the pages. ``field`` is the flat key."""
    if value is None or value == "":
        return ABSENT
    target = normalised(kind, value) if not (kind == "code" and field == "currency") else value
    if target is None:
        return NOT_FOUND
    max_words = 4 if kind in ("amount", "date", "code") else len(str(value).split()) + 2

    best: tuple[float, float, Grounding] | None = None  # (score, rank, grounding)
    for page_no, line_no, words in _lines(pages):
        line_text = " ".join(w.text for w in words)
        bonus = _hint_bonus(field, line_text)
        if prefer_line == (page_no, line_no):
            bonus += 0.1
        for start in range(len(words)):
            for end in range(start + 1, min(start + max_words, len(words)) + 1):
                span_words = words[start:end]
                span = " ".join(w.text for w in span_words)
                score = _score(kind, field, target, span)
                if score < FUZZY_MIN:
                    continue
                # Shorter spans win ties, so "1,180.00" beats "Total 1,180.00".
                rank = score + bonus - 0.001 * len(span_words)
                if best is None or rank > best[1]:
                    best = (
                        score,
                        rank,
                        Grounding(
                            match="exact" if score == 1.0 else "fuzzy",
                            page=page_no,
                            line=line_no,
                            bbox=_union(span_words),
                            ocr_conf=sum(w.conf for w in span_words) / len(span_words),
                            similarity=round(score, 3),
                            raw=span,
                        ),
                    )
    return best[2] if best else NOT_FOUND
