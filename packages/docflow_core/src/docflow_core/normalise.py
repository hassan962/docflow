"""Value normalisation shared by the validators and the eval harness.

Comparing "Rs. 1,180.00" with "1180" or "12/03/2026" with "2026-03-12" as raw strings would
make a correct extraction look wrong. Both the validators and the evals run values through
these functions first, so "correct" means the same thing everywhere in the system.
"""

from __future__ import annotations

import re
import string
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from dateutil import parser as date_parser

CENT = Decimal("0.01")
NBSP = chr(0xA0)  # no-break space, common in PDF text between a currency and its amount

# "Rs." is listed first and without a trailing \b: there is no word boundary between "." and
# the space after it, so "rs\.?\b" would match only "Rs" and leave the dot glued to the number.
_CURRENCY_TOKENS = re.compile(r"(?i)\brs\.|\b(?:rs|inr|usd|eur|gbp)\b|[₹$€£]")
_WHITESPACE = re.compile(r"\s+")
_COMPANY_SUFFIXES = frozenset(
    {
        "pvt", "private", "ltd", "limited", "inc", "incorporated", "llc", "llp", "plc",
        "co", "corp", "corporation", "company", "gmbh", "sa", "ag", "bv",
    }
)  # fmt: skip
_PUNCTUATION = str.maketrans({c: " " for c in string.punctuation})


def normalise_amount(raw: Any) -> Decimal | None:
    """Parse an amount to a Decimal with two places, or None if it is not an amount.

    Handles currency symbols and codes, thousands separators in both the Western
    (1,234,567.00) and Indian (12,34,567.00) styles, the European decimal comma (1.234,56),
    and accounting negatives "(120.00)".
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, Decimal | int):
        return Decimal(raw).quantize(CENT)
    if isinstance(raw, float):
        return Decimal(str(raw)).quantize(CENT)

    text = _CURRENCY_TOKENS.sub("", str(raw)).replace(NBSP, " ").strip()
    if not text:
        return None
    negative = (text.startswith("(") and text.endswith(")")) or text.startswith("-")
    text = text.strip("()+- ").replace(" ", "")

    if "," in text and "." in text:
        # Whichever separator comes last is the decimal point.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        head, _, tail = text.rpartition(",")
        # "12,50" is a decimal comma; "1,180" and "1,00,000" are thousands separators.
        text = f"{head}.{tail}" if len(tail) == 2 and "," not in head else text.replace(",", "")

    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    return (-value if negative else value).quantize(CENT)


def normalise_date(raw: Any, *, dayfirst: bool = True) -> date | None:
    """Parse a date to a ``date``.

    ISO strings are read exactly. Anything else goes through dateutil with ``dayfirst``
    defaulting to True, because the invoices this project targets write 03/04/2026 for
    3 April. The setting is a parameter, not a guess, so a US-format source can flip it.
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        pass
    try:
        parsed: datetime = date_parser.parse(text, dayfirst=dayfirst, fuzzy=False)
        return parsed.date()
    except (ValueError, OverflowError):
        return None


def normalise_name(raw: Any) -> str | None:
    """Lower-case a company or person name, drop punctuation and legal suffixes.

    "ACME Traders Pvt. Ltd." and "Acme Traders" both become "acme traders".
    """
    if raw is None:
        return None
    words = str(raw).lower().translate(_PUNCTUATION).split()
    while words and words[-1] in _COMPANY_SUFFIXES:
        words.pop()
    result = " ".join(words)
    return result or None


def normalise_text(raw: Any) -> str | None:
    """Generic text: trim and collapse internal whitespace, case-insensitive."""
    if raw is None:
        return None
    result = _WHITESPACE.sub(" ", str(raw)).strip().lower()
    return result or None
