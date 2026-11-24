"""Tax-ID formats. Shared by the synthetic generator and the tax_id_format validator.

Indian GSTIN: 15 characters
    2 digits     state code
    10 chars     the holder's PAN (5 letters, 4 digits, 1 letter)
    1 char       entity number for that PAN in the state (1-9, then A-Z)
    'Z'          fixed
    1 char       check character (a mod-36 Luhn variant over the first 14)
"""

from __future__ import annotations

import re

_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_PATTERN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")


def gstin_check_char(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14.upper()):
        product = _CHARS.index(ch) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return _CHARS[(36 - total % 36) % 36]


def is_valid_gstin(value: str) -> bool:
    value = value.strip().upper()
    return bool(GSTIN_PATTERN.match(value)) and gstin_check_char(value[:14]) == value[14]
