"""Tax-ID formats. Shared by the synthetic generator and the tax_id_format validator.

Indian GSTIN: 15 characters
    2 digits     state code
    10 chars     the holder's PAN (5 letters, 4 digits, 1 letter)
    1 char       entity number for that PAN in the state (1-9, then A-Z)
    'Z'          fixed
    1 char       check character (a mod-36 Luhn variant over the first 14)

OCR swaps look-alike characters (O/0, Z/2, I/1, S/5, B/8, G/6). Because every position has a
known class (digit, letter, or 'Z'), a misread usually breaks the format or the check
character, and the format says which look-alike was meant. ``repair_gstin`` tries exactly
those swaps and returns a candidate only when one unique valid GSTIN results.
"""

from __future__ import annotations

import itertools
import re

_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
GSTIN_PATTERN = re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]$")

# Position classes: D digit, L letter, E entity (1-9A-Z), Z the literal Z, C check (0-9A-Z).
_CLASSES = "DDLLLLLDDDDLEZC"
_AS_DIGIT = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
    "T": "7",
}
# A digit printed where a letter belongs can stand for several letters ("0" for O, Q or D);
# the check character decides between them.
_AS_LETTERS = {"0": "OQD", "1": "IL", "2": "Z", "5": "S", "6": "G", "8": "B", "7": "T"}


def gstin_check_char(first14: str) -> str:
    total = 0
    for i, ch in enumerate(first14.upper()):
        product = _CHARS.index(ch) * (2 if i % 2 else 1)
        total += product // 36 + product % 36
    return _CHARS[(36 - total % 36) % 36]


def is_valid_gstin(value: str) -> bool:
    value = value.strip().upper()
    return bool(GSTIN_PATTERN.match(value)) and gstin_check_char(value[:14]) == value[14]


def _options(ch: str, cls: str) -> list[str]:
    if cls == "D":
        return [_AS_DIGIT.get(ch, ch)]
    if cls == "L":
        return list(_AS_LETTERS.get(ch, ch))
    if cls == "Z":
        return ["Z"]
    # Entity and check positions accept both digits and letters, so the printed character
    # and its look-alike are both possible; the check character decides.
    swaps = {ch, _AS_DIGIT.get(ch, ch), *_AS_LETTERS.get(ch, ch)}
    return sorted(swaps)


def _swaps(raw: str) -> set[str]:
    options = (_options(c, k) for c, k in zip(raw, _CLASSES, strict=True))
    return {"".join(chars) for chars in itertools.product(*options)}


def repair_gstin(value: str) -> str | None:
    """The single valid GSTIN reachable by undoing OCR errors, else None.

    Tries look-alike swaps, and for 16 characters also one dropped character (OCR sometimes
    doubles a glyph: "Z2", "O0"). Returns a candidate only when exactly one valid GSTIN is
    reachable, so an ambiguous misread is left for a person.
    """
    raw = re.sub(r"[\s\-]", "", value.upper())
    if not raw.isalnum() or len(raw) not in (15, 16):
        return None
    if is_valid_gstin(raw):
        return raw
    bases = [raw] if len(raw) == 15 else {raw[:i] + raw[i + 1 :] for i in range(len(raw))}
    valid = {c for base in bases for c in _swaps(base) if is_valid_gstin(c)}
    return valid.pop() if len(valid) == 1 else None
