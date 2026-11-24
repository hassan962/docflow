from datetime import date
from decimal import Decimal

import pytest

from docflow_core.normalise import normalise_amount, normalise_date, normalise_name, normalise_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Rs. 1,180.00", "1180.00"),
        ("₹1,180", "1180.00"),
        ("INR 12,34,567.50", "1234567.50"),  # Indian grouping
        ("$1,234,567.89", "1234567.89"),
        ("1.234,56 €", "1234.56"),  # European decimal comma
        ("12,50", "12.50"),
        ("INR" + chr(0xA0) + "1,180.00", "1180.00"),  # no-break space from PDF text
        ("(120.00)", "-120.00"),  # accounting negative
        ("-45", "-45.00"),
        (1180, "1180.00"),
        (Decimal("99.999"), "100.00"),
        (12.5, "12.50"),
    ],
)
def test_amounts(raw: object, expected: str) -> None:
    assert normalise_amount(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", [None, "", "N/A", "total", True, "NaN"])
def test_non_amounts(raw: object) -> None:
    assert normalise_amount(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-03-12", date(2026, 3, 12)),
        ("12/03/2026", date(2026, 3, 12)),  # day first by default
        ("12-Mar-2026", date(2026, 3, 12)),
        ("March 12, 2026", date(2026, 3, 12)),
        (date(2026, 3, 12), date(2026, 3, 12)),
    ],
)
def test_dates(raw: object, expected: date) -> None:
    assert normalise_date(raw) == expected


def test_date_month_first_is_a_parameter() -> None:
    assert normalise_date("03/12/2026", dayfirst=False) == date(2026, 3, 12)


@pytest.mark.parametrize("raw", [None, "", "not a date", "31/02/2026"])
def test_non_dates(raw: object) -> None:
    assert normalise_date(raw) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ACME Traders Pvt. Ltd.", "acme traders"),
        ("Acme Traders", "acme traders"),
        ("Globex, Inc.", "globex"),
        ("  Initech   LLC ", "initech"),
        ("Ltd", None),
    ],
)
def test_names(raw: str, expected: str | None) -> None:
    assert normalise_name(raw) == expected


def test_text() -> None:
    assert normalise_text("  INV-\n 0042 ") == "inv- 0042"
