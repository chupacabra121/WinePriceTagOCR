import pytest

from wine_ocr.normalize import (
    check_price,
    check_unit_price,
    dedupe_key,
    detect_currency,
    discount_pct,
    normalize_vintage,
    normalize_volume,
    parse_price_text,
    parse_unit_price,
    price_per_litre,
    review_reasons,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        # Both separators appear on a single Annabella rail, so both must work.
        ("58.99", 58.99),
        ("54,59", 54.59),
        ("  25,49 lei ", 25.49),
        ("€12,50", 12.50),
        ("$1,299.00", 1299.00),
        ("1.299,00", 1299.00),
        ("19 999,95", 19999.95),
        ("0,99", 0.99),
        ("7", 7.0),
        ("", None),
        (None, None),
        ("no digits here", None),
    ],
)
def test_parse_price_text(text, expected):
    assert parse_price_text(text) == expected


def test_three_digit_tail_is_grouping_not_decimals():
    assert parse_price_text("1.299") == 1299.0
    assert parse_price_text("1,299") == 1299.0


@pytest.mark.parametrize(
    "text,code",
    [
        ("25,49 lei", "RON"),
        ("€12", "EUR"),
        ("12 EUR", "EUR"),
        ("$4", "USD"),
        ("32,59", None),
        (None, None),
    ],
)
def test_detect_currency(text, code):
    assert detect_currency(text) == code


def test_check_price_flags_decimal_misread():
    assert check_price(32.59, "32,59") == "ok"
    assert check_price(3259.0, "32,59") == "mismatch_x100"
    assert check_price(30.00, "32,59") == "mismatch"
    assert check_price(None, None) == "no_price"
    assert check_price(12.0, None) == "unverified"


def test_unit_price_cross_check():
    # 58.99 for 750 ml is 78.65/L, which is what the tag prints.
    assert price_per_litre(58.99, 750) == 78.65
    assert check_unit_price(78.65, "PU 78,65 RON/L") == "ok"
    # Same price read against a 1 L volume gives a clean 0.75 ratio: volume is wrong.
    assert check_unit_price(58.99, "78,65 RON/L") == "mismatch_volume"
    assert check_unit_price(None, "78,65 RON/L") == "unverified"
    assert check_unit_price(78.65, None) == "unverified"


def test_parse_unit_price_ignores_per_kilo():
    assert parse_unit_price("12,00 RON/L") == 12.0
    assert parse_unit_price("12,00 RON/kg") is None


@pytest.mark.parametrize(
    "raw,expected",
    [("0.75L", 750.0), ("750 ml", 750.0), ("75 cl", 750.0), ("1,5 L", 1500.0), ("x", None)],
)
def test_normalize_volume_from_text(raw, expected):
    assert normalize_volume(None, raw) == expected


def test_normalize_volume_repairs_litre_valued_numbers():
    assert normalize_volume(0.75) == 750.0   # model returned litres
    assert normalize_volume(75) == 750.0     # model returned centilitres
    assert normalize_volume(750) == 750.0    # already millilitres


def test_normalize_vintage():
    assert normalize_vintage("2019") == "2019"
    assert normalize_vintage(None, "Terra Romana 2021 rose") == "2021"
    assert normalize_vintage("nv") == "NV"
    assert normalize_vintage(None, "0.75L SEC") is None  # a volume is not a vintage
    assert normalize_vintage("2199") is None             # out of plausible range


def test_discount_and_per_litre():
    assert discount_pct(75.0, 100.0) == 25.0
    assert discount_pct(100.0, 75.0) is None
    assert price_per_litre(30.0, 750) == 40.0
    assert price_per_litre(30.0, None) is None


def test_dedupe_key_ignores_price_case_and_diacritics():
    a = dedupe_key("Annabella", "Feteasca Neagra Sec", "2021", 750)
    b = dedupe_key("annabella", "FETEASCĂ  NEAGRĂ, SEC", "2021", 750)
    assert a == b
    assert a != dedupe_key("Annabella", "Feteasca Neagra Sec", "2020", 750)


def test_review_reasons():
    good = {
        "pairing_confidence": "high", "price": 54.59, "price_check": "ok",
        "wine_name": "Cotnari Euforia Busuioaca Roze", "currency": "RON",
        "volume_ml": 750, "price_per_litre": 72.79,
    }
    assert review_reasons(good) == []

    bad = dict(good, price=None, pairing_confidence="low", wine_name="vin")
    reasons = review_reasons(bad)
    assert "no price" in reasons
    assert "low pairing confidence" in reasons
    assert "name too generic" in reasons

    assert "implausible price per litre" in review_reasons(dict(good, price_per_litre=0.4))


# --------------------------------------------------------------------------
# Decoys on the tag that look like the value being read
# --------------------------------------------------------------------------


def test_a_per_litre_reference_is_not_read_as_the_bottle_volume():
    """Lidl prints '1 L = 53,32 Lei' on every tag; the bottle is still 750ml."""
    tag = "COTNARI EUFORIA 0.75 L  54,59  1 L = 72,79 Lei"
    assert normalize_volume(None, tag) == 750.0


def test_a_kaufland_litre_reference_is_not_read_as_the_volume():
    assert normalize_volume(None, "El Carisma Vin alb (1 litru = 24.52)") is None


def test_a_real_litre_bottle_is_still_read():
    assert normalize_volume(None, "PELIN CARPATIN 1 L") == 1000.0


def test_a_three_litre_box_is_still_read():
    assert normalize_volume(None, "ZURZUR FETEASCA NEAGRA DS 3L") == 3000.0


def test_an_electronic_label_price_verifies_against_its_run_together_text():
    """Carrefour prints '88' then a small '19' and no separator at all."""
    assert check_price(88.19, "8819") == "ok_no_separator"


def test_a_genuine_decimal_misread_is_still_caught():
    """A separator is present, so the run-together excuse does not apply."""
    assert check_price(88.19, "8.819") == "mismatch_x100"
    assert check_price(88.19, "38,19") == "mismatch"
