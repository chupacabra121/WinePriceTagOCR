"""Post-processing: prices, volumes, vintages, and cross-checks.

The model already returns a parsed numeric price. Everything here exists to
*verify* that number against the verbatim ``price_text``, because a
misread decimal separator is the one error that looks completely plausible in a
spreadsheet and silently corrupts a price comparison.
"""

from __future__ import annotations

import re
from typing import Optional

CURRENCY_SYMBOLS = {
    "€": "EUR", "$": "USD", "£": "GBP", "¥": "JPY", "₺": "TRY",
    "₽": "RUB", "₴": "UAH", "zł": "PLN", "₩": "KRW", "₹": "INR",
}

CURRENCY_WORDS = {
    "lei": "RON", "ron": "RON", "leu": "RON",
    "eur": "EUR", "euro": "EUR", "eura": "EUR",
    "usd": "USD", "dollar": "USD", "dollars": "USD",
    "gbp": "GBP", "pound": "GBP",
    "chf": "CHF", "fr": "CHF",
    "pln": "PLN", "zloty": "PLN",
    "huf": "HUF", "ft": "HUF", "forint": "HUF",
    "czk": "CZK", "kc": "CZK",
    "bgn": "BGN", "lv": "BGN",
    "mdl": "MDL", "rsd": "RSD", "sek": "SEK", "nok": "NOK", "dkk": "DKK",
    "cad": "CAD", "aud": "AUD",
}

_NUM = re.compile(r"\d[\d\s., ']*\d|\d")


def detect_currency(*texts: Optional[str]) -> Optional[str]:
    """First currency found across ``texts``, symbols before words."""
    for text in texts:
        if not text:
            continue
        for sym, code in CURRENCY_SYMBOLS.items():
            if sym in text:
                return code
        for word in re.findall(r"[A-Za-zĂÂÎȘȚăâîșț]+", text):
            code = CURRENCY_WORDS.get(word.lower())
            if code:
                return code
    return None


def parse_price_text(text: Optional[str]) -> Optional[float]:
    """Parse a printed price into a float, handling European conventions.

    Rules for the final separator in a number:
      * followed by 1-2 digits  -> decimal separator  ("32,59" -> 32.59)
      * followed by exactly 3   -> thousands grouping ("1.299" -> 1299)
    Any earlier separators are grouping regardless.
    """
    if not text:
        return None
    match = _NUM.search(text.replace(" ", " "))
    if not match:
        return None

    raw = re.sub(r"[\s']", "", match.group(0))
    if not raw:
        return None

    last_sep = max(raw.rfind(","), raw.rfind("."))
    if last_sep == -1:
        try:
            return float(raw)
        except ValueError:
            return None

    tail = raw[last_sep + 1:]
    head = raw[:last_sep]
    if not tail.isdigit():
        return None

    if len(tail) <= 2:
        head = re.sub(r"[.,]", "", head)
        candidate = f"{head}.{tail}" if head else f"0.{tail}"
    else:
        candidate = re.sub(r"[.,]", "", raw)

    try:
        return float(candidate)
    except ValueError:
        return None


def check_price(price: Optional[float], price_text: Optional[str]) -> str:
    """Compare the model's number against a reparse of the printed text."""
    if price is None and not price_text:
        return "no_price"
    if price is None:
        return "text_only"
    reparsed = parse_price_text(price_text)
    if reparsed is None:
        return "unverified"
    if abs(reparsed - price) < 0.005:
        return "ok"
    # A factor-of-100 gap is the classic decimal-separator misread.
    if abs(reparsed - price * 100) < 0.5 or abs(reparsed * 100 - price) < 0.5:
        return "mismatch_x100"
    return "mismatch"


_UNIT_PRICE = re.compile(
    r"(\d[\d\s.,']*)\s*(?:ron|lei|eur|€|usd|\$|gbp|£)?\s*/\s*(l\b|litru|liter|litre|kg)",
    re.I,
)


def parse_unit_price(text: Optional[str]) -> Optional[float]:
    """Pull a per-litre price out of a tag's small print, e.g. '78,65 RON/L'."""
    if not text:
        return None
    if match := _UNIT_PRICE.search(text):
        if match.group(2).lower().startswith("l"):
            return parse_price_text(match.group(1))
    return None


def check_unit_price(
    computed_per_litre: Optional[float], unit_price_text: Optional[str]
) -> str:
    """Cross-check price and volume against the retailer's own per-litre figure.

    This is the strongest validation available on a European shelf tag: the
    retailer prints price-per-litre independently, so agreement confirms that
    *both* the price and the bottle volume were read correctly.
    """
    printed = parse_unit_price(unit_price_text)
    if printed is None or computed_per_litre is None:
        return "unverified"
    if printed <= 0:
        return "unverified"
    ratio = computed_per_litre / printed
    if 0.97 <= ratio <= 1.03:
        return "ok"
    # A clean ratio implicates the volume, not the price.
    for factor, name in ((0.75, "volume"), (1 / 0.75, "volume"), (0.5, "volume"),
                         (2.0, "volume")):
        if abs(ratio - factor) < 0.03:
            return f"mismatch_{name}"
    return "mismatch"


_VINTAGE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def normalize_vintage(value: Optional[str], *fallbacks: Optional[str]) -> Optional[str]:
    for text in (value, *fallbacks):
        if not text:
            continue
        stripped = text.strip()
        if stripped.upper() in {"NV", "N.V.", "NON-VINTAGE", "NONVINTAGE"}:
            return "NV"
        if match := _VINTAGE.search(stripped):
            return match.group(1)
    return None


_VOLUME_PATTERNS = [
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:l|litri|liter|litre)\b", re.I), 1000.0),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:ml|mililitri)\b", re.I), 1.0),
    (re.compile(r"(\d+(?:[.,]\d+)?)\s*cl\b", re.I), 10.0),
]


def normalize_volume(value: Optional[float], *texts: Optional[str]) -> Optional[float]:
    """Volume in ml, from the model's number or by re-reading the raw text."""
    if value:
        # Tags often print "0.75" meaning litres; the model should convert, but
        # a bare 0.75 or 75 is unambiguous enough to repair here.
        if value < 10:
            return round(value * 1000, 1)
        if value < 100:
            return round(value * 10, 1)
        return round(value, 1)
    for text in texts:
        if not text:
            continue
        for pattern, factor in _VOLUME_PATTERNS:
            if match := pattern.search(text):
                try:
                    return round(float(match.group(1).replace(",", ".")) * factor, 1)
                except ValueError:
                    continue
    return None


def price_per_litre(
    price: Optional[float], volume_ml: Optional[float]
) -> Optional[float]:
    if price is None or not volume_ml:
        return None
    return round(price * 1000.0 / volume_ml, 2)


def discount_pct(price: Optional[float], original: Optional[float]) -> Optional[float]:
    if price is None or not original or original <= 0 or original <= price:
        return None
    return round((original - price) / original * 100.0, 1)


_WS = re.compile(r"\s+")
_NOISE = re.compile(r"[^\w\s]", re.UNICODE)
_DIACRITICS = str.maketrans("ăâîșțĂÂÎȘȚáéíóúüöäàèìòù", "aaistAAISTaeiouuoaaeiou")


def dedupe_key(
    store: str,
    wine_name: str,
    vintage: Optional[str],
    volume_ml: Optional[float],
) -> str:
    """Loose identity for the same product seen twice.

    Deliberately excludes price: the same wine at a different price on a
    different day is the same product, and that is the comparison we want.
    """
    name = _NOISE.sub(" ", wine_name.lower().translate(_DIACRITICS))
    name = _WS.sub(" ", name).strip()
    return "|".join(
        [
            store.lower().strip(),
            name,
            (vintage or "").strip(),
            f"{volume_ml:.0f}" if volume_ml else "",
        ]
    )


GENERIC_NAMES = {
    "", "wine", "vin", "red wine", "white wine", "unknown", "n/a", "vin rosu",
    "vin alb", "unreadable", "illegible",
}


def review_reasons(row: dict) -> list[str]:
    """Why a row should be looked at by a human before being trusted."""
    reasons: list[str] = []
    if row.get("pairing_confidence") == "low":
        reasons.append("low pairing confidence")
    elif row.get("pairing_confidence") == "medium":
        reasons.append("unconfirmed pairing")
    if row.get("price") in (None, ""):
        reasons.append("no price")
    if row.get("price_check", "").startswith("mismatch"):
        reasons.append(f"price {row['price_check']}")
    if row.get("unit_price_check", "").startswith("mismatch"):
        reasons.append(f"per-litre {row['unit_price_check']}")
    name = str(row.get("wine_name", "")).strip().lower()
    if name in GENERIC_NAMES or len(name) < 4:
        reasons.append("name too generic")
    if not row.get("currency"):
        reasons.append("no currency")
    if not row.get("volume_ml"):
        reasons.append("no volume")
    ppl = row.get("price_per_litre")
    if isinstance(ppl, (int, float)) and (ppl < 3 or ppl > 3000):
        reasons.append("implausible price per litre")
    return reasons
