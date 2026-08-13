"""Score an extraction against a hand-read ground-truth file.

Ground truth covers a subset of a photo set (typically one shelf rail), so this
measures **recall over the covered subset** and reports everything else as
context rather than as error. Counting uncovered rows as false positives would
make a more complete extraction look worse.

Matching is on price within a store, because the price is the field that can be
transcribed by eye with near-certainty. A matched row is then checked for the
expected name token, which is what actually distinguishes a good read from one
that got the digits right and the product wrong.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DIACRITICS = str.maketrans("ăâîșțĂÂÎȘȚáéíóúüöäàèìòù", "aaistAAISTaeiouuoaaeiou")


def _norm(text: Optional[str]) -> str:
    if not text:
        return ""
    folded = text.translate(_DIACRITICS).upper()
    return re.sub(r"[^A-Z0-9 ]+", " ", folded)


@dataclass
class Expected:
    photo: str
    rail: str
    position: str
    price: float
    currency: str
    volume_ml: Optional[float]
    name_contains: str
    notes: str


@dataclass
class Outcome:
    expected: Expected
    matched: Optional[dict] = None
    name_ok: Optional[bool] = None   # None when no token was asserted
    volume_ok: Optional[bool] = None

    @property
    def found(self) -> bool:
        return self.matched is not None


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)
    extra: list[dict] = field(default_factory=list)
    total_rows: int = 0

    @property
    def found(self) -> int:
        return sum(1 for o in self.outcomes if o.found)

    @property
    def name_checked(self) -> int:
        return sum(1 for o in self.outcomes if o.name_ok is not None)

    @property
    def name_correct(self) -> int:
        return sum(1 for o in self.outcomes if o.name_ok)

    @property
    def volume_correct(self) -> int:
        return sum(1 for o in self.outcomes if o.volume_ok)


def load_truth(path: Path) -> list[Expected]:
    rows: list[Expected] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        # '#' comment lines carry provenance for the transcription.
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
    for raw in csv.DictReader(lines):
        if not (raw.get("price") or "").strip():
            continue
        rows.append(
            Expected(
                photo=(raw.get("photo") or "").strip(),
                rail=(raw.get("rail") or "").strip(),
                position=(raw.get("position") or "").strip(),
                price=float(raw["price"]),
                currency=(raw.get("currency") or "").strip(),
                volume_ml=float(raw["volume_ml"]) if (raw.get("volume_ml") or "").strip() else None,
                name_contains=(raw.get("name_contains") or "").strip(),
                notes=(raw.get("notes") or "").strip(),
            )
        )
    return rows


def load_extracted(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    # Duplicate sightings are the same product; score the primary row only.
    return [r for r in rows if not (r.get("duplicate_of") or "").strip()]


def verify(truth: list[Expected], extracted: list[dict]) -> Report:
    report = Report(total_rows=len(extracted))
    used: set[int] = set()

    for exp in truth:
        outcome = Outcome(expected=exp)

        # Prefer a candidate whose name token also matches, so that two tags at
        # the same price (Castel Huniade appears twice at 27.99) are not matched
        # to each other's row.
        candidates = []
        for i, row in enumerate(extracted):
            if i in used:
                continue
            raw_price = (row.get("price") or "").strip()
            if not raw_price:
                continue
            try:
                price = float(raw_price)
            except ValueError:
                continue
            # NaN would make every comparison False and match everything.
            if price != price or abs(price - exp.price) > 0.005:
                continue
            haystack = _norm(f"{row.get('wine_name','')} {row.get('raw_tag_text','')}")
            token_ok = bool(exp.name_contains) and _norm(exp.name_contains) in haystack
            candidates.append((token_ok, i, row))

        if candidates:
            candidates.sort(key=lambda c: not c[0])  # token matches first
            token_ok, index, row = candidates[0]
            used.add(index)
            outcome.matched = row
            outcome.name_ok = token_ok if exp.name_contains else None
            if exp.volume_ml is not None:
                try:
                    outcome.volume_ok = abs(float(row.get("volume_ml") or 0) - exp.volume_ml) < 1
                except ValueError:
                    outcome.volume_ok = False

        report.outcomes.append(outcome)

    report.extra = [r for i, r in enumerate(extracted) if i not in used]
    return report
