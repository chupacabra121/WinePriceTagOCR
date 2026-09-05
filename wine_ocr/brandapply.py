"""Attach standardized brand columns to a finished wine table.

The reading pass records what a shelf tag said. Analysis needs the opposite —
one stable identity per product, so the same wine seen in nine shops collapses
to nine prices of one thing. The standardized sheet supplies that identity as
Group / Winery / Label, and this module carries it onto every row.

Matching runs on distinct *names*, not rows: 6,670 shelf readings collapse to
about 2,400 distinct names, so a name is judged once and the verdict is shared
by every row that read it.
"""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .brands import BrandIndex, tokens

BRAND_COLUMNS = ["brand_group", "brand_winery", "brand_label", "brand_match"]


def load_reference(path: Path) -> BrandIndex:
    with path.open(encoding="utf-8-sig") as fh:
        return BrandIndex(list(csv.DictReader(fh, delimiter="\t")))


def name_key(row: dict) -> str:
    """The identity a brand verdict is keyed on: the name's brand-bearing words."""
    return " ".join(tokens(row.get("wine_name")))


def load_verdicts(answers_dir: Path, batches_dir: Path) -> dict[str, dict]:
    """Read the reviewed answers back, keyed by name.

    A batch whose answer file is missing or malformed simply contributes
    nothing, leaving its names to the index's own guess — the same
    partial-progress property the reading pass has.
    """
    verdicts: dict[str, dict] = {}
    for batch_file in sorted(batches_dir.glob("*.json")):
        answer_file = answers_dir / batch_file.name
        if not answer_file.exists():
            continue
        try:
            items = json.loads(batch_file.read_text(encoding="utf-8"))
            picks = json.loads(answer_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(picks, dict):                     # occasionally wrapped
            picks = next((v for v in picks.values() if isinstance(v, list)), [])
        by_key = {it["key"]: it for it in items if isinstance(it, dict)}
        for pick in picks:
            if not isinstance(pick, dict):
                continue
            item = by_key.get(pick.get("key"))
            if item is None:
                continue
            choice = pick.get("choice")
            if not isinstance(choice, int) or not 1 <= choice <= len(item["candidates"]):
                verdicts[item["key"]] = {"brand": None,
                                         "confidence": pick.get("confidence") or "low"}
                continue
            verdicts[item["key"]] = {
                "brand": item["candidates"][choice - 1],
                "confidence": pick.get("confidence") or "medium",
            }
    return verdicts


def enrich(rows: list[dict], index: BrandIndex, verdicts: dict[str, dict]) -> dict:
    """Add the brand columns to every row in place; return a small summary.

    Rows the reviewers judged take the reviewed answer. Rows they never saw fall
    back to the index's own pick, marked so the two can be told apart — an
    automatic guess and a checked one should never look alike in a spreadsheet.
    """
    stats = {"reviewed": 0, "auto": 0, "none": 0, "rows": len(rows)}
    cache: dict[str, tuple] = {}

    for row in rows:
        key = name_key(row)
        if key in cache:
            group, winery, label, how = cache[key]
        else:
            verdict = verdicts.get(key)
            if verdict is not None:
                brand = verdict["brand"]
                how = f"reviewed:{verdict['confidence']}" if brand else "reviewed:none"
                parts = [p.strip() for p in brand.split("|")] if brand else ["", "", ""]
            else:
                match = index.match_wine(
                    row.get("wine_name"), row.get("producer"),
                    row.get("raw_tag_text"), row.get("raw_label_text"),
                )
                if match.matched:
                    parts = [match.row.group, match.row.winery, match.row.label]
                    how = "auto"
                else:
                    parts, how = ["", "", ""], "none"
            group, winery, label = (parts + ["", "", ""])[:3]
            cache[key] = (group, winery, label, how)

        row["brand_group"] = group
        row["brand_winery"] = winery
        row["brand_label"] = label
        row["brand_match"] = how
        if how.startswith("reviewed:none") or how == "none":
            stats["none"] += 1
        elif how.startswith("reviewed"):
            stats["reviewed"] += 1
        else:
            stats["auto"] += 1
    return stats
