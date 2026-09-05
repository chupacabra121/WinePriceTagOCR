"""Write results back out in the shape of the input tree.

The photo library is organised by shop — ``Hypermarket - Kaufland/``,
``Discounter - LIDL/`` and so on — and that organisation is the analysis. So the
output mirrors it: one CSV per input folder, sitting in the same relative place
under the output root, plus one flat CSV of everything and a short index.

Reading a single shop's prices should not require filtering a 5000-row sheet.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from .output import COLUMNS, UNMATCHED_COLUMNS, write_csv

INDEX_COLUMNS = [
    "folder", "csv", "photos", "wines", "priced", "needs_review",
    "median_price", "currency",
]


def _folder_of(row: dict) -> str:
    """The input subfolder a row came from, as a relative path.

    Rows whose photo sat directly in the run root land under ``.``, which keeps
    the mirror total: every row is in exactly one folder file.
    """
    parent = Path(str(row.get("photo", ""))).parent
    return str(parent) if str(parent) not in {"", "."} else "."


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 2)
    return round((ordered[mid - 1] + ordered[mid]) / 2, 2)


def write_mirror(
    out_root: Path,
    rows: list[dict],
    unmatched: list[dict],
    *,
    include_duplicates: bool = False,
) -> list[dict]:
    """Write one CSV per source folder and return the index rows."""
    primary = rows if include_duplicates else [r for r in rows if not r.get("duplicate_of")]

    by_folder: dict[str, list[dict]] = defaultdict(list)
    for row in primary:
        by_folder[_folder_of(row)].append(row)

    unmatched_by_folder: dict[str, list[dict]] = defaultdict(list)
    for row in unmatched:
        unmatched_by_folder[_folder_of(row)].append(row)

    index: list[dict] = []
    for folder in sorted(set(by_folder) | set(unmatched_by_folder)):
        folder_rows = by_folder.get(folder, [])
        target_dir = out_root if folder == "." else out_root / folder
        name = "wines.csv" if folder == "." else f"{Path(folder).name}.csv"
        csv_path = target_dir / name
        write_csv(csv_path, folder_rows, COLUMNS)

        stray = unmatched_by_folder.get(folder, [])
        if stray:
            write_csv(target_dir / "unmatched_tags.csv", stray, UNMATCHED_COLUMNS)

        priced = [r for r in folder_rows if isinstance(r.get("price"), (int, float))]
        currencies = {r.get("currency") for r in priced if r.get("currency")}
        index.append({
            "folder": folder,
            "csv": str(csv_path.relative_to(out_root)),
            "photos": len({r.get("photo") for r in folder_rows}),
            "wines": len(folder_rows),
            "priced": len(priced),
            "needs_review": sum(1 for r in folder_rows if r.get("needs_review") == "yes"),
            "median_price": _median([float(r["price"]) for r in priced]),
            "currency": "; ".join(sorted(c for c in currencies if c)),
        })

    write_csv(out_root / "index.csv", index, INDEX_COLUMNS)
    return index
