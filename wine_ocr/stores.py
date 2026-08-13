"""Deciding which store a photo belongs to.

Precedence, highest first:

1. ``--store`` on the command line (applies to the whole run)
2. an alias match in ``config/stores.yaml`` against any folder in the path
3. the parent folder name, when it is not a generic bucket like ``photos`` or ``2026``
4. the retailer name the model read off the photo
5. ``Unknown``

Folder naming beats the photo read because a folder is a deliberate statement by
the person who took the pictures, whereas signage is often partly occluded. The
losing candidates are not discarded — every row carries ``store_source`` and
``store_read_from_photo`` so a wrong call is visible and fixable in the sheet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

GENERIC_FOLDERS = {
    "photos", "photo", "pics", "pictures", "images", "img", "samples", "sample",
    "data", "input", "inputs", "wine", "wines", "shelf", "shelves", "new",
    "unsorted", "misc", "tmp", "temp", "downloads", "camera", "dcim", "batch",
    "trip", "visit", "store", "stores", "out", "raw",
}
_DATEISH = re.compile(r"^[\d\W_]+$")


@dataclass
class StoreResolution:
    store: str
    source: str
    read_from_photo: Optional[str]


def load_store_map(path: Optional[Path]) -> dict[str, str]:
    """Load ``alias -> canonical name`` from a YAML config.

    Accepts either ``{canonical: [aliases]}`` or a flat ``{alias: canonical}``.
    """
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    stores = raw.get("stores", raw)
    mapping: dict[str, str] = {}
    if isinstance(stores, dict):
        for key, value in stores.items():
            if isinstance(value, list):
                mapping[_norm(key)] = key
                for alias in value:
                    mapping[_norm(str(alias))] = key
            elif isinstance(value, str):
                mapping[_norm(key)] = value
                mapping[_norm(value)] = value
    return mapping


def _norm(text: str) -> str:
    return re.sub(r"[\s_\-]+", " ", text.strip().lower())


def prettify(folder: str) -> str:
    cleaned = re.sub(r"[\s_\-]+", " ", folder).strip()
    return " ".join(word.capitalize() if word.islower() else word
                    for word in cleaned.split())


def resolve_store(
    photo: Path,
    root: Path,
    override: Optional[str],
    store_map: dict[str, str],
    read_from_photo: Optional[str],
) -> StoreResolution:
    if override:
        return StoreResolution(override, "cli", read_from_photo)

    try:
        parts = list(photo.relative_to(root).parent.parts)
    except ValueError:
        parts = list(photo.parent.parts)

    for part in reversed(parts):
        if canonical := store_map.get(_norm(part)):
            return StoreResolution(canonical, "folder-map", read_from_photo)

    for part in reversed(parts):
        norm = _norm(part)
        if norm and norm not in GENERIC_FOLDERS and not _DATEISH.match(norm):
            return StoreResolution(prettify(part), "folder", read_from_photo)

    if read_from_photo:
        canonical = store_map.get(_norm(read_from_photo), prettify(read_from_photo))
        return StoreResolution(canonical, "photo", read_from_photo)

    return StoreResolution("Unknown", "none", read_from_photo)
