"""On-disk cache of model responses.

Vision calls dominate the cost of a run, and post-processing changes far more
often than prompts do. Keying on image bytes + prompt version + model + region
means you can iterate on normalisation, output columns and review rules all day
without paying for a single extra call.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


class ResponseCache:
    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def key(self, *parts: Any) -> str:
        digest = hashlib.sha256("||".join(str(p) for p in parts).encode()).hexdigest()
        return digest[:40]

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                self.hits += 1
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            self.misses += 1
            return None

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False)
        tmp.replace(path)
