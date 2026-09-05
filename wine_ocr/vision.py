"""Local text recognition through Apple's Vision framework.

Why this is the foundation of the pipeline
------------------------------------------
A vision model downscales anything longer than 2576px on its long edge. These
photos are 4284x5712, so a whole-frame call loses 2.2x of linear resolution —
and the price digits, while large, are not so large that this is free. Vision
runs locally, costs nothing, and can be pointed at the native frame.

Measured on ``data/samples/annabella/IMG_5755.HEIC`` against the hand-read
ground truth, tiling at 1200px:

* every one of the eleven shelf prices came back at confidence 1.0
* the tag's small article line, which names the wine, came back as noise —
  it is ~12px tall and blurred, and is not legible by eye either
* the bottle labels above the rail came back readable

So local OCR is the price source and a strong name *hint*, and the model is
left with the semantic work it is actually needed for.

The heavy lifting is in ``tools/visionocr.swift``; this module runs it, caches
it, and hands back typed rows.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from .cache import ResponseCache

# Tile size handed to Vision. Vision resamples its own input, so this is the
# real resolution knob rather than a cost knob. 1200 was the smallest tile that
# did not start losing whole prices to over-cropping in the sweep; 2600 and 900
# both scored worse.
DEFAULT_MAX_TILE = 1200
DEFAULT_LANGUAGES = ("ro-RO", "en-US")

BINARY = Path(__file__).resolve().parent.parent / "tools" / "visionocr"

# Bumped when the Swift tool changes in a way that alters its output, so cached
# pages are re-read rather than silently reused.
OCR_VERSION = "2026-08-17.1"


class VisionUnavailable(RuntimeError):
    """The Vision helper is missing or will not run on this machine."""


@dataclass(frozen=True)
class OCRLine:
    """One recognised line, in fractions of the oriented image, origin top-left."""

    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float
    height_px: int

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def x_overlap(self, other: "OCRLine") -> float:
        """Shared width as a fraction of the narrower box. 0 when disjoint."""
        lo = max(self.x0, other.x0)
        hi = min(self.x1, other.x1)
        if hi <= lo:
            return 0.0
        return (hi - lo) / max(1e-9, min(self.width, other.width))

    def to_dict(self) -> dict:
        return {
            "t": self.text, "c": self.conf, "h": self.height_px,
            "b": [self.x0, self.y0, self.x1, self.y1],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OCRLine":
        b = d["b"]
        return cls(d["t"], float(d["c"]), *(float(v) for v in b), int(d["h"]))


@dataclass(frozen=True)
class OCRPage:
    """Everything Vision read out of one photo."""

    path: Path
    width: int
    height: int
    lines: tuple[OCRLine, ...]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "error": self.error,
            "lines": [l.to_dict() for l in self.lines],
        }

    @classmethod
    def from_dict(cls, path: Path, d: dict) -> "OCRPage":
        return cls(
            path=path,
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            lines=tuple(OCRLine.from_dict(x) for x in d.get("lines", [])),
            error=d.get("error"),
        )

    def within(self, x0: float, y0: float, x1: float, y1: float) -> list[OCRLine]:
        """Lines whose centre falls inside the given normalised box."""
        return [
            l for l in self.lines
            if x0 <= l.cx <= x1 and y0 <= l.cy <= y1
        ]


def ensure_binary(binary: Path = BINARY) -> Path:
    """Compile the Swift helper on first use.

    Shipping a checked-in binary would be a 120KB blob in git that only runs on
    one architecture, so it is built on demand instead. swiftc is present on any
    Mac with the command line tools.
    """
    src = binary.with_suffix(".swift")
    if binary.exists() and binary.stat().st_mtime >= src.stat().st_mtime:
        return binary
    if not src.exists():
        raise VisionUnavailable(f"missing {src}")
    try:
        subprocess.run(
            ["swiftc", "-O", str(src), "-o", str(binary)],
            check=True, capture_output=True, timeout=600,
        )
    except FileNotFoundError as exc:
        raise VisionUnavailable(
            "swiftc not found — install Xcode command line tools with "
            "`xcode-select --install`"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise VisionUnavailable(
            f"could not build the Vision helper:\n{exc.stderr.decode(errors='replace')}"
        ) from exc
    return binary


def supported_languages(binary: Path = BINARY) -> list[str]:
    out = subprocess.run(
        [str(ensure_binary(binary)), "--list-languages"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.split()


def _cache_key(cache: ResponseCache, sha: str, max_tile: int, upscale: float,
               languages: Sequence[str]) -> str:
    return cache.key(
        "vision", OCR_VERSION, sha, max_tile, f"{upscale:.2f}", ",".join(languages)
    )


def read_pages(
    paths: Sequence[Path],
    shas: Sequence[str],
    cache: ResponseCache,
    *,
    max_tile: int = DEFAULT_MAX_TILE,
    upscale: float = 1.0,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    binary: Path = BINARY,
) -> Iterator[OCRPage]:
    """OCR each path, yielding pages in the order given.

    One subprocess handles every uncached photo: Vision's first call in a
    process pays a model-loading cost that dwarfs the per-photo work, so paying
    it once for the whole run matters more than parallelism here.
    """
    if len(paths) != len(shas):
        raise ValueError("paths and shas must be the same length")

    keys = [_cache_key(cache, s, max_tile, upscale, languages) for s in shas]
    by_path = {str(p): i for i, p in enumerate(paths)}
    cached: dict[int, OCRPage] = {}
    todo: list[int] = []
    for i, key in enumerate(keys):
        hit = cache.get(key)
        if hit is not None:
            cached[i] = OCRPage.from_dict(paths[i], hit)
        else:
            todo.append(i)

    fresh: dict[str, dict] = {}
    if todo:
        for rec in _stream(paths, todo, max_tile, upscale, languages, binary):
            fresh[rec["path"]] = rec
            # Cache as each photo lands rather than at the end. A library this
            # size takes ten minutes to read, and an interrupted run that had
            # cached nothing would start again from zero.
            index = by_path.get(rec["path"])
            if index is not None:
                cache.put(keys[index], OCRPage.from_dict(paths[index], rec).to_dict())

    for i, path in enumerate(paths):
        if i in cached:
            yield cached[i]
            continue
        rec = fresh.get(str(path))
        if rec is None:
            yield OCRPage(path, 0, 0, (), error="visionocr returned no record")
            continue
        yield OCRPage.from_dict(path, rec)


def _stream(
    paths: Sequence[Path],
    todo: Sequence[int],
    max_tile: int,
    upscale: float,
    languages: Sequence[str],
    binary: Path,
) -> Iterator[dict]:
    """Run the helper on ``todo`` and yield each record as it is printed.

    The helper emits one JSON object per line and flushes after each, so reading
    the pipe gives per-photo progress instead of a ten-minute silence followed
    by everything at once.
    """
    cmd = [
        str(ensure_binary(binary)),
        "--max-tile", str(max_tile),
        "--languages", ",".join(languages),
    ]
    if upscale != 1.0:
        cmd += ["--upscale", f"{upscale}"]

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        proc.stdin.write("\n".join(str(paths[i]) for i in todo) + "\n")
        proc.stdin.close()
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    finally:
        proc.stdout.close()
        stderr = proc.stderr.read() if proc.stderr else ""
        if proc.stderr:
            proc.stderr.close()
        code = proc.wait()
        if code != 0:
            raise VisionUnavailable(f"visionocr exited {code}: {stderr[:2000]}")


def read_page(
    path: Path,
    sha: str,
    cache: ResponseCache,
    **kwargs,
) -> OCRPage:
    return next(iter(read_pages([path], [sha], cache, **kwargs)))
