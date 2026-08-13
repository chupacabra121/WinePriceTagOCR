"""Image loading, EXIF, and the crop/tile geometry.

Why tiling exists
-----------------
Measured on the Annabella sample photos (4284x5712 iPhone HEIC): when the whole
frame is downscaled to the 2576px vision limit, the big price digits stay
readable but the small article line naming the wine degrades to mush. Prices
survive a whole-image call; names do not. So the pipeline crops each shelf band
at native resolution and, if the crop is still wider than the limit, splits it
into overlapping tiles. See README "Why two passes".
"""

from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageOps

try:  # HEIC/HEIF is the iPhone default and very common for this task
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    HEIF_AVAILABLE = False

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp",
}

# Long-edge limit for the high-resolution vision tier (Claude Opus 5 / Sonnet 5).
# Larger inputs are downscaled server-side, so sending more is wasted bytes.
MAX_EDGE_HIRES = 2576
# The pre-4.7 limit. Cheaper per image; fine for the layout pass.
MAX_EDGE_STANDARD = 1568


@dataclass(frozen=True)
class PhotoMeta:
    path: Path
    sha256: str
    width: int
    height: int
    taken_at: Optional[str]
    gps_lat: Optional[float]
    gps_lon: Optional[float]


@dataclass(frozen=True)
class Region:
    """A crop of a source photo, in absolute pixels of the oriented image."""

    label: str
    x0: int
    y0: int
    x1: int
    y1: int
    band_index: int
    tile_index: int
    tile_count: int
    # True/False when the tag rail's position is known (band tiling), None when
    # it is not (grid tiling). A tile known to exclude the rail must not be
    # allowed to invent prices, so this is surfaced in the prompt.
    contains_tag_rail: Optional[bool] = None

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def to_full(self, bx0: float, by0: float, bx1: float, by1: float,
                photo_w: int, photo_h: int) -> tuple[float, float, float, float]:
        """Map a 0-1 box within this region back to 0-1 of the whole photo."""
        return (
            (self.x0 + bx0 * self.width) / photo_w,
            (self.y0 + by0 * self.height) / photo_h,
            (self.x0 + bx1 * self.width) / photo_w,
            (self.y0 + by1 * self.height) / photo_h,
        )


def iter_photos(root: Path, recursive: bool = True) -> list[Path]:
    """All image files under ``root``, sorted, skipping hidden files."""
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    globber = root.rglob("*") if recursive else root.glob("*")
    found = [
        p
        for p in globber
        if p.is_file()
        and p.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith(".") for part in p.parts)
    ]
    return sorted(found)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _rational(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return value.numerator / value.denominator
        except Exception:
            return 0.0


def _gps(exif) -> tuple[Optional[float], Optional[float]]:
    """Decode GPSInfo into signed decimal degrees."""
    try:
        from PIL.ExifTags import GPSTAGS

        raw = exif.get_ifd(0x8825)
        if not raw:
            return None, None
        tags = {GPSTAGS.get(k, k): v for k, v in raw.items()}

        def dms(entry, ref, negative_ref) -> Optional[float]:
            if not entry:
                return None
            d, m, s = (_rational(x) for x in entry)
            val = d + m / 60 + s / 3600
            return -val if ref == negative_ref else val

        lat = dms(tags.get("GPSLatitude"), tags.get("GPSLatitudeRef"), "S")
        lon = dms(tags.get("GPSLongitude"), tags.get("GPSLongitudeRef"), "W")
        return lat, lon
    except Exception:
        return None, None


def load_oriented(path: Path) -> Image.Image:
    """Open an image and apply its EXIF orientation.

    Phone photos are almost always stored rotated with an orientation flag; if
    that is not applied the shelves come out vertical and every coordinate the
    model returns is wrong.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def read_meta(path: Path) -> PhotoMeta:
    img = Image.open(path)
    taken_at = None
    lat = lon = None
    try:
        exif = img.getexif()
        if exif:
            for tag in (36867, 36868, 306):  # DateTimeOriginal, Digitized, DateTime
                if raw := exif.get(tag):
                    try:
                        taken_at = datetime.strptime(
                            str(raw), "%Y:%m:%d %H:%M:%S"
                        ).isoformat(sep=" ")
                        break
                    except ValueError:
                        continue
            lat, lon = _gps(exif)
    except Exception:
        pass
    oriented = ImageOps.exif_transpose(img)
    w, h = oriented.size
    return PhotoMeta(
        path=path,
        sha256=sha256_file(path),
        width=w,
        height=h,
        taken_at=taken_at,
        gps_lat=lat,
        gps_lon=lon,
    )


def encode_jpeg(img: Image.Image, max_edge: int, quality: int = 90) -> tuple[str, int, int]:
    """Downscale to ``max_edge`` if needed and return (base64 jpeg, w, h)."""
    w, h = img.size
    longest = max(w, h)
    if longest > max_edge:
        scale = max_edge / longest
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), img.width, img.height


def image_block(b64: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
    }


def estimate_image_tokens(width: int, height: int, max_edge: int) -> int:
    """Approximate vision token cost of an image after downscaling.

    Roughly (w*h)/750, bounded by the tier ceiling.
    """
    longest = max(width, height)
    if longest > max_edge:
        scale = max_edge / longest
        width, height = width * scale, height * scale
    ceiling = 4784 if max_edge > MAX_EDGE_STANDARD else 1600
    return int(min(ceiling, (width * height) / 750))


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def band_regions(
    photo_w: int,
    photo_h: int,
    bands: Iterable,
    max_edge: int = MAX_EDGE_HIRES,
    pad_frac: float = 0.015,
    overlap_frac: float = 0.14,
) -> list[Region]:
    """Turn model-reported shelf bands into concrete, tiled pixel regions.

    Each band is padded slightly, then split into horizontally overlapping tiles
    so that no tile's long edge exceeds ``max_edge`` — i.e. so that no tile is
    downscaled before it reaches the model.
    """
    regions: list[Region] = []
    for i, band in enumerate(bands):
        if not getattr(band, "contains_wine", True):
            continue
        raw_y0 = min(band.bottles_y0, band.tags_y0)
        raw_y1 = max(band.bottles_y1, band.tags_y1)
        # Reject degenerate reports *before* padding, which would otherwise
        # inflate a zero-height sliver into a plausible-looking crop and spend a
        # call on it.
        if raw_y1 - raw_y0 < 0.01 or band.x1 - band.x0 < 0.01:
            continue

        y0 = _clamp(raw_y0 - pad_frac)
        y1 = _clamp(raw_y1 + pad_frac)
        x0 = _clamp(band.x0 - pad_frac)
        x1 = _clamp(band.x1 + pad_frac)

        px0, py0 = int(x0 * photo_w), int(y0 * photo_h)
        px1, py1 = int(x1 * photo_w), int(y1 * photo_h)
        # Absolute pixel span of this band's tag rail, so tiles can report
        # whether they actually contain it.
        rail = (int(band.tags_y0 * photo_h), int(band.tags_y1 * photo_h))
        regions.extend(
            _tile_region(
                px0, py0, px1, py1, i, band.label, max_edge, overlap_frac, rail
            )
        )
    return regions


def _axis_offsets(length: int, budget: int, overlap_frac: float) -> list[tuple[int, int]]:
    """Split ``length`` into overlapping spans of at most ``budget``.

    ``n`` spans of size ``s`` with fractional overlap ``v`` cover
    ``n*s - (n-1)*v*s``, so solve for the smallest ``n`` whose span fits.
    """
    if length <= budget:
        return [(0, length)]
    n = 2
    while True:
        span = length / (n - (n - 1) * overlap_frac)
        if span <= budget or n > 24:
            break
        n += 1
    step = (length - span) / (n - 1)
    return [(int(i * step), int(min(length, i * step + span))) for i in range(n)]


def _tile_region(
    x0: int, y0: int, x1: int, y1: int, band_index: int, label: str,
    max_edge: int, overlap_frac: float,
    tag_rail: Optional[tuple[int, int]] = None,
) -> list[Region]:
    """Split a crop on both axes so no tile's long edge exceeds ``max_edge``.

    Horizontal splitting alone is not enough: a band taller than the limit —
    a close-up, or a single band covering most of the frame — would still be
    downscaled, which is the exact failure tiling exists to prevent.

    Vertical splits are a last resort because they can separate bottles from the
    tags that price them. When that happens the bottom row keeps the tag rail
    (rails sit at the foot of a band), and the upper tiles are marked as
    tag-free so they are told not to guess at prices.
    """
    width, height = x1 - x0, y1 - y0
    if width <= 0 or height <= 0:
        return []

    cols = _axis_offsets(width, max_edge, overlap_frac)
    rows = _axis_offsets(height, max_edge, overlap_frac)
    total = len(cols) * len(rows)

    out: list[Region] = []
    index = 0
    for ry0, ry1 in rows:
        ay0, ay1 = y0 + ry0, y0 + ry1
        if tag_rail is None:
            has_rail: Optional[bool] = None
        else:
            # Any overlap with the rail counts; a sliver of tag is still a tag.
            has_rail = ay0 < tag_rail[1] and ay1 > tag_rail[0]
        for cx0, cx1 in cols:
            name = label if total == 1 else f"{label} (tile {index + 1}/{total})"
            out.append(
                Region(
                    name, x0 + cx0, ay0, x0 + cx1, ay1,
                    band_index, index, total, has_rail,
                )
            )
            index += 1
    return out


def grid_regions(
    photo_w: int,
    photo_h: int,
    max_edge: int = MAX_EDGE_HIRES,
    overlap_frac: float = 0.14,
) -> list[Region]:
    """Fallback tiling when the layout pass is skipped or fails.

    A plain overlapping grid sized so no tile gets downscaled.
    """
    rows = max(1, -(-photo_h // int(max_edge * 0.86)))
    regions: list[Region] = []
    row_h = photo_h / rows
    overlap_px = int(row_h * overlap_frac)
    for r in range(rows):
        y0 = max(0, int(r * row_h) - overlap_px)
        y1 = min(photo_h, int((r + 1) * row_h) + overlap_px)
        regions.extend(
            _tile_region(
                0, y0, photo_w, y1, r, f"grid row {r + 1}/{rows}", max_edge, overlap_frac
            )
        )
    return regions


def crop_region(img: Image.Image, region: Region) -> Image.Image:
    return img.crop((region.x0, region.y0, region.x1, region.y1))


def crop_bbox(
    img: Image.Image,
    bbox: tuple[float, float, float, float],
    pad: float = 0.04,
) -> Image.Image:
    """Cut a review crop from a normalised full-photo box."""
    w, h = img.size
    x0, y0, x1, y1 = bbox
    px = pad * max(x1 - x0, 0.02)
    py = pad * max(y1 - y0, 0.02)
    return img.crop(
        (
            int(_clamp(x0 - px) * w),
            int(_clamp(y0 - py) * h),
            int(_clamp(x1 + px) * w),
            int(_clamp(y1 + py) * h),
        )
    )
